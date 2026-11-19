import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from ocr_data import BucketBatchSampler, OCRDataset, collate_fn, cer_wer, load_samples,IMG_H,IMG_W
from tokenizer import ctc_decode, Tokenizer
import yaml


# MAX_LR     = 0.0005    # 5e-4（従来の5倍）
# INITIAL_LR = 0.00001   # 1e-5
# MIN_LR     = 0.000005  # 5e-6

MAX_LR     = 0.0001
INITIAL_LR = 0.000001
MIN_LR     = 0.000001

tokenizer = Tokenizer()
vocab_size = len(tokenizer)

# def custom_lr_lambda(current_step, total_steps):
#     # current_step は optimizer.step() が呼ばれた回数
#     pct = current_step*5 / total_steps
    
#     if pct < 0.10:
#         lr_range = MAX_LR - INITIAL_LR
#         current_lr = INITIAL_LR + (lr_range * pct / 0.10)
#         return current_lr / INITIAL_LR
#     elif pct < 0.8:
#         progress = (pct - 0.10) / (0.8 - 0.10)
#         cosine_decay = (1 - np.cos(np.pi * progress)) / 2
#         current_lr = MAX_LR - ((MAX_LR - MIN_LR) * cosine_decay)
#         return current_lr / INITIAL_LR
#     else:
#         return 1.0


def custom_lr_lambda(current_step, total_steps):
    # current_step は optimizer.step() が呼ばれた回数
    pct = current_step*1.2 / total_steps
    
    if pct < 0.10:
        lr_range = MAX_LR - INITIAL_LR
        current_lr = INITIAL_LR + (lr_range * pct / 0.10)
        return current_lr / INITIAL_LR
    else:
        progress = min(1.0, (pct - 0.10) / 0.90)
        current_lr = MAX_LR - ((MAX_LR - MIN_LR) * progress)
        return current_lr / INITIAL_LR

# スケジューラ作成時の例
def create_scheduler(optimizer, dataloader, epochs):
    # 1エポックあたりの「実際の更新回数」を計算
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * epochs
    
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: custom_lr_lambda(step, total_steps)
    )
    return scheduler

def calculate_loss(logits, texts, device, loss_fn):
    labels = tokenizer.encode(texts, device)
    label_lengths = (labels != tokenizer.pad_id).sum(dim=1)
    logits_for_loss = logits.log_softmax(2).permute(1, 0, 2)
    B, T, _ = logits.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    return loss_fn(logits_for_loss, labels, input_lengths, label_lengths)

def check_and_clear_vram(threshold_ratio=0.85):
  """VRAMの使用率が threshold_ratio を超えた場合にキャッシュをクリアする"""
  free_memory, total_memory = torch.cuda.mem_get_info()
  used_memory = total_memory - free_memory
  usage_ratio = used_memory / total_memory
  if usage_ratio > threshold_ratio:
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 2. 学習ループ（手動カウントの step を正しく更新）
# ---------------------------------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=True)
    step = 0  # 重み更新の回数（optimizer.step が実行された回数）
    msg = {}
    
    optimizer.zero_grad()
    
    for i, (images, texts) in enumerate(pbar):
        images = images.to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(images)
        
        loss = calculate_loss(logits.float(), texts, device, loss_fn)        
        loss.backward()
        vloss = loss.item()
        total_loss += vloss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # ここで内部の step カウントが +1 される
        optimizer.zero_grad()
        step += 1
        
        current_lr = scheduler.get_last_lr()[0]
        msg["loss"] = f"{vloss:.4f}"
        msg["avg loss"] = f"{total_loss / (i + 1):.4f}"
        msg["lr"] = f"{current_lr:.7f}"
        pbar.set_postfix(msg)
        if step%5 == 0:
            check_and_clear_vram(0.9)

    return total_loss / len(dataloader)

# ============================================================
# 評価ループ (Evaluate) BF16対応版
# ============================================================
def evaluate(model, dataloader, device):
    model.eval()
    e_chars = 0
    e_words = 0
    chars   = 0
    words   = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, texts in pbar:
            images = images.to(device)
            
            # --- BF16 自動混合精度コンテキスト ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
            
            # デコード処理に渡す前に float32 に戻すのが安全
            preds = ctc_decode(logits.float(), tokenizer)
            
            e_char_cnt, char_cnt, e_word_cnt, word_cnt = cer_wer(preds,texts)

            e_chars += e_char_cnt
            e_words += e_word_cnt

            chars += char_cnt
            words += word_cnt

            msg = f" [CER:{e_chars/chars:.3%} ({e_chars}/{chars} letters) WER:{e_words/words:.3%} ({e_words}/{words} words)]"
            pbar.set_description(msg)

    return e_chars / chars, e_words / words


def print_parameter_count(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Total: {total_num/1000000:.4f} M, Trainable: {trainable_num/1000000:.4f} M')

def load_model(modelName):
    from bestocr import load_config, BestOCR
    print("="*60)
    model = BestOCR(modelName,load_config("bestocr.yaml", "small", num_class=vocab_size))
    cfg = model.config
    if os.path.exists(model.lastName):
        payload = torch.load(model.lastName, map_location="cpu", weights_only=False)
        base_state = payload["model_state"]
        model.load_state_dict(base_state)
        print(f"チェックポイント {model.lastName} から読み込みました。")
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametes: {num_params / 1e6:.2f} M")

    return model, cfg

def load_cvit(modelName):
    from cvit import CViT
    print("="*60)
    with open("cvit.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg["model"]
    cfg["num_classes"] = vocab_size
    model = CViT(modelName, **cfg)
    if os.path.exists(model.lastName):
        payload = torch.load(model.lastName, map_location="cpu", weights_only=False)
        base_state = payload["model_state"]
        model.load_state_dict(base_state)
        print(f"チェックポイント {model.lastName} から読み込みました。")
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametes: {num_params / 1e6:.2f} M")

    return model, cfg


def load_vit(modelName):
    from vit import ViTCTC
    print("="*60)
    with open("vit.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg["model_small"]
    cfg["num_classes"] = vocab_size
    model = ViTCTC(modelName, **cfg)
    if os.path.exists(model.lastName):
        payload = torch.load(model.lastName, map_location="cpu", weights_only=False)
        base_state = payload["model_state"]
        model.load_state_dict(base_state)
        print(f"チェックポイント {model.lastName} から読み込みました。")
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametes: {num_params / 1e6:.2f} M")

    return model, cfg

def load_viptr(modelName):
    from viptrv1 import VTPTRNet
    print("="*60)
    with open("viptrv1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["viptrv1_m"]

    cfg["num_classes"] = vocab_size
    model = VTPTRNet(modelName, **cfg)
    if os.path.exists(model.lastName):
        payload = torch.load(model.lastName, map_location="cpu", weights_only=False)
        base_state = payload["model_state"]
        model.load_state_dict(base_state)
        print(f"チェックポイント {model.lastName} から読み込みました。")
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametes: {num_params / 1e6:.2f} M")

    return model, cfg

def load_fno(modelName):
    from linefno import LineFNO
    print("="*60)
    cfg_path = "linefno.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]
    cfg["num_classes"] = vocab_size
    model = LineFNO(modelName, cfg)
    if os.path.exists(model.lastName):
        payload = torch.load(model.lastName, map_location="cpu", weights_only=False)
        base_state = payload["model_state"]
        model.load_state_dict(base_state,strict=False)
        print(f"チェックポイント {model.lastName} から読み込みました。")
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametes: {num_params / 1e6:.2f} M")

    return model, cfg


def load_modelsvtrv2fa(modelName):
    import yaml
    print("="*60)
    from svtrv2fa import PutukSVTRV2FA
    type ="small"
    with open("svtrv2fa.yml", "r") as f:
        config = yaml.safe_load(f)

    cfg = config[type]
    cfg["num_classes"] = vocab_size
    model = PutukSVTRV2FA(modelName,cfg)
    if os.path.exists(model.lastName):
        payload = torch.load(model.lastName, map_location="cpu", weights_only=False)
        base_state = payload["model_state"]
        model.load_state_dict(base_state)
        print(f"\nチェックポイント {model.lastName} から読み込みました。")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametes: {num_params / 1e6:.2f} M")

    return model, cfg

    
# ============================================================
# メイン実行ブロック
# ============================================================
def main(model, cfg, train_loader, val_loader, EPOCHS):

    print(f"Start training {model.modelName} model...")
    t_state = {}
    train_state = model.stateName
    best_model  = model.bestName
    last_model  = model.lastName

    if os.path.exists(train_state):
        t_state = torch.load(train_state, map_location="cpu", weights_only=False)

    opt_param = t_state.get("optimizer", None)
    sc_param = t_state.get("scheduler", None)
    lastepoch = t_state.get("epoch", 0)
    best_cer = t_state.get("best_cer", float("inf"))

    # EPOCHS = lastepoch+3

    print(f"Trained: {lastepoch} epoches")
    print(f"Best CER : {best_cer:.3%}%")

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    if opt_param:
        optimizer.load_state_dict(opt_param)
        # for pg in optimizer.param_groups:
        #     pg["initial_lr"] = INITIAL_LR        
        pass

    scheduler = create_scheduler(optimizer,train_loader,EPOCHS)
    if sc_param:
        scheduler.load_state_dict(sc_param)
        # scheduler.base_lrs = [INITIAL_LR for _ in optimizer.param_groups]

    loss_fn = nn.CTCLoss(blank=tokenizer.pad_id, reduction='mean', zero_infinity=True).to(device)
    # best_cer, wer_val = evaluate(model, val_loader, device)
    # lastepoch = 10
    for epoch in range(lastepoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        torch.cuda.empty_cache()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
        torch.cuda.empty_cache()
        cer_val, wer_val = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val CER: {cer_val:.3%}% | Val WER: {wer_val:.3%}%")
        
        if cer_val < best_cer:
            best_cer = cer_val
            torch.save({"config": cfg, "model_state": model.state_dict()}, best_model)
            print(f"New best model saved! (CER: {best_cer:.3%}%)")
    
        torch.save({"config": cfg, "model_state": model.state_dict()}, last_model)
        torch.save({"epoch": (epoch + 1),"best_cer": best_cer,"optimizer": optimizer.state_dict(),"scheduler": scheduler.state_dict()}, train_state)

        # if (epoch+1) >=8:
        #     break

if __name__ == '__main__':
    os.makedirs("chkpt", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_epochs = 10

