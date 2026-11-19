import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
# from ocrdata import OCRDataset, cer_wer, collate_fn, load_samples
from ocr_data import OCRDataset, BucketBatchSampler, cer_wer, collate_fn, load_samples
from tokenizer import ctc_decode, Tokenizer
from torch.utils.data import DataLoader

_tokenizer = Tokenizer()
vocab_size = len(_tokenizer)

MAX_LR     = 0.0005    # 5e-4（従来の5倍）
INITIAL_LR = 0.00001   # 1e-5
MIN_LR     = 0.000005  # 5e-6

MAX_LR     = 0.0001
INITIAL_LR = 0.000001
MIN_LR     = 0.00001

def custom_lr_lambda(current_step, total_steps):
    # current_step は optimizer.step() が呼ばれた回数
    pct = current_step / total_steps
    
    if pct < 0.10:
        lr_range = MAX_LR - INITIAL_LR
        current_lr = INITIAL_LR + (lr_range * pct / 0.10)
        return current_lr / INITIAL_LR
    elif pct < 0.8:
        progress = (pct - 0.10) / (0.8 - 0.10)
        cosine_decay = (1 - np.cos(np.pi * progress)) / 2
        current_lr = MAX_LR - ((MAX_LR - MIN_LR) * cosine_decay)
        return current_lr / INITIAL_LR
    else:
        return MIN_LR / INITIAL_LR

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
    labels = _tokenizer.encode(texts, device)
    label_lengths = torch.tensor([len(t) + 2 for t in texts], dtype=torch.long, device=device)
    logits_for_loss = logits.log_softmax(2).permute(1, 0, 2)
    B, T, _ = logits.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    return loss_fn(logits_for_loss, labels, input_lengths, label_lengths)

def check_and_clear_vram(threshold_ratio=0.9):
  """VRAMの使用率が threshold_ratio を超えた場合にキャッシュをクリアする"""
  free_memory, total_memory = torch.cuda.mem_get_info()
  used_memory = total_memory - free_memory
  usage_ratio = used_memory / total_memory
  if usage_ratio > threshold_ratio:
    torch.cuda.empty_cache()


def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=True)
    step = 0
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        current_lr = scheduler.get_last_lr()[0]
        msg["loss"] = f"{vloss:.4f}"
        msg["avg loss"] = f"{total_loss / (i + 1):.4f}"
        msg["lr"] = f"{current_lr:.7f}"
        pbar.set_postfix(msg)

        if step % 10 == 0:
            check_and_clear_vram()

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
            preds = ctc_decode(logits.float(),_tokenizer)
            
            e_char_cnt, char_cnt, e_word_cnt, word_cnt = cer_wer(preds,texts)

            e_chars += e_char_cnt
            e_words += e_word_cnt

            chars += char_cnt
            words += word_cnt

            msg = f" [CER:{e_chars/chars:.3%} ({e_chars}/{chars} letters) WER:{e_words/words:.3%} ({e_words}/{words} words)]"
            pbar.set_description(msg)

    return e_chars / chars, e_words / words


def load_modelsvtrv2fa(ckpt_path):
    import yaml
    from svtrv2fa import PutukSVTRV2FA
    type ="small"
    with open("svtrv2fa.yml", "r") as f:
        config = yaml.safe_load(f)

    cfg = config[type]
    cfg["num_classes"] = vocab_size
    model = PutukSVTRV2FA(cfg)
    if os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base_state = payload["model_state"]
        model.load_state_dict(base_state)
        print(f"チェックポイント {ckpt_path} から読み込みました。")

    return model, cfg


# ============================================================
# メイン実行ブロック
# ============================================================
def main(train_loader, val_loader, EPOCHS):
    last_model = f"chkpt/putuksvtrv2fa_last.pt"
    best_model = f"chkpt/putuksvtrv2fa_best.pt"
    train_state = f"chkpt/putuksvtrv2fa_state.pt"
    model, cfg = load_modelsvtrv2fa(last_model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametes: {num_params / 1e6:.2f} M")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    t_state = {}
    if os.path.exists(train_state):
        t_state = torch.load(train_state, map_location="cpu", weights_only=False)

    opt_param = t_state.get("optimizer", None)
    sc_param = t_state.get("scheduler", None)
    lastepoch = t_state.get("epoch", 0)
    best_cer = t_state.get("best_cer", float("inf"))

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    if opt_param:
        optimizer.load_state_dict(opt_param)
        pass

    scheduler = create_scheduler(optimizer,train_loader,EPOCHS)
    if sc_param:
        scheduler.load_state_dict(sc_param)
        pass

    loss_fn = nn.CTCLoss(blank=_tokenizer.pad_id, reduction='mean', zero_infinity=True).to(device)
    print(f"Best CER : {best_cer:.3%}%")
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

if __name__ == '__main__':
    train_epochs = 30
    MAX_LABEL_LENGTH = 120

    main(train_loader, val_loader, train_epochs)
    torch.cuda.empty_cache()
