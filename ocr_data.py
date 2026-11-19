"""
SVTRv2 向け OCR データパイプライン（パディング不要版）

  MSR (Multi-Scale Resizing)
    高さ 48px 固定。幅はアスペクト比から決め、少数の離散サイズ（バケット幅）に丸める。
    各画像はバケット幅へリサイズされる（比率誤差は最大でも数%）。
  BucketBatchSampler
    同一バケット幅の画像だけで 1 バッチを作る → バッチ内の全画像が同一サイズ。
  collate_fn
    torch.stack のみ。パディングなし、マスクなし、無駄なメモリ確保なし。

使い方:
    samples = load_samples(["train.txt"])
    dset    = OCRDataset(samples, train=True)
    sampler = BucketBatchSampler(dset.widths, width_budget=64 * 400)
    loader  = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn,
                         num_workers=4, pin_memory=True, persistent_workers=True)
    for epoch in range(E):
        for images, texts in loader:          # images: (B, 3, 48, W)  W はバッチ内で共通
            T = images.shape[-1] // 4         # CTC の系列長（全サンプル共通）
"""
import math
import random
from concurrent.futures import ThreadPoolExecutor

from bidi import get_display
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode

IMG_H = 48     # 高さ固定
IMG_W = 1200   # 幅の上限（これを超える行は 1600px にクリップ）
MIN_W = 64     # 幅の下限

MAX_LABEL_LENGTH = 101

def snap_width(w: int) -> int:
    snapped = math.ceil(w / 16) * 16
    return max(MIN_W, min(snapped, IMG_W))

def _read_bucket_width(path: str) -> tuple[int, int]:
    # Image.open は遅延読み込み。サイズ取得だけならデコードしない
    with Image.open(path) as im:
        w, h = im.size
    real_w = min(IMG_W, max(1, round(w * IMG_H / h)))
    return snap_width(real_w), real_w


def scan_widths(samples, num_threads: int = 8) -> np.ndarray:
    """全サンプルの (バケット幅, 実リサイズ幅) を並列取得。大規模データなら .npy に保存して再利用すること。"""
    with ThreadPoolExecutor(num_threads) as ex:
        return np.asarray(list(ex.map(lambda s: _read_bucket_width(s[0]), samples)), dtype=np.int32)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class OCRDataset(Dataset):
    def __init__(self, samples, train: bool = True):
        widths = scan_widths(samples)
        widths = np.asarray(widths, dtype=np.int32)
        assert len(widths) == len(samples)

        self.samples = list(samples)
        self.widths = widths[:, 0]       # バケット幅（サンプラーが参照）
        self.real_widths = widths[:, 1]  # アスペクト比維持の実リサイズ幅
        self.train = train

        base = [T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)]
        if train:
            aug = [
                T.RandomApply([T.RandomAffine(
                    degrees=2, translate=(0.02, 0.02),
                    scale=(0.95, 1.05), shear=3,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            ]
            self.transform = T.Compose(aug + base)
        else:
            self.transform = T.Compose(base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if random.random() < 0.5:    
            img = img.resize((self.real_widths[idx], IMG_H), Image.Resampling.BILINEAR)
            padded = Image.new("RGB", (int(self.widths[idx]), IMG_H), (255,255,255))
            padded.paste(img, (0, 0))
            img = padded
        else:
            img = img.resize((self.widths[idx], IMG_H), Image.Resampling.BILINEAR)

        return self.transform(img), get_display(text, base_dir="R")

# --------------------------------------------------------------------------
# BucketBatchSampler: 同一バケット幅のサンプルだけでバッチを作る
# --------------------------------------------------------------------------
class BucketBatchSampler(Sampler):
    """
    同一バケット幅のサンプルだけでバッチを作るサンプラー。
    バッチ内の画像幅の総和 (bs * W) <= width_budget となるよう
    バッチサイズ bs = max(1, width_budget // W) を動的に決定する。
    """

    def __init__(self, widths, width_budget: int = 64 * 400, shuffle: bool = True):
        widths = np.asarray(widths)
        self.shuffle = shuffle
        self.buckets = [
            (max(1, int(width_budget) // int(w)), np.flatnonzero(widths == w).tolist())
            for w in np.unique(widths)
        ]
        self.num_batches = sum((len(ids) + bs - 1) // bs for bs, ids in self.buckets)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        batches = []
        for bs, ids in self.buckets:
            pool = random.sample(ids, len(ids)) if self.shuffle else ids
            batches.extend(pool[i:i + bs] for i in range(0, len(pool), bs))
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)


# --------------------------------------------------------------------------
# collate_fn: stack のみ
# --------------------------------------------------------------------------
def collate_fn(batch):
    imgs, texts = zip(*batch)
    return torch.stack(imgs), list(texts)


# The following code is from: http://hetland.org/coding/python/levenshtein.py
def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def wer(s1, src):
    sw = src.split()
    return levenshtein(s1.split(),sw), len(sw)

def cer(s1, src):
    return levenshtein(s1,src),len(src)

def cer_wer(preds, targets):
    err_c, lettercnt, err_w, wordcnt = 0,0,0,0
    for pred, target in zip(preds, targets):
        c_er, c_cnt = cer(pred, target)
        w_er, w_cnt = wer(pred, target)
        err_c     += c_er
        lettercnt += c_cnt
        wordcnt   += w_cnt
        err_w     += w_er

    return err_c, lettercnt, err_w, wordcnt

# --------------------------------------------------------------------------
# サンプル読み込み（形式: 画像パス<TAB>ラベル）
# --------------------------------------------------------------------------
def _read_samples(txt_path, max_len):
    samples = []
    with open(txt_path, "rt", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            items = line.split("\t")
            if len(items) < 2:
                continue
            text = items[1].strip()
            if len(text) < max_len:
                samples.append([items[0], text])
    return samples


def load_samples(file_list, max_len=1000):
    files = file_list if isinstance(file_list, (list, tuple)) else [file_list]
    samples = []
    for p in files:
        samples.extend(_read_samples(p, max_len))
    random.shuffle(samples)
    return samples


# --------------------------------------------------------------------------
# 動作確認
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import tempfile
    from torch.utils.data import DataLoader

    tmp = tempfile.mkdtemp()
    rng = random.Random(0)
    samples = []
    for i in range(600):
        w, h = rng.randint(60, 2400), rng.choice([32, 48, 64, 90])
        p = os.path.join(tmp, f"{i}.png")
        Image.new("RGB", (w, h), (255, 255, 255)).save(p)
        samples.append([p, "مۇئاۋىن شۇجىس"])

    dset = OCRDataset(samples, train=True)
    sampler = BucketBatchSampler(dset.widths, width_budget=32 * 400)
    loader = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)

    n = 0
    for images, texts in loader:
        assert images.shape[2] == IMG_H and images.shape[3] % 4 == 0
        n += len(texts)
    print(f"samples={n}/{len(dset)}  batches={len(sampler)}  last shape={tuple(images.shape)}")
