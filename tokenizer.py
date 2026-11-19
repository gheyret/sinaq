from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

class Tokenizer:
    PAD = "[P]"
    UYGHUR_CHAGHATAY = ''' !"#$%&'()*+-./0123456789:;,<=>?@[\\]{}~§©«®±»×÷،؛؟ئابتجخدرزسشغفقكلمنوىيپچژڭگھۆۇۈۋېە‰‹›※€℃№ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳■□▲△◆★ءآأؤإةثحذصضطظعهٱکۀی۰۱۲۳۴۵۶۷۸۹⁘'''

    def __init__(self) -> None:
        # CTC専用: index 0 を Blank (PAD) に配置（PyTorch / cuDNN CTCLoss の標準仕様）
        self._itos = (self.PAD,) + tuple(self.UYGHUR_CHAGHATAY)
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.pad_id = self._stoi[self.PAD]  # 0

    def __len__(self) -> int:
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> list[int]:
        # 辞書外の文字(OOV)が混入してもKeyErrorにならないようフィルタリング
        return [self._stoi[s] for s in tokens if s in self._stoi]

    def _ids2tok(self, token_ids: list[int], join: bool = True) -> str | list[str]:
        tokens = [self._itos[i] for i in token_ids if i != self.pad_id]
        return "".join(tokens) if join else tokens

    def encode(
        self, labels: list[str], device: Optional[torch.device] = None
    ) -> Tensor:
        """文字列リストをCTCターゲット用のパディング済みテンソル (B, S) に変換"""
        batch = [
            torch.as_tensor(
                self._tok2ids(label),
                dtype=torch.long,
                device=device,
            )
            for label in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def ctc_decode(logits: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    """CTCの推論結果 (B, T, Vocab) から連続重複とBlankを削除して文字列に変換"""
    preds_idx = logits.argmax(dim=-1).cpu().tolist()
    pad_id = tokenizer.pad_id
    itos = tokenizer._itos
    texts = []

    for pred_ids in preds_idx:
        chars = []
        prev_id = -1
        for idx in pred_ids:
            if idx != prev_id and idx != pad_id:
                chars.append(itos[idx])
            prev_id = idx
        texts.append("".join(chars))
    return texts
