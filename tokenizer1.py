from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

class Tokenizer:
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"
    UYGHUR_CHAGHATAY     = ''' !"#$%&'()*+-./0123456789:;,<=>?@[\\]{}~§©«®±»×÷،؛؟ئابتجخدرزسشغفقكلمنوىيپچژڭگھۆۇۈۋېە‰‹›※€℃№ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳■□▲△◆★ءآأؤإةثحذصضطظعهٱکۀی۰۱۲۳۴۵۶۷۸۹⁘'''
    def __init__(self) -> None:
        self._itos = (self.EOS,) + tuple(self.UYGHUR_CHAGHATAY) + (self.BOS, self.PAD)
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.pad_id = self._stoi[self.PAD]
        self.bos_id = self._stoi[self.BOS]
        self.eos_id = self._stoi[self.EOS]

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> list[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: list[int], join: bool = True) -> str | list[str]:
        tokens = [self._itos[i] for i in token_ids]
        return "".join(tokens) if join else tokens

    def encode(
        self, labels: list[str], device: Optional[torch.device] = None
    ) -> Tensor:
        batch = [
            torch.as_tensor(
                [self.bos_id] + self._tok2ids(label) + [self.eos_id],
                dtype=torch.long,
                device=device,
            )
            for label in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, list[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)
        ids = ids[:eos_idx]
        probs = probs[: eos_idx + 1]
        return probs, ids

    def decode(
        self, token_dists: Tensor, raw: bool = False
    ) -> tuple[list[str], list[Tensor]]:
        """Decode token distributions into labels and per-token probabilities."""
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


def ctc_decode(logits: torch.Tensor, tokenizer: Tokenizer):
    """CTCの推論結果(B, T, Vocab)から重複とBlankを削除して文字列に変換"""
    preds_idx = logits.argmax(dim=-1).cpu().numpy()
    texts = []

    for pred_id_list in preds_idx:
        char_list = []
        for i in range(len(pred_id_list)):
            char_id = pred_id_list[i]
            
            # Blank以外の処理
            if char_id != tokenizer.pad_id:
                # 連続する同じIDは無視
                if i == 0 or char_id != pred_id_list[i - 1]:
                    # 予測された BOS / EOS も文字列からは除外
                    if char_id not in [tokenizer.bos_id, tokenizer.eos_id]:
                        char_list.append(tokenizer._itos[char_id])
        
        texts.append(''.join(char_list))
    return texts
