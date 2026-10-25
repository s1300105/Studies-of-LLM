#!/usr/bin/env python3
"""K6：nanoGPT の model.py と train.py に、比較実験用の4つのスイッチを足す（元のファイルは .orig に残す）。

    use_wpe  = True   False にすると位置埋め込み wpe を足さない（x = wte[idx] だけ）
    causal   = True   False にすると因果マスクを外す（未来の文字も見える）
    use_ln_f = True   False にすると最終 LayerNorm（ln_f）を通さない
    seed     = 1337   乱数の種（元の train.py は 1337 に固定）。同じ設定を seed だけ変えて回し、ばらつきの大きさを測る

既定値は元の nanoGPT と同じなので、パッチを当てても何も指定しなければ元と同じ計算になる。
ヘッド数は元から --n_head で変えられる。

    python3 patch_nanogpt.py ~/nanoGPT            # パッチを当てる（2回目以降は「当て済み」と表示して何もしない）
    python3 patch_nanogpt.py ~/nanoGPT --revert   # .orig から元に戻す

書き換える場所は「この文字列がちょうど1回出てくる」ことを確かめてから置き換える。
nanoGPT の版が変わって見つからなければ、どこで止まったかを表示して何も書き換えない。
（確認した版：karpathy/nanoGPT コミット 3adf61e、2025-11-12）
"""
import argparse
import shutil
import sys
from pathlib import Path

MARK = "# [K6 patch]"

MODEL_EDITS = [
    ("    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster\n",
     "    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster\n"
     f"    use_wpe: bool = True  {MARK} False: 位置埋め込みを足さない\n"
     f"    causal: bool = True   {MARK} False: 因果マスクを外す\n"
     f"    use_ln_f: bool = True {MARK} False: 最終 LayerNorm を通さない\n"),
    ("        self.dropout = config.dropout\n",
     "        self.dropout = config.dropout\n"
     f"        self.causal = config.causal {MARK}\n"),
    ("dropout_p=self.dropout if self.training else 0, is_causal=True)",
     f"dropout_p=self.dropout if self.training else 0, is_causal=self.causal) {MARK}"),
    ("            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))\n",
     f"            if self.causal: {MARK}\n"
     "                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))\n"),
    ("        x = self.transformer.drop(tok_emb + pos_emb)\n",
     f"        x = self.transformer.drop(tok_emb + pos_emb if self.config.use_wpe else tok_emb) {MARK}\n"),
    ("        x = self.transformer.ln_f(x)\n",
     f"        if self.config.use_ln_f: {MARK}\n"
     "            x = self.transformer.ln_f(x)\n"),
]

TRAIN_EDITS = [
    ("bias = False # do we use bias inside LayerNorm and Linear layers?\n",
     "bias = False # do we use bias inside LayerNorm and Linear layers?\n"
     f"use_wpe = True {MARK}\ncausal = True {MARK}\nuse_ln_f = True {MARK}\nseed = 1337 {MARK}\n"),
    ("torch.manual_seed(1337 + seed_offset)\n",
     f"torch.manual_seed(seed + seed_offset) {MARK}\n"),
    ("                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line\n",
     "                  bias=bias, vocab_size=None, dropout=dropout, # start with model_args from command line\n"
     f"                  use_wpe=use_wpe, causal=causal, use_ln_f=use_ln_f) {MARK}\n"),
]


def apply(path, edits):
    src = path.read_text(encoding="utf-8")
    if MARK in src:
        print(f"{path.name}: 当て済み（何もしない）")
        return None
    out = src
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            print(f"{path.name}: 置き換え元が {n} 回見つかった（1回のはず）。nanoGPT の版が違う可能性があります:\n  {old!r}")
            sys.exit(1)
        out = out.replace(old, new)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nanogpt_dir")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.nanogpt_dir).expanduser()
    files = {"model.py": MODEL_EDITS, "train.py": TRAIN_EDITS}
    if args.revert:
        for name in files:
            orig = root / (name + ".orig")
            if orig.exists():
                shutil.copy(orig, root / name)
                print(f"{name}: .orig から戻した")
        return
    new = {name: apply(root / name, edits) for name, edits in files.items()}   # 全部確かめてから書く
    for name, text in new.items():
        if text is None:
            continue
        shutil.copy(root / name, root / (name + ".orig"))
        (root / name).write_text(text, encoding="utf-8")
        print(f"{name}: パッチを当てた（元は {name}.orig）")


if __name__ == "__main__":
    main()
