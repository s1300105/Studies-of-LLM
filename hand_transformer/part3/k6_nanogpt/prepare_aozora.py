#!/usr/bin/env python3
"""K6：青空文庫のテキストを nanoGPT の文字単位データ（train.bin / val.bin / meta.pkl）にする。

nanoGPT の data/shakespeare_char/prepare.py と同じ形式で書き出す（1文字 = 1トークン、uint16、先頭 90% が train）。
違いは、青空文庫の注記を取り除くことだけ。

    python3 prepare_aozora.py --out ~/nanoGPT/data/aozora_char 789_ruby_5639.zip
    python3 prepare_aozora.py --out ~/nanoGPT/data/aozora_char a.zip b.zip c.txt     # 複数ファイルをつなげる
    python3 prepare_aozora.py --out ... --plain *.md                                   # 注記処理なし（UTF-8 のテキスト）

青空文庫のテキスト（「テキストファイル(ルビあり)」の zip の中身）に対して行う処理:
    1. Shift_JIS（cp932）として読む（読めなければ UTF-8）。改行を \\n にそろえる
    2. 冒頭の「題名・著者名」と、--- の行で囲まれた【テキスト中に現れる記号について】の説明を捨てる
    3. 末尾の「底本：」以降（書誌情報）を捨てる
    4. ルビ《…》、ルビの始まりの記号｜、入力者注［＃…］（外字の ※［＃…］ は ※ ごと）を消す
    5. 3行以上続く空行を1行にする
"""
import argparse
import io
import os
import pickle
import re
import zipfile
from collections import Counter

import numpy as np


def read_bytes(path):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                raise SystemExit(f"{path}: zip の中に .txt が無い")
            return z.read(names[0]), f"{path}:{names[0]}"
    with open(path, "rb") as f:
        return f.read(), path


def decode(b):
    for enc in ("cp932", "utf-8"):
        try:
            return b.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", errors="replace"), "utf-8(置換あり)"


SEP = re.compile(r"^-{20,}\s*$", re.M)


def clean_aozora(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    seps = [m for m in SEP.finditer(text)]
    if len(seps) >= 2:                                   # 記号の説明（--- と --- の間）とその前の題名を捨てる
        text = text[seps[1].end():]
    k = text.find("\n底本：")
    if k >= 0:
        text = text[:k]
    text = re.sub(r"※［＃[^］]*］", "", text)            # 外字（説明ごと消す）
    text = re.sub(r"［＃[^］]*］", "", text)               # 入力者注（字下げ・見出し・傍点など）
    text = re.sub(r"《[^》]*》", "", text)                 # ルビ
    text = text.replace("｜", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n") + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="青空文庫の .zip / .txt（--plain なら任意の UTF-8 テキスト）")
    ap.add_argument("--out", required=True, help="出力先（nanoGPT の data/<名前>）")
    ap.add_argument("--plain", action="store_true", help="青空文庫の注記処理をしない")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    parts = []
    for f in args.files:
        b, name = read_bytes(f)
        t, enc = decode(b)
        if not args.plain:
            t = clean_aozora(t)
        print(f"{name}: {enc}、{len(t):,} 文字")
        parts.append(t)
    data = "".join(parts)
    print(f"合計 {len(data):,} 文字")
    print("先頭 120 文字:", repr(data[:120]))
    leftovers = {s: data.count(s) for s in ("《", "》", "｜", "［＃")}
    if any(leftovers.values()) and not args.plain:
        print("注意: 注記の消し残し", leftovers)

    chars = sorted(set(data))
    vocab_size = len(chars)
    assert vocab_size < 65536, "uint16 に収まらない"
    cnt = Counter(data)
    print(f"語彙（異なる文字の数）: {vocab_size:,}")
    print("多い文字 20 個:", " ".join(f"{c!r}:{n}" for c, n in cnt.most_common(20)))
    once = sum(1 for n in cnt.values() if n == 1)
    print(f"1回しか出ない文字: {once:,} 種類")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    n = len(data)
    split = int(n * (1 - args.val_frac))
    train_ids = np.array([stoi[c] for c in data[:split]], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in data[split:]], dtype=np.uint16)
    unseen = set(data[split:]) - set(data[:split])
    print(f"train {len(train_ids):,} トークン、val {len(val_ids):,} トークン（val にだけ出る文字 {len(unseen)} 種類）")
    os.makedirs(args.out, exist_ok=True)
    train_ids.tofile(os.path.join(args.out, "train.bin"))
    val_ids.tofile(os.path.join(args.out, "val.bin"))
    with open(os.path.join(args.out, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)
    with io.open(os.path.join(args.out, "input.txt"), "w", encoding="utf-8") as f:
        f.write(data)
    print(f"書き出し: {args.out}/train.bin, val.bin, meta.pkl, input.txt")


if __name__ == "__main__":
    main()
