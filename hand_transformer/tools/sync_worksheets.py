"""blank.md の記入用の表を、各ステップの手順書（workbook/step*.md）の「## 記入欄」に同期する。

mini_gpt.py --blank で blank.md を作り直したら、これを実行して手順書側の表も更新する:
  python mini_gpt.py --blank > workbook/blank.md && python tools/sync_worksheets.py
ついでに、表や本文中の生の <s>（GitHub では取り消し線タグになる）を &lt;s&gt; に置き換える。
"""
import re
from pathlib import Path

WB = Path(__file__).resolve().parent.parent / "workbook"
HEAD = re.compile(r"^(#+) (\d\.\d) ")


def blank_sections():
    secs, cur = {}, None
    for line in (WB / "blank.md").read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if m and m.group(1) == "##":
            cur = m.group(2)
            secs[cur] = []
        elif cur:
            secs[cur].append(line)
    return {k: "\n".join(v).strip("\n") for k, v in secs.items()}


def escape_s(line):
    parts = line.split("`")
    for i in range(0, len(parts), 2):          # バッククォートの外側だけ
        parts[i] = parts[i].replace("<s>", "&lt;s&gt;")
    return "`".join(parts)


def sync(path, secs):
    lines = path.read_text(encoding="utf-8").splitlines()
    out, i, in_blank, fence, n = [], 0, False, False, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            in_blank = line.strip() == "## 記入欄"
        m = HEAD.match(line)
        if in_blank and m and m.group(1) == "###" and m.group(2) in secs:
            out += [line, "", secs[m.group(2)], ""]
            n += 1
            i += 1
            while i < len(lines) and not lines[i].startswith(("## ", "### ")):
                i += 1
            continue
        out.append(line if fence else escape_s(line))
        i += 1
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    return n


if __name__ == "__main__":
    secs = blank_sections()
    for f in sorted(WB.glob("step*.md")):
        print(f"{f.name}: {sync(f, secs)} 個の記入欄を同期")
