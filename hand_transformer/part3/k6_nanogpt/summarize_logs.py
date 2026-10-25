#!/usr/bin/env python3
"""K6：run_experiments.sh の logs/*.log から、各実験の train / val 損失を表にする。

    python3 summarize_logs.py ~/nanoGPT/logs

表示：step 0 と最後の評価の train / val、val の最小値とそのステップ、1反復の時間（中央値）。
"""
import re
import statistics
import sys
from pathlib import Path

STEP = re.compile(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)")
ITER = re.compile(r"^iter \d+: loss [\d.]+, time ([\d.]+)ms")


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "logs")
    print(f"| 実験 | train（最初） | train（最後） | val（最後） | val の最小（step） | 1反復 |")
    print("|---|---:|---:|---:|---:|---:|")
    for f in sorted(d.glob("*.log")):
        steps, times = [], []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = STEP.match(line)
            if m:
                steps.append((int(m[1]), float(m[2]), float(m[3])))
            m = ITER.match(line)
            if m:
                times.append(float(m[1]))
        if not steps:
            print(f"| {f.stem} | （評価の行が無い） | | | | |")
            continue
        best = min(steps, key=lambda s: s[2])
        t = f"{statistics.median(times):.0f} ms" if times else "—"
        print(f"| {f.stem} | {steps[0][1]:.4f} | {steps[-1][1]:.4f} | {steps[-1][2]:.4f} | {best[2]:.4f}（{best[0]}） | {t} |")


if __name__ == "__main__":
    main()
