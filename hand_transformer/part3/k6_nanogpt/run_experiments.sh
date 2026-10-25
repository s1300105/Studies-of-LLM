#!/usr/bin/env bash
# K6：比較実験をまとめて回す（nanoGPT のフォルダの中で実行する）。
#   cd ~/nanoGPT && bash /path/to/part3/k6_nanogpt/run_experiments.sh
# 前提：patch_nanogpt.py を当ててあること、data/aozora_char/ を prepare_aozora.py で作ってあること。
# 結果：logs/<名前>.log と out-<名前>/（チェックポイント）。最後に summarize_logs.py で表にする。
# 4コアの CPU で1本あたり約1〜2分（評価を含む）。PY=python3 以外を使うなら PY=... を前に付ける。
set -euo pipefail
PY=${PY:-python3}
mkdir -p logs
COMMON="config/train_shakespeare_char.py --dataset=aozora_char --device=cpu --compile=False
  --eval_iters=200 --log_interval=100 --block_size=64 --batch_size=12
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0"
run () {   # run <名前> <追加の引数...>
  name=$1; shift
  echo "=== $name: $*"
  $PY train.py $COMMON --out_dir=out-$name "$@" > logs/$name.log 2>&1
  grep "^step" logs/$name.log | tail -1
}
run base
run base_s1   --seed=1
run base_s2   --seed=2
run nowpe     --use_wpe=False
run nomask    --causal=False
run head1     --n_head=1
run nolnf     --use_ln_f=False
for name in base nowpe nomask head1 nolnf; do
  echo "=== sample $name"
  $PY sample.py --out_dir=out-$name --device=cpu --num_samples=1 --max_new_tokens=200 --start=吾輩は > logs/sample_$name.txt 2>&1
  tail -n 3 logs/sample_$name.txt
done
