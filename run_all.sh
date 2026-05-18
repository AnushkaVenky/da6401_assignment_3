#!/usr/bin/env bash
#
# run_all.sh - Reproduce every experiment in the W&B report.
# Usage:   bash run_all.sh        # runs everything
#          bash run_all.sh main   # only the canonical training run
#          bash run_all.sh 2.1    # only experiment 2.1, etc.
#
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
TASK="${1:-all}"

echo "[run_all] Task: $TASK"
echo "[run_all] Python: $($PY --version)"

# ---- One-time setup: spaCy models ----
$PY -m spacy download de_core_news_sm 2>/dev/null || true
$PY -m spacy download en_core_web_sm 2>/dev/null || true

run_main() {
  echo "==== Canonical training run (baseline checkpoint) ===="
  $PY train.py --run_name baseline --ckpt_path checkpoint.pt --epochs 15
}

run_2_1() {
  echo "==== 2.1  Noam vs Fixed LR ===="
  $PY experiments/exp_2_1_scheduler.py
}

run_2_2() {
  echo "==== 2.2  1/sqrt(d_k) scaling ablation ===="
  $PY experiments/exp_2_2_scaling.py
}

run_2_3() {
  echo "==== 2.3  Attention head visualisation ===="
  $PY experiments/exp_2_3_attention.py
}

run_2_4() {
  echo "==== 2.4  Positional encoding ablation ===="
  $PY experiments/exp_2_4_posenc.py
}

run_2_5() {
  echo "==== 2.5  Label smoothing ablation ===="
  $PY experiments/exp_2_5_smoothing.py
}

case "$TASK" in
  main) run_main ;;
  2.1)  run_2_1 ;;
  2.2)  run_2_2 ;;
  2.3)  run_2_3 ;;
  2.4)  run_2_4 ;;
  2.5)  run_2_5 ;;
  all)
    run_2_1     # produces ckpt_2_1_noam.pt - reused by 2.3
    run_2_2
    run_2_3
    run_2_4
    run_2_5
    ;;
  *)
    echo "Unknown task '$TASK'.  Choices: all | main | 2.1 | 2.2 | 2.3 | 2.4 | 2.5"
    exit 1
    ;;
esac

echo "[run_all] DONE - all artefacts saved under ./results/ and ckpt_*.pt"
