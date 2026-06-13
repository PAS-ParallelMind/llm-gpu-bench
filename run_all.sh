#!/usr/bin/env bash
# run_all.sh — run every benchmark in the suite for one GPU, writing results/<op>[_<dtype>]_<gpu>.json.
#
#   ./run_all.sh --c-peak 165  --b-peak 1008                 # RTX 4090 (Ada)
#   ./run_all.sh --c-peak 2250 --b-peak 8000                 # GB200 / B200 (Blackwell)
#   ./run_all.sh --c-peak 165 --b-peak 1008 --device 1 --iters 50   # extra args pass through
#
# --c-peak (theoretical TFLOP/s) and --b-peak (theoretical GB/s) are required and GPU-specific
# (the ceiling cancels in the predictor, but the sweep records it). Each benchmark is
# independent: a failure is reported but does not stop the rest. Run from anywhere.
set -uo pipefail

cd "$(dirname "$0")"

CPEAK="" BPEAK="" PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --c-peak) CPEAK="${2:-}"; shift 2 ;;
    --b-peak) BPEAK="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) PASS+=("$1"); shift ;;
  esac
done
if [ -z "$CPEAK" ] || [ -z "$BPEAK" ]; then
  echo "usage: $0 --c-peak <TFLOP/s> --b-peak <GB/s> [extra run.py args...]" >&2
  exit 1
fi

# Every op_dtype the suite supports (see run.py --bench).
BENCHES=(gemm_bf16 attn_bf16 moe_bf16 moe_mxfp4)

FAILED=()
for b in "${BENCHES[@]}"; do
  echo
  echo "========================= $b ========================="
  if python3 run.py --bench "$b" --c-peak "$CPEAK" --b-peak "$BPEAK" ${PASS[@]+"${PASS[@]}"}; then
    :
  else
    echo "!! $b FAILED (continuing)" >&2
    FAILED+=("$b")
  fi
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "all ${#BENCHES[@]} benchmarks done — results in results/"
else
  echo "done, but failed: ${FAILED[*]}" >&2
  exit 1
fi
