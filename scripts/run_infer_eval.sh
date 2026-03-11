#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_xlsum_smoke.yaml}"
OUTPUT_ROOT="${2:-outputs/train/xlsum_smoke}"
RUN_DIR="${3:-}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -dt "$OUTPUT_ROOT"/run-* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "[ERROR] Could not find run directory."
  echo "Usage: $0 <config_path> <output_root> [run_dir]"
  exit 1
fi

INPUT_FILE="$RUN_DIR/input_eval.jsonl"
REFS_FILE="$RUN_DIR/refs_eval.jsonl"
PRED_FILE="$RUN_DIR/preds_eval.jsonl"
REPORT_FILE="$RUN_DIR/eval_report.json"
PER_EXAMPLE_FILE="$RUN_DIR/eval_per_example.csv"
HUMAN_REVIEW_FILE="$RUN_DIR/human_review_template.csv"
ADAPTER_PATH="$RUN_DIR"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "[ERROR] Missing input file: $INPUT_FILE"
  exit 1
fi
if [[ ! -f "$REFS_FILE" ]]; then
  echo "[ERROR] Missing refs file: $REFS_FILE"
  exit 1
fi

echo "[INFO] Using run: $RUN_DIR"

uv run python -m adapter.inference \
  --config "$CONFIG_PATH" \
  --input_file "$INPUT_FILE" \
  --output_file "$PRED_FILE" \
  --adapter_path "$ADAPTER_PATH"

uv run python -m adapter.evaluation \
  --config "$CONFIG_PATH" \
  --pred_file "$PRED_FILE" \
  --ref_file "$REFS_FILE" \
  --report_file "$REPORT_FILE" \
  --per_example_file "$PER_EXAMPLE_FILE" \
  --human_review_file "$HUMAN_REVIEW_FILE"

echo "[DONE] Saved files:"
echo "  - $PRED_FILE"
echo "  - $REPORT_FILE"
echo "  - $PER_EXAMPLE_FILE"
echo "  - $HUMAN_REVIEW_FILE"
