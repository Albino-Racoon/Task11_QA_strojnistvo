#!/usr/bin/env bash
# Prenos datasetov + trening modelov (daljši tek, če uteži še niso v weights/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

echo "==> Dataseti"
python training/download_datasets.py --dataset kolektor || true
python training/download_datasets.py --dataset gc10 || true

echo "==> Trening"
python training/train_anomaly.py --dataset kolektor
python training/train_yolo_gc10.py --epochs "${EPOCHS:-10}" --batch "${BATCH:-4}" --imgsz "${IMGSZ:-320}"

echo "==> Eval"
python training/evaluate.py || true

echo "Končano. Uteži so v weights/. Za demo: bash scripts/quick_start.sh"
