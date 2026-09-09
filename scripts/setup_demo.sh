#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

echo "==> Dataseti"
python training/download_datasets.py --dataset kolektor || true
python training/download_datasets.py --dataset gc10 || true

echo "==> Trening"
python training/train_anomaly.py --dataset kolektor
python training/train_yolo_gc10.py --epochs "${EPOCHS:-10}" --batch "${BATCH:-4}" --imgsz "${IMGSZ:-320}"

echo "==> Eval"
python training/evaluate.py

echo "==> Seed"
python api/seed_demo.py

echo "Končano. Zaženi API: PYTHONPATH=. uvicorn api.main:app --reload --port 8000"
echo "Frontend: cd frontend && npm run dev"
