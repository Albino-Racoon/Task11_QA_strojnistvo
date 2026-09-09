#!/usr/bin/env bash
# Hitri lokalni demo: API + frontend + vzorčne inšpekcije v galerijo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

if [[ ! -d .venv ]]; then
  echo "==> Ustvarjam .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Python odvisnosti"
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [[ ! -d frontend/node_modules ]]; then
  echo "==> npm install"
  (cd frontend && npm install)
fi

if [[ ! -f weights/yolo_gc10.pt ]] || [[ ! -f weights/patchcore_kolektor2/memory_bank.npy ]]; then
  echo "OPOZORILO: manjkajo uteži v weights/ — API bo v demo načinu."
  echo "Za polne modele zaženi: bash scripts/setup_demo.sh"
fi

mkdir -p storage/images storage/heatmaps storage/defective-images

echo "==> Zaganjam API na :8000"
pkill -f "uvicorn api.main:app" 2>/dev/null || true
PYTHONPATH=. uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

echo "==> Čakam na API…"
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null; then
    break
  fi
  sleep 1
done
curl -sf http://127.0.0.1:8000/api/health | head -c 200
echo

echo "==> Pošiljam demo vzorce v /api/inspect"
python scripts/feed_dataset.py samples/demo_examples --limit 20 --line DemoEx

echo ""
echo "Pripravljeno."
echo "  1) V drugem terminalu:  cd frontend && npm run dev"
echo "  2) Odpri:              http://127.0.0.1:5173/zgodovina"
echo "     (filter linije: DemoEx)"
echo ""
echo "API teče v ozadju (pid $API_PID). Ustavi z Ctrl+C."
wait $API_PID
