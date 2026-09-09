# AI Visual Quality Inspector

PoC za **vizualni QC** kovinskih/jeklenih delov: upload/kamera → PASS / FAIL / REVIEW → slovenski dashboard.

## Za šefa: zagon z Codexom (priporočeno)

1. Prenesi / odpri repo v Cursorju (ali Codex CLI).
2. Odpri chat z agentom in prilepi **točno to**:

```text
Poženi ta PoC po navodilih v AGENTS.md.
Ne treniraš modelov. Ko je pripravljeno, odpri http://127.0.0.1:5173/zgodovina
(filter linije DemoEx) in povej mi, da lahko gledam rezultate.
```

3. Počakaj na prvi `pip install` (5–15 min zaradi PyTorch).
4. Ko agent reče, da je OK → odpri povezavo zgoraj.

Agent bere `AGENTS.md` (koraki, preverjanje, troubleshooting).

## Ročni zagon (brez AI)

```bash
git clone https://github.com/Albino-Racoon/Task11_QA_strojnistvo.git
cd Task11_QA_strojnistvo

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1
PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# Terminal 2
PYTHONPATH=. python scripts/feed_dataset.py samples/demo_examples --limit 20 --line DemoEx

# Terminal 3
cd frontend && npm install && npm run dev
```

Odpri **http://127.0.0.1:5173/zgodovina** → linija **DemoEx**.

Ali: `bash scripts/quick_start.sh` nato `cd frontend && npm run dev`.

| Kaj dobiš | Opomba |
|-----------|--------|
| Dashboard + galerija slik | po `feed_dataset` |
| Pravi modeli | datoteke v `weights/` (so v gitu) |
| Brez uteži | UI še vedno teče (demo način) |

**Zahteve:** Python 3.10+, Node 18+, ~3 GB za venv.

## Struktura

```text
AGENTS.md   ← navodila za AI agent (Codex/Cursor)
api/  frontend/  weights/  samples/  decision_engine/  training/
```

## API (kratko)

- `POST /api/inspect` — slika
- `GET /api/inspections` — zgodovina / galerija
- `GET /api/stats/today` — statistika
- `GET /api/health` — status modelov

## Ponovni trening (opcijsko, dolgo)

```bash
bash scripts/setup_demo.sh
```

Samo če hočeš zgraditi uteži znova. Za demo **ni treba**.

## Preklop na podatke podjetja

1. Normalne slike → `training/datasets/customer/normal/`
2. `python training/train_anomaly.py --dataset customer --output weights/patchcore_customer`
3. Posodobi poti v `.env` / `api/config.py` in `POST /api/models/reload`

## Docker

```bash
docker compose up --build
```

UI: http://localhost:5173 · API: http://localhost:8000/docs

## Opombe

- Produkcija potrebuje stabilno optiko/osvetlitev in PLC I/O — to ni del tega PoC.
- Eval: `python training/evaluate.py`
