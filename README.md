# AI Visual Quality Inspector

PoC za **vizualni QC** kovinskih/jeklenih delov: upload/kamera → PASS / FAIL / REVIEW → slovenski dashboard.

## Ali lahko hitro poženeš in vidiš rezultate?

**Da** — v repozitoriju so vzorčne slike + predtrenirane PoC uteži (~11 MB).

```bash
git clone https://github.com/Albino-Racoon/Task11_QA_strojnistvo.git
cd Task11_QA_strojnistvo

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # prvič: 5–15 min (PyTorch)

# Terminal 1 — API
PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# Terminal 2 — napolni galerijo z vzorci
PYTHONPATH=. python scripts/feed_dataset.py samples/demo_examples --limit 20 --line DemoEx

# Terminal 3 — UI
cd frontend && npm install && npm run dev
```

Odpri **http://127.0.0.1:5173/zgodovina** → linija **DemoEx**.

Ali v enem koraku (API + feed; frontend še vedno ločeno):

```bash
bash scripts/quick_start.sh
# nato: cd frontend && npm run dev
```

| Kaj dobiš | Opomba |
|-----------|--------|
| Dashboard + galerija slik | takoj po `feed_dataset` |
| Pravi modeli | če so datoteke v `weights/` (so v gitu) |
| Brez uteži | še vedno teče UI z demo heuristiko |

**Zahteve:** Python 3.10+, Node 18+, ~3 GB prostora za venv (torch).

## Struktura

```text
api/  inference/  decision_engine/  frontend/  samples/  weights/  training/
```

## API (kratko)

- `POST /api/inspect` — slika
- `GET /api/inspections` — zgodovina / galerija
- `GET /api/stats/today` — statistika
- `GET /api/health` — status modelov

## Ponovni trening (opcijsko)

Uteži so že vključene. Če jih hočeš zgraditi znova:

```bash
bash scripts/setup_demo.sh
```

Javni seti: GC10-DET, KolektorSDD / KolektorSDD2. MVTec AD je CC BY-NC-SA (ne za komercialno uporabo).

MSDD (ulitki, ~9 GB, ScienceDB): glej navodila v `training/` po ročnem prenosu.

## Preklop na podatke podjetja

1. Normalne slike → `training/datasets/customer/normal/`
2. `python training/train_anomaly.py --dataset customer --output weights/patchcore_customer`
3. Posodobi poti v `.env` / `api/config.py` in `POST /api/models/reload`

## Docker

```bash
docker compose up --build
```

UI: http://localhost:5173 · API: http://localhost:8000/docs  
(uteži se mountajo iz `./weights`)

## Opombe

- Produkcija potrebuje stabilno optiko/osvetlitev in PLC I/O — to ni del tega PoC.
- Eval metrike: `python training/evaluate.py`
