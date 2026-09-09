# AGENTS.md — navodila za Codex / Cursor / AI agent

Namen: po `git clone` čim hitreje poženi PoC, da uporabnik v brskalniku vidi PASS/FAIL galerijo.

## Cilj (definition of done)

1. API odgovori na `http://127.0.0.1:8000/api/health` s `"status":"ok"`.
2. Frontend teče na `http://127.0.0.1:5173`.
3. Na `http://127.0.0.1:5173/zgodovina` (filter linije **DemoEx**) so vidne slike + statusi SPREJETO/ZAVRNJENO.

**Ne** treniraš modelov znova, razen če uporabnik to eksplicitno zahteva. Uteži so že v `weights/`.

## Predpogoji

- Python **3.10+** (`python3 --version`)
- Node **18+** (`node --version`)
- ~3 GB prostora za `.venv` (PyTorch)
- macOS / Linux (Windows: uporabi WSL)

Če manjka Python/Node: povej uporabniku kaj namestiti; ne ugibaj.

## Hitri zagon (točno ta vrstni red)

Delaj iz korena repozitorija.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows WSL: source .venv/bin/activate
pip install -r requirements.txt    # prvič lahko traja 5–15 min — počakaj

cd frontend && npm install && cd ..
```

Zaženi API v ozadju:

```bash
source .venv/bin/activate
export PYTHONPATH=.
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Preveri:

```bash
curl -sf http://127.0.0.1:8000/api/health
```

Napolni galerijo (obvezno — sicer je Rezultati prazna / brez svežih slik):

```bash
source .venv/bin/activate
export PYTHONPATH=.
python scripts/feed_dataset.py samples/demo_examples --limit 20 --line DemoEx
```

Zaženi UI:

```bash
cd frontend && npm run dev -- --host 127.0.0.1 --port 5173
```

Odpri za uporabnika: **http://127.0.0.1:5173/zgodovina** → izberi linijo **DemoEx**.

Alternativa: `bash scripts/quick_start.sh` (API + feed), nato ločeno `cd frontend && npm run dev`.

## Česa NE delaj

- Ne kliči `scripts/setup_demo.sh` / treninga (dolgo, nepotrebno za demo).
- Ne briši `weights/`.
- Ne commitaj `.venv/`, `storage/*.db`, novih datasetov.
- Ne spreminjaj produkcijskih pragov, razen če uporabnik prosi.

## Troubleshooting

| Simptom | Ukrep |
|---------|--------|
| `api/health` connection refused | API ni zagnan ali napačen port |
| `ModuleNotFoundError` | `source .venv/bin/activate` + `pip install -r requirements.txt` |
| `detector/anomaly available: false` | preveri `weights/yolo_gc10.pt` in `weights/patchcore_kolektor2/` |
| Galerija prazna | ponovi `feed_dataset.py … --line DemoEx` |
| Frontend 404 na `/api/…` | Vite proxy: API mora biti na `:8000` |
| Port zaseden | ustavi star proces ali uporabi drug port + povej URL |

## Struktura (orientacija)

- `api/` — FastAPI
- `frontend/` — React (SI UI): Pregled, **Rezultati** (`/zgodovina`), Napake, Analitika
- `weights/` — PoC modeli (v gitu)
- `samples/demo_examples/` — slike za demo feed
- `decision_engine/` — PASS/FAIL/REVIEW logika

## Jezik

Odgovarjaj uporabniku v **slovenščini**, razen če prosi drugače.
