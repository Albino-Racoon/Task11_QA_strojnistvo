# AI Visual Quality Inspector

Industrijski **visual quality-control + anomaly detection** PoC za kovinske/jeklene dele.

```text
izdelek → kamera/upload → YOLO + PatchCore → PASS/FAIL/REVIEW → dashboard
```

## Kaj vključuje

- **Dual-model**: YOLO11 (znane napake, GC10-DET) + PatchCore-inspired anomaly detector (ResNet18 memory bank; kompatibilen z anomalib-workflowom, implementiran lokalno za zanesljiv PoC)
- **Decision engine**: FAIL ob znani napaki; UNKNOWN DEFECT ob visokem anomaly score; sicer REVIEW/PASS
- **FastAPI** inference + zgodovina + statistika
- **React dashboard** v slovenščini (Pregled, Zgodovina, Napake, Analitika)
- **USB / brskalniška kamera**
- Skripte za **download / train / evaluate**

Trenirani baseline uteži (če ste pognali setup):

- `weights/yolo_gc10.pt`
- `weights/patchcore_kolektor/`

## MSDD — litni / strukturirani deli (priporočeno za “proizvodne kose”)

Glej [training/datasets/msdd/README_DOWNLOAD.md](training/datasets/msdd/README_DOWNLOAD.md).

```bash
# po ročnem prenosu MSDD.rar (~9 GB, ScienceDB login, CC BY 4.0):
brew install unar
PYTHONPATH=. python training/prepare_msdd.py --train --epochs 20
curl -X POST http://127.0.0.1:8000/api/models/reload
PYTHONPATH=. python scripts/feed_dataset.py samples/msdd/nok --limit 10
```

## Hitri start (lokalno)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pripravi podatke + modele (synthetic fallback, če javni seti niso dosegljivi)
python training/download_datasets.py --dataset kolektor
python training/train_anomaly.py --dataset kolektor
python training/train_yolo_gc10.py --epochs 5 --batch 4

# Seed demo statistike + API
python api/seed_demo.py
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

V drugem terminalu:

```bash
cd frontend && npm install && npm run dev
```

Odpri http://localhost:5173

## Docker Compose

```bash
docker compose up --build
```

- UI: http://localhost:5173  
- API: http://localhost:8000/docs  

## Trening na javnih datasetih

| Dataset | Uporaba | Licenca |
|---------|---------|---------|
| **GC10-DET** | YOLO detection | glej vir |
| **KolektorSDD** | anomaly (SI industrija) | glej ViCoS |
| **MVTec AD** | samo interni raziskovalni demo | **CC BY-NC-SA 4.0 — NI za komercialno uporabo** |

```bash
python training/download_datasets.py --dataset gc10
python training/download_datasets.py --dataset kolektor
python training/download_datasets.py --dataset mvtec --mvtec-category metal_nut

python training/train_yolo_gc10.py --epochs 30
python training/train_anomaly.py --dataset kolektor
python training/train_anomaly.py --dataset mvtec --category metal_nut

python training/evaluate.py
```

Po treningu: `POST /api/models/reload` ali restart API.

## API

- `POST /api/inspect` — multipart slika
- `POST /api/inspect/camera` — USB zajem na strežniku
- `GET /api/inspections`
- `GET /api/stats/today`
- `GET /api/stats/defects`
- `GET /api/health`

## Metrike (ne “97 % accuracy”)

`training/evaluate.py` izpiše med drugim:

- defect recall
- false reject rate
- false accept rate
- AUROC (anomaly)
- mAP50 (YOLO, če val set obstaja)
- p50 / p95 latenca

## Preklop na podatke podjetja

Glej [training/datasets/customer/README.md](training/datasets/customer/README.md).

1. Shranite normalne slike v `training/datasets/customer/normal/`
2. `python training/train_anomaly.py --dataset customer --output weights/patchcore_customer`
3. Posodobite `ANOMALY` pot v nastavitvah / `.env` in reload

## Struktura

```text
api/  inference/  decision_engine/  training/  frontend/  camera_service/  docker/
```

## Opombe

- Brez uteži API deluje z **demo fallback** (heuristika) — primerno za UI demo, ne za eval.
- Osvetlitev in optika sta v produkciji pogosto pomembnejši od +2 % modela.
- Faza C (process anomaly / PLC) ni v tem PoC.
