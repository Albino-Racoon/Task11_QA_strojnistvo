# Proizvodni deli — tvoje fotografije

## Hitri demo (brez treninga)
1. Kopiraj slike kosov v `neoznaceno/` (ali `ok/` / `nok/` če že veš rezultat).
2. V UI: Pregled → Izberi sliko.
3. Ali batch:
   ```bash
   PYTHONPATH=. python scripts/feed_dataset.py samples/proizvodni_deli/neoznaceno --line "Linija A"
   ```

Trenutni modeli (GC10 + Kolektor) bodo dali grob PASS/FAIL/REVIEW.
Za resen rezultat potrebuješ fine-tune na tvojih kosih (spodaj).

## Priprava za trening na vaših kosih
Kopiraj:
- dobre kose → `training/datasets/customer/normal/`  (cilj: 500–5000)
- slabe kose → `training/datasets/customer/defect/`  (cilj: 50–500)

Nato:
```bash
PYTHONPATH=. python training/train_anomaly.py --dataset customer --output weights/patchcore_customer
# v .env / api/config: anomaly_weights → weights/patchcore_customer
curl -X POST http://127.0.0.1:8000/api/models/reload
```
