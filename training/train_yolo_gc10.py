#!/usr/bin/env python3
"""Prepare GC10 YOLO dataset and train YOLO11n."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.detector.yolo_detector import GC10_CLASSES

DATA = ROOT / "training" / "datasets" / "gc10"
YOLO_DIR = ROOT / "training" / "datasets" / "gc10_yolo"
WEIGHTS_OUT = ROOT / "weights" / "yolo_gc10.pt"

# Roboflow / alternate naming → canonical GC10 class
CLASS_NAME_MAP = {
    "1_punching_hole": "punching_hole",
    "2_welding_line": "welding_line",
    "3_crescent_gap": "crescent_gap",
    "4_water_spot": "water_spot",
    "5_oil_spot": "oil_spot",
    "6_silk_spot": "silk_spot",
    "7_inclusion": "inclusion",
    "8_rolled_pit": "rolled_pit",
    "9_crease": "crease",
    "10_waist_folding": "waist_folding",
    "punching_hole": "punching_hole",
    "welding_line": "welding_line",
    "crescent_gap": "crescent_gap",
    "water_spot": "water_spot",
    "oil_spot": "oil_spot",
    "silk_spot": "silk_spot",
    "inclusion": "inclusion",
    "rolled_pit": "rolled_pit",
    "crease": "crease",
    "waist_folding": "waist_folding",
}


def _map_class(name: str) -> str | None:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[key]
    for n in GC10_CLASSES:
        if n in key or key in n:
            return n
    # strip leading index "1_..."
    if "_" in key and key.split("_", 1)[0].isdigit():
        return _map_class(key.split("_", 1)[1])
    return None


def _coco_to_yolo(images_dir: Path, ann_file: Path, split_out: Path, class_names: list[str]) -> int:
    data = json.loads(ann_file.read_text())
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    name_to_id = {n: i for i, n in enumerate(class_names)}
    images = {im["id"]: im for im in data["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out = split_out / "images"
    lbl_out = split_out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, im in images.items():
        file_name = im["file_name"]
        src = images_dir / file_name
        if not src.exists():
            matches = list(images_dir.rglob(Path(file_name).name))
            if not matches:
                continue
            src = matches[0]
        dst = img_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            arr = cv2.imread(str(src))
            if arr is None:
                continue
            h, w = arr.shape[:2]
        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_name = cats.get(ann["category_id"], "")
            key = _map_class(str(cat_name))
            if key is None or key not in name_to_id:
                continue
            cls = name_to_id[key]
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (lbl_out / f"{dst.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        count += 1
    return count


def _find_coco_splits(root: Path) -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    for split in ("train", "valid", "val", "test"):
        for ann_name in ("_annotations.coco.json", "annotations.json", f"instances_{split}.json"):
            for ann in root.rglob(ann_name):
                parent = ann.parent
                if parent.name != split and not (split == "valid" and parent.name == "valid"):
                    continue
                mapped = "val" if parent.name in {"valid", "val", "test"} else "train"
                found.append((mapped, parent, ann))
    uniq = {}
    for mapped, images_dir, ann in found:
        uniq[str(ann)] = (mapped, images_dir, ann)
    return list(uniq.values())


def build_synthetic_yolo(out_dir: Path, n: int = 80) -> None:
    """Tiny synthetic detection set so training always works offline."""
    rng = np.random.default_rng(42)
    for split, n_split in [("train", int(n * 0.8)), ("val", max(1, int(n * 0.2)))]:
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_split):
            img = np.full((320, 320, 3), 150, np.uint8)
            img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
            cls = int(rng.integers(0, len(GC10_CLASSES)))
            x1, y1 = int(rng.integers(20, 160)), int(rng.integers(20, 160))
            x2, y2 = x1 + int(rng.integers(40, 100)), y1 + int(rng.integers(20, 80))
            x2, y2 = min(300, x2), min(300, y2)
            color = (20, 20, 20)
            if GC10_CLASSES[cls] in {"crease", "welding_line"}:
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            name = f"syn_{i:03d}.jpg"
            cv2.imwrite(str(img_dir / name), img)
            cx = ((x1 + x2) / 2) / 320
            cy = ((y1 + y2) / 2) / 320
            bw = (x2 - x1) / 320
            bh = (y2 - y1) / 320
            (lbl_dir / f"syn_{i:03d}.txt").write_text(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _parquet_mask_to_yolo(parquet_files: list[Path], out_dir: Path, class_names: list[str]) -> int:
    """Convert Kelvin878-style parquet (image, guide mask, text) to YOLO boxes."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Potreben je pandas+pyarrow za parquet GC10") from exc

    name_to_id = {n: i for i, n in enumerate(class_names)}
    rows = []
    for pf in parquet_files:
        rows.append(pd.read_parquet(pf))
    import pandas as pd

    df = pd.concat(rows, ignore_index=True)
    idxs = list(range(len(df)))
    random.Random(42).shuffle(idxs)
    split_at = max(1, int(len(idxs) * 0.85))
    split_map = {i: ("train" if n < split_at else "val") for n, i in enumerate(idxs)}

    count = 0
    for i, row in df.iterrows():
        split = split_map[int(i)]
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        label = _map_class(str(row.get("text", "")))
        if label is None or label not in name_to_id:
            continue
        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        name = f"gc10_{i:05d}.jpg"
        cv2.imwrite(str(img_dir / name), img)

        lines: list[str] = []
        guide = row.get("guide")
        if isinstance(guide, dict) and guide.get("bytes"):
            garr = np.frombuffer(guide["bytes"], dtype=np.uint8)
            mask = cv2.imdecode(garr, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = max(1, x2 - x1) / w
                    bh = max(1, y2 - y1) / h
                    lines.append(f"{name_to_id[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            # whole-image weak box as last resort
            lines.append(f"{name_to_id[label]} 0.5 0.5 0.8 0.8")
        (lbl_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n")
        count += 1
    return count


def prepare_dataset() -> Path:
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        shutil.rmtree(YOLO_DIR / split, ignore_errors=True)

    splits = _find_coco_splits(DATA) if DATA.exists() else []
    converted = 0
    if splits:
        for mapped, images_dir, ann in splits:
            n = _coco_to_yolo(images_dir, ann, YOLO_DIR / mapped, GC10_CLASSES)
            converted += n
            print(f"Pretvorjenih {n} slik ({mapped}) iz {ann}")
    else:
        parquet_files = list(DATA.rglob("*.parquet")) if DATA.exists() else []
        if parquet_files:
            converted = _parquet_mask_to_yolo(parquet_files, YOLO_DIR, GC10_CLASSES)
            print(f"Pretvorjenih {converted} slik iz parquet GC10")
        else:
            print("GC10 COCO/parquet ni najden — uporabljam synthetic set za baseline trening.")
            build_synthetic_yolo(YOLO_DIR)
            converted = 1

    if converted == 0 or not (YOLO_DIR / "train" / "images").exists():
        print("GC10 pretvorba prazna — synthetic fallback.")
        build_synthetic_yolo(YOLO_DIR)

    if not (YOLO_DIR / "val" / "images").exists():
        shutil.copytree(YOLO_DIR / "train", YOLO_DIR / "val", dirs_exist_ok=True)

    data_yaml = {
        "path": str(YOLO_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(GC10_CLASSES)},
    }
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml, sort_keys=False))
    return yaml_path


def train(epochs: int = 30, imgsz: int = 640, batch: int = 8, model: str = "yolo11n.pt") -> Path:
    yaml_path = prepare_dataset()
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "training" / "runs"),
        name="gc10_yolo",
        exist_ok=True,
        patience=10,
    )
    best = ROOT / "training" / "runs" / "gc10_yolo" / "weights" / "best.pt"
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, WEIGHTS_OUT)
        print(f"Uteži shranjene: {WEIGHTS_OUT}")
    else:
        last = best.with_name("last.pt")
        if last.exists():
            shutil.copy2(last, WEIGHTS_OUT)
            print(f"Uteži (last) shranjene: {WEIGHTS_OUT}")
        else:
            raise FileNotFoundError("YOLO trening ni ustvaril weights.")
    return WEIGHTS_OUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_dataset()
        return
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model=args.model)


if __name__ == "__main__":
    main()
