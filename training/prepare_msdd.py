#!/usr/bin/env python3
"""
MSDD (Metal Surface Defect Dataset) — litni / strukturirani kovinski deli.

Vir: https://doi.org/10.57760/sciencedb.10794  (CC BY 4.0)
Članek: https://doi.org/10.1038/s41597-025-04454-6

Datoteka: MSDD.rar (~9.11 GB). ScienceDB zahteva brezplačno prijavo za download.

Po prenosu:
  1. Shrani MSDD.rar v training/datasets/msdd/
  2. python training/prepare_msdd.py
  3. python training/train_yolo_msdd.py --epochs 20
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
MSDD_ROOT = ROOT / "training" / "datasets" / "msdd"
YOLO_DIR = ROOT / "training" / "datasets" / "msdd_yolo"
WEIGHTS_OUT = ROOT / "weights" / "yolo_msdd.pt"
SAMPLES = ROOT / "samples" / "msdd"

# Table 2 — Scientific Data 2025
MSDD_CLASSES = [
    "misrun",
    "inclusion",
    "dent",
    "parting_line_crack",
    "stamp_collapse",
    "pockmarks",
    "mould_scuffing",
    "cut_marks",
]

CLASS_ALIASES = {
    "misrun": "misrun",
    "inclusion": "inclusion",
    "dent": "dent",
    "parting line crack": "parting_line_crack",
    "parting_line_crack": "parting_line_crack",
    "stamp collapse": "stamp_collapse",
    "stamp_collapse": "stamp_collapse",
    "pockmarks": "pockmarks",
    "mould scuffing": "mould_scuffing",
    "mould_scuffing": "mould_scuffing",
    "mold scuffing": "mould_scuffing",
    "cut marks": "cut_marks",
    "cut_marks": "cut_marks",
}


def find_archive() -> Path | None:
    for name in ("MSDD.rar", "msdd.rar", "MSDD.zip", "msdd.zip"):
        p = MSDD_ROOT / name
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    return None


def find_extracted() -> tuple[Path | None, Path | None]:
    """Return (images_dir, annotations_dir)."""
    for images in [
        MSDD_ROOT / "datasets" / "JPEGImages",
        MSDD_ROOT / "JPEGImages",
        MSDD_ROOT / "datasets" / "Images",
        MSDD_ROOT / "Images",
    ]:
        ann = images.parent / "Annotations"
        if images.exists() and ann.exists():
            return images, ann
    # fallback: search
    for images in MSDD_ROOT.rglob("JPEGImages"):
        if images.is_dir():
            ann = images.parent / "Annotations"
            if ann.exists():
                return images, ann
    return None, None


def extract_archive(archive: Path) -> None:
    MSDD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Ekstrahiram {archive} (lahko traja)…")
    # Prefer extracting only JPEGImages + Annotations if using unar/7z
    if archive.suffix.lower() == ".rar":
        # unar / unrar / 7z
        for cmd in (
            ["unar", "-f", "-o", str(MSDD_ROOT), str(archive)],
            ["unrar", "x", "-o+", str(archive), str(MSDD_ROOT) + "/"],
            ["7z", "x", str(archive), f"-o{MSDD_ROOT}", "-y"],
        ):
            try:
                subprocess.run(cmd, check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        raise RuntimeError(
            "Namesti unar ali 7z za ekstrakcijo RAR:\n"
            "  brew install unar\n"
            "ali ekstrahiraj MSDD.rar ročno v training/datasets/msdd/"
        )
    if archive.suffix.lower() == ".zip":
        shutil.unpack_archive(str(archive), str(MSDD_ROOT))
        return
    raise RuntimeError(f"Nepodprt arhiv: {archive}")


def _map_class(name: str) -> str | None:
    key = name.strip().lower().replace("-", "_")
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    key2 = key.replace("_", " ")
    if key2 in CLASS_ALIASES:
        return CLASS_ALIASES[key2]
    for n in MSDD_CLASSES:
        if n.replace("_", "") in key.replace("_", "") or key.replace(" ", "_") in n:
            return n
    return None


def voc_to_yolo(images_dir: Path, ann_dir: Path, out_dir: Path) -> dict:
    name_to_id = {n: i for i, n in enumerate(MSDD_CLASSES)}
    xmls = sorted(ann_dir.glob("*.xml"))
    items: list[tuple[Path, Path]] = []
    bg_images: list[Path] = []

    # images with annotations
    for xml in xmls:
        stem = xml.stem
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img = cand
                break
        if img is None:
            matches = list(images_dir.rglob(f"{stem}.*"))
            matches = [m for m in matches if m.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            if matches:
                img = matches[0]
        if img is None:
            continue
        items.append((img, xml))

    # defect-free backgrounds (images without xml or empty objects)
    annotated_stems = {p.stem for p, _ in items}
    for img in images_dir.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if img.stem not in annotated_stems:
            bg_images.append(img)

    random.Random(42).shuffle(items)
    random.Random(42).shuffle(bg_images)
    split_at = max(1, int(len(items) * 0.85))
    train_items, val_items = items[:split_at], items[split_at:]
    # keep some backgrounds in both splits
    bg_split = max(1, int(len(bg_images) * 0.85)) if bg_images else 0

    stats = {"train": 0, "val": 0, "boxes": 0, "background": 0}

    def write_split(split: str, pairs: list[tuple[Path, Path]], bgs: list[Path]) -> None:
        img_out = out_dir / split / "images"
        lbl_out = out_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, xml in pairs:
            tree = ET.parse(xml)
            root = tree.getroot()
            size = root.find("size")
            if size is not None:
                w = int(float(size.findtext("width", "0")))
                h = int(float(size.findtext("height", "0")))
            else:
                arr = cv2.imread(str(img))
                if arr is None:
                    continue
                h, w = arr.shape[:2]
            if w <= 0 or h <= 0:
                arr = cv2.imread(str(img))
                if arr is None:
                    continue
                h, w = arr.shape[:2]
            lines = []
            for obj in root.findall("object"):
                raw = obj.findtext("name") or ""
                cls = _map_class(raw)
                if cls is None or cls not in name_to_id:
                    continue
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                xmin = float(bb.findtext("xmin", "0"))
                ymin = float(bb.findtext("ymin", "0"))
                xmax = float(bb.findtext("xmax", "0"))
                ymax = float(bb.findtext("ymax", "0"))
                cx = ((xmin + xmax) / 2) / w
                cy = ((ymin + ymax) / 2) / h
                bw = max(0.0, (xmax - xmin) / w)
                bh = max(0.0, (ymax - ymin) / h)
                lines.append(f"{name_to_id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats["boxes"] += 1
            dst = img_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            (lbl_out / f"{dst.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[split] += 1
        for img in bgs:
            dst = img_out / f"bg_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
            (lbl_out / f"{dst.stem}.txt").write_text("")
            stats["background"] += 1
            stats[split] += 1

    for split in ("train", "val"):
        shutil.rmtree(out_dir / split, ignore_errors=True)

    write_split("train", train_items, bg_images[:bg_split][:500])
    write_split("val", val_items, bg_images[bg_split:][:100])

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(MSDD_CLASSES)},
    }
    (out_dir / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False))
    return stats


def export_samples(images_dir: Path, ann_dir: Path, n: int = 12) -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    ok_dir = SAMPLES / "ok"
    nok_dir = SAMPLES / "nok"
    ok_dir.mkdir(exist_ok=True)
    nok_dir.mkdir(exist_ok=True)
    xmls = list(ann_dir.glob("*.xml"))[: n * 2]
    copied = 0
    for xml in xmls:
        if copied >= n:
            break
        stem = xml.stem
        img = next((images_dir / f"{stem}{e}" for e in (".jpg", ".jpeg", ".png") if (images_dir / f"{stem}{e}").exists()), None)
        if img is None:
            continue
        shutil.copy2(img, nok_dir / img.name)
        copied += 1
    # a few backgrounds
    stems = {x.stem for x in ann_dir.glob("*.xml")}
    bg = 0
    for img in images_dir.glob("*.jpg"):
        if img.stem in stems:
            continue
        shutil.copy2(img, ok_dir / img.name)
        bg += 1
        if bg >= min(8, n):
            break
    print(f"Samples: {copied} NOK → {nok_dir}, {bg} OK → {ok_dir}")


def prepare(force_extract: bool = False) -> Path:
    images, anns = find_extracted()
    if images is None or force_extract:
        archive = find_archive()
        if archive is None:
            manual = MSDD_ROOT / "README_DOWNLOAD.md"
            MSDD_ROOT.mkdir(parents=True, exist_ok=True)
            manual.write_text(
                "# MSDD prenos\n\n"
                "1. Ustvari brezplačen račun na https://www.scidb.cn\n"
                "2. Odpri https://doi.org/10.57760/sciencedb.10794\n"
                "3. Download → **MSDD.rar** (9.11 GB, CC BY 4.0)\n"
                "4. Shrani kot `training/datasets/msdd/MSDD.rar`\n"
                "5. `python training/prepare_msdd.py`\n\n"
                "Po ekstrakciji potrebuješ predvsem mape:\n"
                "- `JPEGImages/` (pseudo-color mešane slike, ~9k)\n"
                "- `Annotations/` (VOC XML)\n"
                "`OrigImages/` (138k) lahko izpustiš / zbrišeš, če ti zmanjka prostora.\n"
            )
            raise FileNotFoundError(f"MSDD.rar ni najden. Glej {manual}")
        extract_archive(archive)
        images, anns = find_extracted()
    if images is None or anns is None:
        raise FileNotFoundError("Po ekstrakciji ni JPEGImages/Annotations. Preveri strukturo arhiva.")

    print(f"Images: {images}\nAnnotations: {anns}")
    stats = voc_to_yolo(images, anns, YOLO_DIR)
    print("Pretvorba:", stats)
    export_samples(images, anns)
    return YOLO_DIR / "data.yaml"


def train(epochs: int = 20, imgsz: int = 640, batch: int = 8) -> Path:
    yaml_path = prepare()
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "training" / "runs"),
        name="msdd_yolo",
        exist_ok=True,
        patience=10,
    )
    best = ROOT / "training" / "runs" / "msdd_yolo" / "weights" / "best.pt"
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    src = best if best.exists() else best.with_name("last.pt")
    if not src.exists():
        raise FileNotFoundError("MSDD YOLO trening ni ustvaril weights.")
    shutil.copy2(src, WEIGHTS_OUT)
    print(f"Uteži: {WEIGHTS_OUT}")
    # also point default detector path optionally
    return WEIGHTS_OUT


def main() -> None:
    parser = argparse.ArgumentParser(description="Pripravi / trenira MSDD")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.train:
        train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    else:
        prepare(force_extract=args.force_extract)


if __name__ == "__main__":
    main()
