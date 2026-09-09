#!/usr/bin/env python3
"""Train PatchCore-inspired anomaly model on KolektorSDD or MVTec."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.anomaly.patchcore import train_patchcore

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTS and "_label" not in p.stem.lower()
    )


def _kolektor_split(root: Path) -> tuple[list[Path], list[Path]]:
    """Split KolektorSDD by label masks: empty mask => normal."""
    import cv2
    import numpy as np

    normals: list[Path] = []
    defects: list[Path] = []
    for img in sorted(root.glob("kos*/Part*.jpg")):
        if "_label" in img.name:
            continue
        label = img.with_name(img.stem + "_label.bmp")
        if not label.exists():
            normals.append(img)
            continue
        mask = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            normals.append(img)
            continue
        if float(np.max(mask)) > 0:
            defects.append(img)
        else:
            normals.append(img)
    return normals, defects


def _kolektor2_split(root: Path) -> tuple[list[Path], list[Path]]:
    """Split KolektorSDD2 (train/test PNGs + *_GT.png masks)."""
    import cv2
    import numpy as np

    normals: list[Path] = []
    defects: list[Path] = []
    for split in ("train", "test"):
        folder = root / split
        if not folder.exists():
            continue
        for img in sorted(folder.glob("*.png")):
            if img.name.endswith("_GT.png"):
                continue
            gt = img.with_name(img.stem + "_GT.png")
            if not gt.exists():
                normals.append(img)
                continue
            mask = cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)
            if mask is not None and float(np.max(mask)) > 0:
                defects.append(img)
            else:
                normals.append(img)
    return normals, defects


def resolve_normal_images(dataset: str, data_root: Path, category: str | None) -> list[Path]:
    if dataset == "kolektor":
        root = data_root / "kolektor"
        normals, defects = _kolektor_split(root)
        if normals:
            print(f"KolektorSDD: {len(normals)} normalnih, {len(defects)} defektnih")
            return normals
        candidates = [
            root / "synthetic" / "OK",
            root / "OK",
            root / "train" / "good",
        ]
        for c in candidates:
            imgs = _collect_images(c)
            if imgs:
                print(f"Kolektor normalne slike: {c} ({len(imgs)})")
                return imgs
        raise FileNotFoundError("Ni Kolektor normalnih slik. Zaženi download_datasets.py --dataset kolektor")

    if dataset == "kolektor2":
        root = data_root / "kolektor2"
        normals, defects = _kolektor2_split(root)
        if normals:
            print(f"KolektorSDD2: {len(normals)} normalnih, {len(defects)} defektnih")
            return normals
        raise FileNotFoundError(
            "Ni KolektorSDD2. Zaženi: python training/download_datasets.py --dataset kolektor2"
        )

    if dataset == "mvtec":
        cat = category or "metal_nut"
        good = data_root / "mvtec" / cat / "train" / "good"
        imgs = _collect_images(good)
        if not imgs:
            raise FileNotFoundError(
                f"MVTec {cat}/train/good ni najden. Prenesi ročno (CC BY-NC-SA)."
            )
        return imgs

    if dataset == "customer":
        good = data_root / "customer" / "normal"
        imgs = _collect_images(good)
        if not imgs:
            raise FileNotFoundError("Pričakovano: training/datasets/customer/normal/*.jpg")
        return imgs

    raise ValueError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["kolektor", "kolektor2", "mvtec", "customer"],
        default="kolektor",
    )
    parser.add_argument("--category", default="metal_nut", help="MVTec kategorija")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-memory", type=int, default=4000)
    parser.add_argument(
        "--output",
        default=None,
        help="Mapa uteži (privzeto odvisno od --dataset)",
    )
    args = parser.parse_args()

    data_root = ROOT / "training" / "datasets"
    normals = resolve_normal_images(args.dataset, data_root, args.category)
    # limit for PoC speed
    if len(normals) > 300:
        normals = normals[:300]
    if args.output:
        out = Path(args.output)
    elif args.dataset == "mvtec":
        out = ROOT / "weights" / f"patchcore_mvtec_{args.category}"
    elif args.dataset == "kolektor2":
        out = ROOT / "weights" / "patchcore_kolektor2"
    else:
        out = ROOT / "weights" / "patchcore_kolektor"
    meta = train_patchcore(
        normals,
        out,
        image_size=args.image_size,
        max_memory=args.max_memory,
    )
    print("Trening končan:", meta)


if __name__ == "__main__":
    main()
