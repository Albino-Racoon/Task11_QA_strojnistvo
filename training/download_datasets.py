#!/usr/bin/env python3
"""Download public datasets for the Metal QC PoC."""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "datasets"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Že obstaja: {dest}")
        return dest
    print(f"Prenašam {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest


def download_gc10(out_dir: Path) -> Path:
    """Download GC10-DET from Hugging Face (COCO format with images)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".downloaded"
    if marker.exists():
        print(f"GC10 že pripravljen: {out_dir}")
        return out_dir

    repos = ["imaadd05/gc10-det", "lzhzj/gc10-det", "Kelvin878/gc10_det"]
    last_err: Exception | None = None
    for repo_id in repos:
        try:
            from huggingface_hub import snapshot_download

            target = out_dir / "hf"
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
            )
            marker.write_text(f"hf:{repo_id}\n")
            print(f"GC10-DET prenešen iz {repo_id} → {out_dir}")
            return out_dir
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"HF {repo_id} ni uspel: {exc}")

    readme = out_dir / "README_MANUAL.txt"
    readme.write_text(
        "Samodejni prenos GC10 ni uspel.\n"
        "Ročno: https://huggingface.co/datasets/imaadd05/gc10-det\n"
        f"Postavi datoteke v: {out_dir}\n"
        f"Zadnja napaka: {last_err}\n"
    )
    raise RuntimeError(f"GC10 prenos ni uspel. Glej {readme}")


def download_kolektor(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".downloaded"
    if marker.exists():
        print(f"KolektorSDD že pripravljen: {out_dir}")
        return out_dir

    # Official ViCoS hosting historically used box/drop-style links; try common mirror.
    urls = [
        "https://www.vicos.si/data/kolektorsdd/KolektorSDD.zip",
        "https://go.vicos.si/kolektorsdd",
    ]
    zip_path = out_dir / "KolektorSDD.zip"
    last_err = None
    for url in urls:
        try:
            _download(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            marker.write_text(url + "\n")
            print(f"KolektorSDD pripravljen: {out_dir}")
            return out_dir
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"Neuspeh {url}: {exc}")

    # Synthetic mini dataset so training scripts remain runnable offline
    synth = out_dir / "synthetic"
    ok_dir = synth / "OK"
    nok_dir = synth / "NOK"
    ok_dir.mkdir(parents=True, exist_ok=True)
    nok_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import cv2

    rng = np.random.default_rng(0)
    for i in range(40):
        img = np.full((256, 256, 3), 140, np.uint8)
        noise = rng.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(ok_dir / f"ok_{i:03d}.png"), img)
    for i in range(12):
        img = np.full((256, 256, 3), 140, np.uint8)
        noise = rng.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.line(img, (30, 40 + i * 3), (220, 200 - i), (20, 20, 20), 2)
        cv2.circle(img, (120, 130), 8, (10, 10, 10), -1)
        cv2.imwrite(str(nok_dir / f"nok_{i:03d}.png"), img)
    marker.write_text(f"synthetic_fallback last_err={last_err}\n")
    print(f"Kolektor ni bil dosegljiv — ustvarjen synthetic set v {synth}")
    return out_dir


def download_kolektor2(out_dir: Path) -> Path:
    """KolektorSDD2 (~850 MB) — hiter javni prenos z ViCoS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".downloaded"
    if marker.exists() and any((out_dir / "train").glob("*.png")):
        print(f"KolektorSDD2 že pripravljen: {out_dir}")
        return out_dir

    urls = [
        "https://go.vicos.si/kolektorsdd2",
        "https://www.vicos.si/data/kolektorsdd2/KolektorSDD2.zip",
    ]
    zip_path = out_dir / "KolektorSDD2.zip"
    last_err = None
    for url in urls:
        try:
            if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
                _download(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            marker.write_text(url + "\n")
            print(f"KolektorSDD2 pripravljen: {out_dir}")
            return out_dir
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"Neuspeh {url}: {exc}")
    raise RuntimeError(f"KolektorSDD2 prenos ni uspel: {last_err}")


def download_mvtec(out_dir: Path, category: str = "metal_nut") -> Path:
    """
    MVTec AD — CC BY-NC-SA 4.0. Samo za interni raziskovalni demo.
    Ni dovoljeno za komercialno uporabo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f".downloaded_{category}"
    if marker.exists():
        print(f"MVTec {category} že pripravljen")
        return out_dir / category

    url = f"https://www.mydrive.ch/publics/1_MvTec_AD/{category}.tar.xz"
    # Official distribution requires accepting license on mvtec.com; try mirror then instruct.
    archive = out_dir / f"{category}.tar.xz"
    try:
        _download(
            f"https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/{category}.tar.xz",
            archive,
        )
    except Exception:
        readme = out_dir / "README_MVTEC.txt"
        readme.write_text(
            "MVTec AD je CC BY-NC-SA 4.0 — samo nekomercialni raziskovalni demo.\n"
            "Prenesite ročno z https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
            f"Razširite kategorijo '{category}' v {out_dir / category}\n"
            "Pričakovana struktura: train/good, test/<defect>, ground_truth/...\n"
        )
        print(readme.read_text())
        raise RuntimeError("MVTec zahteva ročni prenos (licenca).")

    import tarfile

    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(out_dir)
    marker.write_text("ok\n")
    return out_dir / category


def main() -> None:
    parser = argparse.ArgumentParser(description="Prenos datasetov za Metal QC PoC")
    parser.add_argument(
        "--dataset",
        choices=["gc10", "kolektor", "kolektor2", "mvtec", "msdd", "all"],
        default="all",
    )
    parser.add_argument("--mvtec-category", default="metal_nut")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"gc10", "all"}:
        try:
            download_gc10(DATA / "gc10")
        except Exception as exc:  # noqa: BLE001
            print(f"OPOZORILO GC10: {exc}", file=sys.stderr)
    if args.dataset in {"kolektor", "all"}:
        download_kolektor(DATA / "kolektor")
    if args.dataset in {"kolektor2", "all"}:
        try:
            download_kolektor2(DATA / "kolektor2")
        except Exception as exc:  # noqa: BLE001
            print(f"OPOZORILO KolektorSDD2: {exc}", file=sys.stderr)
    if args.dataset in {"mvtec", "all"}:
        try:
            download_mvtec(DATA / "mvtec", category=args.mvtec_category)
        except Exception as exc:  # noqa: BLE001
            print(f"OPOZORILO MVTec: {exc}", file=sys.stderr)
    if args.dataset in {"msdd", "all"}:
        readme = DATA / "msdd" / "README_DOWNLOAD.md"
        print("MSDD zahteva ročni prenos (ScienceDB login, ~9.11 GB, CC BY 4.0).")
        print(f"Navodila: {readme}")
        print("DOI: https://doi.org/10.57760/sciencedb.10794")
        print("Nato: PYTHONPATH=. python training/prepare_msdd.py --train")


if __name__ == "__main__":
    main()
