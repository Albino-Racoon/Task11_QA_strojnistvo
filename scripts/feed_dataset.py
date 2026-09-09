#!/usr/bin/env python3
"""Pošlji slike iz dataseta na /api/inspect."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS and "_label" not in p.name)
    return files[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Datoteka ali mapa s slikami")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--line", default="Linija A")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    files = collect(args.path, args.limit)
    if not files:
        print("Ni slik.", file=sys.stderr)
        sys.exit(1)

    print(f"Pošiljam {len(files)} slik → {args.api}/api/inspect")
    with httpx.Client(timeout=120.0) as client:
        for p in files:
            with p.open("rb") as f:
                res = client.post(
                    f"{args.api}/api/inspect",
                    files={"file": (p.name, f, "image/jpeg")},
                    data={"line": args.line},
                )
            if res.status_code != 200:
                print(f"FAIL {p.name}: {res.status_code} {res.text[:200]}")
                continue
            body = res.json()
            defects = ",".join(d["type"] for d in body.get("defects", [])) or "—"
            print(
                f"{p.name:40s}  {body['status']:6s}  score={body['quality_score']:3d}  "
                f"anom={body['anomaly']['score']:.2f}  {defects}"
            )


if __name__ == "__main__":
    main()
