"""
PatchCore-inspired anomaly detector (ResNet feature memory bank).

Trains on normal images only; scores test images via nearest-neighbour
distance in patch feature space and produces a spatial heatmap.
Compatible with KolektorSDD / MVTec-style folder layouts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from decision_engine.engine import AnomalyResult

logger = logging.getLogger(__name__)


class _FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.layer2 = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        self.layer3 = backbone.layer3
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3_up], dim=1)


class PatchCoreAnomalyDetector:
    def __init__(self, weights_dir: Path, device: str | None = None):
        self.weights_dir = Path(weights_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor: _FeatureExtractor | None = None
        self.memory_bank: np.ndarray | None = None
        self.threshold: float = 0.5
        self.image_size: int = 256
        self.available = False
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._load()

    def _load(self) -> None:
        bank_path = self.weights_dir / "memory_bank.npy"
        meta_path = self.weights_dir / "meta.json"
        if not bank_path.exists() or not meta_path.exists():
            logger.warning("Anomaly uteži niso na voljo: %s", self.weights_dir)
            return
        try:
            self.memory_bank = np.load(bank_path).astype(np.float32)
            meta = json.loads(meta_path.read_text())
            self.threshold = float(meta.get("threshold", 0.5))
            self.image_size = int(meta.get("image_size", 256))
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.extractor = _FeatureExtractor().to(self.device)
            self.available = True
            logger.info(
                "PatchCore naložen (%d embeddingov, thr=%.3f)",
                len(self.memory_bank),
                self.threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Napaka pri nalaganju anomaly modela: %s", exc)
            self.available = False

    @torch.inference_mode()
    def _embedding_map(self, image_bgr: np.ndarray) -> np.ndarray:
        assert self.extractor is not None
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        feats = self.extractor(tensor)[0]  # C,H,W
        feats = F.normalize(feats, dim=0)
        return feats.permute(1, 2, 0).cpu().numpy()  # H,W,C

    def _score_map(self, emb_map: np.ndarray) -> np.ndarray:
        assert self.memory_bank is not None
        h, w, c = emb_map.shape
        flat = emb_map.reshape(-1, c).astype(np.float64)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
        flat = flat / norms
        bank = self.memory_bank.astype(np.float64)
        dists = np.empty(flat.shape[0], dtype=np.float64)
        chunk = 512
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i in range(0, flat.shape[0], chunk):
                part = flat[i : i + chunk]
                sim = part @ bank.T
                sim = np.nan_to_num(np.clip(sim, -1.0, 1.0), nan=-1.0)
                dists[i : i + chunk] = 1.0 - np.max(sim, axis=1)
        return dists.reshape(h, w)

    def predict(
        self,
        image_bgr: np.ndarray,
        heatmap_out: Path | None = None,
    ) -> AnomalyResult:
        if not self.available or self.memory_bank is None or self.extractor is None:
            return AnomalyResult(score=0.0, heatmap_path=None, is_anomaly=False)

        emb = self._embedding_map(image_bgr)
        # normalize patch vectors in embedding map
        h, w, c = emb.shape
        flat = emb.reshape(-1, c)
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        emb = flat.reshape(h, w, c)

        score_map = self._score_map(emb)
        # Top-1% mean is more stable than a single-pixel max (better OK/NOK separation)
        flat_scores = np.sort(score_map.ravel())
        top_n = max(1, int(0.01 * flat_scores.size))
        raw_score = float(np.mean(flat_scores[-top_n:]))
        # Map raw distance so train p90 (self.threshold) ~= 0.5 decision boundary
        thr = max(self.threshold, 1e-6)
        # Slightly sharper sigmoid so borderline normals stay under review band
        norm = float(1.0 / (1.0 + np.exp(-(raw_score - thr) / max(thr * 0.22, 1e-3))))
        norm = float(np.clip(norm, 0.0, 1.0))

        heatmap_path = None
        if heatmap_out is not None:
            heatmap_path = self._save_heatmap(image_bgr, score_map, heatmap_out)

        return AnomalyResult(score=norm, heatmap_path=heatmap_path, is_anomaly=norm >= 0.4)

    def _save_heatmap(
        self,
        image_bgr: np.ndarray,
        score_map: np.ndarray,
        out_path: Path,
    ) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sm = score_map - score_map.min()
        if sm.max() > 0:
            sm = sm / sm.max()
        sm_u8 = (sm * 255).astype(np.uint8)
        sm_u8 = cv2.resize(sm_u8, (image_bgr.shape[1], image_bgr.shape[0]))
        colored = cv2.applyColorMap(sm_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.55, colored, 0.45, 0)
        cv2.imwrite(str(out_path), overlay)
        return str(out_path)

    def info(self) -> dict[str, Any]:
        return {
            "name": "patchcore_resnet18",
            "available": self.available,
            "weights": str(self.weights_dir),
            "threshold": self.threshold,
            "memory_bank_size": 0 if self.memory_bank is None else int(self.memory_bank.shape[0]),
        }


def train_patchcore(
    normal_image_paths: list[Path],
    output_dir: Path,
    *,
    image_size: int = 256,
    max_memory: int = 5000,
    subsample_stride: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    """Build memory bank from normal images and estimate threshold on train max."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = _FeatureExtractor().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    embeddings: list[np.ndarray] = []

    for path in normal_image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.inference_mode():
            feats = extractor(tensor)[0]
            feats = F.normalize(feats, dim=0)
            emb = feats.permute(1, 2, 0).cpu().numpy()  # H,W,C
        sampled = emb[::subsample_stride, ::subsample_stride, :].reshape(-1, emb.shape[-1])
        sampled = sampled.astype(np.float32)
        sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-8
        sampled = sampled / norms
        embeddings.append(sampled)

    if not embeddings:
        raise ValueError("Ni veljavnih normalnih slik za trening anomaly modela.")

    bank = np.concatenate(embeddings, axis=0)
    if bank.shape[0] > max_memory:
        rng = np.random.default_rng(42)
        idx = rng.choice(bank.shape[0], size=max_memory, replace=False)
        bank = bank[idx]
    bank = np.nan_to_num(bank, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Estimate threshold from top-1% mean distances on training normals
    image_scores: list[float] = []
    for path in normal_image_paths[:: max(1, len(normal_image_paths) // 80)]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.inference_mode():
            feats = extractor(tensor)[0]
            feats = F.normalize(feats, dim=0)
            emb = feats.permute(1, 2, 0).cpu().numpy().astype(np.float64)
        flat = emb.reshape(-1, emb.shape[-1])
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        # subsample patches for speed
        flat = flat[::4]
        sim = flat @ bank.astype(np.float64).T
        sim = np.clip(sim, -1.0, 1.0)
        dists = 1.0 - np.max(sim, axis=1)
        top_n = max(1, int(0.01 * dists.size))
        image_scores.append(float(np.mean(np.sort(dists)[-top_n:])))

    finite = [s for s in image_scores if np.isfinite(s)]
    if finite:
        threshold = float(np.percentile(finite, 90))
    else:
        threshold = 0.25
    threshold = float(np.clip(threshold, 0.02, 0.8))

    np.save(output_dir / "memory_bank.npy", bank.astype(np.float32))
    meta = {
        "threshold": threshold,
        "image_size": image_size,
        "memory_bank_size": int(bank.shape[0]),
        "num_train_images": len(normal_image_paths),
        "normal_score_mean": float(np.mean(finite)) if finite else 0.0,
        "normal_score_p90": threshold,
        "score_aggregation": "mean_top_1pct",
        "backbone": "resnet18_layer2_layer3",
        "method": "patchcore_inspired",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
