"""Multi-view CNN + ViT authenticity classifier trained on JaalTaka.

Each JaalTaka note folder holds six camera views. The CNN branch mean-pools
MobileNetV3-Small embeddings across views; the ViT branch reads a single
primary view. A fusion MLP predicts genuine vs counterfeit.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from .config import AUTH_WEIGHTS, DEVICE, JAALTAKA_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 128


def default_transform(train: bool = False) -> transforms.Compose:
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


class TinyViT(nn.Module):
    """Compact ViT used as the second authenticity branch."""

    def __init__(self, img_size: int = IMG_SIZE, patch: int = 16, dim: int = 128, depth: int = 3, heads: int = 4):
        super().__init__()
        assert img_size % patch == 0
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.encoder(x)
        return self.norm(x[:, 0])


class MultiViewCNNVIT(nn.Module):
    """CNN (multi-view) + ViT (primary view) authenticity network."""

    def __init__(self, freeze_cnn: bool = True):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.cnn = backbone.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_dim = 576
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
        self.vit = TinyViT()
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 2),
        )

    def encode_cnn(self, views: torch.Tensor) -> torch.Tensor:
        """views: (B, V, C, H, W) -> (B, cnn_dim)."""
        b, v, c, h, w = views.shape
        feats = self.cnn(views.reshape(b * v, c, h, w))
        feats = self.cnn_pool(feats).flatten(1).reshape(b, v, -1)
        return feats.mean(dim=1)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        cnn_emb = self.encode_cnn(views)
        vit_emb = self.vit(views[:, 0])
        return self.fusion(torch.cat([cnn_emb, vit_emb], dim=1))


def list_jaaltaka_notes() -> list[tuple[Path, int]]:
    """Return (note_dir, label) with label 1=genuine, 0=counterfeit."""
    samples: list[tuple[Path, int]] = []
    for label, folder in ((1, "real_notes"), (0, "fake_notes")):
        root = JAALTAKA_DIR / folder
        if not root.is_dir():
            continue
        for note_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            views = _view_paths(note_dir)
            if views:
                samples.append((note_dir, label))
    return samples


def _view_paths(note_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in note_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def split_notes(
    samples: list[tuple[Path, int]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[Path, int]]]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[tuple[Path, int]]] = {0: [], 1: []}
    for item in samples:
        by_label[item[1]].append(item)
    splits = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n = len(items)
        if n == 0:
            continue
        n_test = max(1, int(n * test_frac)) if n >= 6 else max(0, n // 5)
        n_val = max(1, int(n * val_frac)) if n >= 6 else max(0, n // 5)
        if n_test + n_val >= n:
            n_val = max(0, n // 5)
            n_test = max(0, n // 5)
        test_i = idx[:n_test]
        val_i = idx[n_test : n_test + n_val]
        train_i = idx[n_test + n_val :]
        splits["test"].extend(items[int(i)] for i in test_i)
        splits["val"].extend(items[int(i)] for i in val_i)
        splits["train"].extend(items[int(i)] for i in train_i)
        _ = label
    return splits


class JaalTakaMultiView(Dataset):
    def __init__(self, notes: list[tuple[Path, int]], n_views: int = 2, train: bool = False):
        self.notes = notes
        self.n_views = n_views
        self.tf = default_transform(train=train)

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx: int):
        note_dir, label = self.notes[idx]
        paths = _view_paths(note_dir)
        if not paths:
            raise RuntimeError(f"no views in {note_dir}")
        if len(paths) >= self.n_views:
            idx = np.random.choice(len(paths), size=self.n_views, replace=False)
            chosen = [paths[int(i)] for i in idx]
        else:
            chosen = [paths[i % len(paths)] for i in range(self.n_views)]
        tensors = []
        for p in chosen:
            img = Image.open(p).convert("RGB")
            tensors.append(self.tf(img))
        views = torch.stack(tensors, dim=0)
        return views, torch.tensor(label, dtype=torch.long)


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def bgr_to_views_tensor(crops_bgr: list[np.ndarray], n_views: int = 2) -> torch.Tensor:
    tf = default_transform(train=False)
    if not crops_bgr:
        raise ValueError("need at least one crop")
    views = []
    for i in range(n_views):
        crop = crops_bgr[i % len(crops_bgr)]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        views.append(tf(Image.fromarray(rgb)))
    return torch.stack(views, dim=0).unsqueeze(0)


class AuthenticityClassifier:
    """Runtime wrapper: CNN+ViT genuine/counterfeit on one or more note crops."""

    def __init__(self, weights: Path | None = None, device: torch.device | None = None):
        self.device = device or DEVICE
        self.model = MultiViewCNNVIT(freeze_cnn=True).to(self.device)
        self.loaded = False
        path = weights or AUTH_WEIGHTS
        if path.is_file():
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
            self.loaded = True
        self.model.eval()

    @torch.inference_mode()
    def predict(self, crops_bgr: list[np.ndarray]) -> dict:
        if not crops_bgr:
            return {"label": "unknown", "genuine_prob": 0.5, "loaded": self.loaded}
        x = bgr_to_views_tensor(crops_bgr).to(self.device)
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)[0]
        genuine = float(prob[1].item())
        label = "genuine" if genuine >= 0.5 else "counterfeit"
        return {
            "label": label,
            "genuine_prob": genuine,
            "counterfeit_prob": float(prob[0].item()),
            "loaded": self.loaded,
        }
