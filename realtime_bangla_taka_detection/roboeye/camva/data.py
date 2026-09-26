"""Variable-view JaalTaka datasets, corruptions, collate."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..authenticity import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, default_transform

_VIEW_CACHE = Path(__file__).resolve().parents[2] / "cache" / "rgb128"
_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))


def cached_resized_rgb(path: Path) -> Image.Image | None:
    """Load a view as a 128x128 RGB image, caching the resized array on disk."""
    _VIEW_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    fp = _VIEW_CACHE / f"{digest}.npy"
    if fp.is_file():
        try:
            return Image.fromarray(np.load(fp))
        except (OSError, ValueError):
            pass
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = _RESIZE(Image.fromarray(rgb))
    array = np.asarray(image, dtype=np.uint8)
    tmp = fp.with_suffix(".tmp.npy")
    np.save(tmp, array)
    tmp.replace(fp)
    return Image.fromarray(array)


def apply_corruption(bgr: np.ndarray, name: str, severity: float) -> np.ndarray:
    img = bgr.copy()
    if name == "gaussian_blur":
        k = int(max(1, round(severity))) * 2 + 1
        return cv2.GaussianBlur(img, (k, k), severity)
    if name == "motion_blur":
        k = max(3, int(severity) * 2 + 1)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel)
    if name == "brightness":
        return np.clip(img.astype(np.float32) * float(severity), 0, 255).astype(np.uint8)
    if name == "low_light":
        return np.clip(img.astype(np.float32) * float(severity), 0, 255).astype(np.uint8)
    if name == "occlusion":
        h, w = img.shape[:2]
        frac = float(severity)
        rh, rw = max(1, int(h * frac ** 0.5)), max(1, int(w * frac ** 0.5))
        y0 = np.random.randint(0, max(h - rh, 1))
        x0 = np.random.randint(0, max(w - rw, 1))
        img[y0 : y0 + rh, x0 : x0 + rw] = 0
        return img
    if name == "jpeg":
        q = int(np.clip(severity, 10, 95))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if name == "rotation":
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(severity), 1.0)
        return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if name == "perspective":
        h, w = img.shape[:2]
        s = float(severity)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32(
            [
                [s * w, s * h],
                [w - s * w, 0],
                [w, h - s * h],
                [0, h],
            ]
        )
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if name == "scale":
        h, w = img.shape[:2]
        f = float(severity)
        nh, nw = max(8, int(h * f)), max(8, int(w * f))
        small = cv2.resize(img, (nw, nh))
        return cv2.resize(small, (w, h))
    if name == "contrast":
        mean = float(img.mean())
        return np.clip((img.astype(np.float32) - mean) * float(severity) + mean, 0, 255).astype(np.uint8)
    if name == "glare":
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-((yy - h * 0.3) ** 2 + (xx - w * 0.7) ** 2) / (2 * (min(h, w) * 0.15) ** 2))
        return np.clip(img.astype(np.float32) + 255.0 * float(severity) * blob[..., None], 0, 255).astype(np.uint8)
    if name == "sensor_noise":
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, float(severity), img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


class NoteViewDataset(Dataset):
    """One physical note per item. Returns all views (padded later) plus mask."""

    def __init__(
        self,
        note_ids: list[str],
        records: dict[str, dict],
        n_views: int | None = None,
        train: bool = False,
        view_order: str = "fixed",
        corruption: str | None = None,
        corruption_severity: float = 0.0,
        seed: int = 42,
    ):
        self.ids = list(note_ids)
        self.records = records
        self.n_views = n_views
        self.tf = default_transform(train=train)
        self._jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self._to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )
        self.train = train
        self.view_order = view_order
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def _ordered_paths(self, rec: dict) -> list[Path]:
        paths = [Path(p) for p in rec["view_paths"]]
        if self.view_order == "random":
            idx = self.rng.permutation(len(paths))
            paths = [paths[int(i)] for i in idx]
        return paths

    def __getitem__(self, i: int):
        nid = self.ids[i]
        rec = self.records[nid]
        paths = self._ordered_paths(rec)
        if self.n_views is not None:
            paths = paths[: self.n_views]
        tensors = []
        for p in paths:
            if self.corruption:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                bgr = apply_corruption(bgr, self.corruption, self.corruption_severity)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = _RESIZE(Image.fromarray(rgb))
            else:
                image = cached_resized_rgb(p)
                if image is None:
                    continue
            if self.train:
                if torch.rand(1).item() < 0.5:
                    image = transforms.functional.hflip(image)
                image = self._jitter(image)
            tensors.append(self._to_tensor(image))
        if not tensors:
            raise RuntimeError(f"no readable views for {nid}")
        views = torch.stack(tensors, dim=0)
        label = int(rec["label"])
        return {
            "note_id": nid,
            "views": views,
            "n": views.size(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_notes(batch: list[dict]) -> dict:
    max_v = max(item["n"] for item in batch)
    c, h, w = batch[0]["views"].shape[1:]
    views = torch.zeros(len(batch), max_v, c, h, w)
    mask = torch.zeros(len(batch), max_v, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    ids = []
    for i, item in enumerate(batch):
        v = item["n"]
        views[i, :v] = item["views"]
        mask[i, :v] = 1
        labels[i] = item["label"]
        ids.append(item["note_id"])
    return {"views": views, "mask": mask, "label": labels, "note_id": ids}
