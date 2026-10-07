"""Data: a synthetic detection dataset that trains on a laptop CPU, and label-format I/O.

SyntheticShapes renders squares, circles and triangles of varied size, colour and position
onto noisy backgrounds. It is small enough to train in minutes yet has small, medium and
large objects in the COCO sense, occlusion, and class confusion (rounded squares look like
circles at small sizes) — enough to watch every metric in the book move.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CLASSES = ("square", "circle", "triangle")


def _draw(img, cls, cx, cy, r, color):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if cls == 0:
        m = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    elif cls == 1:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    else:  # upward triangle inscribed in the (2r)x(2r) square
        t = (yy - (cy - r)) / max(2 * r, 1)
        m = (yy >= cy - r) & (yy <= cy + r) & (np.abs(xx - cx) <= t * r)
    img[m] = color
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    # tight box from the rendered pixels (continuous coords: pixel i spans [i, i+1))
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def make_sample(rng: np.random.Generator, size=256, max_objects=6, min_r=4, max_r=48):
    """One image [size, size, 3] uint8 with boxes [N, 4] xyxy float and labels [N]."""
    bg = rng.integers(0, 90)
    img = (rng.normal(bg, 12, (size, size, 3))).clip(0, 255).astype(np.uint8)
    boxes, labels = [], []
    for _ in range(int(rng.integers(1, max_objects + 1))):
        cls = int(rng.integers(0, len(CLASSES)))
        # log-uniform radius: as many small objects as large ones, like real data
        r = int(np.exp(rng.uniform(np.log(min_r), np.log(max_r))))
        cx, cy = (int(v) for v in rng.integers(r, size - r, 2))
        color = rng.integers(110, 256, 3)
        b = _draw(img, cls, cx, cy, r, color)
        if b is not None:
            boxes.append(b), labels.append(cls)
    # objects drawn later may occlude earlier ones: recompute nothing, like real annotators
    return img, np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


class SyntheticShapes(torch.utils.data.Dataset):
    """Deterministic synthetic dataset. ``augment`` is a callable (img, boxes, labels) -> same."""

    def __init__(self, n=512, size=256, seed=0, augment=None, **kw):
        rng = np.random.default_rng(seed)
        self.samples = [make_sample(rng, size, **kw) for _ in range(n)]
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, boxes, labels = self.samples[i]
        if self.augment is not None:
            img, boxes, labels = self.augment(img.copy(), boxes.copy(), labels.copy())
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return x, torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.as_tensor(labels, dtype=torch.long)


def collate(batch):
    imgs, boxes, labels = zip(*batch)
    return torch.stack(imgs), list(boxes), list(labels)


# ---------------------------------------------------------------------------
# Label formats
# ---------------------------------------------------------------------------


def to_yolo_lines(boxes: np.ndarray, labels: np.ndarray, width: int, height: int) -> list[str]:
    """xyxy pixels -> YOLO txt lines: 'class cx cy w h' normalised to [0, 1]."""
    lines = []
    for (x1, y1, x2, y2), c in zip(boxes, labels):
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def from_yolo_lines(lines: list[str], width: int, height: int):
    """YOLO txt lines -> (boxes xyxy pixels, labels)."""
    rows = [list(map(float, ln.split())) for ln in lines if ln.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    a = np.array(rows, dtype=np.float64)
    c, cx, cy, w, h = a[:, 0], a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    boxes = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1).astype(np.float32)
    return boxes, c.astype(np.int64)


def coco_to_yolo(coco: dict, out_dir: str | Path) -> dict:
    """Convert a COCO-format dict to one YOLO .txt per image. Returns {coco_cat_id: yolo_class}.

    COCO category ids are sparse (1..90 for the 80 COCO classes); YOLO wants 0..K-1. Crowd
    annotations are dropped — YOLO's format has no way to mark 'ignore' regions.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat_ids = sorted(c["id"] for c in coco["categories"])
    remap = {cid: i for i, cid in enumerate(cat_ids)}
    imgs = {im["id"]: im for im in coco["images"]}
    per_img: dict[int, list[str]] = {i: [] for i in imgs}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        per_img[a["image_id"]].append(f"{remap[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    for i, lines in per_img.items():
        (out / (Path(imgs[i]["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    return remap


def to_coco_dict(samples, categories=CLASSES) -> dict:
    """List of (img, boxes, labels) -> minimal COCO ground-truth dict (for pycocotools)."""
    images, anns = [], []
    for i, (img, boxes, labels) in enumerate(samples):
        h, w = img.shape[:2]
        images.append({"id": i, "width": w, "height": h, "file_name": f"{i:06d}.png"})
        for b, c in zip(boxes, labels):
            x1, y1, x2, y2 = map(float, b)
            anns.append(
                {"id": len(anns) + 1, "image_id": i, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}
            )
    return {"images": images, "annotations": anns, "categories": [{"id": i + 1, "name": n} for i, n in enumerate(categories)]}


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj))
