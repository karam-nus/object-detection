"""Where predictions live: grid points, anchor boxes, and k-means anchor fitting."""

from __future__ import annotations

import torch

from .boxes import pairwise_iou


def make_grid_points(feature_sizes, strides, offset: float = 0.5):
    """Anchor *points* for anchor-free heads (FCOS, YOLOX, YOLOv6/8/11/26).

    feature_sizes: [(H3, W3), (H4, W4), (H5, W5)], strides: [8, 16, 32].
    Returns points [A, 2] in *pixels* (cell centres) and strides [A, 1].
    A = sum(H_l * W_l); for a 640x640 input with strides 8/16/32: 80^2 + 40^2 + 20^2 = 8400.
    """
    pts, strs = [], []
    for (h, w), s in zip(feature_sizes, strides):
        ys = (torch.arange(h, dtype=torch.float32) + offset) * s
        xs = (torch.arange(w, dtype=torch.float32) + offset) * s
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts.append(torch.stack((xx, yy), -1).reshape(-1, 2))
        strs.append(torch.full((h * w, 1), float(s)))
    return torch.cat(pts), torch.cat(strs)


def make_anchor_boxes(feature_size, stride: int, sizes=(32, 64, 128), ratios=(0.5, 1.0, 2.0)):
    """Anchor *boxes* for anchor-based heads (Faster R-CNN RPN, SSD, RetinaNet, YOLOv2-v5, v7).

    Every cell gets len(sizes) * len(ratios) boxes with area size^2 and aspect h/w = ratio.
    Returns [H * W * A, 4] xyxy in pixels, ordered (row, col, anchor).
    """
    h, w = feature_size
    shapes = []
    for s in sizes:
        for r in ratios:
            bw, bh = s / r**0.5, s * r**0.5
            shapes.append((bw, bh))
    shapes = torch.tensor(shapes)  # [A, 2]
    cy = (torch.arange(h, dtype=torch.float32) + 0.5) * stride
    cx = (torch.arange(w, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(cy, cx, indexing="ij")
    centres = torch.stack((xx, yy), -1).reshape(-1, 1, 2)  # [HW, 1, 2]
    half = shapes[None] / 2  # [1, A, 2]
    return torch.cat((centres - half, centres + half), -1).reshape(-1, 4)


def wh_iou(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    """IoU between boxes that share a centre — only widths and heights matter. [N,2]x[M,2] -> [N,M]."""
    inter = torch.minimum(wh1[:, None], wh2[None]).prod(-1)
    return inter / (wh1.prod(-1)[:, None] + wh2.prod(-1)[None] - inter)


def kmeans_anchors(wh: torch.Tensor, k: int = 9, iters: int = 300, seed: int = 0):
    """YOLOv2-style k-means on box shapes with distance d = 1 - IoU(box, centroid).

    Euclidean k-means on (w, h) over-weights large boxes; the IoU distance is scale-invariant
    in the sense that matters for matching. Returns (anchors [k, 2] sorted by area, mean best IoU).
    """
    g = torch.Generator().manual_seed(seed)
    wh = wh.float()
    centroids = wh[torch.randperm(len(wh), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = wh_iou(wh, centroids).argmax(1)
        new = torch.stack([wh[assign == j].median(0).values if (assign == j).any() else centroids[j] for j in range(k)])
        if torch.allclose(new, centroids):
            break
        centroids = new
    centroids = centroids[centroids.prod(1).argsort()]
    fitness = wh_iou(wh, centroids).max(1).values.mean().item()
    return centroids, fitness


def best_possible_recall(wh: torch.Tensor, anchors: torch.Tensor, thr: float = 4.0) -> float:
    """YOLOv5 'autoanchor' check: fraction of gt boxes matchable by some anchor.

    A gt matches an anchor when both side ratios are within [1/thr, thr] (anchor_t = 4).
    """
    r = wh[:, None] / anchors[None]
    worst = torch.minimum(r, 1 / r).min(2).values  # worst side ratio per (gt, anchor)
    return (worst.max(1).values > 1 / thr).float().mean().item()


def anchor_coverage(gt_boxes: torch.Tensor, anchors: torch.Tensor, pos_thr: float = 0.5) -> float:
    """Fraction of gt boxes having at least one anchor with IoU >= pos_thr (two-stage view)."""
    if len(gt_boxes) == 0:
        return 1.0
    return (pairwise_iou(gt_boxes, anchors).max(1).values >= pos_thr).float().mean().item()
