"""Box-aware augmentation, written without OpenCV so every step is visible.

Images are numpy uint8 arrays [H, W, 3]; boxes are numpy float arrays [N, 4] xyxy in pixels.
Semantics follow the Ultralytics implementations (LetterBox, Mosaic, RandomPerspective,
MixUp, RandomHSV) so the numbers in the book match what YOLO training actually does.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PAD_VALUE = 114  # the grey YOLO pads with: (114, 114, 114)


def resize(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Bilinear resize (antialiased when shrinking)."""
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    shrink = new_w < img.shape[1] or new_h < img.shape[0]
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=shrink)
    return t[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def letterbox(img: np.ndarray, new_shape=640, stride: int = 32, auto: bool = False, scaleup: bool = True, center: bool = True):
    """Resize keeping aspect ratio, then pad to new_shape with grey.

    Returns (image, r, (left, top)). A box maps as  x' = x * r + left,  y' = y * r + top.
    auto=True pads only to the next multiple of ``stride`` (minimum-rectangle inference).
    """
    h, w = img.shape[:2]
    nh, nw = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
    r = min(nh / h, nw / w)
    if not scaleup:
        r = min(r, 1.0)
    uw, uh = int(round(w * r)), int(round(h * r))
    dw, dh = nw - uw, nh - uh
    if auto:
        dw, dh = dw % stride, dh % stride
    if center:
        dw, dh = dw / 2, dh / 2
    if (uw, uh) != (w, h):
        img = resize(img, uw, uh)
    top, bottom = (int(round(dh - 0.1)), int(round(dh + 0.1))) if center else (0, int(dh))
    left, right = (int(round(dw - 0.1)), int(round(dw + 0.1))) if center else (0, int(dw))
    out = np.full((uh + top + bottom, uw + left + right, 3), PAD_VALUE, dtype=np.uint8)
    out[top : top + uh, left : left + uw] = img
    return out, r, (left, top)


def letterbox_boxes(boxes: np.ndarray, r: float, pad) -> np.ndarray:
    """Original-image boxes -> letterboxed-image boxes."""
    out = boxes * r
    out[:, [0, 2]] += pad[0]
    out[:, [1, 3]] += pad[1]
    return out


def unletterbox_boxes(boxes: np.ndarray, r: float, pad, orig_shape) -> np.ndarray:
    """Letterboxed-image boxes (model output) -> original-image boxes. The step people forget."""
    out = boxes.copy().astype(np.float64)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / r
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_shape[1])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_shape[0])
    return out


def hflip(img: np.ndarray, boxes: np.ndarray):
    w = img.shape[1]
    out = boxes.copy()
    out[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img[:, ::-1].copy(), out


def random_hsv(img: np.ndarray, hgain=0.015, sgain=0.7, vgain=0.4, rng=np.random):
    """Multiply hue/saturation/value by random factors in [1-gain, 1+gain] (hue wraps)."""
    if not (hgain or sgain or vgain):
        return img
    r = rng.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hsv = np.array(Image.fromarray(img).convert("HSV")).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return np.array(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


def _warp_affine(img: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Inverse-map warp with bilinear sampling; out-of-image pixels get PAD_VALUE."""
    Minv = np.linalg.inv(M)
    ys, xs = np.meshgrid(np.arange(out_h) + 0.5, np.arange(out_w) + 0.5, indexing="ij")
    pts = np.stack((xs, ys, np.ones_like(xs)), -1) @ Minv.T
    sx, sy = pts[..., 0] / pts[..., 2], pts[..., 1] / pts[..., 2]
    h, w = img.shape[:2]
    grid = torch.from_numpy(np.stack((sx / w * 2 - 1, sy / h * 2 - 1), -1)).float()[None]
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float() - PAD_VALUE
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + PAD_VALUE
    return out[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def box_candidates(b_before: np.ndarray, b_after: np.ndarray, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16) -> np.ndarray:
    """Keep boxes that are still >2 px, kept >10% of their (scaled) area, and are not slivers."""
    w1, h1 = b_before[:, 2] - b_before[:, 0], b_before[:, 3] - b_before[:, 1]
    w2, h2 = b_after[:, 2] - b_after[:, 0], b_after[:, 3] - b_after[:, 1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def random_affine(
    img: np.ndarray,
    boxes: np.ndarray,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    border=(0, 0),
    rng=np.random,
):
    """Ultralytics RandomPerspective: M = T @ S @ R @ P @ C, applied to image and box corners.

    border < 0 crops (used after mosaic: a 2s x 2s canvas with border=-s/2 becomes s x s).
    Returns (image, boxes, keep_mask).
    """
    out_h = img.shape[0] + border[0] * 2
    out_w = img.shape[1] + border[1] * 2
    C = np.eye(3)
    C[0, 2], C[1, 2] = -img.shape[1] / 2, -img.shape[0] / 2
    P = np.eye(3)
    P[2, 0], P[2, 1] = rng.uniform(-perspective, perspective), rng.uniform(-perspective, perspective)
    R = np.eye(3)
    a, s = rng.uniform(-degrees, degrees), rng.uniform(1 - scale, 1 + scale)
    ca, sa = math.cos(math.radians(a)) * s, math.sin(math.radians(a)) * s
    R[:2, :2] = [[ca, sa], [-sa, ca]]
    S = np.eye(3)
    S[0, 1] = math.tan(math.radians(rng.uniform(-shear, shear)))
    S[1, 0] = math.tan(math.radians(rng.uniform(-shear, shear)))
    T = np.eye(3)
    T[0, 2] = rng.uniform(0.5 - translate, 0.5 + translate) * out_w
    T[1, 2] = rng.uniform(0.5 - translate, 0.5 + translate) * out_h
    M = T @ S @ R @ P @ C
    img = _warp_affine(img, M, out_w, out_h)
    n = len(boxes)
    if n == 0:
        return img, boxes, np.zeros(0, dtype=bool)
    corners = np.ones((n * 4, 3))
    corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    corners = corners @ M.T
    corners = (corners[:, :2] / corners[:, 2:3]).reshape(n, 8)
    xs, ys = corners[:, [0, 2, 4, 6]], corners[:, [1, 3, 5, 7]]
    new = np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), 1)
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, out_w)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, out_h)
    keep = box_candidates(boxes * s, new)
    return img, new, keep


def load_resized(img: np.ndarray, size: int) -> np.ndarray:
    """Resize so the long side equals ``size`` (what YOLO dataloaders do before mosaic)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    return resize(img, max(1, int(round(w * r))), max(1, int(round(h * r)))) if r != 1 else img, r


def mosaic4(images, boxes_list, labels_list, size=640, rng=np.random, **affine):
    """4-image mosaic on a 2s x 2s canvas around a random centre, then random_affine crop to s x s.

    Every training image now contains objects at four different scales/contexts and the
    batch-norm statistics see four images per sample.
    """
    s = size
    xc, yc = (int(rng.uniform(s // 2, 3 * s // 2)) for _ in range(2))
    canvas = np.full((2 * s, 2 * s, 3), PAD_VALUE, dtype=np.uint8)
    all_b, all_l = [], []
    for i, (img, b, l) in enumerate(zip(images, boxes_list, labels_list)):
        img, r = load_resized(img, s)
        h, w = img.shape[:2]
        if i == 0:
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw, padh = x1a - x1b, y1a - y1b
        bb = np.asarray(b, dtype=np.float64) * r
        if len(bb):
            bb[:, [0, 2]] += padw
            bb[:, [1, 3]] += padh
        all_b.append(bb.reshape(-1, 4)), all_l.append(np.asarray(l).reshape(-1))
    boxes = np.concatenate(all_b).clip(0, 2 * s)
    labels = np.concatenate(all_l)
    img, boxes, keep = random_affine(canvas, boxes, border=(-s // 2, -s // 2), rng=rng, **affine)
    return img, boxes[keep], labels[keep]


def mixup(img1, boxes1, labels1, img2, boxes2, labels2, rng=np.random):
    """Blend two (mosaicked) images with r ~ Beta(32, 32) (≈ 0.5) and keep all boxes of both."""
    r = rng.beta(32.0, 32.0)
    img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)
    return img, np.concatenate((boxes1, boxes2)), np.concatenate((labels1, labels2))
