"""Box representations and the IoU family.

Conventions used everywhere in odlab:
    * boxes are float tensors with a trailing dimension of 4
    * the canonical format is ``xyxy`` in pixels: (x1, y1, x2, y2), x2 > x1, y2 > y1
    * no "+1" pixel convention (COCO / pycocotools / torchvision all use continuous coordinates)

Every IoU variant is implemented once, element-wise, on broadcastable inputs.
The pairwise [N, M] matrix is the element-wise function applied to a[:, None] and b[None, :].
"""

from __future__ import annotations

import math

import torch

EPS = 1e-7

# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------


def xyxy_to_xywh(b: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) -> (x1, y1, w, h). The COCO annotation format."""
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack((x1, y1, x2 - x1, y2 - y1), -1)


def xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """(x1, y1, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = b.unbind(-1)
    return torch.stack((x, y, x + w, y + h), -1)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) -> (cx, cy, w, h). The YOLO / DETR internal format."""
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), -1)


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)


def normalize(b: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Pixel boxes -> [0, 1] boxes (any 4-tuple format whose entries alternate x, y)."""
    scale = b.new_tensor([width, height, width, height])
    return b / scale


def denormalize(b: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """[0, 1] boxes -> pixel boxes."""
    scale = b.new_tensor([width, height, width, height])
    return b * scale


def box_area(b: torch.Tensor) -> torch.Tensor:
    """Area of xyxy boxes; degenerate boxes have zero area."""
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def clip_boxes(b: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Clip xyxy boxes to the image rectangle."""
    out = b.clone()
    out[..., 0::2] = out[..., 0::2].clamp(0, width)
    out[..., 1::2] = out[..., 1::2].clamp(0, height)
    return out


# ---------------------------------------------------------------------------
# The IoU family (element-wise on broadcastable inputs)
# ---------------------------------------------------------------------------


def _intersection_union(a: torch.Tensor, b: torch.Tensor):
    iw = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    ih = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter, union


def _enclosing(a: torch.Tensor, b: torch.Tensor):
    """Width and height of the smallest axis-aligned box enclosing a and b."""
    cw = torch.maximum(a[..., 2], b[..., 2]) - torch.minimum(a[..., 0], b[..., 0])
    ch = torch.maximum(a[..., 3], b[..., 3]) - torch.minimum(a[..., 1], b[..., 1])
    return cw, ch


def _center_dist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dx = (a[..., 0] + a[..., 2] - b[..., 0] - b[..., 2]) / 2
    dy = (a[..., 1] + a[..., 3] - b[..., 1] - b[..., 3]) / 2
    return dx * dx + dy * dy


def iou(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Plain IoU = |A ∩ B| / |A ∪ B|."""
    inter, union = _intersection_union(a, b)
    return inter / (union + eps)


def giou(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Generalized IoU (Rezatofighi et al., 2019): IoU - |C \\ (A ∪ B)| / |C|.

    Range (-1, 1]. Non-zero gradient even when the boxes do not overlap.
    """
    inter, union = _intersection_union(a, b)
    cw, ch = _enclosing(a, b)
    c_area = cw * ch
    return inter / (union + eps) - (c_area - union) / (c_area + eps)


def diou(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Distance IoU (Zheng et al., 2020): IoU - rho^2 / c^2.

    rho = distance between box centres, c = diagonal of the enclosing box.
    """
    cw, ch = _enclosing(a, b)
    c2 = cw * cw + ch * ch + eps
    return iou(a, b, eps) - _center_dist2(a, b) / c2


def ciou(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Complete IoU (Zheng et al., 2020): DIoU - alpha * v.

    v measures aspect-ratio disagreement; alpha = v / ((1 - IoU) + v) is treated as a
    constant in the backward pass (as in the paper's reference code and torchvision).
    """
    i = iou(a, b, eps)
    w1, h1 = a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]
    w2, h2 = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    v = (4 / math.pi**2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - i + v + eps)
    cw, ch = _enclosing(a, b)
    c2 = cw * cw + ch * ch + eps
    return i - _center_dist2(a, b) / c2 - alpha * v


def eiou(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Efficient IoU (Zhang et al., 2021): DIoU with separate width and height penalties."""
    w1, h1 = a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]
    w2, h2 = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    cw, ch = _enclosing(a, b)
    c2 = cw * cw + ch * ch + eps
    return iou(a, b, eps) - _center_dist2(a, b) / c2 - (w1 - w2) ** 2 / (cw * cw + eps) - (h1 - h2) ** 2 / (ch * ch + eps)


def siou(a: torch.Tensor, b: torch.Tensor, theta: float = 4.0, eps: float = EPS) -> torch.Tensor:
    """SCYLLA-IoU (Gevorgyan, 2022): IoU - (distance_cost + shape_cost) / 2.

    The angle cost steers the predicted centre toward the nearer axis first.
    ``a`` is treated as the prediction and ``b`` as the target.
    """
    i = iou(a, b, eps)
    w1, h1 = a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]
    w2, h2 = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    s_cw = (b[..., 0] + b[..., 2] - a[..., 0] - a[..., 2]) / 2
    s_ch = (b[..., 1] + b[..., 3] - a[..., 1] - a[..., 3]) / 2
    sigma = torch.sqrt(s_cw**2 + s_ch**2) + eps
    sin_alpha = torch.abs(s_ch) / sigma
    sin_beta = torch.abs(s_cw) / sigma
    thr = math.sqrt(2) / 2
    sin_x = torch.where(sin_alpha > thr, sin_beta, sin_alpha)
    angle_cost = torch.cos(2 * torch.arcsin(sin_x.clamp(-1, 1)) - math.pi / 2)
    cw, ch = _enclosing(a, b)
    rho_x = (s_cw / (cw + eps)) ** 2
    rho_y = (s_ch / (ch + eps)) ** 2
    gamma = 2 - angle_cost
    distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)
    omega_w = torch.abs(w1 - w2) / torch.maximum(w1, w2).clamp(min=eps)
    omega_h = torch.abs(h1 - h2) / torch.maximum(h1, h2).clamp(min=eps)
    shape_cost = (1 - torch.exp(-omega_w)) ** theta + (1 - torch.exp(-omega_h)) ** theta
    return i - (distance_cost + shape_cost) / 2


def nwd(a: torch.Tensor, b: torch.Tensor, constant: float = 12.8) -> torch.Tensor:
    """Normalized Gaussian Wasserstein Distance (Wang et al., 2021) for tiny objects.

    Each box is modelled as a 2-D Gaussian N((cx, cy), diag(w^2/4, h^2/4)). The squared
    2-Wasserstein distance between two such Gaussians has the closed form below, and
    NWD = exp(-sqrt(W2^2) / C). Unlike IoU it is smooth and non-zero for disjoint boxes,
    and a 1-pixel offset costs a 6x6 box no more than it costs a 60x60 box.
    ``constant`` is dataset-dependent; 12.8 is the AI-TOD value from the paper.
    """
    ca, cb = xyxy_to_cxcywh(a), xyxy_to_cxcywh(b)
    va = torch.stack((ca[..., 0], ca[..., 1], ca[..., 2] / 2, ca[..., 3] / 2), -1)
    vb = torch.stack((cb[..., 0], cb[..., 1], cb[..., 2] / 2, cb[..., 3] / 2), -1)
    w2 = ((va - vb) ** 2).sum(-1)
    return torch.exp(-torch.sqrt(w2 + EPS) / constant)


_KINDS = {"iou": iou, "giou": giou, "diou": diou, "ciou": ciou, "eiou": eiou, "siou": siou, "nwd": nwd}


def elementwise_iou(a: torch.Tensor, b: torch.Tensor, kind: str = "iou") -> torch.Tensor:
    """IoU variant between matched pairs a[i] <-> b[i]; shapes broadcast."""
    return _KINDS[kind](a, b)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor, kind: str = "iou") -> torch.Tensor:
    """[N, 4] x [M, 4] -> [N, M] matrix of the chosen IoU variant."""
    return _KINDS[kind](a[:, None, :], b[None, :, :])
