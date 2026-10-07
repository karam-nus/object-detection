"""Detection losses: classification (focal family), box regression (IoU family, DFL), helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .boxes import elementwise_iou

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def sigmoid_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
    """Focal loss (Lin et al., 2017): FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t).

    gamma=0, alpha=-1 recovers plain BCE. With gamma=2 an easy negative at p=0.01 is
    down-weighted by (0.01)^2 = 1e-4 relative to BCE — that is how 100k background anchors
    stop drowning 20 foreground ones.
    """
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return _reduce(loss, reduction)


def quality_focal_loss(logits, soft_targets, beta: float = 2.0, reduction: str = "none"):
    """QFL (Li et al., GFL, 2020): BCE against a *soft* target y in [0,1] (the IoU),
    modulated by |y - sigma|^beta. Joins classification score and localization quality
    into one number, so NMS ranks by something that reflects box quality.
    """
    p = logits.sigmoid()
    loss = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none") * (soft_targets - p).abs().pow(beta)
    return _reduce(loss, reduction)


def varifocal_loss(logits, soft_targets, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "none"):
    """VFL (Zhang et al., VarifocalNet, 2021): asymmetric — positives weighted by their target q,
    negatives by alpha * p^gamma. Used by PP-YOLOE, RT-DETR (as an option), YOLOv6, DAMO-YOLO.
    """
    p = logits.sigmoid()
    pos = (soft_targets > 0).to(p)
    weight = soft_targets * pos + alpha * p.pow(gamma) * (1 - pos)
    loss = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none") * weight
    return _reduce(loss, reduction)


def bce_with_soft_targets(logits, soft_targets, reduction: str = "none"):
    """What YOLOv8/11/26 actually use: plain BCE against TAL's soft (alignment-weighted) targets."""
    return _reduce(F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none"), reduction)


# ---------------------------------------------------------------------------
# Box regression
# ---------------------------------------------------------------------------


def iou_loss(pred_xyxy, target_xyxy, kind: str = "ciou", weight=None, reduction: str = "mean"):
    """1 - IoU-variant between matched pairs."""
    loss = 1.0 - elementwise_iou(pred_xyxy, target_xyxy, kind)
    if weight is not None:
        loss = loss * weight
    return _reduce(loss, reduction)


def bbox2dist(points, boxes, reg_max: int | None = None):
    """xyxy boxes -> (left, top, right, bottom) distances from points, in the same units.

    When reg_max is given (DFL heads), distances are clamped to [0, reg_max - 1 - 0.01]
    because the distribution only has bins 0..reg_max-1.
    """
    lt = points - boxes[..., :2]
    rb = boxes[..., 2:] - points
    d = torch.cat((lt, rb), -1)
    return d.clamp(0, reg_max - 1 - 0.01) if reg_max is not None else d


def dist2bbox(points, dist):
    """(l, t, r, b) distances -> xyxy boxes."""
    lt, rb = dist.chunk(2, -1)
    return torch.cat((points - lt, points + rb), -1)


def dfl_decode(pred_dist_logits, reg_max: int = 16):
    """Distribution -> expected distance. [..., 4 * reg_max] logits -> [..., 4] in stride units.

    Each side is a softmax over bins {0, 1, ..., reg_max-1}; the prediction is its mean.
    This is a fixed 1x1 'conv' with weights 0..reg_max-1 after a softmax — the op that
    YOLO26 removed because it is awkward for some compilers and for INT8.
    """
    shape = pred_dist_logits.shape[:-1]
    probs = pred_dist_logits.view(*shape, 4, reg_max).softmax(-1)
    bins = torch.arange(reg_max, dtype=probs.dtype, device=probs.device)
    return (probs * bins).sum(-1)


def dfl_loss(pred_dist_logits, target_dist, reg_max: int = 16):
    """Distribution Focal Loss (Li et al., GFL, 2020).

    A continuous target y lying between bins y_l = floor(y) and y_r = y_l + 1 is encoded as the
    two-hot distribution with weights (y_r - y) and (y - y_l); the loss is cross-entropy against
    it. Its minimum puts all mass on the two neighbouring bins, and the expectation of that
    distribution equals y exactly. pred [M, 4*reg_max] logits, target [M, 4] -> [M] loss.
    """
    target = target_dist.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()
    tr = tl + 1
    wl = tr.to(target) - target
    wr = 1 - wl
    logp = F.log_softmax(pred_dist_logits.view(-1, 4, reg_max), -1)
    loss = -(logp.gather(-1, tl[..., None])[..., 0] * wl + logp.gather(-1, tr[..., None])[..., 0] * wr)
    return loss.mean(-1)


def l1_distance_loss(pred_dist, target_dist, norm=None):
    """YOLO26's DFL replacement: L1 on (l, t, r, b) distances, normalised by image size (``norm``)."""
    if norm is not None:
        pred_dist, target_dist = pred_dist / norm, target_dist / norm
    return (pred_dist - target_dist).abs().mean(-1)


def _reduce(x, reduction):
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    return x
