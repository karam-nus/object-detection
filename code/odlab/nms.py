"""Post-processing: greedy NMS and its variants.

All functions take xyxy boxes [N, 4] and scores [N]. They are written for clarity
(Python loops over kept boxes), not speed; ``torchvision.ops.nms`` is the fast path.
"""

from __future__ import annotations

import torch

from .boxes import box_area, diou, pairwise_iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """Greedy NMS. Returns indices of kept boxes, sorted by decreasing score.

    Repeatedly take the highest-scoring remaining box, keep it, and delete every
    remaining box whose IoU with it exceeds ``iou_thr``. O(N^2) in the worst case.
    """
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = pairwise_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.stack(keep) if keep else boxes.new_zeros(0, dtype=torch.long)


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """Class-aware NMS via the coordinate-offset trick.

    Shift each class's boxes by (label * (max_coord + 1)) so boxes of different classes can
    never overlap, then run a single class-agnostic NMS. This is what torchvision and
    Ultralytics do; it turns C small NMS calls into one.
    """
    if boxes.numel() == 0:
        return boxes.new_zeros(0, dtype=torch.long)
    offset = labels.to(boxes) * (boxes.max() + 1)
    return nms(boxes + offset[:, None], scores, iou_thr)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    iou_thr: float = 0.3,
    score_thr: float = 0.001,
    method: str = "gaussian",
):
    """Soft-NMS (Bodla et al., 2017): decay overlapping scores instead of deleting boxes.

    linear:   s_j <- s_j * (1 - IoU)            if IoU > iou_thr
    gaussian: s_j <- s_j * exp(-IoU^2 / sigma)  for every remaining box
    Returns (kept indices in selection order, their decayed scores).
    """
    scores = scores.clone()
    idx = torch.arange(len(scores))
    keep, kept_scores = [], []
    while idx.numel() > 0:
        top = scores[idx].argmax()
        i = idx[top]
        keep.append(i)
        kept_scores.append(scores[i].clone())
        idx = torch.cat((idx[:top], idx[top + 1 :]))
        if idx.numel() == 0:
            break
        ious = pairwise_iou(boxes[i : i + 1], boxes[idx])[0]
        if method == "linear":
            decay = torch.where(ious > iou_thr, 1 - ious, torch.ones_like(ious))
        else:
            decay = torch.exp(-(ious**2) / sigma)
        scores[idx] = scores[idx] * decay
        idx = idx[scores[idx] > score_thr]
    if not keep:
        return boxes.new_zeros(0, dtype=torch.long), scores.new_zeros(0)
    return torch.stack(keep), torch.stack(kept_scores)


def diou_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """DIoU-NMS (Zheng et al., 2020): suppress when IoU - rho^2/c^2 > threshold.

    Two heavily overlapping boxes whose centres are far apart (two people side by side,
    one partly behind the other) are penalised less, so both survive more often.
    """
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        d = diou(boxes[i : i + 1].expand(order.numel() - 1, 4), boxes[order[1:]])
        order = order[1:][d <= iou_thr]
    return torch.stack(keep) if keep else boxes.new_zeros(0, dtype=torch.long)


def matrix_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor | None = None,
    kernel: str = "gaussian",
    sigma: float = 2.0,
) -> torch.Tensor:
    """Matrix NMS (Wang et al., SOLOv2, 2020), applied to boxes.

    Fully parallel: every box's score is decayed by its worst overlap with any
    higher-scoring box, compensated by how suppressed that higher box itself is.
    Returns decayed scores in the original order; threshold them afterwards.
    """
    n = len(scores)
    if n == 0:
        return scores
    order = scores.argsort(descending=True)
    b, s = boxes[order], scores[order]
    ious = pairwise_iou(b, b).triu(diagonal=1)  # ious[i, j] for i scored above j
    if labels is not None:
        lab = labels[order]
        ious = ious * (lab[:, None] == lab[None, :]).to(ious)
    comp = ious.max(dim=0).values  # how much each box is itself overlapped from above
    comp = comp[:, None].expand(n, n)  # compensation for row i (the suppressor)
    if kernel == "linear":
        decay = ((1 - ious) / (1 - comp)).min(dim=0).values
    else:
        decay = torch.exp(-sigma * (ious**2 - comp**2)).min(dim=0).values
    out = torch.empty_like(scores)
    out[order] = s * decay
    return out


def weighted_boxes_fusion(
    boxes_list: list[torch.Tensor],
    scores_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
):
    """Weighted Boxes Fusion (Solovyev et al., 2019) for ensembling several detectors.

    Instead of discarding overlapping boxes, average their coordinates weighted by score.
    Fused confidence = mean score * min(#boxes in cluster, #models) / #models.
    Inputs: one tensor per model. Returns fused (boxes, scores, labels).
    """
    n_models = len(boxes_list)
    all_b, all_s, all_l = [], [], []
    for b, s, l in zip(boxes_list, scores_list, labels_list):
        m = s >= skip_box_thr
        all_b.append(b[m]), all_s.append(s[m]), all_l.append(l[m])
    B, S, L = torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)
    out_b, out_s, out_l = [], [], []
    for c in L.unique():
        m = L == c
        bc, sc = B[m], S[m]
        order = sc.argsort(descending=True)
        bc, sc = bc[order], sc[order]
        clusters: list[list[int]] = []
        fused: list[torch.Tensor] = []
        for i in range(len(bc)):
            if fused:
                ious = pairwise_iou(bc[i : i + 1], torch.stack(fused))[0]
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    clusters[j].append(i)
                    w = sc[clusters[j]]
                    fused[j] = (bc[clusters[j]] * w[:, None]).sum(0) / w.sum()
                    continue
            clusters.append([i])
            fused.append(bc[i].clone())
        for cl, fb in zip(clusters, fused):
            conf = sc[cl].mean() * min(len(cl), n_models) / n_models
            out_b.append(fb), out_s.append(conf), out_l.append(c)
    if not out_b:
        return B.new_zeros(0, 4), S.new_zeros(0), L.new_zeros(0)
    return torch.stack(out_b), torch.stack(out_s), torch.stack(out_l)


def topk_select(scores: torch.Tensor, k: int = 300):
    """End-to-end (NMS-free) post-processing: take the k best (anchor, class) pairs.

    ``scores`` is [A, C] class probabilities from a one-to-one head. Returns
    (anchor indices, class labels, scores). This is the entire post-processor of
    YOLOv10 / YOLO26-e2e / DETR-family models.
    """
    flat = scores.flatten()
    k = min(k, flat.numel())
    vals, idx = flat.topk(k)
    c = scores.shape[1]
    return idx // c, idx % c, vals


def nms_flops_estimate(n: int) -> int:
    """Worst-case pairwise IoU evaluations for greedy NMS over n boxes: n(n-1)/2."""
    return n * (n - 1) // 2


__all__ = [
    "nms",
    "batched_nms",
    "soft_nms",
    "diou_nms",
    "matrix_nms",
    "weighted_boxes_fusion",
    "topk_select",
    "nms_flops_estimate",
    "box_area",
]
