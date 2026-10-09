"""Label assignment: deciding which predictions are responsible for which objects.

Single-image implementations, written to be read side by side:

    max_iou_assign       Faster R-CNN / RetinaNet / SSD   (static, IoU thresholds on anchors)
    atss_assign          ATSS (Zhang et al., 2020)        (static, statistics-adaptive threshold)
    simota_assign        YOLOX (Ge et al., 2021)           (dynamic, cost + dynamic-k)
    task_aligned_assign  TOOD / YOLOv6 / v8 / 11 / 26     (dynamic, s^alpha * u^beta, top-k)
    hungarian_assign     DETR family                      (one-to-one, bipartite matching)

Conventions: N anchors/points/queries, G ground-truth boxes, C classes.
Return value: ``gt_idx`` [N] long, -1 = background (and -2 = ignore where defined).
"""

from __future__ import annotations

import torch

from .boxes import ciou, giou, pairwise_iou, xyxy_to_cxcywh

# ---------------------------------------------------------------------------
# Static, IoU-threshold assignment
# ---------------------------------------------------------------------------


def max_iou_assign(anchors, gt_boxes, pos_thr=0.5, neg_thr=0.4, min_pos_iou=0.0, allow_low_quality=True):
    """Anchor-based assignment by IoU thresholds (RetinaNet defaults 0.5 / 0.4; RPN uses 0.7 / 0.3).

    IoU >= pos_thr -> positive; IoU < neg_thr -> negative; in between -> ignored (-2).
    allow_low_quality: every gt also claims its best anchor even below pos_thr, so no gt
    is left without a positive (this is where most tiny objects get their only anchor).
    """
    n = len(anchors)
    gt_idx = torch.full((n,), -1, dtype=torch.long)
    if len(gt_boxes) == 0:
        return gt_idx
    ious = pairwise_iou(anchors, gt_boxes)  # [N, G]
    best_iou, best_gt = ious.max(1)
    gt_idx[(best_iou >= neg_thr) & (best_iou < pos_thr)] = -2
    pos = best_iou >= pos_thr
    gt_idx[pos] = best_gt[pos]
    if allow_low_quality:
        gt_best_iou, gt_best_anchor = ious.max(0)
        for g in range(len(gt_boxes)):
            if gt_best_iou[g] >= min_pos_iou and gt_best_iou[g] > 0:
                gt_idx[gt_best_anchor[g]] = g
    return gt_idx


def atss_assign(anchors, num_per_level, gt_boxes, topk=9):
    """Adaptive Training Sample Selection.

    For each gt: take the k anchors per level whose centres are closest to the gt centre,
    compute their IoUs, and use mean + std of those IoUs as that gt's own threshold.
    Positives must also have their centre inside the gt. Almost hyperparameter-free.
    """
    n = len(anchors)
    gt_idx = torch.full((n,), -1, dtype=torch.long)
    if len(gt_boxes) == 0:
        return gt_idx
    ious = pairwise_iou(anchors, gt_boxes)  # [N, G]
    ac = xyxy_to_cxcywh(anchors)[:, :2]
    gc = xyxy_to_cxcywh(gt_boxes)[:, :2]
    dist = torch.cdist(ac, gc)  # [N, G]
    cand = []
    start = 0
    for n_l in num_per_level:
        k = min(topk, n_l)
        cand.append(dist[start : start + n_l].topk(k, dim=0, largest=False).indices + start)
        start += n_l
    cand = torch.cat(cand, 0)  # [k * L, G]
    cand_iou = ious.gather(0, cand)
    thr = cand_iou.mean(0) + cand_iou.std(0)  # per-gt adaptive threshold
    is_pos = cand_iou >= thr[None]
    cx, cy = ac[cand, 0], ac[cand, 1]
    inside = (cx > gt_boxes[None, :, 0]) & (cx < gt_boxes[None, :, 2]) & (cy > gt_boxes[None, :, 1]) & (cy < gt_boxes[None, :, 3])
    is_pos &= inside
    # an anchor chosen by several gts goes to the one with the highest IoU
    pos_iou = torch.full_like(ious, -1.0)
    for g in range(len(gt_boxes)):
        sel = cand[is_pos[:, g], g]
        pos_iou[sel, g] = ious[sel, g]
    best_iou, best_gt = pos_iou.max(1)
    gt_idx[best_iou > -1] = best_gt[best_iou > -1]
    return gt_idx


# ---------------------------------------------------------------------------
# Dynamic assignment
# ---------------------------------------------------------------------------


def simota_assign(pred_scores, pred_boxes, points, strides, gt_boxes, gt_labels, center_radius=2.5, iou_weight=3.0, candidate_topk=10):
    """SimOTA (YOLOX): optimal-transport-flavoured assignment solved greedily.

    1. Candidates: points inside the gt box OR within center_radius * stride of its centre.
    2. Cost = BCE(cls prob, onehot) + iou_weight * (-log IoU) + 1e5 * (not in both regions).
    3. Dynamic k for each gt = max(1, int(sum of its top-10 candidate IoUs)).
       A well-predicted, large object gets more positives; a hard one gets few.
    4. Each gt takes its k lowest-cost points; a point claimed twice goes to the cheaper gt.
    pred_scores: [N, C] probabilities. Returns (gt_idx [N], matched IoU [N]).
    """
    n = len(points)
    gt_idx = torch.full((n,), -1, dtype=torch.long)
    matched_iou = torch.zeros(n)
    if len(gt_boxes) == 0:
        return gt_idx, matched_iou
    px, py = points[:, 0:1], points[:, 1:2]
    in_box = (px > gt_boxes[None, :, 0]) & (px < gt_boxes[None, :, 2]) & (py > gt_boxes[None, :, 1]) & (py < gt_boxes[None, :, 3])
    gc = xyxy_to_cxcywh(gt_boxes)
    r = center_radius * strides  # [N, 1]
    in_ctr = (px > gc[None, :, 0] - r) & (px < gc[None, :, 0] + r) & (py > gc[None, :, 1] - r) & (py < gc[None, :, 1] + r)
    cand = (in_box | in_ctr).any(1)  # [N]
    both = (in_box & in_ctr)[cand]  # [n_cand, G]
    ious = pairwise_iou(pred_boxes[cand], gt_boxes)  # [n_cand, G]
    probs = pred_scores[cand].clamp(1e-7, 1 - 1e-7)  # [n_cand, C]
    p = probs[:, gt_labels]  # [n_cand, G] prob of each gt's class
    # sum_c BCE(p_c, onehot_c) = -log p_gt - sum_{c != gt} log(1 - p_c)
    all_neg = -torch.log(1 - probs).sum(1, keepdim=True)  # [n_cand, 1]
    cls_cost = all_neg - torch.log(p) + torch.log(1 - p)
    cost = cls_cost + iou_weight * (-torch.log(ious + 1e-8)) + 1e5 * (~both)
    k_iou = ious.topk(min(candidate_topk, len(ious)), dim=0).values
    dyn_k = k_iou.sum(0).int().clamp(min=1)  # [G]
    match = torch.zeros_like(cost, dtype=torch.bool)
    for g in range(len(gt_boxes)):
        idx = cost[:, g].topk(int(dyn_k[g]), largest=False).indices
        match[idx, g] = True
    multi = match.sum(1) > 1
    if multi.any():
        best = cost[multi].argmin(1)
        match[multi] = False
        match[multi.nonzero()[:, 0], best] = True
    cand_idx = cand.nonzero()[:, 0]
    has = match.any(1)
    gt_idx[cand_idx[has]] = match[has].float().argmax(1)
    matched_iou[cand_idx[has]] = ious[has].gather(1, gt_idx[cand_idx[has]][:, None])[:, 0]
    return gt_idx, matched_iou


def task_aligned_assign(
    pred_scores,
    pred_boxes,
    points,
    gt_boxes,
    gt_labels,
    topk=10,
    alpha=0.5,
    beta=6.0,
    topk2=None,
    small_side_floor=None,
    eps=1e-9,
):
    """Task-Aligned Assigner (TOOD, Feng et al., 2021) as used by YOLOv6/v8/11/26.

    t = s^alpha * u^beta, where s is the predicted probability of the gt's class at that point
    and u the IoU (CIoU, clamped >= 0) between the predicted box and the gt. A point is a
    good positive only if it is BOTH confident and well-localized — the two heads are
    pushed to agree. Ultralytics uses alpha=0.5, beta=6.0; the TOOD paper used alpha=1.

    small_side_floor (STAL, YOLO26): gt sides shorter than this many pixels are enlarged to it
        (about the centre) when deciding which points lie "inside" the gt, so tiny objects
        that fall between grid points still receive positives.
    topk2: after resolving conflicts, keep only the best topk2 points per gt (YOLO26's
        one-to-one branch uses topk=7, topk2=1).

    Returns dict(gt_idx [N], fg [N] bool, target_labels [N], target_boxes [N,4],
                 target_scores [N, C] soft targets = normalised alignment).
    """
    n, c = pred_scores.shape
    g = len(gt_boxes)
    out = dict(
        gt_idx=torch.full((n,), -1, dtype=torch.long),
        fg=torch.zeros(n, dtype=torch.bool),
        target_labels=torch.zeros(n, dtype=torch.long),
        target_boxes=torch.zeros(n, 4),
        target_scores=torch.zeros(n, c),
    )
    if g == 0:
        return out
    cand_boxes = gt_boxes.clone()
    if small_side_floor is not None:
        cxcywh = xyxy_to_cxcywh(cand_boxes)
        cxcywh[:, 2:] = cxcywh[:, 2:].clamp(min=small_side_floor)
        cand_boxes = torch.cat((cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2), 1)
    px, py = points[None, :, 0], points[None, :, 1]
    inside = (px - cand_boxes[:, 0:1] > eps) & (py - cand_boxes[:, 1:2] > eps) & (cand_boxes[:, 2:3] - px > eps) & (cand_boxes[:, 3:4] - py > eps)  # [G, N]
    u = ciou(gt_boxes[:, None, :], pred_boxes[None, :, :]).clamp(min=0) * inside  # [G, N]
    s = pred_scores[:, gt_labels].T * inside  # [G, N]
    t = s.pow(alpha) * u.pow(beta)  # alignment metric
    k = min(topk, n)
    top = t.topk(k, dim=1).indices  # [G, k]
    mask = torch.zeros(g, n, dtype=torch.bool)
    mask.scatter_(1, top, True)
    mask &= inside
    # a point selected by several gts keeps only the gt it overlaps most
    multi = mask.sum(0) > 1
    if multi.any():
        best = u[:, multi].argmax(0)
        mask[:, multi] = False
        mask[best, multi.nonzero()[:, 0]] = True
    if topk2 is not None and topk2 != topk:
        keep = (t * mask).topk(min(topk2, n), dim=1).indices
        m2 = torch.zeros_like(mask)
        m2.scatter_(1, keep, True)
        mask &= m2
    fg = mask.any(0)
    gt_idx = mask.float().argmax(0)
    out["fg"] = fg
    out["gt_idx"] = torch.where(fg, gt_idx, torch.full_like(gt_idx, -1))
    out["target_labels"] = gt_labels[gt_idx]
    out["target_boxes"] = gt_boxes[gt_idx]
    # soft target: alignment normalised so each gt's best point gets that gt's best IoU
    tm = t * mask
    norm = (tm * u.mul(mask).amax(1, keepdim=True) / (tm.amax(1, keepdim=True) + eps)).amax(0)  # [N]
    out["target_scores"][fg, gt_labels[gt_idx[fg]]] = norm[fg]
    return out


# ---------------------------------------------------------------------------
# One-to-one assignment (DETR)
# ---------------------------------------------------------------------------


def detr_match_cost(pred_logits, pred_boxes, gt_labels, gt_boxes, w_cls=2.0, w_l1=5.0, w_giou=2.0, alpha=0.25, gamma=2.0):
    """Matching cost used by Deformable-DETR / DINO / RT-DETR / D-FINE (focal-style class cost).

    pred_boxes and gt_boxes are xyxy normalised to [0, 1]. Returns cost [Q, G].
    """
    p = pred_logits.sigmoid()[:, gt_labels]  # [Q, G]
    neg = (1 - alpha) * p.pow(gamma) * (-(1 - p + 1e-8).log())
    pos = alpha * (1 - p).pow(gamma) * (-(p + 1e-8).log())
    cost_cls = pos - neg
    cost_l1 = torch.cdist(xyxy_to_cxcywh(pred_boxes), xyxy_to_cxcywh(gt_boxes), p=1)
    cost_giou = -giou(pred_boxes[:, None], gt_boxes[None])
    return w_cls * cost_cls + w_l1 * cost_l1 + w_giou * cost_giou


def hungarian_assign(cost):
    """Optimal one-to-one matching (Kuhn-Munkres via scipy). cost [Q, G] -> gt_idx [Q] (-1 unmatched)."""
    from scipy.optimize import linear_sum_assignment

    q = cost.shape[0]
    gt_idx = torch.full((q,), -1, dtype=torch.long)
    if cost.numel() == 0:
        return gt_idx
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    gt_idx[torch.as_tensor(rows)] = torch.as_tensor(cols)
    return gt_idx
