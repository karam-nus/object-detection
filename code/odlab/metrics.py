"""Detection metrics from scratch: COCO-style AP/AR, VOC AP, PR curves, error breakdown.

``coco_evaluate`` re-implements the matching and accumulation logic of pycocotools'
COCOeval (bbox mode) so every step is readable. The test-suite checks that it agrees
with pycocotools to 1e-6 on random data.

Input format (per image, all tensors on CPU):
    gts[i]   = {"boxes": [G,4] xyxy, "labels": [G], optional "iscrowd": [G] bool, optional "area": [G]}
    preds[i] = {"boxes": [D,4] xyxy, "scores": [D], "labels": [D]}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .boxes import box_area, pairwise_iou

COCO_IOU_THRS = np.linspace(0.5, 0.95, 10)
COCO_REC_THRS = np.linspace(0.0, 1.0, 101)
COCO_AREA_RNG = {"all": (0, 1e10), "small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e10)}


# ---------------------------------------------------------------------------
# Single-curve AP (the three interpolation conventions)
# ---------------------------------------------------------------------------


def average_precision(recall: np.ndarray, precision: np.ndarray, method: str = "coco101") -> float:
    """Area under a PR curve given cumulative recall/precision arrays (score-sorted).

    coco101    : mean of the interpolated precision at 101 recall points 0, 0.01, ..., 1
    voc11      : PASCAL VOC 2007 — mean of max precision at recall >= {0, 0.1, ..., 1}
    all_points : PASCAL VOC 2010+ — exact area under the monotone (interpolated) envelope
    """
    recall, precision = np.asarray(recall, float), np.asarray(precision, float)
    if method == "coco101":
        # pycocotools: make precision monotone, then read it at the first detection whose
        # recall reaches each threshold; thresholds never reached contribute 0.
        pr = np.flip(np.maximum.accumulate(np.flip(precision))) if len(precision) else precision
        q = np.zeros(len(COCO_REC_THRS))
        for ri, pi in enumerate(np.searchsorted(recall, COCO_REC_THRS, side="left")):
            if pi < len(pr):
                q[ri] = pr[pi]
        return float(q.mean())
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([0.0], precision, [0.0]))
    p_env = np.flip(np.maximum.accumulate(np.flip(p)))  # p_env[i] = max(p[i:])
    if method == "all_points":
        i = np.where(r[1:] != r[:-1])[0]
        return float(np.sum((r[i + 1] - r[i]) * p_env[i + 1]))
    if method == "voc11":
        return float(np.mean([p[r >= t].max() for t in np.linspace(0, 1, 11)]))
    raise ValueError(f"unknown method {method!r}")


def pr_curve(scores: np.ndarray, is_tp: np.ndarray, n_gt: int):
    """Cumulative precision/recall after sorting detections by descending score."""
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return recall, precision, scores[order]


# ---------------------------------------------------------------------------
# COCO-style evaluation
# ---------------------------------------------------------------------------


@dataclass
class CocoResult:
    """The 12 standard COCO numbers plus the raw precision array."""

    stats: dict
    precision: np.ndarray = field(repr=False)  # [T, R, K, A, M]
    recall: np.ndarray = field(repr=False)  # [T, K, A, M]
    categories: list = field(repr=False)

    def summary(self) -> str:
        names = [
            ("AP", "AP @[.50:.95] all  maxDets=100"),
            ("AP50", "AP @.50 all  maxDets=100"),
            ("AP75", "AP @.75 all  maxDets=100"),
            ("APs", "AP small"),
            ("APm", "AP medium"),
            ("APl", "AP large"),
            ("AR1", "AR maxDets=1"),
            ("AR10", "AR maxDets=10"),
            ("AR100", "AR maxDets=100"),
            ("ARs", "AR small"),
            ("ARm", "AR medium"),
            ("ARl", "AR large"),
        ]
        return "\n".join(f"{k:>6} = {self.stats[k]:.4f}   ({d})" for k, d in names)


def _evaluate_image(g, d, cat, area_rng, max_det, iou_thrs):
    """Match detections to ground truth for one (image, category, area range). Mirrors COCOeval.evaluateImg."""
    gm = g["labels"] == cat
    dm = d["labels"] == cat
    gb = g["boxes"][gm]
    db, ds = d["boxes"][dm], d["scores"][dm]
    crowd = g.get("iscrowd", torch.zeros(len(g["labels"]), dtype=torch.bool))[gm].numpy().astype(bool)
    garea = g.get("area", box_area(g["boxes"]))[gm].numpy()
    if len(gb) == 0 and len(db) == 0:
        return None
    g_ignore = crowd | (garea < area_rng[0]) | (garea > area_rng[1])
    # sort gt: non-ignored first (stable); sort dets by score, keep top max_det
    g_order = np.argsort(g_ignore, kind="mergesort")
    gb, crowd, g_ignore = gb[g_order], crowd[g_order], g_ignore[g_order]
    d_order = np.argsort(-ds.numpy(), kind="mergesort")[:max_det]
    db, ds = db[d_order], ds[d_order].numpy()
    ious = pairwise_iou(db, gb).numpy() if len(db) and len(gb) else np.zeros((len(db), len(gb)))
    if crowd.any() and len(db):
        # for crowd gt pycocotools uses intersection / area(det)
        inter = ious * 0
        for j in np.where(crowd)[0]:
            a, b = db, gb[j : j + 1]
            iw = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
            ih = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
            inter[:, j] = (iw * ih / box_area(a).clamp(min=1e-12)).numpy()
        ious[:, crowd] = inter[:, crowd]
    T, D, G = len(iou_thrs), len(db), len(gb)
    gt_matched = np.zeros((T, G))
    dt_matched = np.zeros((T, D))
    dt_ignore = np.zeros((T, D), dtype=bool)
    for ti, t in enumerate(iou_thrs):
        for di in range(D):
            best = min(t, 1 - 1e-10)
            m = -1
            for gi in range(G):
                if gt_matched[ti, gi] > 0 and not crowd[gi]:
                    continue  # already matched (crowd regions can absorb many dets)
                if m > -1 and not g_ignore[m] and g_ignore[gi]:
                    break  # we have a real match; ignored gts come later in the order
                if ious[di, gi] < best:
                    continue
                best, m = ious[di, gi], gi
            if m == -1:
                continue
            dt_ignore[ti, di] = g_ignore[m]
            dt_matched[ti, di] = 1
            gt_matched[ti, m] = 1
    darea = box_area(db).numpy()
    d_out = (darea < area_rng[0]) | (darea > area_rng[1])
    dt_ignore = dt_ignore | ((dt_matched == 0) & d_out[None, :])
    return {"scores": ds, "matched": dt_matched, "ignore": dt_ignore, "n_gt": int((~g_ignore).sum())}


def coco_evaluate(preds, gts, iou_thrs=COCO_IOU_THRS, max_dets=(1, 10, 100), area_rng=COCO_AREA_RNG) -> CocoResult:
    """COCO bbox evaluation. Returns AP, AP50, AP75, APs/m/l, AR1/10/100, ARs/m/l."""
    cats = sorted({int(c) for g in gts for c in g["labels"]} | {int(c) for p in preds for c in p["labels"]})
    T, R, K, A, M = len(iou_thrs), len(COCO_REC_THRS), len(cats), len(area_rng), len(max_dets)
    precision = -np.ones((T, R, K, A, M))
    recall = -np.ones((T, K, A, M))
    for k, cat in enumerate(cats):
        for a, rng in enumerate(area_rng.values()):
            per_img = [_evaluate_image(g, d, cat, rng, max_dets[-1], iou_thrs) for g, d in zip(gts, preds)]
            per_img = [e for e in per_img if e is not None]
            if not per_img:
                continue
            for m, md in enumerate(max_dets):
                scores = np.concatenate([e["scores"][:md] for e in per_img])
                order = np.argsort(-scores, kind="mergesort")
                matched = np.concatenate([e["matched"][:, :md] for e in per_img], axis=1)[:, order]
                ignore = np.concatenate([e["ignore"][:, :md] for e in per_img], axis=1)[:, order]
                n_gt = sum(e["n_gt"] for e in per_img)
                if n_gt == 0:
                    continue
                tps = np.logical_and(matched, ~ignore)
                fps = np.logical_and(~matched.astype(bool), ~ignore)
                tp_sum = np.cumsum(tps, axis=1).astype(float)
                fp_sum = np.cumsum(fps, axis=1).astype(float)
                for t in range(T):
                    tp, fp = tp_sum[t], fp_sum[t]
                    rc = tp / n_gt
                    pr = tp / (fp + tp + np.spacing(1))
                    recall[t, k, a, m] = rc[-1] if len(tp) else 0
                    pr = np.flip(np.maximum.accumulate(np.flip(pr))) if len(pr) else pr
                    q = np.zeros(R)
                    inds = np.searchsorted(rc, COCO_REC_THRS, side="left")
                    for ri, pi in enumerate(inds):
                        if pi < len(pr):
                            q[ri] = pr[pi]
                    precision[t, :, k, a, m] = q

    def _ap(t=None, area="all", md=100):
        a, m = list(area_rng).index(area), list(max_dets).index(md)
        p = precision[:, :, :, a, m] if t is None else precision[[list(iou_thrs).index(t) if t in iou_thrs else int(np.argmin(np.abs(iou_thrs - t)))], :, :, a, m]
        p = p[p > -1]
        return float(p.mean()) if p.size else -1.0

    def _ar(area="all", md=100):
        a, m = list(area_rng).index(area), list(max_dets).index(md)
        r = recall[:, :, a, m]
        r = r[r > -1]
        return float(r.mean()) if r.size else -1.0

    stats = {
        "AP": _ap(),
        "AP50": _ap(0.5),
        "AP75": _ap(0.75),
        "APs": _ap(area="small"),
        "APm": _ap(area="medium"),
        "APl": _ap(area="large"),
        "AR1": _ar(md=max_dets[0]),
        "AR10": _ar(md=max_dets[1]),
        "AR100": _ar(md=max_dets[2]),
        "ARs": _ar(area="small"),
        "ARm": _ar(area="medium"),
        "ARl": _ar(area="large"),
    }
    return CocoResult(stats, precision, recall, cats)


# ---------------------------------------------------------------------------
# Per-class matching at a single IoU threshold (VOC style / PR curves / F1)
# ---------------------------------------------------------------------------


def match_detections(preds, gts, iou_thr: float = 0.5):
    """Greedy score-ordered matching per class at one IoU threshold.

    Returns {class: (scores, is_tp, n_gt)} pooled over images — the raw material for
    PR curves, F1-vs-confidence curves and VOC AP.
    """
    out: dict[int, list] = {}
    for g, d in zip(gts, preds):
        for c in set(map(int, g["labels"])) | set(map(int, d["labels"])):
            gb = g["boxes"][g["labels"] == c]
            m = d["labels"] == c
            db, ds = d["boxes"][m], d["scores"][m]
            order = ds.argsort(descending=True)
            db, ds = db[order], ds[order]
            used = torch.zeros(len(gb), dtype=torch.bool)
            tp = np.zeros(len(db), dtype=bool)
            if len(gb) and len(db):
                ious = pairwise_iou(db, gb)
                for i in range(len(db)):
                    cand = ious[i].clone()
                    cand[used] = -1
                    j = int(cand.argmax())
                    if cand[j] >= iou_thr:
                        used[j] = True
                        tp[i] = True
            e = out.setdefault(c, [[], [], 0])
            e[0].append(ds.numpy()), e[1].append(tp)
            e[2] += len(gb)
    return {c: (np.concatenate(s), np.concatenate(t), n) for c, (s, t, n) in out.items()}


def voc_map(preds, gts, iou_thr: float = 0.5, method: str = "all_points") -> float:
    """PASCAL-VOC-style mAP at a single IoU threshold."""
    aps = []
    for c, (s, tp, n) in match_detections(preds, gts, iou_thr).items():
        if n == 0:
            continue
        rec, prec, _ = pr_curve(s, tp, n)
        aps.append(average_precision(rec, prec, method))
    return float(np.mean(aps)) if aps else 0.0


def best_f1_threshold(preds, gts, iou_thr: float = 0.5):
    """Confidence threshold that maximises micro-F1 — the number deployments actually need."""
    m = match_detections(preds, gts, iou_thr)
    s = np.concatenate([v[0] for v in m.values()])
    tp = np.concatenate([v[1] for v in m.values()])
    n = sum(v[2] for v in m.values())
    order = np.argsort(-s, kind="mergesort")
    s, tp = s[order], tp[order]
    ctp = np.cumsum(tp)
    prec = ctp / np.arange(1, len(tp) + 1)
    rec = ctp / max(n, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1)) if len(f1) else 0
    return (float(s[i]), float(f1[i])) if len(f1) else (0.0, 0.0)


# ---------------------------------------------------------------------------
# Error breakdown (a simplified TIDE: counts, not delta-AP)
# ---------------------------------------------------------------------------


def error_breakdown(preds, gts, score_thr: float = 0.3, fg_iou: float = 0.5, bg_iou: float = 0.1):
    """Classify every false positive and every missed ground truth (Bolya et al., 2020 taxonomy).

    cls  : IoU >= fg with a gt of another class
    loc  : IoU in [bg, fg) with a gt of the right class
    both : IoU in [bg, fg) with a gt of another class
    dupe : IoU >= fg with a right-class gt that was already matched
    bkg  : IoU < bg with every gt
    miss : gt never matched by any detection
    """
    counts = dict(tp=0, cls=0, loc=0, both=0, dupe=0, bkg=0, miss=0)
    for g, d in zip(gts, preds):
        keep = d["scores"] >= score_thr
        db, ds, dl = d["boxes"][keep], d["scores"][keep], d["labels"][keep]
        order = ds.argsort(descending=True)
        db, dl = db[order], dl[order]
        gb, gl = g["boxes"], g["labels"]
        matched = torch.zeros(len(gb), dtype=torch.bool)
        ious = pairwise_iou(db, gb) if len(db) and len(gb) else torch.zeros(len(db), len(gb))
        for i in range(len(db)):
            same = gl == dl[i]
            iou_same = torch.where(same, ious[i], torch.full_like(ious[i], -1)) if len(gb) else ious[i]
            iou_other = torch.where(~same, ious[i], torch.full_like(ious[i], -1)) if len(gb) else ious[i]
            free_same = torch.where(same & ~matched, ious[i], torch.full_like(ious[i], -1)) if len(gb) else ious[i]
            if len(gb) and free_same.max() >= fg_iou:
                matched[free_same.argmax()] = True
                counts["tp"] += 1
            elif len(gb) and iou_same.max() >= fg_iou:
                counts["dupe"] += 1
            elif len(gb) and iou_other.max() >= fg_iou:
                counts["cls"] += 1
            elif len(gb) and iou_same.max() >= bg_iou:
                counts["loc"] += 1
            elif len(gb) and iou_other.max() >= bg_iou:
                counts["both"] += 1
            else:
                counts["bkg"] += 1
        counts["miss"] += int((~matched).sum())
    return counts
