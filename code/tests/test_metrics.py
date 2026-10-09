import contextlib
import io

import numpy as np
import pytest
import torch

from odlab.metrics import average_precision, best_f1_threshold, coco_evaluate, error_breakdown, pr_curve, voc_map


def make_case(seed, n_img=12, n_cls=3, crowd=False):
    rng = np.random.default_rng(seed)
    gts, preds = [], []
    for i in range(n_img):
        G = int(rng.integers(0, 7))
        xy = rng.uniform(0, 300, (G, 2))
        wh = np.exp(rng.uniform(np.log(4), np.log(200), (G, 2)))
        gb = np.concatenate((xy, xy + wh), 1)
        gl = rng.integers(0, n_cls, G)
        ic = (rng.random(G) < 0.15) if crowd else np.zeros(G, bool)
        # predictions: jittered copies of gts (some with wrong class) + random false positives
        db, dl, ds = [], [], []
        for b, l in zip(gb, gl):
            for _ in range(int(rng.integers(0, 3))):
                j = b + rng.normal(0, 0.08, 4) * np.r_[wh[0], wh[0]] if G else b
                db.append(j), dl.append(l if rng.random() > 0.15 else rng.integers(0, n_cls)), ds.append(rng.random())
        for _ in range(int(rng.integers(0, 5))):
            xy2 = rng.uniform(0, 300, 2)
            wh2 = rng.uniform(5, 100, 2)
            db.append(np.r_[xy2, xy2 + wh2]), dl.append(rng.integers(0, n_cls)), ds.append(rng.random() * 0.6)
        db = np.array(db, dtype=np.float64).reshape(-1, 4)
        db[:, 2:] = np.maximum(db[:, 2:], db[:, :2] + 1)
        gts.append({"boxes": torch.tensor(gb, dtype=torch.float64), "labels": torch.tensor(gl), "iscrowd": torch.tensor(ic)})
        preds.append({"boxes": torch.tensor(db), "scores": torch.tensor(ds, dtype=torch.float64), "labels": torch.tensor(np.array(dl, dtype=np.int64))})
    return preds, gts


def pycoco_stats(preds, gts):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cats = sorted({int(c) for g in gts for c in g["labels"]} | {int(c) for p in preds for c in p["labels"]})
    images, anns, dets = [], [], []
    for i, (g, p) in enumerate(zip(gts, preds)):
        images.append({"id": i, "width": 640, "height": 640})
        for b, l, c in zip(g["boxes"].tolist(), g["labels"].tolist(), g["iscrowd"].tolist()):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": l, "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": int(c)})
        for b, l, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
            dets.append({"image_id": i, "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in cats]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats


@pytest.mark.parametrize("seed,crowd", [(0, False), (1, False), (2, True), (3, True), (4, False)])
def test_coco_evaluate_matches_pycocotools(seed, crowd):
    preds, gts = make_case(seed, crowd=crowd)
    ref = pycoco_stats(preds, gts)
    mine = coco_evaluate(preds, gts).stats
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    for k, r in zip(keys, ref):
        assert abs(mine[k] - r) < 1e-6, (k, mine[k], r)


def test_perfect_predictions_give_ap_one():
    gts = [{"boxes": torch.tensor([[0.0, 0, 50, 50], [60, 60, 200, 200]]), "labels": torch.tensor([0, 1])}]
    preds = [{"boxes": gts[0]["boxes"].clone(), "scores": torch.tensor([0.9, 0.8]), "labels": torch.tensor([0, 1])}]
    s = coco_evaluate(preds, gts).stats
    assert s["AP"] == pytest.approx(1.0) and s["AP50"] == pytest.approx(1.0)


def test_interpolation_conventions_on_textbook_curve():
    # 5 detections, 4 gts: TP FP TP TP FP
    is_tp = np.array([1, 0, 1, 1, 0], bool)
    rec, prec, _ = pr_curve(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), is_tp, 4)
    assert np.allclose(rec, [0.25, 0.25, 0.5, 0.75, 0.75])
    # all-points: 0.25*1 + 0.5*0.75 (envelope at recall .5 and .75 is 3/4) = 0.625
    assert average_precision(rec, prec, "all_points") == pytest.approx(0.625)
    # VOC11: thresholds 0..0.2 ->1.0 (3 pts), 0.3..0.7 -> 0.75 (5 pts), 0.8..1.0 -> 0 = 6.75/11
    assert average_precision(rec, prec, "voc11") == pytest.approx(6.75 / 11)
    # COCO101: r in [0, .25] -> 1 (26 pts), (.25, .75] -> .75 (50 pts) = (26 + 37.5)/101
    assert average_precision(rec, prec, "coco101") == pytest.approx((26 + 37.5) / 101)


def test_voc_map_and_f1_and_breakdown_run():
    preds, gts = make_case(7)
    assert 0 <= voc_map(preds, gts) <= 1
    thr, f1 = best_f1_threshold(preds, gts)
    assert 0 <= thr <= 1 and 0 <= f1 <= 1
    c = error_breakdown(preds, gts, score_thr=0.0)
    n_det = sum(len(p["scores"]) for p in preds)
    assert c["tp"] + c["cls"] + c["loc"] + c["both"] + c["dupe"] + c["bkg"] == n_det
