"""Same predictions, three scorers: pycocotools, odlab (pycocotools-exact), Ultralytics' val metric."""
import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import sys, numpy as np, torch
sys.path.insert(0, REPO + "/code")
from odlab.metrics import coco_evaluate
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.models.yolo.detect.val import DetectionValidator

rng = np.random.default_rng(0)
def make(n_img=200, kmax=12, dup=0.15, fp=3, jitter=0.08):
    preds, gts = [], []
    for _ in range(n_img):
        k = rng.integers(1, kmax)
        wh = rng.uniform(8, 160, (k, 2)); xy = rng.uniform(0, 480, (k, 2))
        g = np.concatenate([xy, xy + wh], 1); lab = rng.integers(0, 5, k)
        pb, ps, pl = [], [], []
        for b, l in zip(g, lab):
            if rng.random() < 0.85:  # detected
                s = b[2:] - b[:2]
                for d in range(1 + (rng.random() < dup)):
                    j = b + rng.normal(0, jitter, 4) * np.r_[s, s]
                    pb.append(j); ps.append(rng.uniform(0.3, 1.0) * (0.6 if d else 1)); pl.append(l if rng.random() < 0.93 else rng.integers(0, 5))
        for _ in range(rng.integers(0, fp + 1)):
            xy0 = rng.uniform(0, 500, 2); pb.append(np.r_[xy0, xy0 + rng.uniform(8, 100, 2)]); ps.append(rng.uniform(0.001, 0.6)); pl.append(rng.integers(0, 5))
        preds.append({"boxes": torch.tensor(np.array(pb), dtype=torch.float32).reshape(-1, 4), "scores": torch.tensor(ps, dtype=torch.float32), "labels": torch.tensor(pl)})
        gts.append({"boxes": torch.tensor(g, dtype=torch.float32), "labels": torch.tensor(lab)})
    return preds, gts

def ultralytics_map(preds, gts):
    v = DetectionValidator.__new__(DetectionValidator)
    v.iouv = torch.linspace(0.5, 0.95, 10)
    tps, confs, pcls, tcls = [], [], [], []
    for p, g in zip(preds, gts):
        iou = box_iou(g["boxes"], p["boxes"]) if len(p["boxes"]) else torch.zeros(len(g["boxes"]), 0)
        tps.append(v.match_predictions(p["labels"], g["labels"], iou).numpy())
        confs.append(p["scores"].numpy()); pcls.append(p["labels"].numpy()); tcls.append(g["labels"].numpy())
    out = ap_per_class(np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.concatenate(tcls))
    ap = out[5]
    return ap[:, 0].mean(), ap.mean(1).mean(), out[2].mean(), out[3].mean()

for name, kw in [("clean", dict(dup=0.0, fp=0, jitter=0.04)), ("typical", dict()), ("noisy", dict(dup=0.4, fp=8, jitter=0.12)), ("crowded", dict(kmax=150, fp=40))]:
    preds, gts = make(**kw)
    r = coco_evaluate(preds, gts).stats
    u50, u, P, R = ultralytics_map(preds, gts)
    print(f"{name:8s} COCO AP {r['AP']:.4f} AP50 {r['AP50']:.4f} | Ultralytics mAP50-95 {u:.4f} mAP50 {u50:.4f} | diff {u - r['AP']:+.4f} | P {P:.3f} R {R:.3f}")
