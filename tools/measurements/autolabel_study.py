"""How good are open-vocabulary auto-labels? YOLOE-26 text prompts vs human COCO128 labels."""
import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import sys, glob, os, json, numpy as np, torch
sys.path.insert(0, REPO + "/code")
from odlab.metrics import coco_evaluate, match_detections
from ultralytics import YOLOE
S = WORK + "/"
D = S + "datasets/coco128/"
os.chdir(S + "weights")
COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
imgs = sorted(glob.glob(D + "images/train2017/*.jpg"))
def gt_of(p):
    from PIL import Image
    w, h = Image.open(p).size
    lp = p.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    b, l = [], []
    if os.path.exists(lp):
        for line in open(lp):
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            b.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]); l.append(int(c))
    return {"boxes": torch.tensor(b, dtype=torch.float32).reshape(-1, 4), "labels": torch.tensor(l, dtype=torch.long)}
gts = [gt_of(p) for p in imgs]
out = {}
for wname in ("yoloe-26s-seg.pt",):
    m = YOLOE(wname); m.set_classes(COCO)
    preds = []
    for p in imgs:
        r = m.predict(p, conf=0.001, max_det=100, verbose=False)[0]
        preds.append({"boxes": r.boxes.xyxy.cpu(), "scores": r.boxes.conf.cpu(), "labels": r.boxes.cls.cpu().long()})
    st = coco_evaluate(preds, gts).stats
    rec = {"AP": round(st["AP"], 4), "AP50": round(st["AP50"], 4)}
    for thr in (0.1, 0.25, 0.4):
        pk = [{k: v[d["scores"] > thr] for k, v in d.items()} for d in preds]
        mm = match_detections(pk, gts, iou_thr=0.5)
        tp = sum(int(t.sum()) for _, t, _ in mm.values()); nd = sum(len(t) for _, t, _ in mm.values()); ng = sum(n for _, _, n in mm.values())
        small_tp = 0
        rec[f"@{thr}"] = {"precision": round(tp / max(nd, 1), 3), "recall": round(tp / max(ng, 1), 3), "dets": nd, "gt": ng,
                          "per_class_recall_lowest": sorted([(COCO[c], round(int(t.sum()) / n, 2), n) for c, (_, t, n) in mm.items() if n >= 5], key=lambda x: x[1])[:5]}
    out[wname] = rec
    print(wname, json.dumps(rec, default=str), flush=True)
json.dump(out, open(S + "weights/autolabel_study.json", "w"), indent=1, default=str)
