"""Bootstrap CI of COCO mAP vs validation-set size, on synthetic 'typical' predictions."""
import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import sys, numpy as np, time
sys.path.insert(0, REPO + "/code")
from odlab.metrics import coco_evaluate
import torch
rng = np.random.default_rng(0)
_mc = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_os.path.dirname(__file__), "metric_compare.py")
exec(open(_mc).read().split("def ultralytics_map")[0].split("rng = np.random.default_rng(0)")[1])  # reuse make()
preds, gts = make(n_img=1000)
for n in (50, 200, 1000):
    idx0 = np.arange(n)
    full = coco_evaluate([preds[i] for i in idx0], [gts[i] for i in idx0]).stats["AP"]
    b = []
    t0 = time.time()
    for _ in range(100):
        idx = rng.choice(idx0, n, replace=True)
        b.append(coco_evaluate([preds[i] for i in idx], [gts[i] for i in idx]).stats["AP"])
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"n={n:5d} AP {full:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  width {hi - lo:.4f}  std {np.std(b):.4f}  ({time.time() - t0:.0f}s)", flush=True)
