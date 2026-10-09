"""Train TinyYOLO on SyntheticShapes and evaluate it with odlab's own COCO metric.

    python -m odlab.train --epochs 30                 # DFL head + NMS (YOLOv8/11-style)
    python -m odlab.train --epochs 30 --reg-max 1 --end2end   # YOLO26-style: no DFL, NMS-free

On a laptop CPU the default run takes a few minutes. Expect AP50 well above 0.8 and
AP in the 0.5-0.7 range; small objects (AP_s) lag, exactly as on COCO.
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np
import torch

from .augment import hflip, random_affine
from .data import SyntheticShapes, collate
from .metrics import coco_evaluate
from .model import DetectionLoss, TinyYOLO, count_params


class ModelEMA:
    """Exponential moving average of weights with Ultralytics' warm-up ramp:
    decay(t) = d * (1 - exp(-t / tau)). Early on the EMA tracks the model closely; later it
    averages over ~1/(1-d) steps. Evaluation and export always use the EMA weights.
    """

    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        self.updates, self.d, self.tau = 0, decay, tau
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.d * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def light_augment(img, boxes, labels, rng=np.random):
    """Flip + mild scale/translate. Mosaic is left out to keep CPU runs short (see Ch. 28)."""
    if rng.rand() < 0.5:
        img, boxes = hflip(img, boxes)
    img, new, keep = random_affine(img, boxes, translate=0.1, scale=0.3, rng=rng)
    return img, new[keep].astype(np.float32), labels[keep]


@torch.no_grad()
def evaluate(model, dataset, conf=0.001, batch=32, use_o2o=None):
    model.eval()
    preds, gts = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate)
    for x, boxes, labels in loader:
        preds += model.predict(x, conf=conf, iou=0.7, use_o2o=use_o2o)
        gts += [{"boxes": b, "labels": l} for b, l in zip(boxes, labels)]
    return coco_evaluate(preds, gts)


def train(epochs=30, n_train=768, n_val=128, img=256, batch=16, lr=2e-3, reg_max=16, end2end=False, seed=0, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = SyntheticShapes(n_train, img, seed=seed, augment=light_augment)
    val_ds = SyntheticShapes(n_val, img, seed=seed + 1)
    model = TinyYOLO(nc=3, reg_max=reg_max, end2end=end2end, img_size=img)
    log(f"TinyYOLO: {count_params(model) / 1e6:.2f} M params, reg_max={reg_max}, end2end={end2end}")
    loss_fn = DetectionLoss(model)
    # parameter groups as in YOLO: no weight decay on BN and biases
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate, drop_last=True)
    steps = epochs * len(loader)
    warmup = max(len(loader), 50)
    ema = ModelEMA(model, tau=max(steps // 10, 50))
    step = 0
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), torch.zeros(3)
        for x, boxes, labels in loader:
            # linear warm-up then cosine decay to 1% of lr
            f = step / warmup if step < warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))
            for g in opt.param_groups:
                g["lr"] = lr * f
            out = model(x)
            o2m_w = 0.8 - 0.7 * ep / max(epochs - 1, 1)  # ProgLoss-style schedule (only used if end2end)
            loss, items = loss_fn(out, boxes, labels, o2m_weight=o2m_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ema.update(model)
            run += items
            step += 1
        run /= len(loader)
        msg = f"epoch {ep + 1:3d}/{epochs}  box {run[0]:.3f}  cls {run[1]:.3f}  {'dfl' if reg_max > 1 else 'l1 '} {run[2]:.3f}  ({time.time() - t0:.1f}s)"
        if (ep + 1) % max(epochs // 5, 1) == 0 or ep + 1 == epochs:
            r = evaluate(ema.ema, val_ds)
            msg += f"  | val AP {r.stats['AP']:.3f}  AP50 {r.stats['AP50']:.3f}  APs {r.stats['APs']:.3f}"
        log(msg)
    return ema.ema, evaluate(ema.ema, val_ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-train", type=int, default=768)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--end2end", action="store_true")
    a = ap.parse_args()
    model, res = train(a.epochs, a.n_train, 128, a.img, a.batch, a.lr, a.reg_max, a.end2end)
    print(res.summary())
    if a.end2end:
        print("\none-to-many head + NMS for comparison:")
        print(evaluate(model, SyntheticShapes(128, a.img, seed=1), use_o2o=False).summary())


if __name__ == "__main__":
    main()
