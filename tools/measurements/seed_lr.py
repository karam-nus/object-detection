"""LR x seed sweep for TinyYOLO: data fixed (seed 0/1), init + order + augmentation vary with run seed."""
import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import sys, math, time, json
import numpy as np, torch
sys.path.insert(0, REPO + "/code")
from odlab.data import SyntheticShapes, collate
from odlab.model import TinyYOLO, DetectionLoss, count_params
from odlab.train import evaluate, ModelEMA, light_augment

def run(lr, seed, epochs=12, img=256, batch=16, n_train=768):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = SyntheticShapes(n_train, img, seed=0, augment=light_augment)
    val_ds = SyntheticShapes(128, img, seed=1)
    g = torch.Generator(); g.manual_seed(seed)
    model = TinyYOLO(nc=3, reg_max=16, img_size=img)
    loss_fn = DetectionLoss(model)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate, drop_last=True, generator=g)
    steps = epochs * len(loader); warmup = max(len(loader), 50)
    ema = ModelEMA(model, tau=max(steps // 10, 50)); step = 0
    for ep in range(epochs):
        model.train()
        for x, boxes, labels in loader:
            f = step / warmup if step < warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))
            for gr in opt.param_groups: gr["lr"] = lr * f
            loss, _ = loss_fn(model(x), boxes, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); ema.update(model); step += 1
    r = evaluate(ema.ema, val_ds)
    return {k: round(float(v), 4) for k, v in r.stats.items()}

if __name__ == "__main__":
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    lrs = [float(x) for x in sys.argv[1].split(",")]
    for lr in lrs:
        for seed in (0, 1, 2):
            t0 = time.time()
            s = run(lr, seed)
            print(json.dumps({"lr": lr, "seed": seed, "AP": s["AP"], "AP50": s["AP50"], "APs": s["APs"], "sec": round(time.time() - t0)}), flush=True)
