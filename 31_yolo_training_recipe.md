---
title: "Chapter 31 — YOLO Axis 5: Training Recipe"
---

[← Back to Table of Contents](./README.md)

# Chapter 31 — YOLO Axis 5: Training Recipe

> *"Two teams with the same architecture and the same data can be 3 AP apart. The difference is usually in this chapter."*

## Overview

The training recipe is everything between "here is a loss" and "here is a checkpoint": optimiser,
learning-rate schedule, warm-up, batch size and gradient accumulation, EMA, mixed precision, gradient
clipping, epochs, early stopping, initialisation from pre-trained weights and the order in which
augmentations switch off. This chapter reads each mechanism from the Ultralytics trainer, gives the
official recipes of the main YOLO versions, and ends with recipe cards for common situations. Tuning the
numbers is Chapter 32.

<div class="diagram">
<div class="diagram-title">One Ultralytics training iteration, and what happens each epoch</div>
<div class="flow">
  <div class="flow-h">
    <div class="flow-node">batch (mosaic …)</div>
    <div class="flow-arrow"></div>
    <div class="flow-node blue">forward<small>AMP autocast</small></div>
    <div class="flow-arrow"></div>
    <div class="flow-node accent">loss × batch</div>
    <div class="flow-arrow"></div>
    <div class="flow-node">backward<small>scaled</small></div>
  </div>
  <div class="flow-arrow"></div>
  <div class="flow-h">
    <div class="flow-node purple">accumulate until ≈ 64 images<small>nbs</small></div>
    <div class="flow-arrow"></div>
    <div class="flow-node">unscale · clip norm 10</div>
    <div class="flow-arrow"></div>
    <div class="flow-node accent">optimizer step<small>SGD / AdamW / MuSGD</small></div>
    <div class="flow-arrow"></div>
    <div class="flow-node green">EMA update</div>
  </div>
  <div class="flow-arrow"></div>
  <div class="flow-node wide">each epoch: LR schedule step · E2E loss-weight decay · mosaic off for the last 10 · validate the EMA model · fitness = mAP50-95 · save last/best · early-stop check</div>
</div>
</div>

---

## Optimiser

### `optimizer=auto`

The default is `auto`. The trainer estimates the number of optimiser steps and picks:

```python
iterations = ceil(len(train_set) / max(batch, nbs)) * epochs          # nbs = 64
if iterations > 10_000:  MuSGD, lr0 = 0.01, momentum = 0.9
else:                    AdamW, lr0 = round(0.002 * 5 / (4 + nc), 6), momentum (β1) = 0.9
```

The user's `lr0` is **ignored** in auto mode, which surprises people. Worked examples:

| Dataset | Epochs | Steps | Chosen | lr0 |
|---|---|---|---|---|
| 800 images, 3 classes | 100 | 13 × 100 = 1,300 | AdamW | 0.002 × 5 / 7 = 0.001429 |
| 12,000 images, 10 classes | 100 | 188 × 100 = 18,800 | MuSGD | 0.01 |
| COCO (118k), 80 classes | 100 | 1,849 × 100 ≈ 185k | MuSGD | 0.01 |

The AdamW formula lowers the learning rate as the class count grows (more classes, more parameters in the
class head receiving gradient). Older releases chose plain SGD for long runs. Current releases choose
MuSGD.

### Parameter groups

Parameters are split into groups with different weight decay:

| Group | Contents | Weight decay |
|---|---|---|
| weights | conv and linear weights (or, under MuSGD, any 2-D/4-D parameter → the Muon group) | `weight_decay` (scaled, below) |
| bn | BatchNorm (and other norm) weights, `logit_scale` | 0 |
| bias | every bias | 0 |

Under MuSGD, every group is split again and the **classification head (`cv3`, `one2one_cv3`) gets
3× the learning rate**. The class head adapts fastest when fine-tuning to new classes.

### MuSGD

MuSGD combines **Muon** (momentum, then orthogonalise the update matrix with five Newton–Schulz
iterations so that all its singular values are close to 1) with ordinary Nesterov SGD on the same
parameters:

$$
\theta \leftarrow \theta \;-\; \eta\, w_{muon}\, \text{Orth}\big(\text{Nesterov-EMA}(g)\big)\cdot\sqrt{\max(1, \tfrac{\text{rows}}{\text{cols}})} \;-\; \eta\, w_{sgd}\, \text{SGD-Nesterov}(g + \lambda\theta)
$$

Conv filters are reshaped to 2-D (out × in·k·k) for the orthogonalisation. Weight decay goes only into
the SGD part. Parameters that are not 2-D/4-D (BN, biases) get SGD only. In the public package
`w_muon = 0.2` and `w_sgd = 1.0`. The released YOLO26 checkpoints used other values from an
experimental branch (`muon_w` 0.44–0.53, `sgd_w` 0.48–0.67, recorded in their `train_args`).

Why it helps: an orthogonalised update moves every direction of a weight matrix by a similar amount, so
rare directions are not starved by the dominant gradient direction. Muon came from fast language-model
training (Jordan et al., 2024; Kimi K2 used it at scale). YOLO26 is its first mainstream use in
detection. Chapter 37 discusses what is and is not established about its benefit for detectors.

---

## Batch size, nominal batch and accumulation

Ultralytics defines hyperparameters for a **nominal batch size `nbs = 64`**:

```python
accumulate   = max(round(nbs / batch), 1)                    # optimiser step every `accumulate` batches
weight_decay = weight_decay * batch * accumulate / nbs       # scale decay with the effective batch
loss         = per_image_mean_loss * batch                   # returned by the criterion
```

| `batch` | `accumulate` | Images per step | Decay multiplier | Gradient scale vs nbs |
|---|---|---|---|---|
| 8 | 8 | 64 | 1.0 | 1.0 |
| 16 (default) | 4 | 64 | 1.0 | 1.0 |
| 32 | 2 | 64 | 1.0 | 1.0 |
| 64 | 1 | 64 | 1.0 | 1.0 |
| 128 | 1 | 128 | 2.0 | **2.0** |

Below 64 the effective batch is always 64, so `lr0` means the same thing whether you train with batch 8
or 32. Above 64 the summed loss doubles the gradient, so the step grows linearly with batch: the linear
scaling rule is applied implicitly. That is one reason the YOLO26 checkpoints, trained at batch 128,
were fine-tuned with lower `lr0` values than the defaults.

**BatchNorm is the exception.** Accumulation does not enlarge BN's batch. BN statistics come from each
forward batch, so batch 4 with accumulation 16 is *not* equivalent to batch 64. Below about 8 images per
GPU, BN statistics get noisy. Prefer a smaller `imgsz` or a smaller model over a tiny batch.

**AutoBatch.** `batch=-1` profiles the model and picks the batch that uses about 60% of GPU memory.
`batch=0.8` targets 80%. It runs once at the start. Two safety nets exist for out-of-memory errors. During the
first epoch on a single GPU, the trainer halves the batch and retries, up to three times (it logs the
new batch size, which also changes `accumulate`). Inside the loss, `TaskAlignedAssigner` retries one
image at a time after a CUDA OOM, which happens on images with hundreds of objects.

---

## Learning-rate schedule

```python
lf = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf           # linear (default)
lf = one_cycle(1, lrf, epochs)                                        # cos_lr=True: cosine from 1 to lrf
lr(epoch) = lr0 * lf(epoch)
```

`lrf` is the **final fraction**, not the final LR. With defaults (`lr0 = 0.01`, `lrf = 0.01`) the LR
falls linearly from 0.01 to 0.0001. The YOLO26 COCO-stage recipes show two different strategies:

| | `lr0` | `lrf` | LR at start → end | Interpretation |
|---|---|---|---|---|
| Default | 0.01 | 0.01 | 0.01 → 0.0001 | Train from scratch or a long fine-tune |
| YOLO26-N COCO stage | 0.0054 | 0.0495 | 0.0054 → 0.00027 | Substantial re-learning on top of Objects365 |
| YOLO26-S…X COCO stage | 0.00038 | 0.882 | 0.00038 → 0.00034 | **Almost constant, very low LR**: gentle adaptation |

Linear vs cosine rarely matters by more than a few tenths of AP. The end LR (`lrf`) and the total
length matter more.

---

## Warm-up

For the first `warmup_epochs` (default 3, i.e. `round(3 × batches_per_epoch)` iterations, always
leaving at least one epoch after warm-up), three things are interpolated linearly per iteration:

| Quantity | From | To |
|---|---|---|
| Bias learning rate | `warmup_bias_lr` = 0.1 | the scheduled LR |
| All other learning rates | 0 | the scheduled LR |
| Momentum | `warmup_momentum` = 0.8 | `momentum` = 0.937 |
| Accumulation | 1 | `nbs / batch` |

The bias LR starts *high* and falls, while weight LRs start at zero and rise. In early iterations the
model mostly learns its output biases (the class prior of Chapter 29 and the box-distance offsets) while
the randomly initialised or newly attached weights are protected from large steps. With `optimizer=auto`,
`warmup_bias_lr` is set to 0 (a 0.1 bias LR is too high for Adam-family optimisers).

YOLO26's recipes use about **one epoch** of warm-up (0.98–0.99 in the COCO stage; 1, or 2 for X, on
Objects365). With pre-trained weights and batch 128, three epochs of warm-up are unnecessary.

---

## EMA: the model you actually ship

After every optimiser step, an exponential moving average of the weights is updated:

$$
\theta_{EMA} \leftarrow d\,\theta_{EMA} + (1 - d)\,\theta, \qquad d = 0.9999\,\big(1 - e^{-\text{updates}/2000}\big)
$$

The ramp keeps the EMA from being dominated by its random initial value: $d$ is 0.39 after 1,000
updates, 0.86 after 4,000, and approaches 0.9999 after about 20,000. At full decay the average spans
roughly the last 10,000 steps. **Validation, `best.pt` and `last.pt` all use the EMA weights.** The raw
weights are saved only to resume.

EMA smooths the validation curve and usually gives a slightly better final model than the raw weights.
It also explains a common confusion: early in training the validation mAP lags the training loss, because the EMA still
averages over older, worse weights.

---

## Mixed precision and clipping

- **AMP** (`amp=True`): FP16 autocast with a dynamic gradient scaler. Before training, `check_amp`
  runs a reference `yolo26n.pt` on a test image in FP32 and AMP and disables AMP if the outputs differ by
  more than 0.5 absolute. This catches GPUs and drivers with broken FP16 paths. `amp="bf16"` uses
  bfloat16 on hardware that supports it (no scaler needed).
- **Gradient clipping**: the gradient norm is clipped to **10.0** after unscaling, every step. It
  rarely triggers in a healthy run but prevents one mosaic with hundreds of objects from wrecking the
  weights.
- **NaN losses** usually point to AMP on an incompatible device, a learning rate far too high for the
  optimiser (SGD-level `lr0` with AdamW), or invalid labels (zero-size boxes). Try `amp=False` first to
  separate the causes.

---

## Epochs, stopping and checkpoint selection

| Setting | Default | Notes |
|---|---|---|
| `epochs` | 100 | Official COCO recipes: v5 300; v8 500; YOLOX 300; v7 300; YOLO26 150 (Objects365) + 40–245 (COCO) |
| `close_mosaic` | 10 | Mosaic off for the last 10 epochs (Chapter 28) |
| `patience` | 100 | Stop after N epochs without fitness improvement |
| `time` | — | Hours. Overrides `epochs` and adapts the schedule to the time budget |
| `save_period` | −1 | Save every N epochs |
| `val` | True | Validate every epoch (`False` validates only at the end) |

**Fitness**, which decides `best.pt` and early stopping, is **mAP50-95 alone** in the current code
(weights `[P, R, mAP50, mAP50-95] = [0, 0, 0, 1]`). YOLOv5 and earlier Ultralytics releases used
`0.1 × mAP50 + 0.9 × mAP50-95`. If your deployment metric is recall at a fixed threshold, the "best"
checkpoint is not necessarily best for you. Validate the last few checkpoints on your own metric
(Chapter 33).

---

## Starting weights and transfer

| Start from | When | Notes |
|---|---|---|
| `yolo26s.pt` (COCO) | Default for most custom datasets | COCO's 80 classes; strong for everyday objects |
| `yolo26s-objv1-150.pt` (Objects365) | Unusual domains; many classes | 365-class prior, not specialised to COCO (Chapter 27) |
| `yolo26s.yaml` (scratch) | Very large datasets, or architectures without weights | Needs 300+ epochs and the full augmentation recipe |
| Your own previous checkpoint | Continual data growth | Same classes: lower `lr0`, fewer epochs |

What happens at load:

- **Shape-matched transfer.** Only tensors whose names and shapes match are copied. The log reports
  "Transferred 691/697 items". The missing ones are usually the final class convs, because `nc` differs.
- **`cls_remap=True`** (default): when `nc` differs, the loader copies class-head rows **by class name**
  from the checkpoint. A custom dataset with classes `person`, `forklift` and `pallet`, starting from
  COCO weights, starts its `person` row from COCO's trained `person` row. Only `forklift` and `pallet`
  start from scratch. Matching names exactly ("person", not "Person" or "human") is free accuracy.
- **First-conv adaptation.** If the number of input channels differs (for example 4-channel RGB-IR),
  the overlapping channels of the first conv are copied.
- **`freeze=N`** freezes the first N layers (indices from the YAML: `freeze=10` freezes YOLO26 layers 0–9,
  the backbone through SPPF, while C2PSA at index 10 stays trainable; Chapter 29). The fixed DFL conv is always frozen. Freezing saves memory and
  time and resists over-fitting on tiny datasets. It costs accuracy when the domain is far from the
  pre-training data.

---

## Reproducibility and run-to-run variance

`seed=0` and `deterministic=True` are defaults. Determinism is best-effort: some CUDA kernels (for
example the backward pass of upsampling) are non-deterministic, multi-worker data loading interacts with
seeds, and DDP adds more ordering effects. Two "identical" runs can differ by a few tenths of mAP on
COCO, and by **1–3 points on datasets with a few hundred validation images**. Before concluding that a
change helped:

1. Run at least 2–3 seeds of the baseline and of the change on small datasets.
2. Compare the mean difference with the seed spread.
3. Prefer metrics with more support (mAP over all classes, not one rare class's AP).

---

## Throughput

| Bottleneck | Symptom | Fix |
|---|---|---|
| JPEG decode / augmentation | GPU utilisation below 70%, CPU cores saturated | `workers` up to the number of cores, `cache=ram` or `cache=disk`, DALI (Chapter 28) |
| Disk | Low CPU, low GPU | `cache=ram`; local SSD instead of network storage |
| GPU compute | GPU utilisation ~100% | AMP on, larger batch, `compile=True` (`torch.compile`; benefit varies) |
| Validation every epoch | Long pauses between epochs | `val=False` until the end, or validate on a subset |
| Multi-GPU | — | `device=0,1,2,3` launches DDP; `batch` is the total across GPUs |

---

## Built-in knowledge distillation

The trainer accepts `distill_model=<teacher.pt>`. It wraps student and teacher in a
`DistillationModel`, hooks the neck features that feed both Detect heads, aligns them with a small MLP
projector, and adds a score-weighted L2 feature loss with weight `dis = 6.0`. The teacher is frozen and
removed from saved checkpoints. Chapter 42 covers when distillation pays off and how to measure it.

```python
YOLO("yolo26n.pt").train(data="widgets.yaml", epochs=100, distill_model="runs/detect/train_x/weights/best.pt")
```

---

## Recipe cards

<div class="diagram-grid cols-2">
  <div class="diagram-card accent"><div class="card-title">A. Small custom dataset (≤ 1,000 images)</div><div class="card-desc"><code>model=yolo26s.pt epochs=100 patience=20 optimizer=AdamW lr0=0.001 mosaic=0.5 mixup=0 copy_paste=0</code> · try <code>freeze=10</code> · 3 seeds · compare COCO vs Objects365 starting weights</div></div>
  <div class="diagram-card blue"><div class="card-title">B. Medium dataset (5k–50k images)</div><div class="card-desc">Defaults (<code>optimizer=auto</code> → MuSGD) · <code>epochs=100–200</code> · batch as large as fits up to 64 · <code>cache=ram</code> if it fits · tune <code>scale</code> and HSV for the domain</div></div>
  <div class="diagram-card green"><div class="card-title">C. Large dataset (> 50k) or new domain at scale</div><div class="card-desc">Follow the YOLO26 pre-training recipe: MuSGD, batch 128, ~1 epoch warm-up, mosaic 1.0, <code>scale=0.9</code> (s+), copy-paste and mixup scaled with model size, <code>close_mosaic=10</code></div></div>
  <div class="diagram-card purple"><div class="card-title">D. Nano model for the edge</div><div class="card-desc">Train n from an Objects365 or COCO checkpoint at the deployment <code>imgsz</code> · lighter augmentation (<code>scale=0.5</code>, <code>mixup=0</code>) · consider distilling from s/m (<code>distill_model</code>) · validate the exported INT8 model, not just the PyTorch one (Chapter 45)</div></div>
</div>

---

## Key Takeaways

- `optimizer=auto` chooses MuSGD (lr 0.01) for runs over 10,000 steps and AdamW with
  `lr0 = 0.01 / (4 + nc)` otherwise, and ignores your `lr0`. Set the optimiser explicitly when you
  set `lr0`.
- Hyperparameters are defined for a nominal batch of 64: below it, gradients are accumulated; above it,
  the step and the weight decay grow with the batch.
- Warm-up starts the bias LR high (0.1) and every other LR at zero, ramps momentum 0.8 → 0.937, and
  ramps accumulation. YOLO26 recipes use about one epoch.
- The EMA model (decay 0.9999 with a 2,000-update ramp) is what gets validated and saved. `best.pt` is
  chosen by mAP50-95 alone.
- Transfer copies shape-matched tensors and, with `cls_remap`, class-head rows by class name. Use the
  same class names as the pre-training dataset where they apply.
- On small datasets, seed variance (1–3 mAP) is often larger than the effect of the change being tested.

## Check Yourself

<details class="check"><summary>You set lr0=0.02 for a 600-image dataset and the log shows AdamW with lr=0.001429. Why?</summary>
With optimizer=auto, the trainer estimates the number of optimiser steps (here well under 10,000),
picks AdamW and computes lr0 = 0.002 × 5 / (4 + nc). Your lr0 is ignored. Set optimizer=SGD (or AdamW,
or MuSGD) explicitly to use your own lr0.</details>

<details class="check"><summary>You move from batch 16 on one GPU to batch 128 on eight GPUs and accuracy drops. Which recipe mechanism changed?</summary>
At batch 16 the trainer accumulates 4 batches to reach the nominal 64, so each step sees 64 images. At
128 there is no accumulation and the loss is summed over 128 images, so each step is about twice as large
and the weight decay is doubled. Lower lr0 (roughly halve it) or keep the per-step image count at 64.
Also check warm-up: fewer steps per epoch means fewer warm-up iterations.</details>

<details class="check"><summary>Why does the validation mAP in the first few epochs look much worse than the training loss suggests?</summary>
Validation uses the EMA weights. Early on, the EMA decay ramps up but still averages over recent,
poorer weights, and the bias-heavy warm-up has only just ended. The gap closes as training continues.</details>

<details class="check"><summary>Your dataset has the classes "Person", "car" and "dog". Starting from yolo26s.pt, which class-head rows start from trained weights?</summary>
With cls_remap, rows are matched by class name. "car" and "dog" match COCO names and inherit trained
rows. "Person" (capital P) does not match COCO's "person" and starts from scratch. Renaming it to
"person" gives it the trained row for free.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics `engine/trainer.py` (`build_optimizer`, `_setup_train`, `_do_train`, `optimizer_step`) | Ultralytics | github.com/ultralytics/ultralytics | Auto optimiser, parameter groups, nbs scaling, warm-up, clipping, close_mosaic, early stopping, distillation hook |
| Ultralytics `optim/muon.py` (`MuSGD`, `muon_update`, `zeropower_via_newtonschulz5`) | Ultralytics | github.com/ultralytics/ultralytics | MuSGD update rule |
| Ultralytics `utils/torch_utils.py` (`ModelEMA`), `utils/checks.py` (`check_amp`), `utils/autobatch.py` | Ultralytics | github.com/ultralytics/ultralytics | EMA formula, AMP check, AutoBatch |
| Ultralytics `utils/metrics.py` (`fitness`) | Ultralytics | github.com/ultralytics/ultralytics | Checkpoint-selection metric |
| Ultralytics `nn/tasks.py` (`load`, `_remap_cls_by_names`), `nn/distill_model.py` | Ultralytics | github.com/ultralytics/ultralytics | Transfer, class-name remapping, distillation |
| Ultralytics `cfg/default.yaml` | Ultralytics | github.com/ultralytics/ultralytics | Defaults |
| YOLO26 training recipe guide | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Two-stage recipe, per-size LR/epochs/warm-up, internal MuSGD weights, fine-tuning advice |
| Muon optimizer | Jordan et al., 2024 | kellerjordan.github.io/posts/muon | Orthogonalised momentum updates |
| Accurate, Large Minibatch SGD | Goyal et al., 2017 | arXiv:1706.02677 | Linear scaling rule and warm-up |
| Mean teachers / weight averaging | Tarvainen, Valpola, 2017 | arXiv:1703.01780 | EMA of weights |

---

**Next:** [Chapter 32 — Axis 6: Hyperparameter Tuning](./32_yolo_hyperparameter_tuning.md) — which of
these numbers deserve a search, and how the Ultralytics tuner searches them.
