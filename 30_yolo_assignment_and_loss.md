---
title: "Chapter 30 — YOLO Axis 4: Assignment & Loss"
---

[← Back to Table of Contents](./README.md)

# Chapter 30 — YOLO Axis 4: Assignment & Loss

> *"Most of the AP gained between YOLOv3 and YOLOv8 came from deciding better which predictions should learn from which object."*

## Overview

Chapters 5 and 6 explain assignment strategies and loss functions in general. This chapter follows the
YOLO line through them, version by version. For each version it gives the exact rule that decides which
predictions are positives, the target each positive receives, the loss terms and their gains, and how
the total is normalised. The details come from each project's training code, not its paper, because the
two often differ. The chapter closes with what the three numbers in an Ultralytics training log mean
and which knobs you can safely turn.

<div class="diagram">
<div class="diagram-title">Assignment through the YOLO line</div>
<div class="timeline">
  <div class="timeline-item"><div class="timeline-year">2016</div><div class="timeline-title">v1</div><div class="timeline-desc">Centre cell, best of 2 predictors · SSE</div></div>
  <div class="timeline-item"><div class="timeline-year">2017–18</div><div class="timeline-title">v2–v3</div><div class="timeline-desc">Best anchor in centre cell (one positive) · ignore zone · BCE</div></div>
  <div class="timeline-item"><div class="timeline-year">2020</div><div class="timeline-title">v4–v5</div><div class="timeline-desc">Many anchors per object (IoU or shape ratio) · neighbour cells · CIoU</div></div>
  <div class="timeline-item"><div class="timeline-year">2021</div><div class="timeline-title">YOLOX</div><div class="timeline-desc">Anchor-free · SimOTA dynamic k</div></div>
  <div class="timeline-item"><div class="timeline-year">2022–23</div><div class="timeline-title">v6–v8</div><div class="timeline-desc">TAL soft targets · DFL · no objectness</div></div>
  <div class="timeline-item"><div class="timeline-year">2024</div><div class="timeline-title">v10</div><div class="timeline-desc">Dual assignment: top-10 + top-1</div></div>
  <div class="timeline-item"><div class="timeline-year">2026</div><div class="timeline-title">YOLO26</div><div class="timeline-desc">ProgLoss · top-7→1 · STAL · L1 distances</div></div>
</div>
</div>

---

## The common skeleton

Every YOLO loss, from v1 to YOLO26, does the same five things:

1. **Assign**: decide for each object which predictions (cell × anchor, or point) are positive, which
   are negative, and which are ignored.
2. **Build targets**: box target, class target (hard 1 or a soft quality score), objectness target if
   there is an objectness branch.
3. **Compute terms**: box (SSE → IoU-family), classification (SSE → BCE → soft BCE / VFL),
   objectness (versions up to v7 and YOLOX), distribution or distance (v6+).
4. **Normalise**: by batch size, number of positives, or sum of soft targets.
5. **Weight** with gains (`box`, `cls`, `obj`/`dfl`) and sum.

The versions differ mainly in step 1, and step 1 changes what the other steps mean.

---

## YOLOv1 to YOLOv3: one positive per object

**YOLOv1.** The cell containing the object centre is responsible. Of its two box predictors, the one
whose current prediction has the higher IoU with the object becomes the positive. Everything is
sum-squared error: box terms weighted by $\lambda_{coord} = 5$, no-object confidence by
$\lambda_{noobj} = 0.5$, and widths and heights regressed as square roots so that a 10-pixel error
costs more on a small box than on a large one.

**YOLOv2.** Five anchors per cell from k-means on box shapes. The positive is the anchor in the centre
cell with the highest *shape* IoU (anchor and box aligned at a common centre). The box is decoded as
$b_x = \sigma(t_x) + c_x$, $b_w = p_w e^{t_w}$. The objectness target is the IoU of the prediction with
the object. Predictions that already overlap any object by more than 0.6 are excluded from the
no-object loss.

**YOLOv3.** Nine anchors, three per scale. Each object gets **exactly one positive**: the best-matching
anchor across *all* scales, in the centre cell of that scale. Other predictions whose IoU with the object
exceeds the ignore threshold (0.5 in the paper, `ignore_thresh = .7` in the released `yolov3.cfg`) are
**ignored**, so they receive neither positive nor negative objectness loss. Class and
objectness become independent **BCE** terms (multi-label friendly). The box loss stays squared error on
$t_x, t_y, t_w, t_h$, scaled by $(2 - w h)$ (normalised width × height) to up-weight small boxes.

The defect of this era: **one positive per object** is very sparse supervision. A 300-pixel object
covers hundreds of cells, and one of them learns. The rest are told "background" or ignored.

---

## YOLOv4 and YOLOv5: many positives per object

**YOLOv4** (Darknet) lets every anchor whose IoU with the object exceeds `iou_thresh = 0.213` become
positive, not only the best one. It adds CIoU loss and "grid sensitivity" scaling (`scale_x_y`) so that
predicted centres can actually reach cell borders.

**YOLOv5** replaced IoU matching with a shape-ratio test and added neighbouring cells. From
`ComputeLoss.build_targets`:

```python
r = t[..., 4:6] / anchors[:, None]                      # w, h ratio of object to anchor
j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # anchor_t = 4.0: both ratios within 4×
# then: the centre cell + the 2 nearest neighbouring cells (offset g = 0.5)
```

- **Anchor test**: an anchor is a candidate if the object's width *and* height are within 4× of the
  anchor's (`anchor_t = 4.0`). This is scale tolerance, not overlap.
- **Neighbour cells**: besides the centre cell, the two adjacent cells nearest to the centre (left or
  right, above or below) also become positive. One object can have up to 3 cells × 3 anchors × 3
  levels = 27 positives.
- **Decoding** was changed to match: $b_{xy} = 2\sigma(t_{xy}) - 0.5 + c$ (range −0.5 to 1.5, so a
  neighbour cell can reach the centre) and $b_{wh} = (2\sigma(t_{wh}))^2 \cdot p_{wh}$ (0 to 4× the
  anchor, bounded, unlike $e^{t}$, which can explode).
- **Losses**: CIoU box loss; BCE objectness with target = CIoU of the prediction (detached, clamped
  at 0, `gr = 1.0`); BCE class loss on positives. Objectness is weighted per level by
  `balance = [4.0, 1.0, 0.4]` (P3, P4, P5).
- **Gains** (`hyp.scratch-low.yaml`): `box = 0.05`, `cls = 0.5`, `obj = 1.0`. `train.py` rescales them:
  `box × 3/nl`, `cls × nc/80 × 3/nl`, `obj × (imgsz/640)² × 3/nl`. The summed loss is multiplied by the
  batch size.

<div class="callout warn"><span class="callout-title">Same name, different meaning</span>YOLOv5's
<code>cls_pw</code> is the BCE <em>positive weight</em>. In the current Ultralytics package,
<code>cls_pw</code> is the <em>power</em> of inverse-frequency class weighting (Chapter 27). Copying a
YOLOv5 hyperparameter file into Ultralytics silently changes what the number does.</div>

---

## YOLOX: anchor-free with SimOTA

YOLOX removed anchors (one point per cell) and replaced static matching with **SimOTA** (Chapter 5):
the cost of assigning point $i$ to object $j$ is the classification BCE plus 3 × the IoU loss
($-\log \text{IoU}$), plus a huge penalty outside the object's centre region. Each object takes its
$k_j$ cheapest points, where $k_j$ is the sum of its top-10 IoUs, at least 1. In the current code the
centre region has a radius of 1.5 strides (2.5 in the original release).

Targets and losses: the class target is one-hot × the IoU of the matched prediction (a quality-aware
soft target, before TAL made this standard); the objectness target is 1 for positives; the IoU loss is
weighted by 5.0. An extra **L1 loss** on the raw box outputs is switched on only for the final 15
no-augmentation epochs. Everything is normalised by the number of positives.

---

## YOLOv6 and YOLOv7

**YOLOv6** (3.0) uses **ATSS** (top-9 per level) for the first 4 epochs, while predictions are still
meaningless, then switches to **TAL** (top-13, α = 1, β = 6, the TOOD defaults). The classification
loss is **VariFocal** (Chapter 6). The box loss is GIoU or SIoU (set per model config) plus DFL for the
m/l models. Loss weights: class 1.0, IoU 2.5, DFL 0.5.

**YOLOv7** keeps anchors and YOLOv5-style candidate selection, then refines the candidates with a
SimOTA-like cost. Its **lead-guided** assignment builds the auxiliary head's targets from the lead
head's predictions with a looser (coarse) candidate set, and the lead head's with a strict (fine) set
(Chapter 25).

---

## YOLOv8 and YOLO11: TAL + DFL, no objectness

Ultralytics' `v8DetectionLoss` and `TaskAlignedAssigner`, which YOLOv8, YOLOv9 (main branch) and YOLO11
share:

```python
# assignment (no gradient)
align = score[gt_class] ** 0.5 * CIoU(pred_box, gt_box).clamp(0) ** 6.0   # α = 0.5, β = 6
candidates = points whose centre is strictly inside gt_box                 # all pyramid levels together
positives  = top-10 candidates by align, per object                        # tal_topk = 10
conflict   = a point claimed by several objects keeps the one with the highest IoU
soft_target = align * max_IoU_of_object / max_align_of_object             # per positive, per object

# loss
cls  = BCE(logits, soft_target).sum() / soft_target.sum()                  # every point, every class
box  = ((1 - CIoU) * w).sum() / soft_target.sum()                          # w = soft target of the positive
dfl  = (DFL(dist_logits, target_ltrb) * w).sum() / soft_target.sum()      # reg_max = 16
loss = (7.5 * box + 0.5 * cls + 1.5 * dfl) * batch_size
```

Things the pseudo-code makes visible:

- **Assignment uses the network's own predictions**, scores and boxes, detached. The same object can
  get different positives in consecutive iterations.
- **There is no level assignment by size.** Candidates come from all three levels at once, and the top-10
  ranking decides. A large object usually ends up on P4/P5 because those points predict it better, not
  because a rule sends it there.
- **No objectness.** The class score is trained toward a localisation-quality target, so one number
  ranks boxes by "right class and well localised".
- **Every term is normalised by the sum of soft targets**, not by the number of positives. Images with
  many poorly matched objects contribute less than their positive count would suggest.
- **The assigner uses CIoU** (clamped at 0), not plain IoU, in its ranking metric.

### How many positives does an object get?

The number of *candidates* (grid points strictly inside the box) at 640 × 640, computed for boxes
centred near (323, 212):

| Object size (px) | P3 (stride 8) | P4 (16) | P5 (32) | Candidates | Positives (top-10) | With STAL (YOLO26) |
|---|---|---|---|---|---|---|
| 6 × 6 | 0–1 | 0 | 0 | **0–1** | 0–1 | 2–5 |
| 12 × 10 | 1–4 | 0–1 | 0 | 2–4 | 2–4 | 2–5 |
| 20 × 20 | 4–9 | 1–4 | 0 | 8–10 | 8–10 | same |
| 40 × 30 | 15–16 | 4 | 1–2 | 20–22 | 10 | same |
| 100 × 80 | 117–130 | 24–30 | 9–12 | ~160 | 10 | same |
| 300 × 240 | ~1,100 | ~270 | 70 | ~1,450 | 10 | same |

A 6 × 6 object centred on a grid line contains **no grid point at all** and is never learned. Objects
of 20–40 px get about as many positives as they have candidates. Above that, every object gets exactly
10, whatever its size. Chapter 41 builds on this table.

---

## YOLOv10: two heads, two assignments

`E2EDetectLoss` runs the same `v8DetectionLoss` twice: once with `tal_topk = 10` for the one-to-many
head, once with `tal_topk = 1` for the one-to-one head, and **adds them with equal weight**. Using the
same α and β for both (the "consistent matching metric") means the single one-to-one positive is the
top one-to-many positive, so the two heads never receive contradictory supervision.

---

## YOLO26: ProgLoss, top-k then top-1, STAL, L1

YOLO26's `E2ELoss` changes four things:

| Change | Code | Effect |
|---|---|---|
| **Progressive loss weights (ProgLoss)** | `total = o2m · L_o2m + o2o · L_o2o`; `o2m` starts at 0.8 and decays linearly per epoch to 0.1 at the last epoch; `o2o = 1 − o2m` | Dense supervision early, the deployed head's loss late |
| **Softer one-to-one assignment** | one-to-one branch: `tal_topk = 7`, `topk2 = 1` | Choose 7 candidates, resolve conflicts among them, *then* keep the best 1. Fewer objects lose their only positive to a neighbour |
| **STAL** (small-target-aware) | in `select_candidates_in_gts`, box sides below `stride[1]` (16 px) are enlarged to 16 px around the centre, for candidate selection only | Every object has at least one candidate point. The regression target is still the true box |
| **L1 instead of DFL** | `reg_max = 1`; `BboxLoss` computes L1 on (l, t, r, b) distances divided by image width/height, weighted by the soft target, gain `dfl` | Smaller head, simpler export; distances unbounded |

Two implementation details matter when you read logs or modify the code:

- **The one-to-one head trains on detached features.** In `Detect.forward`, the one-to-one branch
  receives `x.detach()` during training. Its loss updates only its own head. The backbone and neck are
  shaped by the one-to-many loss (and by the one-to-one loss only indirectly, through the shared
  assignment metric).
- **The logged losses are the one-to-one branch's.** `E2ELoss` returns the weighted sum for the
  gradient but reports the one-to-one loss items. The log shows the deployed head's training loss,
  not the total.

The released checkpoints were trained on an experimental branch with a few extra internal parameters
recorded in their `train_args`: an `o2m` weight (1.0 for N, 0.705 for S–X), a `topk` (8 for N, 5 for
S–X) and a classification weight `cls_w`. They are not accepted by the public package, and the YOLO26
training-recipe guide says training on `main` "lands within a negligible distance of the published
metrics". The loss gains, unlike these, are public (table below).

---

## Loss gains through the versions

| Version | Classification | Box | Distance / objectness | Normaliser |
|---|---|---|---|---|
| v1 | SSE, 1 | SSE, λ_coord 5 | conf SSE, λ_noobj 0.5 | — |
| v3 | BCE, 1 | SSE on t, ×(2 − wh) | obj BCE, 1 | — (sum) |
| v5 | BCE, 0.5 · nc/80 · 3/nl | CIoU, 0.05 · 3/nl | obj BCE, 1.0 · (img/640)² · 3/nl; level balance 4/1/0.4 | mean per term, × batch |
| YOLOX | BCE (IoU-soft), 1 | IoU, 5 | obj BCE 1; L1 1 (last 15 epochs) | # positives |
| v6 3.0 | VFL, 1 | GIoU/SIoU, 2.5 | DFL 0.5 (m/l) | sum of targets |
| v8 / 11 | BCE (TAL-soft), 0.5 | CIoU, 7.5 | DFL, 1.5 | sum of soft targets, × batch |
| v10 | same as v8, two heads summed | | | |
| YOLO26 default.yaml | BCE (TAL-soft), 0.5 | CIoU, 7.5 | L1 distances, 1.5 | sum of soft targets, × batch |
| YOLO26 Objects365 stage | 0.5 | 7.5 | 6.0 | |
| YOLO26-N COCO stage | 0.56 | 5.63 | 9.04 | |
| YOLO26-S…X COCO stage | 0.65 | 9.83 | 0.96 | |

The YOLO26 COCO-stage gains came from evolutionary search per size (Chapter 32). The N model puts
nearly 10× more weight on the distance term than S–X do, a reminder that gains are tuned per model
and per normaliser, not universal.

---

## Reading the loss columns in an Ultralytics log

```text
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100      7.43G      1.012     0.6124      1.031        147        640
```

| Column | What it is | Typical behaviour | Red flag |
|---|---|---|---|
| `box_loss` | 7.5 × weighted mean (1 − CIoU) over positives | Falls fast, then slowly; 1.0 means mean CIoU ≈ 0.87 | Flat from the start: labels misaligned (Chapter 27 checklist) |
| `cls_loss` | 0.5 × BCE over all points and classes / sum of targets | Highest early; keeps falling longest | Rising while mAP rises: soft targets growing as boxes improve, often harmless |
| `dfl_loss` (v8/11) | 1.5 × DFL | Cannot reach 0: plateaus at 1.5 × (target entropy + ambiguity) | Well below about 0.75 (1.5 × the ~0.5-nat floor) is suspicious: check box targets |
| `l1_loss` (YOLO26) | gain × L1 on image-normalised distances | Small numbers (hundredths) | Not comparable to v8's `dfl_loss` column |

Two runs from this book's TinyYOLO (Chapter 39, 30 epochs on synthetic shapes):

| Run | Epoch 1 (box / cls / dist) | Epoch 30 | Val AP |
|---|---|---|---|
| DFL head + NMS (`reg_max=16`) | 2.64 / 4.26 / 2.91 | 0.37 / 0.34 / **0.87** | 0.833 |
| YOLO26-style L1 + one-to-one (`reg_max=1`, e2e) | 2.85 / 5.99 / 0.034 | 0.64 / 0.45 / **0.006** | 0.787 |

The DFL column ends at 0.87, which is 0.58 nats after dividing by the 1.5 gain, close to the ~0.5-nat
entropy floor of Chapter 6. The L1 column means something else entirely. The e2e run's box and class
losses are higher because the logged values come from the one-to-one branch, whose single positive per
object is harder to fit.

---

## Knobs you can turn, and when

| Knob | Default | Turn it when | Watch out |
|---|---|---|---|
| `box` | 7.5 | Localisation matters more than recall (measurement, robotics): raise to 10–15 | AP50 may drop as classification gets relatively less weight |
| `cls` | 0.5 | Many fine-grained classes confuse each other: raise to 1.0 | Scores become more peaked; recheck the threshold |
| `dfl` | 1.5 | Rarely. For YOLO26 it weights L1, not DFL | Values tuned for v8 do not transfer to YOLO26 |
| `cls_pw` | 0.0 | Strong class imbalance: try 0.5–1.0 | Rare-class false positives rise |
| `single_cls` | False | Class identity doesn't matter or labels are inconsistent | Merges all classes into one |
| TAL top-k, α, β | 10, 0.5, 6.0 | Not exposed in `default.yaml`; edit `v8DetectionLoss` | Changing them changes the meaning of every gain |

<div class="callout field"><span class="callout-title">Field note</span>Before touching gains, check
the positives. Most "loss doesn't go down" or "small objects never detected" reports trace back to
assignment: objects under 8 px at the training resolution, or boxes in the wrong coordinate convention,
receive zero or wrong positives. Gains cannot fix supervision that never arrives. The grid lab
(<a href="./labs">Interactive Labs</a>) shows which points fall inside a box at each stride.</div>

---

## Failure modes that come from assignment

| Symptom | Assignment cause | Fix |
|---|---|---|
| Tiny objects never detected | No grid point inside the box (no STAL) | Higher `imgsz`, P2 head (Chapter 41), YOLO26 (STAL) |
| Crowded scenes: some objects missed | A point inside two boxes goes to the higher-IoU object; small, occluded objects lose their points | P2 level, higher resolution, crowd-aware models (Chapter 43) |
| Large objects detected late in training | Only 10 positives regardless of size; early predictions are poor | Usually resolves; check that `scale` augmentation covers large sizes |
| One-to-one head misses objects the o2m head finds | Single positive; conflicts in dense scenes | Use `nms=None` (o2m + NMS) if the latency budget allows; train longer |
| Duplicate boxes from a one-to-one head | o2o head undertrained (short fine-tune) | More epochs; keep a light NMS as a safety net (Chapter 7) |

---

## Key Takeaways

- YOLO assignment went from one positive per object (v1–v3), to many static positives (v4/v5), to
  dynamic prediction-aware positives (SimOTA, TAL), to dual heads (v10, YOLO26).
- v5 matches on width/height ratio (`anchor_t = 4`) and adds two neighbouring cells. Its decoding
  `2σ − 0.5` and `(2σ)²` exists to support that matching.
- v8/11/26 use TAL with α = 0.5, β = 6, top-10, CIoU in the metric, candidates from all levels at once,
  and soft targets that make the class score a localisation-quality score.
- YOLO26's `E2ELoss`: o2m weight 0.8 → 0.1 linearly over epochs, one-to-one top-7 → top-1, STAL
  enlarges sub-16-px boxes for candidate selection, and L1 on normalised distances replaces DFL. The
  one-to-one head trains on detached features, and the log shows its loss.
- Loss gains are specific to version, normaliser and model size. Compare logs only within the same
  loss definition.

## Check Yourself

<details class="check"><summary>A 7 × 7 px object at 640 × 640 is never detected by your YOLO11 model, but YOLO26 finds it. Which mechanism explains the difference?</summary>
TAL candidates are grid points strictly inside the box. At stride 8 a 7-pixel box can contain zero grid
centres, so it gets no positives and is never learned. YOLO26's STAL enlarges boxes smaller than 16 px
to 16 px for candidate selection, so the object always contains one to four stride-8 points and gets
positives (with the true box as the regression target).</details>

<details class="check"><summary>Why does YOLOv5 decode centres as 2σ(t) − 0.5 instead of σ(t)?</summary>
YOLOv5 assigns each object to its centre cell and to the two nearest neighbouring cells. A neighbouring
cell must be able to predict a centre outside its own cell. σ(t) only covers 0–1, while 2σ(t) − 0.5
covers −0.5 to 1.5, so the neighbour can reach the object's centre. The (2σ)² width decoding similarly
bounds sizes to 0–4× the anchor, matching anchor_t = 4.</details>

<details class="check"><summary>Your YOLO26 run logs box_loss 1.4 while the same data on YOLO11 logs 1.1. Is YOLO26 localising worse?</summary>
Not necessarily. YOLO26's E2ELoss reports the one-to-one branch's losses, which have one positive per
object and are harder to fit than YOLO11's top-10 one-to-many losses. The checkpoints also used different
gains. Compare mAP and AP75 on the same validation set, not logged loss values across versions.</details>

<details class="check"><summary>In TAL, does a 300 × 240 object get more positives than a 40 × 30 one?</summary>
No. Both get the top-10 by alignment metric (a 300 × 240 box has about 1,450 candidate points, a
40 × 30 box about 20). Larger objects have more candidates to choose from, not more positives. The
normalisation by the sum of soft targets also keeps any single object from dominating.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| YOLOv1 | Redmon et al., 2016 | arXiv:1506.02640 | Responsible cell, SSE weights, square-root sizes |
| YOLO9000 (v2) | Redmon, Farhadi, 2017 | arXiv:1612.08242 | Anchors, sigmoid offsets, IoU objectness target |
| YOLOv3 | Redmon, Farhadi, 2018 | arXiv:1804.02767 | One anchor per object, ignore threshold, BCE |
| YOLOv4 | Bochkovskiy, Wang, Liao, 2020 | arXiv:2004.10934 | Multiple anchors per object, CIoU, grid sensitivity |
| YOLOv5 `utils/loss.py`, `data/hyps/hyp.scratch-low.yaml` | Ultralytics | github.com/ultralytics/yolov5 | `build_targets`, decoding, gains, balance |
| YOLOX `yolox/models/yolo_head.py` | Megvii | github.com/Megvii-BaseDetection/YOLOX | SimOTA as implemented, loss weights, L1 phase |
| YOLOv6 `yolov6/models/losses/loss.py` | Meituan | github.com/meituan/YOLOv6 | ATSS warm-up, TAL settings, VFL, loss weights |
| YOLOv7 | Wang, Bochkovskiy, Liao, 2022 | arXiv:2207.02696 | Lead-guided assignment |
| TOOD | Feng et al., 2021 | arXiv:2108.07755 | Task-aligned assignment |
| YOLOv10 | Wang et al., 2024 | arXiv:2405.14458 | Consistent dual assignment |
| Ultralytics `utils/tal.py`, `utils/loss.py` (`TaskAlignedAssigner`, `v8DetectionLoss`, `BboxLoss`, `DFLoss`, `E2EDetectLoss`, `E2ELoss`), `nn/modules/head.py` | Ultralytics | github.com/ultralytics/ultralytics | Exact v8/v10/YOLO26 assignment and losses, detached one-to-one features |
| YOLO26 training recipe guide | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Per-size gains, internal `o2m`/`topk` values |
| YOLO26 model docs; end-to-end detection guide | Ultralytics, 2026 | docs/en/models/yolo26.md; docs/en/guides/end2end-detection.md | ProgLoss and STAL naming; o2m vs o2o accuracy |
| `odlab/assign.py`, `odlab/model.py` + training logs | this book | code/odlab | Positive counts, measured loss curves |

---

**Next:** [Chapter 31 — Axis 5: Training Recipe](./31_yolo_training_recipe.md) — optimiser, schedule,
warm-up, EMA, AMP and everything else that turns a loss into a trained model.
