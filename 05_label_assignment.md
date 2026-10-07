---
title: "Chapter 5 — Label Assignment"
---

[← Back to Table of Contents](./README.md)

# Chapter 5 — Label Assignment

> *"Architecture decides what a detector can represent. Assignment decides what it actually learns."*

## Overview

A dense detector has thousands of candidates and an image has a handful of objects. Before any loss
can be computed, each candidate must be labelled **positive** (learn to fire for object $j$, and
regress its box), **negative** (learn background), or **ignored**. That labelling is *label
assignment*. It is invisible at inference, absent from most architecture diagrams, and responsible
for some of the largest accuracy differences in the field. This chapter derives seven assigners in
historical order, works each one through with numbers, and maps them to the detectors that use them.
Each one is implemented in [`odlab/assign.py`](https://github.com/karam-nus/object-detection/blob/main/code/odlab/assign.py).

<div class="diagram">
<div class="diagram-title">Three axes of assignment</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card blue"><div class="card-title">Static ↔ Dynamic</div><div class="card-desc">Fixed geometric rule, or a rule that uses the network's current predictions (OTA, SimOTA, TAL, Hungarian)</div></div>
  <div class="diagram-card green"><div class="card-title">Fixed k ↔ Adaptive k</div><div class="card-desc">Same number of positives per object, or more for easy/large objects and fewer for hard ones (ATSS, SimOTA)</div></div>
  <div class="diagram-card purple"><div class="card-title">One-to-many ↔ One-to-one</div><div class="card-desc">Several positives per object (needs NMS), or exactly one (NMS-free: DETR, YOLOv10/26 o2o head)</div></div>
</div>
</div>

---

## Why assignment is hard

Consider one 100 × 60-pixel car in a 640 YOLO. Its centre region overlaps perhaps 70 P3 cells, 18 P4
cells and 5 P5 cells. Which of those ~90 candidates should be positive?

- **Too few positives** (one point): the gradient signal is sparse and training is slow and
  unstable. DETR's 500 epochs are the extreme case.
- **Too many positives** (every point inside the box): border points are asked to regress a box whose
  evidence they barely see, and the classification head learns to fire on poor-quality locations.
  Those later win NMS with bad boxes.
- **Wrong level**: a car assigned to P3 competes with the small objects P3 should specialise in.
- **Misaligned tasks**: the location with the highest class score is not necessarily the one with
  the best box. If they differ, NMS keeps the confident-but-sloppy box. This is TOOD's
  "task misalignment".

Every assigner below is a different answer to these four problems.

---

## 1. Max-IoU assignment (Faster R-CNN, SSD, RetinaNet)

For anchor $a$ and ground truths $\{g_j\}$:

$$\text{label}(a) = \begin{cases} \arg\max_j \text{IoU}(a, g_j) & \text{if } \max_j \text{IoU} \geq \tau_{pos} \\ \text{ignore} & \text{if } \tau_{neg} \leq \max_j \text{IoU} < \tau_{pos} \\ \text{background} & \text{if } \max_j \text{IoU} < \tau_{neg} \end{cases}$$

RetinaNet: $\tau_{pos} = 0.5$, $\tau_{neg} = 0.4$. RPN: 0.7 / 0.3. A **low-quality match** rule then
gives every ground truth its best anchor, even below $\tau_{pos}$.

**Failure mode.** Small and oddly shaped objects. A 12 × 12 object against a 32 × 32 anchor has IoU
$144/1024 = 0.14$. It gets exactly one positive, from the low-quality rule. `test_max_iou_every_gt_gets_an_anchor`
exercises this case. The assignment is static: it ignores what the network predicts, so a good
prediction from an unassigned anchor is treated as a false positive and suppressed in training.

---

## 2. Points-in-box with scale ranges (FCOS)

A point is positive for $g$ if it lies inside $g$ (or within a centre-sampling radius) **and**
$\max(l, t, r, b)$ falls in its level's range ($[0, 64]$ for P3 … $[512, \infty)$ for P7). If a point
lies in several boxes, the smallest box wins. Simple, but the ranges are hand-set, and large boxes get
hundreds of positives while small ones get a few.

---

## 3. ATSS — let statistics set the threshold

Zhang et al. (2020) showed that the RetinaNet–FCOS gap came from these definitions, not from anchors.
Their **Adaptive Training Sample Selection**:

1. On each level, take the $k = 9$ anchors whose centres are closest to $g$'s centre.
2. Compute their IoUs with $g$: mean $m_g$, standard deviation $v_g$.
3. Threshold $\tau_g = m_g + v_g$. Candidates with IoU ≥ $\tau_g$ **and** centre inside $g$ are positive.

A well-matched object (high mean, low spread) gets a high threshold and keeps only its best anchors.
A poorly matched one (low mean, high spread) gets a low threshold and still receives positives. The
only hyperparameter is $k$, and results are insensitive to it.

**Numerical check.** Candidate IoUs for one object: $\{0.64, 0.60, 0.57, 0.41, 0.33, 0.20, 0.15,
0.11, 0.05\}$. Mean $= 0.340$. The standard deviation is $0.226$, using the unbiased estimator as
`torch.std` and the reference code do. So $\tau = 0.566$, and three anchors are positive. For a tiny
object whose candidate IoUs are all low, mean and spread fall together, the threshold falls with
them, and the best one or two anchors still qualify. `odlab.assign.atss_assign` implements it
(`test_atss_assigns_inside_boxes`).

---

## 4. OTA and SimOTA — use the network's own opinion

**OTA** (Ge et al., 2021) casts assignment as optimal transport. Each ground truth is a *supplier*
with $k_j$ units of "positive label". The background is a supplier with the rest. Every anchor is a
*demander* of one unit. The transport cost is the loss the anchor would incur if assigned to that
ground truth. The Sinkhorn–Knopp algorithm solves it. It is accurate, and too slow to run every
iteration.

**SimOTA** (YOLOX) keeps the idea and solves it greedily:

1. **Candidates**: points inside $g$ *or* within $2.5 \times s_l$ of its centre.
2. **Cost** for candidate $i$ and object $j$:
   $$C_{ij} = \underbrace{\sum_k \text{BCE}(p_{ik}, \mathbb{1}[k = k_j])}_{\text{classification}} + 3 \cdot \underbrace{(-\log \text{IoU}(\hat{b}_i, g_j))}_{\text{regression}} + 10^5 \cdot \mathbb{1}[\text{not in both regions}]$$
3. **Dynamic $k$**: $k_j = \max\!\left(1, \left\lfloor \sum \text{top-10 IoUs of } g_j \right\rfloor\right)$.
4. Each $g_j$ takes its $k_j$ lowest-cost candidates. A candidate claimed twice goes to the cheaper object.

**Why dynamic $k$ works.** The sum of the best IoUs estimates how many *good* predictions an object
already has. A large, well-predicted car whose top-10 IoUs are
$\{0.82, 0.78, 0.71, 0.66, 0.52, 0.40, 0.31, 0.22, 0.15, 0.08\}$ (sum 4.65) gets $k = 4$. A partly
occluded pedestrian with top-10 IoUs summing to 1.3 gets $k = 1$. Supervision follows prediction
quality, and it changes as training progresses. `test_simota_dynamic_k` checks that the larger, better-covered
object receives more positives.

---

## 5. Task-Aligned Assignment (TOOD → YOLOv6/8/11/26)

Feng et al. (2021) targeted *task misalignment*: the most confident location and the best-localised
location should be the same location. Define, for candidate $i$ and object $j$ with class $k_j$,

$$t_{ij} = p_{i,k_j}^{\;\alpha} \cdot u_{ij}^{\;\beta}$$

where $p$ is the predicted class probability and $u$ the IoU (Ultralytics uses CIoU clamped at 0)
between the *predicted* box and the ground truth. Then:

1. **Candidates**: points whose centre lies strictly inside $g_j$.
2. **Positives**: for each $g_j$, the **top-$k$** candidates by $t_{ij}$ ($k = 10$ in Ultralytics).
3. **Conflicts**: a point selected by several objects keeps the one with the highest IoU.
4. **Soft targets**: the classification target is not 1 but a normalised alignment score,
   $$\hat{t}_{ij} = t_{ij} \cdot \frac{\max_{i'} u_{i'j}}{\max_{i'} t_{i'j}}$$
   so the best-aligned point of each object gets that object's best IoU as its target, and the others
   get proportionally less.

Ultralytics uses $\alpha = 0.5$ and $\beta = 6.0$ (TOOD's paper used $\alpha = 1$).

**Numerical check: why $\beta = 6$.** Two candidates for the same object:

| Candidate | $p$ | $u$ | $p^{0.5}$ | $u^{6}$ | $t$ |
|---|:---:|:---:|:---:|:---:|:---:|
| A: confident, mediocre box | 0.90 | 0.60 | 0.949 | 0.0467 | **0.044** |
| B: hesitant, good box | 0.60 | 0.80 | 0.775 | 0.262 | **0.203** |

$B$ wins by 4.6×. With $\beta = 6$, localisation quality dominates the ranking. The soft target then
teaches B's classification score to rise toward 0.8. Over training, high scores come to *mean*
good boxes. This is why NMS on TAL-trained models keeps well-localised boxes, and part of why
TAL-based YOLOs gained AP$_{75}$ over their predecessors.

**Normalisation check.** If an object's best $t$ is 0.203 and its best IoU 0.80, a positive with
$t = 0.10$ gets soft target $0.10 \times 0.80 / 0.203 = 0.394$.

**Where it lives.** `odlab.assign.task_aligned_assign` reproduces Ultralytics' `TaskAlignedAssigner`
for one image, including the CIoU clamp, the highest-IoU conflict rule, the normalisation, and two
YOLO26 options covered below (`test_tal_positives_lie_inside_gt_and_stal_rescues_tiny`).

<div class="callout note"><span class="callout-title">Note</span>TAL's candidates are points
<em>inside</em> the box, not inside a centre radius. A tiny object can contain zero grid centres
at stride 8. Then it is simply not learned. YOLO26's STAL (below) and the grid lab in
<a href="./04_anchors_points_queries.md">Chapter 4</a> show the fix.</div>

---

## 6. Hungarian matching (DETR family)

With $N$ queries and $G \le N$ objects, find the permutation $\sigma$ that minimises total matching
cost:

$$\hat{\sigma} = \arg\min_{\sigma} \sum_{j=1}^{G} \mathcal{C}(\hat{y}_{\sigma(j)}, y_j), \qquad \mathcal{C} = \lambda_{cls} C_{cls} + \lambda_{L1} \|\hat{b} - b\|_1 + \lambda_{giou} (-\text{GIoU})$$

with $\lambda_{cls} = 2, \lambda_{L1} = 5, \lambda_{giou} = 2$ in Deformable-DETR, DINO, RT-DETR and
D-FINE. The class cost is focal-style. The Kuhn–Munkres algorithm solves this exactly in $O(N^3)$;
`scipy.optimize.linear_sum_assignment` is the standard call (`odlab.assign.hungarian_assign`,
`detr_match_cost`). Matched queries are positives, and every other query is background.

**What one-to-one buys.** The loss now penalises duplicates. A second query firing on the same car is
a background query with a high score, so the model learns to suppress its own duplicates and no NMS
is needed.

**What it costs.** One positive per object is sparse supervision, and early matching is unstable
because queries swap targets between iterations. The DETR descendants spent years compensating:
auxiliary losses at every decoder layer, denoising queries (DN-DETR/DINO), and extra one-to-many
branches during training only (H-DETR, Group-DETR, Co-DETR). DEIM's **Dense O2O** keeps one-to-one
matching but raises the number of *objects* per image with mosaic-style augmentation, so that more
queries get positives.

---

## 7. Dual assignment: one-to-many for learning, one-to-one for inference

**YOLOv10 (2024)** attaches two heads to the same neck:

- a **one-to-many** head trained with TAL ($k = 10$), which gives rich supervision,
- a **one-to-one** head trained with the *same* metric but $k = 1$.

At inference only the one-to-one head runs, and NMS is not needed. The paper's *consistent matching
metric* argument: if both heads use the same $(\alpha, \beta)$, the one-to-one positive is
also the top-ranked one-to-many positive. The two heads' supervision then agrees instead of
competing. Ultralytics' `E2EDetectLoss` implements exactly this (`tal_topk=10` and `tal_topk=1`).

**YOLO26** refines it in three ways, visible in Ultralytics' `E2ELoss` and `TaskAlignedAssigner`:

| Mechanism | What the code does | Purpose |
|---|---|---|
| **Progressive loss (ProgLoss)** | total $= w \cdot L_{o2m} + (1 - w) L_{o2o}$, with $w$ decaying linearly from 0.8 to 0.1 over the epochs | Learn from dense supervision early; finish by optimising the head you deploy |
| **One-to-one top-k then top-k2** | o2o branch uses `topk=7, topk2=1`: select 7 candidates, resolve conflicts, keep the best 1 | Softer one-to-one selection that is less brittle early on |
| **STAL** (small-target-aware) | ground-truth sides below `stride[1]` (16 px) are enlarged to 16 px *only for candidate selection* | Tiny objects always contain at least one grid point and so receive a positive |

The YOLO26 training-recipe page records the per-size values actually used. The one-to-many loss
weight `o2m` was 0.705–1.0, and the one-to-one top-k was 5–8 depending on the size (Chapter 31).
The accuracy cost of using the one-to-one head is 0.6–0.8 COCO AP across YOLO26 sizes. The YOLO27
preview claims to cut this to 0.4 AP on N and S with a training-only "foreground alignment" branch
(preliminary).

---

## The assigners side by side

| Assigner | Static / dynamic | Positives per object | Hyperparameters | Used by | Choose this when |
|---|---|---|---|---|---|
| **Max-IoU** | static | variable, often 0–1 for tiny objects | $\tau_{pos}, \tau_{neg}$ | Faster R-CNN, SSD, RetinaNet | Legacy two-stage pipelines |
| **FCOS ranges** | static | ∝ area | level ranges, radius | FCOS | Simple anchor-free baselines |
| **ATSS** | static, adaptive | adaptive | $k$ (insensitive) | ATSS, GFL, PP-YOLOE (warm-up), PicoDet | Strong static baseline; early training of dynamic assigners |
| **SimOTA** | dynamic | dynamic $k$ | radius 2.5, IoU weight 3 | YOLOX, YOLOv6 (early), DAMO-YOLO (AlignedOTA), RTMDet (dynamic soft label) | Fast convergence with prediction-aware cost |
| **TAL** | dynamic | top-$k$ = 10 | $\alpha, \beta, k$ | TOOD, YOLOv6 3.0, YOLOv8, YOLO11, YOLO26, PP-YOLOE | The modern YOLO default |
| **Hungarian** | dynamic, one-to-one | exactly 1 | cost weights | DETR family, D-FINE, RF-DETR | NMS-free, query-based |
| **Dual (o2m + o2o)** | dynamic | 10 for training head, 1 for deployed head | as TAL + schedule | YOLOv10, YOLO26 | NMS-free deployment of dense CNN detectors |

---

## Debugging assignment

Assignment bugs never crash. They show up as low recall for one size or class. Check these numbers:

1. **Positives per ground truth, by size bucket.** Log `fg.sum()` per object. If small objects average
   0.3 positives, they are mostly unassigned. Fix with resolution, a P2 head, STAL or a centre radius.
2. **Fraction of objects with zero positives.** It should be ~0 after the first few epochs for TAL.
3. **Level distribution.** Which strides do positives come from for each size bucket?
4. **Soft-target magnitudes.** If TAL targets stay below 0.2 for a class, its predicted boxes are poor.
   Look at box-loss components for that class.

`odlab.assign.task_aligned_assign` returns `fg`, `gt_idx` and `target_scores`, which is all the
material for these checks.

---

## Key Takeaways

- Assignment labels every candidate as positive (for which object), negative or ignored. It is the
  hidden hyperparameter behind many architecture comparisons.
- Static IoU thresholds starve small and unusually shaped objects. ATSS replaces fixed thresholds with
  per-object statistics.
- Dynamic assigners (SimOTA, TAL) use the network's predictions. SimOTA adapts the *number* of
  positives; TAL ranks by $p^{\alpha}u^{\beta}$ and supervises with IoU-aware soft targets.
- With $\beta = 6$, TAL lets localisation quality dominate, which teaches the score to mean "good box".
- Hungarian matching gives one positive per object and removes NMS, at the price of sparse
  supervision and slower convergence.
- YOLOv10 and YOLO26 train one-to-many and one-to-one heads together. YOLO26 shifts weight
  progressively to the one-to-one head and protects tiny objects with STAL.

## Check Yourself

<details class="check"><summary>Compute the TAL metric for p = 0.5, u = 0.9 and for p = 0.95, u = 0.7 with α = 0.5, β = 6. Which wins?</summary>
First: 0.707 × 0.531 = 0.376. Second: 0.975 × 0.118 = 0.115. The first wins by more than 3×. The
better-localised candidate is chosen even though its class score is lower.</details>

<details class="check"><summary>Why does DETR not need NMS, while a TAL-trained YOLO does?</summary>
TAL gives each object up to 10 positives. All of them are trained to fire, so duplicates are the
intended behaviour and NMS removes them. Hungarian matching gives each object exactly one positive.
Any other query firing on the same object is trained as background, so the model learns not to
produce duplicates.</details>

<details class="check"><summary>A 6×6-pixel object occupies [29,29,35,35] in a 640 image. With strides 8/16/32, how many TAL candidates does it have, and what does STAL change?</summary>
Grid centres are at 4, 12, 20, 28, 36… (P3), 8, 24, 40… (P4) and 16, 48… (P5). None of them lies
strictly inside (29, 35), on any level, so the object has zero candidates and can never become a
positive. STAL enlarges it to 16×16 about its centre (32, 32), giving [24,24,40,40] for candidate
selection. The four P3 centres (28 or 36, 28 or 36) now lie inside, so the object gets positives.
Regression targets still use the true 6×6 box.</details>

<details class="check"><summary>What does YOLO26's ProgLoss weight look like at the start and end of training?</summary>
In Ultralytics' E2ELoss the one-to-many weight starts at 0.8 (one-to-one 0.2) and decays linearly to
0.1 (one-to-one 0.9) by the last epoch. Training starts with dense supervision and finishes by
optimising the head that is deployed without NMS.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Faster R-CNN; Focal Loss | Ren et al., 2015; Lin et al., 2017 | arXiv:1506.01497; 1708.02002 | Max-IoU thresholds 0.7/0.3 and 0.5/0.4 |
| FCOS | Tian et al., 2019 | arXiv:1904.01355 | Regression-range level assignment |
| ATSS | Zhang et al., 2020 | arXiv:1912.02424 | mean + std threshold; assignment explains anchor-free gap |
| OTA: Optimal Transport Assignment | Ge et al., 2021 | arXiv:2103.14259 | Assignment as optimal transport |
| YOLOX | Ge et al., 2021 | arXiv:2107.08430 | SimOTA, dynamic k, cost weights |
| TOOD | Feng et al., 2021 | arXiv:2108.07755 | Task-aligned metric and soft targets |
| DETR | Carion et al., 2020 | arXiv:2005.12872 | Hungarian set loss |
| DINO; DN-DETR | Zhang et al., 2022; Li et al., 2022 | arXiv:2203.03605; 2203.01305 | Denoising queries, cost weights 2/5/2 |
| H-DETR; Group-DETR; Co-DETR | Jia 2023; Chen 2023; Zong 2023 | arXiv:2207.13080; 2207.13085; 2211.12860 | One-to-many auxiliary supervision |
| DEIM | Huang et al., 2024 | arXiv:2412.04234 | Dense O2O |
| YOLOv10 | Wang et al., 2024 | arXiv:2405.14458 | Consistent dual assignment |
| Ultralytics `utils/tal.py`, `utils/loss.py` (`E2ELoss`, `E2EDetectLoss`) | Ultralytics, 2026 | github.com/ultralytics/ultralytics | α = 0.5, β = 6, top-10, CIoU clamp, STAL floor, topk2, ProgLoss decay |
| YOLO26 training recipe guide | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Per-size o2m weights and top-k |
| `odlab/assign.py` + tests | this book | code/odlab | Implementations of all seven assigners |

---

**Next:** [Chapter 6 — Losses](./06_losses.md) — assignment produced targets; now we choose the
functions that pull predictions toward them.
