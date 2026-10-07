---
title: "Chapter 6 — Losses"
---

[← Back to Table of Contents](./README.md)

# Chapter 6 — Losses

> *"A detection loss is three losses arguing about who gets the gradient."*

## Overview

Assignment turned every candidate into a target: background, or (class, box, maybe a quality score).
The loss turns targets into gradients. A detection loss always has a **classification** term and a
**box** term. Older designs add an **objectness** or **centre-ness** term, and newer ones an
**auxiliary** term. Each term has its own failure mode: imbalance for classification, scale
sensitivity for boxes, misalignment between the two. This chapter derives the losses that won, shows
with numbers what each one fixes, and lists the exact combinations used by YOLO and DETR families.
Implementations are in [`odlab/losses.py`](https://github.com/karam-nus/object-detection/blob/main/code/odlab/losses.py).

<div class="diagram">
<div class="diagram-title">Anatomy of a modern dense-detector loss</div>
<div class="flow-h">
  <div class="flow-node blue">classification<small>BCE / focal / QFL / VFL vs soft targets</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">+ box: IoU-type<small>CIoU / GIoU, weighted by target score</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">+ box: distance-type<small>DFL / L1 / FDR</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">÷ normaliser<small># positives or Σ soft targets</small></div>
</div>
</div>

---

## Classification: the imbalance problem

A YOLO at 640 scores 8,400 candidates × $K$ classes per image. RetinaNet scores about 100,000
anchors. A typical COCO image has around 7 objects. With binary cross-entropy every candidate
contributes, and the sum is dominated by the easy background.

### Focal loss, derived

With $p = \sigma(x)$, target $y \in \{0, 1\}$, and $p_t = p$ if $y = 1$ else $1 - p$:

$$\text{BCE} = -\log p_t, \qquad \text{FL} = -\alpha_t (1 - p_t)^{\gamma} \log p_t$$

The modulating factor $(1 - p_t)^{\gamma}$ is near 1 for misclassified examples and near 0 for easy
ones. Lin et al. used $\gamma = 2$, $\alpha = 0.25$ (the weight on positives; $1 - \alpha = 0.75$ on
negatives).

**Numerical check: what focal loss does to the balance.** One image, 10 positives that the model
currently scores $p = 0.3$, and 100,000 easy negatives scored $p = 0.01$:

| | Per-example BCE | Total BCE | Focal weight | Total focal |
|---|:---:|:---:|:---:|:---:|
| 10 positives | $-\ln 0.3 = 1.204$ | 12.0 | $0.25 \times 0.7^2 = 0.1225$ | **1.47** |
| 100,000 negatives | $-\ln 0.99 = 0.01005$ | **1,005** | $0.75 \times 0.01^2 = 7.5\times10^{-5}$ | 0.075 |

Under BCE the background contributes 98.8% of the loss. Under focal loss it contributes 4.9%. That
reversal let a one-stage detector match two-stage accuracy for the first time. `odlab.losses.sigmoid_focal_loss`
reduces exactly to BCE when $\gamma = 0$ and $\alpha$ is disabled (`test_focal_reduces_to_bce_and_dfl_expectation`),
and matches `torchvision.ops.sigmoid_focal_loss` otherwise.

### From hard labels to quality-aware soft labels

The classification score is used to rank boxes in NMS and AP. A hard target of 1 teaches "there is an
object here". It does not teach "this box is good". Three closely related fixes:

| Loss | Target | Weighting | Used by |
|---|---|---|---|
| **QFL** (GFL, 2020) | IoU of the predicted box (soft, in $[0,1]$) | $\lvert y - p\rvert^{\beta}$, $\beta = 2$ | GFL, NanoDet, PicoDet |
| **VFL** (VarifocalNet, 2021) | IoU for positives, 0 for negatives | positives weighted by $q$; negatives by $\alpha p^{\gamma}$ ($\alpha = 0.75$, $\gamma = 2$) | PP-YOLOE, YOLOv6, DAMO-YOLO, RT-DETR |
| **BCE vs TAL soft targets** | normalised alignment $\hat{t}$ (Chapter 5) | none (plain BCE) | YOLOv8, YOLO11, YOLO26 |

VFL is **asymmetric**: it down-weights negatives in focal style but keeps full weight on positives,
because positives are rare and precious. YOLOv8's approach is simpler. TAL already produces
quality-aware targets and selects at most $k$ positives per object, so plain BCE suffices. The class
scores of all non-positive candidates are pushed to zero, but the normaliser (below) prevents them
from dominating.

### Objectness and centre-ness: the separate quality head

YOLOv3–v5 and v7 predict an **objectness** logit per candidate and score detections by
objectness × class. YOLOv5 trains objectness with an *IoU-aware* target: the target for a positive is
its current IoU (detached), not 1, scaled by `gr = 1.0`. FCOS trains a **centre-ness** branch
(Chapter 4). Both are separate quality estimators. QFL, VFL and TAL fold quality into the class score
itself and removed the extra branch from YOLOX onwards. That simplifies the head and removes an
inconsistency: the two branches were trained on different samples but multiplied at inference.

---

## Box regression

### Smooth L1 and L1

Fast R-CNN used smooth L1 on the anchor deltas: quadratic below 1, linear above, robust to outliers.
DETR uses plain L1 on normalised $(c_x, c_y, w, h)$. L1 alone is not scale-invariant. A 5-pixel error
counts the same on a 10-pixel and a 500-pixel box. So it is always paired with an IoU term, and
DETR-family losses are $5 \cdot L1 + 2 \cdot \text{GIoU}$.

### IoU losses

$\mathcal{L} = 1 - \text{IoU}_{\text{variant}}$, derived in Chapter 3. Ultralytics weights each
positive's CIoU loss by its soft target score, so better-aligned positives drive the box gradient:

$$\mathcal{L}_{box} = \frac{\sum_{i \in \text{pos}} \hat{t}_i \,(1 - \text{CIoU}_i)}{\sum_i \hat{t}_i}$$

### Distribution Focal Loss (DFL)

Li et al. (GFL, 2020) argued that a box edge is often *ambiguous* (occlusion, blur, a dog's tail). A
single scalar regression cannot express that uncertainty. Predict instead a **distribution** over
discretised distances. For each side, the head outputs $R$ logits over bins $\{0, 1, \dots, R-1\}$
(in stride units; $R$ is `reg_max`, 16 in YOLOv8/11). The prediction is the expectation:

$$\hat{d} = \sum_{n=0}^{R-1} n \cdot \text{softmax}(z)_n$$

For a continuous target $y$ between bins $y_l = \lfloor y \rfloor$ and $y_r = y_l + 1$, DFL is the
cross-entropy against the two-hot distribution:

$$\text{DFL}(z, y) = -\big[(y_r - y)\log P_{y_l} + (y - y_l)\log P_{y_r}\big]$$

**Numerical check.** $y = 3.3$: target mass 0.7 on bin 3 and 0.3 on bin 4, and the expectation of
that distribution is $3 \times 0.7 + 4 \times 0.3 = 3.3$. The optimum reproduces $y$ exactly
(`test_focal_reduces_to_bce_and_dfl_expectation` constructs these logits and checks
`dfl_decode` returns the target).

**The loss floor.** At the optimum, DFL equals the *entropy* of the two-hot target,
$-(0.7 \ln 0.7 + 0.3 \ln 0.3) = 0.611$ nats, not zero. If the fractional parts of the targets are
uniformly distributed, the expected floor is

$$\mathbb{E}_f[H] = -\int_0^1 \big(f \ln f + (1-f)\ln(1-f)\big)\,df = 2 \cdot \tfrac{1}{4} = 0.5 \text{ nats}$$

That is why `dfl_loss` in a YOLOv8 training log plateaus around 0.8–1.0 rather than approaching 0.
TinyYOLO in Chapter 39 shows the same plateau. It is not a sign of under-fitting.

**DFL's hidden constraint: a bounded range.** Distances are clamped to $[0, R - 1 - 0.01]$ strides.
With $R = 16$, a P5 point (stride 32) can reach at most $15 \times 32 = 480$ pixels to each side.
That covers a 640 input, but objects close to 1,000 pixels in a 1,280 input can exceed it. The YOLO26
documentation states the design goal of removing DFL explicitly: "reducing detection-head complexity
while preserving an **unconstrained regression range**".

**DFL's deployment cost.** At inference DFL is a reshape, a softmax over 16 bins and a fixed 1×1
convolution with weights $0, 1, \dots, 15$, applied to $4 \times 8400$ values. It is cheap on GPUs.
On some NPUs it is awkward: softmax on an odd axis, a fixed conv that the quantiser must keep
precise, and a layout change. Removing it is one of YOLO26's export-driven decisions (Chapter 36).

### YOLO26: L1 on normalised distances instead of DFL

With `reg_max = 1`, the YOLO26 head regresses $(l, t, r, b)$ directly. In Ultralytics' `BboxLoss` the
`dfl` gain is reused to weight an **L1 loss on distances normalised by image width and height**:

$$\mathcal{L}_{dist} = \frac{\sum_i \hat{t}_i \cdot \frac{1}{4}\sum_{e \in \{l,t,r,b\}} \left| \frac{\hat{d}_{i,e} - d_{i,e}}{W\text{ or }H} \right|}{\sum_i \hat{t}_i}$$

The YOLO26 training-recipe page confirms that the checkpoints' `dfl` hyperparameter "weights an L1
loss on normalized box distances rather than distribution focal loss". `odlab`'s TinyYOLO switches to
the same loss when `reg_max=1`.

### D-FINE's Fine-grained Distribution Refinement (FDR)

D-FINE (2024) takes the distribution idea further for DETR decoders. Each decoder layer predicts a
*residual* probability distribution over edge offsets relative to the previous layer's box. The bins
are non-uniform, dense near zero and sparse far away, so late layers make fine corrections. **GO-LSD**
then distils the final layer's distributions into earlier layers (a self-distillation loss, "DDF").
This adds no inference cost and gives one of the larger single-paper AP jumps among real-time DETRs
(Chapter 19).

---

## Normalisation and balancing

**Normaliser.** Divide by the number of positives (RetinaNet, DETR) or by $\sum \hat{t}$, the sum of
soft targets (Ultralytics). Without it, an image with 50 objects contributes five times the loss of an
image with 10. Ultralytics then multiplies the per-image-average loss by the batch size, so the
logged loss values are per-image while gradients scale with batch.

**Loss gains**, as shipped:

| Detector | Classification | Box (IoU) | Distance / other | Objectness |
|---|---|---|---|---|
| **YOLOv5** | BCE, `cls = 0.5` | CIoU, `box = 0.05` | — | BCE vs IoU, `obj = 1.0`, per-level balance [4.0, 1.0, 0.4] |
| **YOLOv8 / YOLO11** | BCE vs TAL soft, `cls = 0.5` | CIoU, `box = 7.5` | DFL, `dfl = 1.5` | — |
| **YOLO26-N (checkpoint)** | BCE, `cls = 0.56` | CIoU, `box = 5.63` | L1 distances, `dfl = 9.04` | — |
| **YOLO26-S…X (checkpoint)** | BCE, `cls = 0.65` | CIoU, `box = 9.83` | L1 distances, `dfl = 0.96` | — |
| **DETR / DINO / RT-DETR** | focal or VFL, weight 1–2 | GIoU, 2 | L1, 5 | — |
| **D-FINE / DEIM** | VFL / MAL | GIoU, 2 | L1 5 + FGL + DDF | — |
| **FCOS** | focal | GIoU | — | centre-ness BCE |

The YOLO26 values come from the training arguments embedded in the released checkpoints (Chapter 31).
The N model weights the distance term far more heavily than the others do. It was found by
evolutionary search and illustrates that **loss gains are tuned per model size**, not universal
constants.

<div class="callout warn"><span class="callout-title">Trap</span>Loss gains interact with the
normaliser and with the number of positives per object (TAL's top-k). Copy a gain from one codebase
into another and you can be off by an order of magnitude. YOLOv5's <code>box = 0.05</code> and
YOLOv8's <code>box = 7.5</code> are not comparable numbers.</div>

---

## Auxiliary losses

| Mechanism | Where | Purpose | Inference cost |
|---|---|---|---|
| **Deep supervision** on every decoder layer | DETR family | Each layer's predictions are Hungarian-matched and supervised | none (extra layers' heads dropped or kept for refinement) |
| **Denoising loss** (noised GT boxes as queries) | DN-DETR, DINO, RT-DETR, D-FINE | Stable, direct box-refinement signal | none |
| **One-to-many auxiliary heads** | H-DETR, Co-DETR, YOLOv10/26 | Dense supervision for a one-to-one model | none (heads removed) |
| **Lead-guided auxiliary head** | YOLOv7 | Coarse-to-fine soft labels for an auxiliary head | none |
| **PGI auxiliary reversible branch** | YOLOv9 | Gradient information preserved through deep paths | none |
| **Foreground alignment branch** | YOLO27 *(preliminary)* | Close the one-to-many / one-to-one gap | none |

Every row is "train-time only". Since 2022 the field's main trick has been spending training compute
on supervision that is removed before deployment.

---

## Key Takeaways

- Background candidates outnumber objects by four orders of magnitude. Focal loss changes the
  background share of the loss from about 99% to about 5% in a typical case.
- Quality-aware classification (QFL, VFL, or BCE against TAL soft targets) makes the score reflect box
  quality, which is what ranking for NMS and AP needs.
- IoU-type box losses provide scale invariance. L1 or distribution losses provide well-conditioned
  gradients near the optimum. Strong recipes use both.
- DFL predicts a distribution per box edge. Its loss floor is the target entropy (≈0.5 nats on
  average), its range is bounded by `reg_max − 1` strides, and it is awkward on some NPUs. YOLO26
  replaced it with L1 on normalised distances.
- Loss gains and normalisers are codebase-specific and size-specific. YOLO26's per-size gains were
  found by evolutionary search.
- Most recent gains come from train-time-only auxiliary supervision: denoising, one-to-many heads,
  PGI, alignment branches.

## Check Yourself

<details class="check"><summary>Compute the focal-loss weight for an easy positive (p = 0.95) and a hard positive (p = 0.2) with γ = 2, α = 0.25. How much more does the hard one count, per example, than under BCE?</summary>
Easy: 0.25 × 0.05² = 6.25e-4, times BCE −ln 0.95 = 0.0513 → 3.2e-5. Hard: 0.25 × 0.8² = 0.16, times
−ln 0.2 = 1.609 → 0.257. Under BCE the ratio is 1.609 / 0.0513 ≈ 31. Under focal it is about 8,000.
Focal loss concentrates training on hard examples.</details>

<details class="check"><summary>Your YOLOv8 training log shows dfl_loss stuck at 0.9 while mAP keeps rising. Is something wrong?</summary>
Probably not. DFL's minimum is the entropy of the two-hot targets (about 0.5 nats on average). It is
also weighted and averaged over four sides, and ambiguous edges keep it above that floor. Watch
validation mAP, especially AP75, rather than the absolute DFL value.</details>

<details class="check"><summary>Why do DETR-family losses use both L1 and GIoU instead of one of them?</summary>
L1 on normalised coordinates gives a smooth, well-conditioned gradient everywhere but is not
scale-invariant. GIoU is scale-invariant and aligned with the evaluation metric, but its gradient is
weak or degenerate in some configurations (containment, near-optimum). The sum (5·L1 + 2·GIoU) gets
both properties.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Fast R-CNN | Girshick, 2015 | arXiv:1504.08083 | Smooth L1, multi-task loss |
| Focal Loss for Dense Object Detection | Lin et al., 2017 | arXiv:1708.02002 | FL, γ = 2, α = 0.25 |
| Generalized Focal Loss | Li et al., 2020 | arXiv:2006.04388 | QFL, DFL |
| VarifocalNet | Zhang et al., 2021 | arXiv:2008.13367 | VFL, asymmetric weighting |
| TOOD | Feng et al., 2021 | arXiv:2108.07755 | Task-aligned soft targets |
| D-FINE | Peng et al., 2024 | arXiv:2410.13842 | FDR, GO-LSD |
| DEIM | Huang et al., 2024 | arXiv:2412.04234 | Matchability-aware loss (MAL) |
| Ultralytics `utils/loss.py` (`DFLoss`, `BboxLoss`, `v8DetectionLoss`) | Ultralytics, 2026 | github.com/ultralytics/ultralytics | DFL formula, L1 replacement when reg_max = 1, soft-target weighting, normaliser |
| Ultralytics `cfg/default.yaml`; YOLOv5 `hyp.scratch-low.yaml` | Ultralytics | github.com/ultralytics | Default gains (7.5/0.5/1.5; 0.05/0.5/1.0) |
| YOLO26 training recipe | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Per-size loss gains, dfl = L1 note |
| YOLO26 model docs | Ultralytics, 2026 | docs/en/models/yolo26.md | "unconstrained regression range" |
| `odlab/losses.py` + tests | this book | code/odlab | Implementations and checks |

---

**Next:** [Chapter 7 — Post-Processing & NMS](./07_post_processing.md) — one-to-many training
leaves many confident boxes per object. At inference they must be reduced to one.
