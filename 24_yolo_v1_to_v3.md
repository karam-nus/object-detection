---
title: "Chapter 24 — YOLOv1–v3: The Founding Ideas"
---

[← Back to Table of Contents](./README.md)

# Chapter 24 — YOLOv1–v3: The Founding Ideas

> *"Treat detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities." — Redmon et al., 2016*

## Overview

Three papers between 2015 and 2018 defined the YOLO formulation: a grid of cells, each responsible for
the objects whose centres fall in it, predicting boxes, confidences and classes in a single forward pass.
Almost every later YOLO keeps this skeleton and changes its parts. This chapter goes through each paper's
mechanism, loss and training details, measured results, and the limitations that motivated the next
version. Read it for the *why* behind design choices that later chapters take for granted: sigmoid
offsets, k-means anchors, multi-scale heads, BCE classes.

<div class="diagram">
<div class="diagram-title">Three steps</div>
<div class="flow-h">
  <div class="flow-node blue">YOLOv1 (2016)<small>7×7 grid, 2 boxes/cell, FC head, SSE loss</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">YOLOv2 (2017)<small>BN, k-means anchors, σ offsets, passthrough, multi-scale</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">YOLOv3 (2018)<small>Darknet-53, 3 scales × 3 anchors, BCE classes</small></div>
</div>
</div>

---

## YOLOv1 — detection as one regression

| | |
|---|---|
| **Introduced** | Redmon, Divvala, Girshick, Farhadi, 2016 (arXiv:1506.02640, CVPR 2016) |
| **Lineage** | GoogLeNet-inspired CNN; Darknet |
| **Paradigm** | grid cells, $B$ boxes per cell, fully connected head |
| **Assignment** | the cell containing the object centre; within it, the predictor with the highest current IoU |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | VOC 2007: 63.4 mAP (YOLO, 45 FPS); 52.7 mAP (Fast YOLO, 155 FPS) |
| **Latency** | real-time on a Titan X |
| **Pre-training** | ImageNet (first 20 conv layers at 224 px) |
| **License** | Darknet (permissive) |
| **Known failure modes** | small objects in groups; one class per cell; at most 98 boxes; coarse localisation |

### The formulation

Divide the 448 × 448 input into an $S \times S$ grid ($S = 7$). Each cell predicts:

- $B = 2$ boxes $(x, y, w, h)$: $(x, y)$ relative to the cell, $(w, h)$ relative to the image;
- a confidence per box, trained to equal $\Pr(\text{object}) \cdot \text{IoU}(\text{pred}, \text{truth})$;
- $C = 20$ conditional class probabilities $\Pr(\text{class}_i \mid \text{object})$, **one set per cell**.

The output is $S \times S \times (5B + C) = 7 \times 7 \times 30$. At test time the class-specific score
of a box is $\Pr(\text{class}_i \mid \text{object}) \cdot \text{confidence}$.

### The loss

A sum of squared errors, with indicators $\mathbb{1}_{ij}^{\text{obj}}$ (predictor $j$ in cell $i$ is
responsible for an object) and $\mathbb{1}_{ij}^{\text{noobj}}$:

$$\begin{aligned}
\mathcal{L} = \;& \lambda_{\text{coord}} \sum_{i,j} \mathbb{1}_{ij}^{\text{obj}} \big[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\big] + \lambda_{\text{coord}} \sum_{i,j} \mathbb{1}_{ij}^{\text{obj}} \big[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\big] \\
&+ \sum_{i,j} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 + \lambda_{\text{noobj}} \sum_{i,j} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2 + \sum_i \mathbb{1}_i^{\text{obj}} \sum_{c} \big(p_i(c) - \hat{p}_i(c)\big)^2
\end{aligned}$$

with $\lambda_{\text{coord}} = 5$ and $\lambda_{\text{noobj}} = 0.5$. Three ideas here persist in some form:

1. **Imbalance weighting.** Most cells are empty. $\lambda_{\text{noobj}} = 0.5$ down-weights their
   confidence loss and $\lambda_{\text{coord}} = 5$ up-weights coordinates. This is the ancestor of
   focal loss and of the separate box/cls gains in every modern YOLO.
2. **Square roots of width and height** make the same absolute error cost more on small boxes. It is a
   crude form of the scale invariance that IoU losses provide properly (Chapter 3).
3. **Responsibility by current IoU.** Within a cell, the predictor whose box currently overlaps the
   object best becomes responsible, so the two predictors specialise in shapes. It is a prediction-aware
   assignment, an ancestor of SimOTA and TAL (Chapter 5).

**Numerical check (square-root trick).** A 4-pixel width error on a 20-pixel box contributes
$(\sqrt{24} - \sqrt{20})^2 = 0.182$. On a 200-pixel box it contributes $(\sqrt{204} - \sqrt{200})^2 =
0.020$, about 9× less, where plain SSE would weight them equally.

### Training details

ImageNet pre-training of the first 20 conv layers at 224 px. Then 4 conv + 2 FC layers were added and
the model trained at 448 px for **135 epochs** on VOC 2007+2012, batch 64, momentum 0.9, weight decay
0.0005, dropout 0.5, scale/translation and HSV augmentation. The learning rate warmed up from
$10^{-3}$ to $10^{-2}$, held for 75 epochs, then $10^{-3}$ for 30 and $10^{-4}$ for 30. Leaky ReLU
(slope 0.1).

### What the error analysis showed

The paper's error analysis on VOC 2007 (following Hoiem et al.) compared YOLO with Fast R-CNN:

| Error type | Fast R-CNN | YOLO |
|---|:---:|:---:|
| Localisation errors | 8.6% | **19.0%** |
| Background errors | **13.6%** | 4.75% |

YOLO sees the whole image, so it rarely mistakes background for objects. Its coarse grid and FC head
localise poorly. Combining the two (rescoring Fast R-CNN boxes with YOLO) gave +3.2 mAP. Every later
version is, in part, an attack on that 19% localisation figure.

**Hard limits.** One class set per cell, two boxes per cell: at most 98 detections, and two small
objects of different classes in one cell cannot both be detected. Downsampling to 7×7 loses small
objects entirely.

---

## YOLOv2 — better, faster, stronger

| | |
|---|---|
| **Introduced** | Redmon & Farhadi, 2017 (arXiv:1612.08242, CVPR 2017 "YOLO9000") |
| **Lineage** | YOLOv1 + anchor boxes |
| **Paradigm** | anchors on a 13×13 grid (416 px), fully convolutional |
| **Assignment** | the cell containing the centre; the anchor with the best shape IoU |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | VOC 2007: 78.6 mAP at 544 px (40 FPS); 69.0 at 288 px (91 FPS); COCO test-dev 21.6 AP |
| **Latency** | 40–91 FPS on a Titan X depending on input size |
| **Pre-training** | ImageNet (Darknet-19) |
| **License** | Darknet |
| **Known failure modes** | single output scale; small objects still weak |

The paper is a sequence of ablations, each a change to the v1 recipe:

| Change | Effect (VOC 2007) |
|---|---|
| **Batch normalisation** on every conv; dropout removed | +2% mAP |
| **High-resolution classifier**: fine-tune the backbone at 448 on ImageNet before detection | +4% mAP |
| **Anchor boxes** (remove FC layers, predict per anchor) | recall 81% → 88%, mAP 69.5 → 69.2 |
| **Dimension clusters**: k-means anchors with $d = 1 - \text{IoU}$; $k = 5$ | 5 clusters give avg IoU 61.0, about the same as 9 hand-picked anchors (60.9) |
| **Direct location prediction** (sigmoid offsets, below) | with clusters, ~+5% over anchor version |
| **Passthrough layer**: reorganise 26×26×512 into 13×13×2048 and concatenate | +1% |
| **Multi-scale training**: every 10 batches pick an input size from {320, 352, …, 608} | one model, speed/accuracy trade-off at test time |
| **Darknet-19** backbone: 19 conv + 5 max-pool, 5.58 B operations | faster than VGG-16 at similar accuracy |

### Direct location prediction

Unconstrained anchor offsets ($x = x_a + t_x w_a$) let early predictions jump anywhere, which made
training unstable. YOLOv2 bounds the centre to the responsible cell $(c_x, c_y)$:

$$b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y, \quad b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}$$

with anchor (prior) size $(p_w, p_h)$, in grid units. YOLOv3 kept this, and YOLOv5 widened it to
$2\sigma(t) - 0.5$ so that neighbouring cells can predict an object (Chapter 30).

### YOLO9000

The same paper trained jointly on COCO detection and ImageNet classification using **WordTree**, a
hierarchy built from WordNet. Detection images back-propagate the full loss, while classification images
back-propagate only the class loss at their level of the tree. It detected 9,000+ categories, with 19.7
mAP on the ImageNet detection validation set (16.0 on the 156 classes absent from COCO). This was an
early form of the open-vocabulary detection of Chapter 20.

---

## YOLOv3 — an incremental improvement

| | |
|---|---|
| **Introduced** | Redmon & Farhadi, 2018 (arXiv:1804.02767) |
| **Lineage** | YOLOv2 + residual backbone + FPN-like heads |
| **Paradigm** | anchors at three scales |
| **Assignment** | per GT, the single best-IoU anchor (across all scales) is positive; other anchors with IoU > 0.5 are ignored |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | 608 px: 33.0 AP, 57.9 AP$_{50}$ at 51 ms; 320 px: 28.2 AP at 22 ms (Titan X) |
| **Latency** | AP$_{50}$ comparable to RetinaNet (57.5) at ~3.8× the speed |
| **Pre-training** | ImageNet (Darknet-53) |
| **License** | Darknet |
| **Known failure modes** | weak at high IoU (AP$_{75}$); "YOLOv3 struggles to get the boxes perfectly aligned" |

### What changed

- **Darknet-53**: 53 conv layers with residual connections and 3×3/1×1 alternation. The paper reports
  accuracy comparable to ResNet-152 at roughly twice the speed.
- **Three output scales** (strides 32, 16, 8; 13×13, 26×26, 52×52 at 416 px) with FPN-style upsampling
  and **concatenation** of earlier features. Each scale has 3 anchors, 9 in total from k-means (the
  famous list in Chapter 4).
- Per-scale output: $N \times N \times [3 \times (4 + 1 + 80)] = N \times N \times 255$.
- **Objectness by logistic regression**: target 1 for the best anchor of each object. Anchors that
  overlap a ground truth by more than 0.5 without being the best are **ignored** rather than penalised,
  a three-way positive/negative/ignore assignment.
- **Independent logistic (BCE) classifiers** instead of softmax. Labels can overlap ("woman" and
  "person"), which mattered for Open Images.

### What did not work (the paper says so)

The paper lists failed attempts: linear (instead of sigmoid) offset activations, **focal loss**
("dropped our mAP about 2 points"), and dual IoU thresholds. Its explanation for focal loss: the
objectness term and conditional class predictions already separate easy negatives. Its candour set a
standard that later papers rarely matched.

### The AP$_{50}$ vs AP story

YOLOv3's AP$_{50}$ (57.9) matched RetinaNet's, but its COCO AP (33.0) was well below RetinaNet's 39.1.
High-IoU localisation remained YOLO's weakness. Closing this gap took CIoU (v4), better assignment
(SimOTA, TAL) and distribution-based box regression (DFL in v6/v8). By YOLOv8, AP$_{75}$ / AP$_{50}$
ratios were comparable to two-stage detectors.

---

## What v1–v3 established

| Idea | Origin | Still present in 2026? |
|---|---|---|
| Grid responsibility (object centre ↔ cell) | v1 | Yes: anchor points + centre-inside-box candidates (TAL) |
| Imbalance weighting of empty cells | v1 | Replaced by focal-family losses and soft targets |
| Prediction-aware responsibility | v1 | Yes: SimOTA, TAL |
| Sigmoid-bounded centre offsets | v2 | Replaced by distance regression in anchor-free heads |
| k-means anchors | v2 | Only in anchor-based lines (v5, v7); anchor-free since YOLOX/v8 |
| Multi-scale training | v2 | Yes (`multi_scale` option), less used with mosaic |
| Three-scale FPN-like heads | v3 | Yes: P3–P5 everywhere (YOLO27-N/S may drop P4) |
| BCE multi-label classes | v3 | Yes (sigmoid class scores) |
| Ignore band in assignment | v3 | Largely replaced by dynamic assignment |

---

## Key Takeaways

- YOLOv1 framed detection as one regression over a 7×7 grid. It made far fewer background errors than
  Fast R-CNN (4.75% vs 13.6%) and far more localisation errors (19.0% vs 8.6%).
- Its loss already weighted empty cells down, used square roots for scale, and assigned responsibility by
  current IoU: early forms of focal weighting, scale invariance and dynamic assignment.
- YOLOv2 was an ablation-driven recipe: BatchNorm, high-res pre-training, k-means anchors, sigmoid
  offsets, passthrough, multi-scale training. 78.6 mAP on VOC at 40 FPS.
- YOLOv3 added Darknet-53, three scales with 3 anchors each, an ignore band, and BCE classes. It matched
  RetinaNet at AP$_{50}$ but not at high IoU.
- The high-IoU localisation gap is the thread that connects v1 to v8.

## Check Yourself

<details class="check"><summary>Why can YOLOv1 output at most 98 boxes, and why is that a problem for COCO but less so for VOC?</summary>
7 × 7 cells × 2 boxes = 98. Each cell also predicts only one class distribution. COCO images can contain
dozens of small objects, often several per cell, and VOC images rarely do. COCO evaluation also keeps up
to 100 detections per image.</details>

<details class="check"><summary>In YOLOv2's parameterisation, what range can b_x take for an anchor in cell column 6, and why was that a stability improvement?</summary>
b_x = σ(t_x) + 6 lies in (6, 7) grid units: the centre stays inside its cell. Unbounded offsets could move
a prediction anywhere early in training, which destabilised optimisation. Bounding them makes each anchor
responsible only for its own cell.</details>

<details class="check"><summary>YOLOv3 tried focal loss and lost 2 mAP. Why might focal loss help RetinaNet but not YOLOv3?</summary>
YOLOv3 separates objectness from conditional class probabilities and ignores ambiguous anchors with IoU >
0.5, so the class loss is computed only on positives and easy negatives are already handled by the
objectness term. Focal loss's down-weighting then mostly removes useful gradient. RetinaNet has no
objectness branch, and all of its ~100k anchors enter the class loss directly.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| You Only Look Once: Unified, Real-Time Object Detection | Redmon et al., 2016 | arXiv:1506.02640 | Formulation, loss, λ values, training schedule, error analysis, 63.4 / 52.7 mAP |
| YOLO9000: Better, Faster, Stronger | Redmon & Farhadi, 2017 | arXiv:1612.08242 | Ablation table, k-means, σ offsets, passthrough, multi-scale, WordTree, 78.6 mAP |
| YOLOv3: An Incremental Improvement | Redmon & Farhadi, 2018 | arXiv:1804.02767 | Darknet-53, three scales, ignore band, BCE classes, failed attempts, 33.0 AP |
| Diagnosing Error in Object Detectors | Hoiem et al., 2012 | ECCV 2012 | Error-analysis methodology used by YOLOv1 |
| Focal Loss (RetinaNet) | Lin et al., 2017 | arXiv:1708.02002 | 39.1 AP / 57.5 AP50 comparison |

---

**Next:** [Chapter 25 — YOLOv4–v7 and the Forks](./25_yolo_v4_to_v7.md) — from 2020 the YOLO name
fragmented, and the training pipeline became the main battleground.
