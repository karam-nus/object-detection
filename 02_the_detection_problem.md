---
title: "Chapter 2 — The Detection Problem, Formally"
---

[← Back to Table of Contents](./README.md)

# Chapter 2 — The Detection Problem, Formally

> *"Half of all detection bugs are coordinate bugs wearing a model-accuracy costume."*

## Overview

This chapter fixes the vocabulary used for the rest of the book: what a detector outputs, the eight
common ways to write down a box, the coordinate conventions that silently differ between libraries,
and the tensor shapes that flow from an image to a list of detections. None of it is glamorous. All
of it is where real pipelines break.

<div class="diagram">
<div class="diagram-title">From pixels to a set of detections</div>
<div class="flow-h">
  <div class="flow-node">image<small>H×W×3 uint8, RGB?</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">preprocess<small>letterbox, normalise</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">network<small>N candidates</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">decode<small>raw → xyxy pixels</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">select<small>threshold + NMS / top-k</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">un-letterbox<small>back to original pixels</small></div>
</div>
</div>

---

## The output: a set, not a list

A detector maps an image $I \in \mathbb{R}^{H \times W \times 3}$ to a finite set

$$\mathcal{D}(I) = \{(b_i, k_i, s_i)\}_{i=1}^{M}, \qquad b_i \in \mathbb{R}^4,\; k_i \in \{1,\dots,K\},\; s_i \in [0,1]$$

Three properties matter:

1. **$M$ varies per image** and is unknown at inference. The network outputs a fixed $N \geq M$
   candidates ($N = 8{,}400$ for a 640-pixel YOLO; $N = 300$ queries for RT-DETR, D-FINE and
   YOLO26's one-to-one head), and a selection step produces $\mathcal{D}$.
2. **Order is meaningless.** Any loss must be *permutation-invariant* with respect to the ground
   truth. Dense detectors get this from spatial assignment. DETR gets it from bipartite matching
   (Chapter 5).
3. **Scores rank; they are not necessarily probabilities.** AP depends only on the ordering of
   scores (Chapter 8). A detector can have excellent AP and terribly calibrated scores. Deployment
   thresholds care about calibration (Chapter 47).

### What exactly is the score?

| Detector family | Score definition | Consequence |
|---|---|---|
| **YOLOv3–v5, v7** | objectness × class probability | Two heads must agree; objectness learns "is there any object here" |
| **YOLOv8 / 11 / 26, YOLOX-style** | class probability only (sigmoid), trained against an IoU-aware soft target | One head; the score already encodes localisation quality |
| **FCOS** | class probability × centre-ness | Down-weights off-centre predictions before NMS |
| **GFL / VFL family** | IoU-aware classification score (QFL/VFL) | Ranking reflects box quality, which helps AP at high IoU |
| **DETR family** | sigmoid focal class score per query | One query per object; no NMS needed |
| **MLLMs** | often *no* score; boxes are generated text | AP is ill-defined without a score; papers report F1 at fixed IoU (Chapter 21) |

**Multi-label.** Sigmoid per-class scores (YOLOv8+, DETRs) let one box carry several classes, for
example `person` and `pedestrian` in a hierarchical taxonomy. Ultralytics exposes this as
`multi_label` in its NMS. Softmax heads (Faster R-CNN, YOLOv1) force a single class per box.

---

## Eight ways to write down a box

| Format | Tuple | Units | Used by | Watch out for |
|---|---|---|---|---|
| **xyxy** | $(x_1, y_1, x_2, y_2)$ | pixels | torchvision, pycocotools internals, NMS kernels, `odlab` | $x_2 > x_1$ must hold; degenerate boxes give NaN in some losses |
| **xywh** | $(x_1, y_1, w, h)$ | pixels | **COCO annotation JSON** | Top-left corner, *not* centre |
| **cxcywh** | $(c_x, c_y, w, h)$ | pixels | YOLO internals, DETR internals | Same letters as xywh, different meaning |
| **YOLO txt** | $(k, c_x/W, c_y/H, w/W, h/H)$ | $[0,1]$ | Ultralytics / Darknet label files | Class index first; zero-based classes |
| **ltrb distances** | $(l, t, r, b)$ from a point | pixels or strides | FCOS, YOLOv6/8/11/26 heads | Must be ≥ 0; DFL clamps to `reg_max − 1` |
| **anchor deltas** | $(t_x, t_y, t_w, t_h)$ | normalised by anchor | Faster R-CNN, RetinaNet, YOLOv2–v5 | Log-space sizes; variance scaling in SSD |
| **oriented** | $(c_x, c_y, w, h, \theta)$ | pixels, radians | DOTA, YOLO-OBB | Angle range conventions; boundary discontinuity (Chapter 43) |
| **quantised tokens** | four integers in $[0, 999]$ | 1/1000 of image | Qwen-VL (0–1000), Rex-Omni, Florence-2 | Coordinates relative to the *resized* or the *original* image depending on the model |

The conversions are one line each: see `xyxy_to_cxcywh`, `cxcywh_to_xyxy`, `xyxy_to_xywh` in
[`odlab/boxes.py`](https://github.com/karam-nus/object-detection/blob/main/code/odlab/boxes.py), and
`to_yolo_lines` / `from_yolo_lines` in
[`odlab/data.py`](https://github.com/karam-nus/object-detection/blob/main/code/odlab/data.py), which
`test_yolo_txt_roundtrip` checks in both directions.

### Anchor deltas, derived

For an anchor with centre $(a_x, a_y)$ and size $(a_w, a_h)$ and a ground truth $(g_x, g_y, g_w, g_h)$:

$$t_x = \frac{g_x - a_x}{a_w},\quad t_y = \frac{g_y - a_y}{a_h},\quad t_w = \log\frac{g_w}{a_w},\quad t_h = \log\frac{g_h}{a_h}$$

Dividing offsets by the anchor size makes the target scale-invariant. The log makes widths
symmetric: a box twice as wide as the anchor and one half as wide give $t_w = \pm 0.693$.

**Numerical check.** Anchor $(100, 100, 64, 64)$, ground truth $(110, 96, 128, 32)$:
$t_x = 10/64 = 0.156$, $t_y = -4/64 = -0.0625$, $t_w = \ln 2 = 0.693$, $t_h = \ln 0.5 = -0.693$.

YOLOv2/v3 replaced the unbounded centre offset with $b_x = \sigma(t_x) + c_x$, which keeps the centre
inside its grid cell. YOLOv5 widened this to $b_x = 2\sigma(t_x) - 0.5 + c_x$ so that neighbouring
cells can also predict the object (Chapter 30).

---

## Coordinate conventions: the silent disagreements

### Pixel indices vs continuous coordinates

There are two ways to say where a pixel is:

- **Discrete index**: pixel $(i, j)$ is a sample; a box $[x_1, x_2]$ covers pixels $x_1 \dots x_2$
  *inclusive*, so its width is $x_2 - x_1 + 1$. This is the legacy PASCAL VOC / Detectron-v1
  convention.
- **Continuous**: pixel $i$ spans the interval $[i, i+1)$ and its centre is at $i + 0.5$. Width is
  $x_2 - x_1$. This is COCO / pycocotools / torchvision / Ultralytics / `odlab`.

**Numerical check: why it matters.** Boxes `[0, 0, 9, 9]` and `[1, 1, 10, 10]`:

| Convention | Areas | Intersection | Union | IoU |
|---|:---:|:---:|:---:|:---:|
| +1 (inclusive) | 100, 100 | 9 × 9 = 81 | 119 | **0.681** |
| continuous | 81, 81 | 8 × 8 = 64 | 98 | **0.653** |

A 0.03 IoU difference on a 10-pixel object is enough to flip a match at the 0.65 or 0.70 threshold.
COCO evaluates at ten thresholds, so mixing conventions between training labels and evaluation
costs a measurable amount of AP on small objects with no visible bug.

### Where is the centre of the first grid cell?

Anchor points are placed at cell centres: $(j + 0.5)\,s_l$ for column $j$ at stride $s_l$.
`make_grid_points` in `odlab/anchors.py` uses `offset=0.5`, as Ultralytics' `make_anchors` does. The
first point at stride 8 is $(4, 4)$, the last one in a 640 image at stride 32 is $(624, 624)$ (checked
by `test_grid_points_count_8400`). An off-by-half error here shifts every predicted box by
$s_l/2$, which is 16 pixels at stride 32.

### Axis orientation and image orientation

Image $y$ points **down**. Rotations by positive angles in OpenCV/`odlab.augment.random_affine` are
counter-clockwise *on screen*. Two practical traps:

- **EXIF orientation.** Phone JPEGs store a rotation flag. PIL's `Image.open` ignores it unless you
  call `ImageOps.exif_transpose`. Annotation tools usually *apply* it. The result is labels that
  belong to a rotated image. Ultralytics applies `exif_transpose` when loading. Plain
  `cv2.imread` does by default; `cv2.IMREAD_IGNORE_ORIENTATION` does not.
- **BGR vs RGB.** OpenCV loads BGR, PIL loads RGB, and models are trained on one of them. A model fed
  the wrong channel order still runs and loses a few AP. People rarely notice. YOLO26-N's training
  recipe even uses a `bgr=0.106` augmentation (Chapter 31), which is a small hedge against this.

---

## Tensor shapes, end to end

<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 170px;">Input batch</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim generic">3</span><span class="ts-sep">,</span><span class="ts-dim feature">640</span><span class="ts-sep">,</span><span class="ts-dim feature">640</span><span class="ts-bracket">]</span></div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">float, RGB, /255</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 170px;">Pyramid P3 / P4 / P5</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim heads">C_l</span><span class="ts-sep">,</span><span class="ts-dim feature">80|40|20</span><span class="ts-sep">,</span><span class="ts-dim feature">80|40|20</span><span class="ts-bracket">]</span></div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">strides 8, 16, 32</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 170px;">YOLO raw (one-to-many)</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim vocab">4 + K</span><span class="ts-sep">,</span><span class="ts-dim seq">8400</span><span class="ts-bracket">]</span></div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">YOLOv8/11/26 export, before NMS</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 170px;">YOLO26 one-to-one / DETR</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">300</span><span class="ts-sep">,</span><span class="ts-dim generic">6</span><span class="ts-bracket">]</span></div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">x1 y1 x2 y2 score class — final</span>
  </div>
</div>

Where 8,400 comes from: $80^2 + 40^2 + 20^2 = 6{,}400 + 1{,}600 + 400$. The YOLO26 docs state the
two output contracts explicitly: the default one-to-many head exports `(N, nc + 4, 8400)` and needs
NMS; `nms=False` selects the one-to-one head and exports `(N, 300, 6)`. A deployment that swaps one
model for the other without changing its post-processor breaks silently.

---

## Variants of the task

| Variant | Output per object | Where in this book |
|---|---|---|
| Class-agnostic / objectness detection | box + score | proposals, open-world (Ch 14, 20) |
| Closed-set detection | box + one of $K$ classes | most of the book |
| Multi-label detection | box + subset of classes | YOLOv8+ `multi_label`, hierarchical taxonomies |
| Open-vocabulary / grounding | box + free-text phrase | Ch 20 |
| Referring expression comprehension | the *one* box matching a sentence | Ch 21 |
| Oriented (rotated) detection | 5-tuple box | Ch 43 |
| Instance segmentation / pose | box + mask / keypoints | Ch 38 |
| 3D detection | 7-DoF box in metric space | Ch 43 |
| Counting | integer per class | FOMO-style centroids (Ch 15), density maps |

### Annotation edge cases every dataset has

- **Truncated objects** at the image border: label the visible part. Augmentation crops create more
  of them (`box_candidates` drops boxes that kept less than 10% of their area).
- **Occlusion**: label the full extent or the visible extent? COCO labels visible extent. Some
  pedestrian datasets (CrowdHuman) provide both.
- **Crowd regions**: COCO marks dense groups as `iscrowd=1`. Detections inside them are *ignored*,
  not counted as false positives (Chapter 8). The YOLO label format has no way to express this.
  `coco_to_yolo` in `odlab/data.py` drops crowd annotations, and so does every converter.
- **"Difficult" objects** in PASCAL VOC are excluded from both TP and FP counting.

---

## Key Takeaways

- A detector outputs a set of $(b, k, s)$ of unknown size. Networks output a fixed number of
  candidates and select from them.
- Scores determine ranking and therefore AP. Their meaning (objectness × class, IoU-aware class,
  focal query score, or absent) differs across families and affects threshold choice.
- Learn the eight box formats by heart. COCO JSON is `xywh` top-left; YOLO txt is normalised
  `cxcywh` with the class first.
- Use continuous coordinates (no +1). On a 10-pixel object the legacy convention changes IoU by about
  0.03.
- Shape contracts differ between YOLO26's two heads, `(N, 4+K, 8400)` vs `(N, 300, 6)`. Swapping
  models without swapping post-processing breaks silently.
- EXIF rotation and BGR/RGB mismatches are common and invisible ways to lose accuracy.

## Check Yourself

<details class="check"><summary>A COCO annotation has bbox [50, 40, 100, 80]. Write it in xyxy and in YOLO format for a 640×480 image.</summary>
COCO is xywh with a top-left corner, so xyxy = [50, 40, 150, 120]. The centre is (100, 80) and the
size 100×80. YOLO: class cx/W cy/H w/W h/H = k 0.15625 0.16667 0.15625 0.16667.</details>

<details class="check"><summary>Why does YOLOv2 pass the centre offset through a sigmoid, and why did YOLOv5 change it to 2σ(t) − 0.5?</summary>
The sigmoid bounds the predicted centre to the responsible cell, which stabilised early training (an
unbounded offset can send boxes anywhere). YOLOv5 assigns each object to up to three cells, its own
plus two neighbours. A prediction from a neighbouring cell must reach slightly beyond its own
borders, and the range (−0.5, 1.5) allows that.</details>

<details class="check"><summary>Your exported model outputs [1, 300, 6] but your C++ post-processor expects [1, 84, 8400]. What happened?</summary>
The model was exported with its one-to-one (end-to-end) head: YOLO26 with nms=False, YOLOv10, or a
DETR. The output is already the final top-300 detections (x1, y1, x2, y2, score, class). Drop the NMS
stage and read the rows directly, after confidence thresholding and un-letterboxing.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Microsoft COCO: Common Objects in Context | Lin et al., 2014 | arXiv:1405.0312 | xywh annotation format, crowd regions |
| pycocotools `cocoeval.py` / `mask.py` | COCO API, 2015– | github.com/cocodataset/cocoapi | Continuous-coordinate IoU, crowd IoU |
| Faster R-CNN | Ren et al., 2015 | arXiv:1506.01497 | Anchor delta parameterisation |
| YOLO9000 | Redmon & Farhadi, 2017 | arXiv:1612.08242 | Sigmoid-bounded centre offsets |
| FCOS | Tian et al., 2019 | arXiv:1904.01355 | ltrb distances, centre-ness score |
| Ultralytics docs — YOLO26 dual-head output shapes | Ultralytics, 2026 | docs/en/models/yolo26.md | (N, nc+4, 8400) vs (N, 300, 6) |
| Ultralytics `utils/tal.py` `make_anchors` | Ultralytics | github.com/ultralytics/ultralytics | Grid offset 0.5 |
| `odlab/boxes.py`, `odlab/data.py`, `odlab/anchors.py` | this book | code/odlab | Conversions and their tests |

---

**Next:** [Chapter 3 — IoU and Its Family](./03_iou_family.md) — every assigner, loss, NMS and metric
in the book is built on one similarity function between boxes. We derive it and its seven variants.
