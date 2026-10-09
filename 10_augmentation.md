---
title: "Chapter 10 — Augmentation"
---

[← Back to Table of Contents](./README.md)

# Chapter 10 — Augmentation

> *"In classification, augmentation changes the pixels. In detection, it also has to move the truth."*

## Overview

Augmentation multiplies the effective size of a dataset. For detection it has one extra obligation:
every geometric change to the image must be applied **identically to the boxes**, and boxes that no
longer make sense afterwards must be dropped. This chapter catalogues the transforms that matter, gives
the geometry of box transforms (including why rotation loosens boxes), explains the composite
augmentations that made modern YOLOs work (mosaic, mixup, copy-paste, large-scale jitter), and covers
schedules and test-time augmentation. Chapter 28 then goes through YOLO's exact implementation and
per-version defaults.

<div class="diagram">
<div class="diagram-title">Augmentation families</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card blue"><div class="card-title">Geometric</div><div class="card-desc">flip, scale, translate, crop, rotate, shear, perspective — boxes must follow</div></div>
  <div class="diagram-card green"><div class="card-title">Photometric</div><div class="card-desc">HSV, brightness, contrast, blur, noise, JPEG, channel swap — boxes unchanged</div></div>
  <div class="diagram-card purple"><div class="card-title">Occlusion</div><div class="card-desc">cutout / random erasing, grid mask — simulate partial visibility</div></div>
  <div class="diagram-card accent"><div class="card-title">Composite</div><div class="card-desc">mosaic, mixup, cutmix, copy-paste — change context and object density</div></div>
</div>
</div>

---

## Moving the truth: box transforms

For any affine or projective transform $M$ (a 3×3 matrix), transform the four box **corners** and take
their axis-aligned bounding box:

$$\text{corners}' = M \cdot [x, y, 1]^\top, \qquad b' = \big(\min x', \min y', \max x', \max y'\big)$$

then clip to the image and filter. `odlab.augment.random_affine` builds
$M = T \cdot S \cdot R \cdot P \cdot C$ (centre, perspective, rotation + scale, shear, translate), exactly
as Ultralytics' `RandomPerspective` does. It warps the image by inverse mapping and transforms the box
corners with the same matrix.

### Rotation loosens boxes

The bounding box of a rotated box is larger than the object. A $w \times h$ box rotated by $\theta$
has an axis-aligned bounding box of

$$w' = w|\cos\theta| + h|\sin\theta|, \qquad h' = w|\sin\theta| + h|\cos\theta|$$

**Numerical check.** A 100 × 100 square rotated by 45° has a 141 × 141 bounding box: twice the area.
The IoU between the object's true extent and its new "ground-truth" box is $10{,}000/20{,}000 = 0.5$.
For round or irregular objects the true extent is smaller still. Large rotations therefore teach the
model *loose* boxes and hurt AP$_{75+}$. This is why YOLO recipes keep `degrees` near zero for natural
images. YOLO26 S–X use `degrees ≈ 0`, and only the N model uses 1.11° (Chapter 31). It is also why
aerial datasets with arbitrary orientation use **oriented** boxes or 90° rotations, which are exact,
rather than free rotation.

### Filtering degenerate boxes

After cropping, translation or mosaic, many boxes are cut down to slivers. Ultralytics'
`box_candidates` (also in `odlab.augment`) keeps a box only if:

- width and height > 2 pixels,
- its area is > 10% of its (scaled) original area,
- its aspect ratio is < 100.

The 10% rule matters. A dog cropped down to its tail would otherwise be a positive that is impossible
to learn.

---

## The catalogue

| Transform | What it teaches | Box handling | Typical strength (YOLO-style) | Failure mode |
|---|---|---|---|---|
| **Horizontal flip** | left–right invariance | mirror x | p = 0.5 | Breaks text, arrows, left/right classes |
| **Vertical flip** | top–bottom invariance | mirror y | 0 for natural images; 0.5 for aerial/microscopy | Unrealistic for ground-level scenes |
| **Scale jitter / random resize** | scale robustness | scale | ±50% (YOLO `scale=0.5`), up to 0.9–0.95 in YOLO26 | Too strong shrinks small objects below the stride |
| **Translate** | position invariance near borders | shift | ±10% (`translate=0.1`) | — |
| **Random crop** (SSD min-IoU crop) | zoom-in, partial objects | crop + filter | crop keeps ≥ x IoU with some object | Can remove all objects |
| **Rotation / shear / perspective** | viewpoint | corners → AABB | ≈ 0 for natural images | Loose boxes (above) |
| **HSV / colour jitter** | lighting, camera | none | h 0.015, s 0.7, v 0.4 (YOLO default) | Hue shifts break colour-defined classes (traffic lights) |
| **Blur, noise, JPEG** | sensor and compression robustness | none | small | — |
| **Cutout / random erasing** | occlusion | none (or drop fully-erased boxes) | small patches | Erasing a whole small object leaves a positive with no evidence |
| **Mosaic** | multi-scale context, small objects, more objects per image | stitch + crop + filter | p = 1.0 then off for the last epochs | Unnatural seams and contexts |
| **MixUp** | soft decision boundaries | union of both images' boxes | p 0.0–0.43 in YOLO26 by size | Ghosted objects; harms small models |
| **Copy-Paste** | more instances, rare classes | paste instance masks and boxes | p 0.1–0.6 by size | Needs masks; pasted objects float out of context |
| **Large-scale jitter (LSJ)** | strong scale variation for long schedules | resize 0.1–2.0×, crop/pad to fixed size | ViTDet, Copy-Paste paper | Needs long training |

### Mosaic

Introduced in YOLOv4 and made the default by YOLOv5: four images are resized and tiled around a
random centre on a $2s \times 2s$ canvas, then randomly affined and cropped back to $s \times s$.
`odlab.augment.mosaic4` implements the Ultralytics geometry (`test_mosaic_boxes_inside_canvas`). Why
it works:

- **Context diversity.** Objects appear next to unrelated scenes, so the model cannot lean on co-occurrence.
- **Scale.** Each source image is effectively downscaled, so mosaic produces many small objects. This
  is good for AP$_S$ and bad if your deployment objects are always large.
- **Batch statistics.** Every sample contains four images, which stabilises BatchNorm at small batch sizes.
- **More positives per image.** DEIM's *Dense O2O* uses exactly this to give one-to-one DETR training
  more matched queries per image.

The cost is distribution shift. Mosaic images do not look like test images. Hence `close_mosaic`:
Ultralytics turns mosaic off for the last 10 epochs by default (YOLO26's Objects365 stage used 8),
and YOLOX turned off strong augmentation for its last 15 epochs. The model then re-adapts to natural
images.

### MixUp and Copy-Paste

MixUp blends two images, $x = r x_1 + (1-r) x_2$ with $r \sim \text{Beta}(32, 32)$ (so $r \approx 0.5$),
and keeps both sets of boxes. It regularises large models and confuses small ones. The YOLO26 checkpoints
use mixup 0.012 for N and 0.427 for M/L/X.

**Copy-Paste** (Ghiasi et al., 2021) pastes segmented instances from one image into another. With LSJ
and a long schedule it added several AP on COCO in the original paper. It needs instance masks: YOLO
applies it when segment labels exist. The YOLO26 recipe uses copy-paste 0.075 (N) up to 0.404 (S, L, X).
Synthetic-data pipelines are copy-paste at scale.

---

## Augmentation strength should scale with model capacity

Augmentation is regularisation. Small models are capacity-limited and *under*-fit, so heavy
augmentation hurts them. Large models over-fit, so they need more. The YOLO26 checkpoint recipe shows
this directly:

| Setting | N | S | M | L | X |
|---|:---:|:---:|:---:|:---:|:---:|
| mosaic | 0.909 | 0.992 | 0.992 | 0.992 | 0.992 |
| mixup | 0.012 | 0.05 | 0.427 | 0.427 | 0.427 |
| copy_paste | 0.075 | 0.404 | 0.304 | 0.404 | 0.404 |
| scale | 0.562 | 0.9 | 0.95 | 0.95 | 0.95 |

YOLOv5 encoded the same idea as separate `hyp.scratch-low/med/high.yaml` files for small, medium and
large models. RTMDet and PP-YOLOE likewise use weaker augmentation for their tiny variants.

---

## Letterbox: the augmentation that isn't

Every YOLO resizes keeping the aspect ratio and pads to a square (or to a stride multiple) with grey
(114, 114, 114). Getting the forward and inverse mapping right is the most common deployment bug in
detection. The model is accurate, and the drawn boxes are offset by the padding.

<div class="lab" data-lab="letterbox"></div>

The mapping is $x' = r x + \text{left}$, with $r = \min(S/H, S/W)$. The inverse is
$x = (x' - \text{left})/r$, and `test_letterbox_roundtrip` checks that it round-trips exactly. With
`auto=True` (the "minimum rectangle" used for inference and Ultralytics' `rect` validation), padding
only reaches the next multiple of 32. A 1920 × 1080 frame at 640 becomes 640 × 384 instead of 640 × 640: 40% fewer pixels than a full
square. Chapter 28 covers rectangular training batches.

---

## Test-time augmentation (TTA)

Run the model on several transformed copies of the image, map predictions back, and fuse them with
NMS or WBF (Chapter 7). Ultralytics' `augment=True` uses scales {1, 0.83, 0.67} with a flip on the
middle scale. TTA typically adds about 1–2 AP on COCO at 3× or more the inference cost. It is
standard in competitions, in leaderboard entries for large detectors (Chapter 17) and in offline
labelling pipelines (Chapter 49). It is almost never worth it in a latency-bound deployment.

---

## Libraries

| Library | Box support | Notes |
|---|---|---|
| **Ultralytics** (built-in) | YOLO/xyxy internally | Mosaic, mixup, copy-paste, `RandomPerspective`, HSV; tightly integrated with `close_mosaic` |
| **Albumentations** | `pascal_voc`, `coco`, `yolo`, `albumentations` formats via `BboxParams` | `min_visibility` / `min_area` filters; huge transform zoo |
| **torchvision.transforms.v2** | `tv_tensors.BoundingBoxes` | Box-aware transforms in core PyTorch; `SanitizeBoundingBoxes` drops degenerate boxes |
| **Kornia** | GPU tensors | Differentiable, batched, on GPU |
| **MMDetection / MMYOLO pipelines** | config-driven | `Mosaic`, `YOLOXMixUp`, `CachedMosaic` |

---

## Domain-specific choices

| Domain | Add | Remove |
|---|---|---|
| **Aerial / satellite** | vertical flip, 90° rotations (exact), scale | perspective (camera is far away) |
| **Medical** | modest intensity changes matched to the scanner | strong hue/colour, flips if anatomy is lateralised |
| **Industrial inspection** | blur, noise, exposure, small rotations | mosaic if objects are always centred and large |
| **Driving** | weather/lighting, horizontal flip | vertical flip, large rotations |
| **Documents** | small rotations, scanning noise, JPEG | flips (text) |
| **Underwater / night** | strong HSV / low-light simulation | — |

---

## Key Takeaways

- Geometric augmentations must transform boxes through their four corners, then clip and filter
  (> 2 px, > 10% of original area, aspect < 100).
- Rotation turns a 100 × 100 box into a 141 × 141 one at 45°, an IoU of 0.5 with the truth.
  Keep free rotation near zero unless you use oriented boxes.
- Mosaic gives context diversity, small objects, multi-image batch statistics and more positives. Turn
  it off for the final epochs to re-adapt to natural images.
- Augmentation strength should grow with model capacity. The YOLO26 checkpoints use mixup 0.012 for N
  and 0.427 for M/L/X.
- Letterbox mapping and its inverse are part of the model contract. Most drawn-box offsets in
  production come from getting them wrong.
- TTA buys about 1–2 AP for several times the compute: for offline work, not for latency budgets.

## Check Yourself

<details class="check"><summary>A 60 × 20 box (w × h) is rotated by 30°. What is the new axis-aligned box size and its area relative to the original?</summary>
w' = 60·cos30 + 20·sin30 = 51.96 + 10 = 61.96; h' = 60·sin30 + 20·cos30 = 30 + 17.32 = 47.32. The
area is 2,932 vs 1,200, about 2.4× larger, so the label is now very loose.</details>

<details class="check"><summary>Your objects are always large and centred (a conveyor-belt inspection). Mosaic is on by default. What might go wrong?</summary>
Mosaic shrinks and relocates objects. The model spends capacity learning small, off-centre, partial
objects it will never see, and training batches no longer resemble deployment. Reduce mosaic
probability or disable it, use milder scale jitter, and validate on unaugmented images.</details>

<details class="check"><summary>Why is close_mosaic usually beneficial, even though mosaic helped during most of training?</summary>
Mosaic images are not natural. They have seams, unusual contexts and skewed scale distributions. The
last epochs without mosaic let BatchNorm statistics and the head adapt to the distribution of real
test images. This usually adds a few tenths of AP for little cost.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| SSD | Liu et al., 2016 | arXiv:1512.02325 | Min-IoU random crop |
| mixup | Zhang et al., 2018 | arXiv:1710.09412 | Beta-blended images |
| Learning Data Augmentation Strategies for Object Detection | Zoph et al., 2019 | arXiv:1906.11172 | Learned detection policies |
| YOLOv4 | Bochkovskiy et al., 2020 | arXiv:2004.10934 | Mosaic |
| YOLOX | Ge et al., 2021 | arXiv:2107.08430 | Strong aug off for last 15 epochs |
| Simple Copy-Paste | Ghiasi et al., 2021 | arXiv:2012.07177 | Copy-paste + LSJ |
| Exploring Plain ViT Backbones (ViTDet) | Li et al., 2022 | arXiv:2203.16527 | LSJ for long schedules |
| DEIM | Huang et al., 2024 | arXiv:2412.04234 | Dense O2O via mosaic |
| Ultralytics `data/augment.py` (`Mosaic`, `RandomPerspective`, `box_candidates`, `LetterBox`, `MixUp`) | Ultralytics | github.com/ultralytics/ultralytics | Exact geometry and filters |
| YOLO26 training recipe | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Per-size augmentation table |
| Albumentations | Buslaev et al., 2020 | arXiv:1809.06839 | Box-aware augmentation library |
| `odlab/augment.py` + tests | this book | code/odlab | Letterbox, affine, mosaic, mixup |

---

**Next:** [Chapter 11 — Backbones](./11_backbones.md) — Part II opens the network itself, starting
with the feature extractor that sets the accuracy ceiling.
