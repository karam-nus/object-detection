---
title: "Chapter 28 — YOLO Axis 2: Preprocessing & Augmentation"
---

[← Back to Table of Contents](./README.md)

# Chapter 28 — YOLO Axis 2: Preprocessing & Augmentation

> *"A YOLO never sees your image. It sees a letterboxed, mosaicked, affine-warped, HSV-jittered collage that contains your image."*

## Overview

This axis follows a training image from disk to tensor in the Ultralytics pipeline: the exact transform
order, what each transform does to boxes, how the pipeline changes for the last epochs, and how
validation and inference preprocessing differ from training. A table compares augmentation defaults
across YOLO versions. General augmentation theory (box geometry, why mosaic works) is in Chapter 10.
This chapter is the implementation.

<div class="diagram">
<div class="diagram-title">Ultralytics training pipeline (v8_transforms), in order</div>
<div class="flow">
  <div class="flow-node blue wide">load_image — resize so the long side = imgsz (keep aspect)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Mosaic (p = mosaic) → [CopyPaste (segments)] → RandomPerspective (degrees, translate, scale, shear, perspective)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">MixUp (p = mixup) · CutMix (p = cutmix) — each with its own mosaic+affine of another sample</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Albumentations (if installed): Blur, MedianBlur, ToGray, CLAHE at p = 0.01 each</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">RandomHSV (hsv_h, hsv_s, hsv_v) → RandomFlip vertical (flipud) → RandomFlip horizontal (fliplr)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Format — boxes to normalised xywh, image to CHW uint8 tensor (÷255 on GPU)</div>
</div>
</div>

---

## Step by step

### 1. Load and resize

`load_image` reads the image (EXIF-aware), then resizes it so that its **long side equals `imgsz`**,
keeping the aspect ratio. Upscaling of small images is allowed in training. Labels are normalised
xywh, so they need no change. With `cache=ram` the resized array is kept in memory. This is the biggest
dataloader speed-up for small datasets.

### 2. Mosaic

`Mosaic(dataset, imgsz, p=mosaic)` builds a $2s \times 2s$ canvas filled with grey 114. It picks a
random centre $(x_c, y_c) \in [s/2, 3s/2]^2$ and places four images (the current one and three random
ones) in its quadrants, cropping each one where it hits the centre. Boxes are shifted by each image's
placement offset and clipped to the canvas. A 9-image variant exists (`n=9`). The geometry is
reproduced in `odlab.augment.mosaic4`.

### 3. RandomPerspective (with border crop)

Applied to the mosaic canvas with `border = (-s/2, -s/2)`, so the output is $s \times s$: a random
crop around the centre of the $2s$ canvas, plus the affine. The matrix is
$M = T \cdot S \cdot R \cdot P \cdot C$ (Chapter 10):

| Parameter | Effect | Default |
|---|---|:---:|
| `degrees` | rotation ± deg | 0.0 |
| `translate` | shift ± fraction of size | 0.1 |
| `scale` | zoom factor in [1 − scale, 1 + scale] | 0.5 |
| `shear` | shear ± deg | 0.0 |
| `perspective` | projective ± fraction | 0.0 |

Boxes are transformed by their four corners, then filtered with `box_candidates`: > 2 px, > 10% of the
scaled original area (1% when segments are present), aspect < 100.

### 4. MixUp and CutMix

`MixUp` draws *another* sample, runs it through the same mosaic + perspective pre-transform, and blends
the two images with $r \sim \text{Beta}(32, 32)$, keeping both label sets. `CutMix` pastes a rectangular
region of another sample. Both default to 0.0 in `default.yaml`. YOLO26's checkpoints used mixup up to
0.427 for M/L/X, and cutmix stays near zero.

### 5. Albumentations (optional)

If the `albumentations` package is installed, Ultralytics applies a light default set: Blur, MedianBlur,
ToGray and CLAHE at probability 0.01 each. Brightness/contrast, gamma and JPEG compression are present at
p = 0. Custom pipelines can be passed with the `augmentations=` argument.

### 6. HSV and flips

HSV gains (`hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`) multiply hue, saturation and value by random
factors in $[1-g, 1+g]$, using lookup tables (fast). Vertical flip (`flipud=0.0`) and horizontal flip
(`fliplr=0.5`) follow. For pose data the flip also swaps left/right keypoints via `flip_idx` in
`data.yaml`, and without it flips are disabled with a warning.

### 7. BGR channel swap

`bgr` (default 0.0) swaps RGB↔BGR with some probability. YOLO26-N's checkpoint used 0.106 as a hedge
against channel-order bugs in deployment (Chapter 2).

---

## The last epochs: `close_mosaic`

`close_mosaic = 10` (default): for the final 10 epochs, the dataset's `close_mosaic()` sets `mosaic`,
`copy_paste`, `mixup` and `cutmix` to 0 and rebuilds the transforms. The pipeline then becomes
**letterbox → affine (translate/scale only) → HSV → flips** on single, natural images. YOLOX used the same
idea earlier ("no augmentation for the last 15 epochs"). YOLO26 used `close_mosaic = 8` for Objects365
pre-training and 10 on COCO.

The training curve typically shows a visible step when mosaic turns off: box and class losses drop,
because the images are easier, and validation mAP rises by a few tenths. A model trained without the
closing phase is slightly mismatched to natural images.

---

## Validation and inference preprocessing

Validation is **not** augmented and does **not** use square inputs:

| Stage | Resize | Padding | Batch shape |
|---|---|---|---|
| **Training** | long side = imgsz (mosaic, affine) | to $s \times s$ | square |
| **Validation** (`model.val`) | `LetterBox(scaleup=False)` | **rectangular batches**: images grouped by aspect ratio, each batch padded to its own stride-multiple shape (`pad = 0.5`) | per batch |
| **Predict, PyTorch** | letterbox, `auto=True` | minimum padding to a multiple of 32 | per image |
| **Predict, exported static model** | letterbox to the export size | full padding to the fixed shape | fixed |

Two practical consequences:

1. **`model.val()` and an exported fixed-shape model can differ slightly in AP.** Rectangular
   validation batches see less padding than a fixed 640×640 square. Validate the *exported* model
   (Ultralytics can run `val` on most formats) before trusting the PyTorch number.
2. **Inference latency depends on aspect ratio** for dynamic-shape models. A 1920×1080 frame becomes
   640×384 with `auto=True`, 40% fewer pixels than 640×640 (Chapter 10's letterbox lab).

Input normalisation is simply **÷ 255 in RGB**. There is no ImageNet mean/std subtraction in YOLO
detection models. Feeding mean/std-normalised inputs to an exported YOLO is a common integration bug
with no error message.

---

## `rect` training and `multi_scale`

- **`rect=True`** trains with rectangular batches, as validation does. It reduces padding (faster, closer
  to inference), but it is incompatible with shuffling and with mosaic. Ultralytics disables both and
  warns. It suits datasets with consistent, non-square aspect ratios that are fine-tuned briefly.
- **`multi_scale`** (a fraction of `imgsz`, default 0.0) randomly varies the training size per batch,
  rounded to stride multiples, the idea from YOLOv2. With mosaic and scale jitter already providing scale
  variety, its benefit is modest. It slows training (larger batches at the high end) and is off by
  default.

---

## Augmentation defaults across versions

| Setting | YOLOv5 (scratch-low) | YOLOX (base) | YOLOv7 (p5) | v8 / 11 (`default.yaml`) | YOLOv12 (N → X) | YOLO26 COCO stage (N / S–X) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mosaic | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.909 / 0.992 |
| mixup | 0.0 | 1.0 (off for tiny/nano) | 0.15 | 0.0 | 0.0 → 0.2 | 0.012 / 0.05–0.427 |
| copy_paste | 0.0 | — | 0.0 | 0.0 | 0.1 → 0.6 | 0.075 / 0.30–0.40 |
| scale | 0.5 | mosaic scale (0.1, 2) | 0.9 | 0.5 | 0.5 → 0.9 | 0.562 / 0.9–0.95 |
| translate | 0.1 | 0.1 | 0.2 | 0.1 | 0.1 | 0.071 / 0.275 |
| degrees | 0.0 | 10.0 | 0.0 | 0.0 | 0.0 | 1.11 / ≈0 |
| shear | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 1.46 / ≈0 |
| hsv_h / s / v | .015 / .7 / .4 | HSV on | .015 / .7 / .4 | .015 / .7 / .4 | defaults | .014 / .645 / .566 · .013 / .353 / .194 |
| fliplr | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.606 / 0.304 |
| final no-mosaic phase | — | last 15 epochs | — | `close_mosaic=10` | 10 | 10 |

Sources: YOLOv5 `hyp.scratch-low.yaml`; YOLOX `yolox_base.py`; YOLOv7 `hyp.scratch.p5.yaml`; Ultralytics
`default.yaml`; the YOLOv12 README training snippet; the YOLO26 training-recipe guide. YOLOv12's
per-size values (mixup 0/0.05/0.15/0.15/0.2, copy_paste 0.1/0.15/0.4/0.5/0.6, scale 0.5/0.9/0.9/0.9/0.9)
are identical to YOLO26's Objects365-stage table. The two schedules share an ancestor in Ultralytics'
internal recipes.

Two patterns stand out:

1. **Strength grows with model size**, most clearly in mixup, copy-paste and scale (Chapter 10's
   capacity argument).
2. **The YOLO26-N recipe is an outlier.** It is the only size with meaningful rotation, shear and BGR
   swap, and with much stronger HSV jitter. These values came from evolutionary search
   (Chapter 32). Small models benefit from different regularisation than large ones.

---

## Dataloader performance

Augmentation runs on the CPU, in `workers` processes per GPU. When GPU utilisation is low and epochs are
slow:

| Symptom | Fix |
|---|---|
| CPU at 100%, GPU idle | more `workers`; `cache=ram`; fewer heavy augmentations; faster JPEG decode |
| Disk-bound (network storage) | `cache=disk` (stores decoded `.npy`) or copy the dataset locally |
| Large images (4K) decoded every epoch | pre-resize the dataset to ~1.5× `imgsz` offline |
| Very large batch on fast GPUs | GPU-side augmentation (Ultralytics has an NVIDIA DALI guide) |

---

## Key Takeaways

- Ultralytics' training pipeline order is: resize → mosaic → (copy-paste) → affine with border crop →
  mixup/cutmix → light Albumentations → HSV → flips → format.
- `close_mosaic` disables mosaic, copy-paste, mixup and cutmix for the last N epochs (10 by default) so
  the model finishes on natural images.
- Validation uses unaugmented rectangular batches. Inference uses minimal-padding letterbox for PyTorch
  and fixed shapes for exported static models. Validate the exported model itself.
- Inputs are RGB ÷ 255 with no mean/std normalisation, a frequent integration bug.
- Augmentation strength scales with model size across v12 and YOLO26 recipes. YOLO26-N's searched
  recipe is the exception (rotation, shear, BGR, strong HSV).
- If GPUs are idle, the dataloader is the bottleneck: workers, caching, pre-resizing or GPU augmentation.

## Check Yourself

<details class="check"><summary>Your exported ONNX model's mAP on the validation set is 0.6 lower than `model.val()` in PyTorch. Name two preprocessing reasons that need no bug.</summary>
model.val() uses rectangular batches with minimal padding, while the exported model may use a fixed
square input with full padding, which shrinks objects relative to the input. The exported model's
letterbox may also differ in rounding or scale-up behaviour. Precision (FP16) and NMS settings are
non-preprocessing causes worth ruling out too.</details>

<details class="check"><summary>Why does `rect=True` disable mosaic?</summary>
Mosaic produces square 2s×2s canvases cropped to s×s, and rect training uses per-batch rectangular
shapes chosen by aspect ratio. The two are geometrically incompatible. Rect batches also need images
grouped by aspect ratio, which conflicts with random shuffling.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics `data/augment.py` (`v8_transforms`, `Mosaic`, `RandomPerspective`, `MixUp`, `CutMix`, `Albumentations`, `RandomHSV`, `RandomFlip`, `LetterBox`) | Ultralytics | github.com/ultralytics/ultralytics | Pipeline order and defaults |
| Ultralytics `data/dataset.py`, `data/base.py` | Ultralytics | github.com/ultralytics/ultralytics | `close_mosaic`, rect batch shapes |
| Ultralytics `cfg/default.yaml` | Ultralytics | github.com/ultralytics/ultralytics | Defaults |
| YOLOv5 `data/hyps/hyp.scratch-low.yaml` | Ultralytics | github.com/ultralytics/yolov5 | v5 column |
| YOLOX `yolox/exp/yolox_base.py` | Megvii | github.com/Megvii-BaseDetection/YOLOX | YOLOX column, no-aug epochs |
| YOLOv7 `data/hyp.scratch.p5.yaml` | WongKinYiu | github.com/WongKinYiu/yolov7 | v7 column |
| YOLOv12 README | Tian et al., 2025 | github.com/sunsmarterjie/yolov12 | v12 per-size values |
| YOLO26 training recipe | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | YOLO26 column |
| Ultralytics NVIDIA DALI guide | Ultralytics | docs/en/guides/nvidia-dali.md | GPU-side preprocessing |

---

**Next:** [Chapter 29 — Axis 3: Architecture Anatomy](./29_yolo_architecture.md) — the tensor enters
the network. Every block, every shape.
