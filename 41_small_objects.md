---
title: "Chapter 41 — Small Objects"
---

[← Back to Table of Contents](./README.md)

# Chapter 41 — Small Objects

> *"A 12-pixel object has 144 pixels of evidence, one or two grid points of supervision, and an IoU that halves when the box slips by three pixels."*

## Overview

Small objects are the most common reason a detector that looks good on COCO fails in the field: drones,
traffic cameras, satellite tiles, inspection images, distant people. This chapter explains the failure
as five separate mechanisms (pixels, stride, assignment, IoU sensitivity and preprocessing), measures
each where possible, and maps every mechanism to its fixes: higher resolution, a stride-4 head, tiling
with SAHI, small-object-aware assignment and losses, and dataset practice. It ends with a decision table
keyed to object size in pixels at the deployment resolution.

<div class="diagram">
<div class="diagram-title">Five reasons small objects fail</div>
<div class="layer-stack">
  <div class="layer red">1. Preprocessing: a 4K frame letterboxed to 640 shrinks every object 6×</div>
  <div class="layer orange">2. Pixels: little appearance evidence; texture and context matter more</div>
  <div class="layer yellow">3. Stride: the finest map is stride 8; an 8-px object is one cell</div>
  <div class="layer green">4. Assignment: no grid point inside the box → no positive → never learned</div>
  <div class="layer blue">5. IoU sensitivity: a 2-px error drops IoU from 1.0 to 0.6 on an 8-px box</div>
</div>
</div>

---

## How small is small?

| Definition | Small | Notes |
|---|---|---|
| COCO | area < 32² px | Measured in the **original** image, not at the network input |
| AI-TOD | very tiny 2–8 px, tiny 8–16, small 16–32 | Aerial; mean object size about 12.8 px |
| TinyPerson | people under 20 px | Long-range surveillance |
| Practical (this book) | **under ~16 px at the network input** | What the detector actually sees after resizing |

The last definition is the one that predicts failure. A 40-pixel person in a 3840 × 2160 frame becomes
**6.7 px** after letterboxing to 640. COCO would call it small, and the network sees it as tiny.

---

## Mechanism 1: preprocessing shrinks objects

Detectors take a fixed input size. The letterbox scale is `640 / max(H, W)`:

| Camera | Scale to 640 | A 30-px object becomes |
|---|---|---|
| 1280 × 720 | 0.50 | 15 px |
| 1920 × 1080 | 0.33 | 10 px |
| 3840 × 2160 | 0.17 | 5 px |
| 8000 × 6000 (aerial) | 0.08 | 2.4 px |

**First diagnostic:** histogram your object sizes **after** resizing to the training/deployment `imgsz`
(Chapter 27). If a large share is under 8–10 px, no architecture change fixes it at that resolution.

---

## Mechanisms 2–4: stride and assignment

The finest standard feature map is stride 8. At 640 input, an 8 × 8 object occupies one cell, and the
receptive field of a P3 feature spans far more than the object. Worse, TAL-style assigners only use grid
points **inside** the box. Chapter 30 computed candidate counts at 640:

| Object | Grid points inside (P3, stride 8) | Positives |
|---|---|---|
| 6 × 6 | 0–1 | 0–1: may never be learned |
| 12 × 10 | 1–4 | 1–4 |
| 20 × 20 | 4–9 | up to 10 |

Try it in the grid lab: drag a small box across the grid and watch the number of points that can "see"
it change with sub-pixel position.

<div class="lab" data-lab="grid"></div>

**YOLO26's STAL** enlarges boxes under 16 px to 16 px for candidate selection only, guaranteeing a few
positives (Chapter 30). In other codebases, centre sampling with a minimum radius (FCOS, YOLOX's
centre region) has a similar effect.

---

## Mechanism 5: IoU is harsh on small boxes

A horizontal shift of d pixels on an s × s box gives IoU = (s − d) / (s + d):

| Box size | 1 px off | 2 px off | 4 px off |
|---|---|---|---|
| 4 px | 0.60 | 0.33 | 0.00 |
| 8 px | 0.78 | 0.60 | 0.33 |
| 16 px | 0.88 | 0.78 | 0.60 |
| 32 px | 0.94 | 0.88 | 0.78 |
| 64 px | 0.97 | 0.94 | 0.88 |

A 2-pixel error, which is within annotation noise, makes an 8-px detection fail AP75 and barely pass
AP50. This affects three things:

1. **Evaluation**: small-object AP is mechanically low at strict thresholds. Some aerial benchmarks
   report AP50 for this reason.
2. **Assignment**: IoU-based metrics rank small-object candidates noisily.
3. **Loss**: IoU losses give jumpy gradients at small scales.

**NWD (normalised Wasserstein distance)** models boxes as 2-D Gaussians and compares them with a
Wasserstein distance, which degrades smoothly with offset regardless of size. It can replace IoU in
assignment, NMS and the loss for tiny objects (Wang et al., 2021, on AI-TOD). Chapter 3 gives the
formula.

---

## Fixes, from cheapest

### 1. Train and infer at a higher resolution

The most reliable fix. Cost grows with the square of the side:

| Model | 640 | 1280 |
|---|---|---|
| YOLO26n (GFLOPs, fused) | 5.5 | 23.4 |
| YOLO26s | 20.9 | 86.7 |

Train at the resolution you deploy at. A model trained at 640 and run at 1280 sees objects at sizes it
never trained on. Rectangular `imgsz` (for example `[736, 1280]` for 16:9) avoids wasting compute on
padding.

### 2. Add a stride-4 (P2) level

`yolo26-p2.yaml` adds an upsampling path to stride 4 and a fourth detection input (Chapter 29). Measured
on the fused models:

| Model | Params | GFLOPs at 640 | Grid points at 640 |
|---|---|---|---|
| YOLO26n | 2.41M | 5.5 | 8,400 |
| YOLO26n-P2 | 2.47M | **7.5** (+36%) | **34,000** |
| YOLO26s | 9.50M | 20.9 | 8,400 |
| YOLO26s-P2 | 9.46M | **25.0** (+20%) | 34,000 |

Parameters barely change. FLOPs, memory and post-processing grow, since four times as many candidates
reach the top-k or NMS. P2 helps when objects are 4–12 px at the input. Above about 16 px, P3 already
works.

### 3. Tile: SAHI and sliced training

**SAHI** (Slicing Aided Hyper Inference) runs the detector on overlapping crops of the full-resolution
image, maps every box back to image coordinates, optionally adds a full-image pass for large objects,
and merges duplicates across tiles (greedy non-maximum merging with intersection-over-smaller matching
by default). The SAHI paper reports, on VisDrone and xView, **+6.8, +5.1 and +5.3 AP** for FCOS, VFNet
and TOOD from sliced inference alone, and **+12.7, +13.4 and +14.5 AP** cumulatively with sliced
fine-tuning.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

det = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path="best.pt", confidence_threshold=0.3)
res = get_sliced_prediction("frame_4k.jpg", det, slice_height=640, slice_width=640,
                            overlap_height_ratio=0.2, overlap_width_ratio=0.2)
```

| Choice | Guidance |
|---|---|
| Tile size | The training resolution (so objects appear at the size seen in training) |
| Overlap | 0.2 is the common default; at least the size of the largest object you need to keep whole |
| Full-image pass | Keep it if large objects matter (they get cut by tiles) |
| Merge | Greedy merging with IoS handles objects split across tiles better than plain NMS |
| Cost | Tiles × per-tile latency: a 3840 × 2160 frame in 640 tiles with 0.2 overlap is 8 × 4 = 32 tiles, roughly 32× the compute of one 640 pass |
| Training | Train on tiles too (sliced fine-tuning), or the model sees full-frame statistics it will not see at inference |

### 4. Assignment and loss changes

STAL or a minimum centre radius for tiny boxes, NWD in assignment and loss, and a higher `box` gain
when localisation matters. These are cheap at inference (zero cost) but need training runs to validate.

### 5. Data practice for tiny objects

- **Label consistently**: at 6 px, annotators disagree by 1–2 px, which is already 0.2 IoU.
  Write a labelling rule (tight to visible pixels) and audit a sample.
- **Avoid augmentation that shrinks objects further**: limit downscaling (`scale`), use mosaic
  carefully (it halves objects in each quadrant), keep copy-paste of small instances.
- **Ignore regions** for objects too small to label reliably, instead of leaving them as background.

### 6. Architecture changes worth knowing

Feature-fusion necks that bring high-level semantics down to fine maps (BiFPN, Gold-YOLO's
gather-and-distribute), dilated or large-kernel convs for context, and super-resolution front-ends.
The last is rarely worth its cost compared with simply running at a higher resolution.

---

## Decision table

| Objects at the deployment input are… | Do this |
|---|---|
| Mostly > 32 px | Nothing special. Check `scale` augmentation covers your sizes |
| 16–32 px | Higher `imgsz` if latency allows; YOLO26 (STAL) or STAL-like assignment |
| 8–16 px | Higher `imgsz` **or** P2 head; STAL; consider NWD; label audit |
| < 8 px, high-resolution source | Tiling (SAHI) at training resolution with sliced fine-tuning; P2 head on tiles |
| < 8 px, low-resolution source | Change the optics or camera position if you can. The information may not be there |

Benchmarks for this regime: **VisDrone** (drone video, 10 classes), **AI-TOD** (aerial, tiny),
**DOTA** (aerial, OBB, Chapter 38), **xView** (satellite, 60 classes), **TinyPerson**, **SODA-D**
(driving).

---

## Key Takeaways

- Measure object size in pixels **at the network input**. Letterboxing a 4K frame to 640 shrinks
  objects about 6×.
- Small objects fail through preprocessing, too few pixels, stride, assignment (no grid point inside
  the box) and IoU sensitivity (a 2-px error halves IoU at 4 px and drops it to 0.6 at 8 px).
- Higher resolution is the most reliable fix. A P2 head costs +20–36% FLOPs at 640 and four times as
  many candidates. SAHI tiling scales to arbitrary image sizes at the cost of one pass per tile.
- STAL-like assignment and NWD are free at inference and target the assignment and IoU mechanisms.
- Label tiny objects consistently, limit augmentations that shrink them further, and use ignore regions
  for the unlabellable.

## Check Yourself

<details class="check"><summary>Your 1920 × 1080 camera sees people at 30–60 px. Your YOLO26n at imgsz 640 misses many. What is the cheapest first experiment?</summary>
Compute the size after letterboxing: 640 / 1920 = 0.33, so people are 10–20 px at the input, in the
small-to-tiny regime. Train and run at a higher, rectangular input (for example [736, 1280]) and compare
recall on small people against the latency budget. A P2 head is the alternative if the latency budget
cannot absorb the larger input.</details>

<details class="check"><summary>Why does SAHI merge detections with intersection-over-smaller rather than IoU?</summary>
An object cut by a tile border is detected as a partial box in one tile and a fuller box in another.
Their IoU can be low because one box is much smaller than the other, but the smaller box lies almost
entirely inside the larger one. IoS measures that containment, so the partial duplicates are merged.</details>

<details class="check"><summary>A 6 × 6 object at 640 gets zero positives in YOLO11. Name two fixes that do not change the input size.</summary>
STAL-style candidate enlargement (YOLO26 does this for boxes under 16 px), or a P2 head whose stride-4
grid puts points inside the box. NWD-based assignment also helps rank the few candidates more stably.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| SAHI | Akyon, Altinuc, Temizel, 2022 | arXiv:2202.06934; github.com/obss/sahi | Sliced inference and fine-tuning, reported gains |
| NWD for tiny object detection | Wang et al., 2021 | arXiv:2110.13389 | Gaussian Wasserstein box similarity |
| AI-TOD | Wang et al., 2021 | ICPR 2020 / github.com/jwwangchn/AI-TOD | Tiny-object size bins |
| VisDrone | Zhu et al., 2021 | arXiv:2001.06303 | Drone benchmark |
| xView | Lam et al., 2018 | arXiv:1802.07856 | Satellite benchmark |
| TinyPerson | Yu et al., 2020 | arXiv:1912.10664 | Tiny-person benchmark |
| DOTA | Xia et al., 2018 | arXiv:1711.10398 | Aerial OBB benchmark |
| Ultralytics `cfg/models/26/yolo26-p2.yaml`, `utils/tal.py` (STAL) | Ultralytics | github.com/ultralytics/ultralytics | P2 layout; measured cost; STAL |
| COCO detection evaluation | Lin et al., 2014 | cocodataset.org | Area ranges |

---

**Next:** [Chapter 42 — Label-Efficient Learning & Distillation](./42_label_efficient_and_distillation.md)
— getting more accuracy per labelled image.
