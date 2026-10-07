---
title: "Chapter 27 — YOLO Axis 1: Data"
---

[← Back to Table of Contents](./README.md)

# Chapter 27 — YOLO Axis 1: Data

> *"The cheapest AP you will ever buy is a correct label on a hard example."*

## Overview

This axis covers everything up to the moment pixels enter the augmentation pipeline: the YOLO dataset
layout and `data.yaml`, the label format and its conventions, how Ultralytics caches and validates
labels, class imbalance handling, how much data is enough, which data the official checkpoints were
trained on (COCO from scratch for v5–v12, Objects365 then COCO for YOLO26), and how to build a dataset
for a new task with auto-annotation. General dataset practice (guidelines, noise, splits) is in
Chapter 9. This chapter is the YOLO-specific version.

<div class="diagram">
<div class="diagram-title">The YOLO dataset contract</div>
<div class="flow-h">
  <div class="flow-node">images/train/*.jpg</div>
  <div class="flow-node">labels/train/*.txt<small>same stem, one line per object</small></div>
  <div class="flow-node accent">data.yaml<small>paths + class names</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">*.cache<small>validated labels, image sizes</small></div>
</div>
</div>

---

## Layout and `data.yaml`

```text
datasets/widgets/
├── images/
│   ├── train/  000001.jpg …
│   └── val/    …
└── labels/
    ├── train/  000001.txt …     # same stem as the image; "images" → "labels" in the path
    └── val/    …
```

```yaml
# widgets.yaml
path: datasets/widgets        # dataset root
train: images/train           # relative to path (dirs, list files, or lists of either)
val: images/val
test:                          # optional
names:
  0: widget
  1: gadget
  2: sprocket
```

Ultralytics finds each image's label by replacing `/images/` with `/labels/` in its path and changing
the extension to `.txt`. A missing label file means **background image** (no objects). That is legal and
useful, but a silent failure when it is accidental: a typo in a directory name turns your whole training
set into background images. The training log's line `… N images, M backgrounds, K corrupt` is the first
thing to read.

### The label line

```text
<class> <cx> <cy> <w> <h>          # detection: normalised to [0, 1] by image width / height
<class> <x1> <y1> <x2> <y2> …      # segmentation: normalised polygon
<class> <cx> <cy> <w> <h> <px> <py> [<v>] …   # pose: box + keypoints (+ visibility)
<class> <x1> <y1> … <x4> <y4>      # OBB: four corners, normalised
```

Classes are **zero-based integers** indexing `names`. Coordinates are **normalised centre–size**.
The conversion from COCO JSON (sparse category ids, xywh top-left, crowd flags) is in
`odlab.data.coco_to_yolo` and in Ultralytics' `convert_coco`. `test_yolo_txt_roundtrip` checks that the
format conversions are exact.

---

## Label caching and validation

On first use, Ultralytics scans every image–label pair and writes a `labels.cache`. For each pair it:

- verifies the image opens, reads its size (EXIF-corrected), and flags corrupt JPEGs,
- checks that label values are in range, that boxes have five columns and segments are well formed,
- **removes duplicate rows** (identical labels) with a warning,
- stores the parsed arrays so later epochs skip the parsing.

Images that fail are reported as *corrupt* and excluded. The cache is keyed by a hash of file paths and
sizes, so editing labels in place without changing sizes can leave a stale cache. Deleting `*.cache`
forces a rescan. Large datasets can also cache *images* in RAM or on disk (`cache=ram` or `cache=disk`),
which removes JPEG decoding from the loop (Chapter 31).

---

## How much data?

The Ultralytics training tips give rules of thumb: **≥ 1,500 images per class** and **≥ 10,000
labelled instances per class**, plus **0–10% background images** to reduce false positives. These are
targets for *training from scratch* with COCO-like variety. With a pre-trained checkpoint the
requirement drops sharply. Useful working numbers:

| Regime | Data | What to expect | Recipe adjustments (Chapter 31) |
|---|---|---|---|
| **Proof of concept** | 50–200 images, few classes | Works on narrow, consistent tasks; high variance between runs | Start from COCO or Objects365 weights; reduce mosaic/mixup; `freeze` early layers; multiple seeds |
| **Small production** | 500–2,000 images/class | Solid on in-distribution data | Fine-tune defaults; tune `scale`, HSV for domain |
| **Large** | 10k+ images | From-scratch training becomes competitive with fine-tuning | Match the pre-training recipe more closely |

The **YOLO26 training-recipe guide** gives the official fine-tuning advice by dataset size: for
small datasets (< 1,000 images), reduce augmentation (`mosaic=0.5`, `mixup=0.0`, `copy_paste=0.0`),
switch to AdamW with `lr0=0.001`, use fewer epochs with patience, and consider `freeze=10`. For large
datasets (> 50,000 images), match the pre-training recipe and consider MuSGD with stronger augmentation.

---

## What the official checkpoints were trained on

| Model | Detection data | Schedule | Consequence |
|---|---|---|---|
| **YOLOv5 / v8 / v9 / v10 / 11 / v12** | COCO train2017 only (from scratch) | 300–600 epochs | Pure COCO baseline; fine-tunes well on COCO-like imagery |
| **YOLO26** | **Objects365v1** (150 epochs) → COCO (40–245 epochs by size) | two-stage | Broader prior from 365 classes; weights `yolo26*-objv1-150.pt` also published |
| **YOLO-World / YOLOE / YOLOE-26** | Objects365 + GoldG (+ CC3M pseudo-labels for World) | — | Open-vocabulary priors (Chapter 20) |
| **YOLO-NAS** | Objects365 + pseudo-labelled COCO unlabeled set | — | Weights non-commercial |

The YOLO26 guide is explicit: "No YOLO26 checkpoint was trained on COCO from random weights, which is
why the COCO stage is short for most sizes", and "a from-scratch comparison against those numbers is
not like for like".

<div class="callout field"><span class="callout-title">Field note</span>For a new domain, try
<em>both</em> starting points. The COCO checkpoint (<code>yolo26s.pt</code>) is tuned to COCO's 80
classes. The Objects365 checkpoint (<code>yolo26s-objv1-150.pt</code>) has seen 365 classes and has
not been specialised to COCO. Which fine-tunes better depends on how COCO-like your data is. It costs
one extra run to find out.</div>

---

## Class imbalance

Real datasets are long-tailed: 20,000 "car" boxes and 40 "fire extinguisher" boxes. YOLO's default
loss treats every positive equally, so rare classes get little gradient. The options, from least to
most invasive:

1. **Collect more of the rare class.** Mine it with an open-vocabulary model (Chapter 20).
2. **Oversample images** containing rare classes (repeat-factor sampling, as in LVIS training,
   Chapter 40).
3. **Class weights in the loss.** Ultralytics exposes `cls_pw` ("class weights power for handling class
   imbalance": 0.0 disables it, 1.0 is full inverse-frequency weighting). The resulting per-class weights
   multiply the BCE classification loss (`class_weights` in `v8DetectionLoss`). `cls_pw` is also in the
   tuner's default search space.
4. **Copy-paste** rare objects into other images (needs masks or careful box pasting).
5. **Per-class confidence thresholds** at deployment (Chapter 47). They fix the operating point, not
   the model.

Measure the effect per class (per-class AP and recall at the deployment threshold), not with overall mAP,
which is dominated by frequent classes.

---

## Building a dataset fast: auto-annotation

Ultralytics ships `auto_annotate`, which runs a detector and then SAM to produce segmentation labels.
The open-vocabulary route (YOLOE / YOLO-World text prompts, Grounding DINO, SAM 3) produces box labels
for classes no pre-trained detector knows:

```python
from ultralytics import YOLOE
model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(["pallet", "forklift", "safety cone"])
for r in model.predict("unlabelled/", conf=0.35, stream=True):
    r.save_txt(f"labels/{r.path.stem}.txt")     # YOLO-format pre-labels for human review
```

Treat these as **pre-labels**. Review them in CVAT or Label Studio, fix misses (open-vocabulary
models under-detect small and unusual objects) and remove false positives before training. Chapter 49
measures what auto-labels cost in final accuracy.

---

## Dataset health checklist

| Check | How | Typical bug it catches |
|---|---|---|
| Background count matches expectation | training log "backgrounds" | wrong label directory → everything is background |
| Class histogram per split | count label lines per class | a class missing from val; train/val class-index mismatch |
| Box size histogram | `w × h` in pixels at training resolution | objects below ~8 px at 640 → need higher `imgsz` or P2 (Chapter 41) |
| Aspect-ratio extremes | `max(w/h, h/w)` | slivers > 20:1 that augmentation will discard |
| Duplicate images across splits | perceptual hash | leakage inflating validation AP |
| Visual spot check | `yolo` plotting of `train_batch*.jpg` mosaics after epoch 0 | wrong normalisation, swapped x/y, wrong class names |
| Label after EXIF rotation | view images with labels drawn, using EXIF-aware loading | rotated phone photos with unrotated labels |

---

## Key Takeaways

- A YOLO dataset is images + same-stem `.txt` labels (zero-based class, normalised cxcywh) + a
  `data.yaml`. Missing label files mean background images, intentionally or not.
- Ultralytics validates and caches labels: it drops corrupt files and duplicates. Read the scan summary
  line every time.
- The ≥ 1,500 images / ≥ 10,000 instances per class guidance is for training from scratch. Fine-tuning
  from strong checkpoints works with far less, with reduced augmentation and a lower learning rate.
- YOLO26 checkpoints were pre-trained on Objects365v1 before COCO. Earlier Ultralytics YOLOs trained COCO
  from scratch, which matters for comparisons and for choosing a starting checkpoint.
- Handle imbalance with data first (mining, oversampling, copy-paste), then loss weights, then per-class
  thresholds. Evaluate per class.
- Auto-annotation with open-vocabulary models gives pre-labels, not labels. Budget human review.

## Check Yourself

<details class="check"><summary>Training reports "0 images, 1,200 backgrounds". What happened?</summary>
Ultralytics found the images but no label files. The `labels/` directory is missing or mis-named, the
`images` → `labels` path substitution fails (for example a directory called "imgs"), or the labels have
a different stem or extension. Every image is being treated as background.</details>

<details class="check"><summary>You have 300 images of a niche part. Which starting checkpoint and which three recipe changes would you try first?</summary>
Start from a pre-trained checkpoint (both yolo26s.pt and yolo26s-objv1-150.pt are worth a run). Then
reduce augmentation (mosaic around 0.5, mixup and copy-paste 0), use AdamW with a lower lr0 (~0.001) and
early stopping (patience), and consider freezing early backbone layers (freeze=10). Run several seeds:
variance is high at this size.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics dataset docs (detect format, `data.yaml`) | Ultralytics | docs/en/datasets/detect | Layout, label format |
| Ultralytics `data/utils.py`, `data/dataset.py` | Ultralytics | github.com/ultralytics/ultralytics | images→labels substitution, label verification, cache, duplicate removal |
| Ultralytics "Tips for Best Training Results" | Ultralytics | docs/en/guides/model-training-tips.md | Images/instances per class, background fraction |
| YOLO26 training recipe guide | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Objects365 → COCO; fine-tuning advice by dataset size |
| Ultralytics `cfg/default.yaml`, `engine/tuner.py`, `utils/loss.py` | Ultralytics | github.com/ultralytics/ultralytics | `cls_pw` setting, class weights in the BCE term |
| Ultralytics auto-annotation and YOLOE docs | Ultralytics | docs/en/models/yoloe.md | Pre-labelling |
| `odlab/data.py` + tests | this book | code/odlab | Format conversions |

---

**Next:** [Chapter 28 — Axis 2: Preprocessing & Augmentation](./28_yolo_preprocessing_augmentation.md) —
the pipeline every image passes through before the network sees it.
