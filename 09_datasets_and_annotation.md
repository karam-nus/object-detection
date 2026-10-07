---
title: "Chapter 9 — Datasets, Formats & Annotation"
---

[← Back to Table of Contents](./README.md)

# Chapter 9 — Datasets, Formats & Annotation

> *"Your detector's ceiling is set by your annotators' worst afternoon."*

## Overview

Every number in this book was measured against human-drawn boxes. This chapter covers the
benchmark datasets and what each one is for, the label formats and the conversions between them,
how to write annotation guidelines that produce consistent boxes, how label noise shows up in metrics,
and how foundation models changed the economics of labelling. For the YOLO-specific dataset
pipeline (`data.yaml`, label caching, class imbalance in Ultralytics), see Chapter 27. For every
dataset in one table, see [Appendix D](./appendix_d_datasets.md).

<div class="diagram">
<div class="diagram-title">The data lifecycle</div>
<div class="cycle">
  <div class="cycle-step accent">collect</div><div class="cycle-arrow"></div>
  <div class="cycle-step green">guidelines</div><div class="cycle-arrow"></div>
  <div class="cycle-step purple">label (human + model)</div><div class="cycle-arrow"></div>
  <div class="cycle-step orange">QA & split</div><div class="cycle-arrow"></div>
  <div class="cycle-step accent">train & evaluate</div><div class="cycle-arrow"></div>
  <div class="cycle-step green">mine failures</div>
</div>
</div>

---

## The benchmark datasets

| Dataset | Images | Classes | Boxes | Why it matters | Choose it when |
|---|---|:---:|---|---|---|
| **PASCAL VOC 2007/2012** | ~10k / ~11.5k trainval | 20 | ~25k / ~27k | Historical baseline; AP$_{50}$ | Reproducing pre-2016 work; quick sanity checks |
| **COCO 2017** | 118k train / 5k val / 41k test | 80 | ~860k train | *The* benchmark; AP@[.5:.95]; masks; crowd regions | Comparing with the literature |
| **Objects365** (v1 / v2) | ~600k / ~2M | 365 | ~10M / ~30M | The pre-training set behind YOLO26, LW-DETR, RF-DETR, D-FINE-O365, DINO | Pre-training a detector backbone + head |
| **Open Images V7** | ~1.9M with boxes | 600 boxable | ~16M | Scale, hierarchy, group-of boxes | Broad-coverage pre-training; hierarchical labels |
| **LVIS v1** | ~164k (COCO images) | 1,203 | ~1.3M (masks) | Long tail; federated evaluation; AP$_r$ | Open-vocabulary and long-tail research |
| **CrowdHuman** | 15k train | 1 (+ visible/full/head) | ~470k humans, ~22.6/image | Crowds and occlusion | Pedestrian detection, NMS research |
| **WIDER FACE** | 32k | 1 | ~394k faces | Extreme scale range | Face detection |
| **DOTA v1.0 / v2.0** | 2.8k / 11k aerial | 15 / 18 | 188k / 1.8M oriented | Oriented boxes, tiny objects, huge images | OBB and aerial (Ch 43) |
| **VisDrone-DET** | 10k | 10 | dense, tiny | Drone imagery | Small objects (Ch 41) |
| **SKU-110K** | 11.7k | 1 | ~1.7M, ~147/image | Extreme density | Retail shelves; query-count limits |
| **KITTI / BDD100K / nuImages** | 7.5k / 100k / 93k | 3–10 | — | Driving | Automotive |
| **xView** | ~1k satellite scenes | 60 | > 1M | Satellite, 0.3 m GSD | Remote sensing |
| **RF100 / RF100-VL** | 100 datasets | varied | varied | Transfer across domains; fine-tuning benchmark | Testing whether a model *fine-tunes* well, not just whether it scores well on COCO |
| **ODinW-13 / 35** | 13 / 35 datasets | varied | varied | Open-vocabulary transfer | Zero/few-shot evaluation (Ch 20) |

Two trends matter more than any single row:

1. **Pre-training moved from ImageNet to detection-scale data.** Every YOLO26 checkpoint was
   fine-tuned on COCO from an Objects365v1 checkpoint trained for 150 epochs. D-FINE-X goes from 55.8
   to 59.3 AP with Objects365. Comparing a COCO-from-scratch model with an Objects365-pretrained one is
   comparing two experiments (Chapter 34).
2. **Benchmarks moved from "how good on COCO" to "how good after fine-tuning on *your* data".**
   RF100-VL (Roboflow, 2025) evaluates fine-tuning across 100 diverse datasets. On it the ranking of
   real-time detectors differs from COCO. RF-DETR's README lists YOLO26-N at 52.0 AP$_{50:95}$ and
   D-FINE-N at 58.2 on RF100-VL, even though they are 40.3 and 42.7 on COCO under the same harness.

---

## Label formats

### COCO JSON

```json
{
  "images":      [{"id": 1, "file_name": "000001.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 7, "image_id": 1, "category_id": 18,
                   "bbox": [x, y, w, h], "area": 4123.5, "iscrowd": 0,
                   "segmentation": [[...polygon...]]}],
  "categories":  [{"id": 18, "name": "dog", "supercategory": "animal"}]
}
```

`bbox` is **xywh with a top-left corner** in pixels. `area` drives the size buckets (mask area on COCO).
`iscrowd=1` marks ignore regions. `category_id`s are arbitrary integers: the 80 COCO classes use ids
1–90 with gaps. Detection results are a flat list of
`{"image_id", "category_id", "bbox", "score"}`.

### YOLO txt

One `.txt` per image in a parallel `labels/` directory, one object per line:

```text
# class  cx      cy      w       h        (normalised to [0, 1])
0        0.4812  0.6230  0.1250  0.3104
16       0.2101  0.5512  0.3400  0.2208
```

plus a `data.yaml` naming the splits and classes (Chapter 27). Classes are **zero-based and dense**.
There is no `iscrowd`, no ignore regions, no per-image metadata. An image with no objects has an empty
(or missing) label file and is a **background image**. Polygons (segmentation), keypoints and oriented
boxes extend the same line format.

### Others you will meet

| Format | Box encoding | Where |
|---|---|---|
| **Pascal VOC XML** | `xmin ymin xmax ymax` (1-based, inclusive in the original devkit) | legacy datasets, LabelImg |
| **CreateML JSON** | centre x, y, width, height in pixels | Apple tooling |
| **TFRecord** (TF Object Detection API) | normalised ymin, xmin, ymax, xmax | TensorFlow pipelines |
| **LabelMe JSON** | polygons / rectangles as point lists | LabelMe, X-AnyLabeling |
| **Datumaro / FiftyOne** | internal; convert between all of the above | dataset tooling |

Conversions are mechanical, but each one has a trap: 1-based VOC coordinates, normalised vs pixel
units, ymin-first TFRecords, and remapping sparse COCO ids to dense YOLO indices.
`odlab.data.coco_to_yolo` shows the full conversion, including the id remap and dropping crowd boxes,
in 25 lines. Ultralytics ships `ultralytics.data.converter.convert_coco` for production use.

---

## Writing annotation guidelines

Models learn the *annotators'* definition of an object, including its inconsistencies. A guideline
document is the cheapest accuracy improvement in this book. It must answer:

| Question | Example decision |
|---|---|
| **What is in the class?** | Is a toy car a `car`? Is a person's reflection a `person`? |
| **Occlusion** | Box the *visible* extent (COCO, modal) or the *full* extent (amodal)? |
| **Truncation** | Label objects cut by the image border? Minimum visible fraction? |
| **Minimum size** | Ignore objects under 8 px? Mark them `ignore` rather than leaving them unlabelled? |
| **Crowds** | When does a group become one crowd box? |
| **Tightness** | Box to the outermost pixel; include or exclude thin protrusions (antennas, tails)? |
| **Ambiguity** | What do annotators do when unsure: skip, flag, or best guess? |

Measure **inter-annotator agreement** by having two people label the same 100 images and computing the
IoU of matched boxes and the F1 of their matching. If two careful humans agree at IoU 0.85, a model
cannot be meaningfully evaluated at AP$_{90}$ on that data.

<div class="callout field"><span class="callout-title">Field note</span>Label 200 images,
train a quick model, and look at its false positives with the guideline open. Most "model errors" in
the first round are guideline gaps: two annotators made different choices, or a case was never
defined. Fix the document, re-label, and only then scale up.</div>

---

## Label noise and what it does to metrics

| Noise type | Effect on training | Effect on evaluation |
|---|---|---|
| **Missing labels** (object present, no box) | The object's candidates are trained as background, so recall drops for that appearance | Correct detections count as FPs; precision and AP drop; TIDE shows "Bkg" errors |
| **Loose or shifted boxes** | Regression learns the noise; AP$_{75+}$ drops | High-IoU thresholds become unreachable |
| **Wrong class** | Class confusion, especially between similar classes | Cls errors in TIDE |
| **Inconsistent crowd / ignore handling** | Contradictory gradients in dense regions | Unpredictable |

Dense detectors are particularly sensitive to **missing labels**. An unlabelled object is an explicit
negative for every candidate that sees it. Dataset audits typically start by sorting each image's
highest-confidence false positives. A large share of them turn out to be unlabelled true objects.

---

## Splits, leakage and duplicates

- **Split by source, not by frame.** Consecutive video frames are near-duplicates. A random split puts
  frame 101 in train and frame 102 in validation and inflates validation AP. Split by video,
  camera, site or day.
- **Deduplicate** with perceptual hashes or embedding similarity before splitting.
- **Stratify** so that rare classes appear in validation at all.
- **Freeze a test set** that nobody tunes on, and refresh it from the deployment stream.
- **Background images.** The Ultralytics tips recommend about 0–10% images with no objects, to teach
  what *not* to detect. They also recommend ≥ 1,500 images and ≥ 10,000 instances per class for best
  results. These are rules of thumb, not requirements, and fine-tuning from strong pre-training needs
  far less (Chapter 40).

---

## Labelling tools

| Tool | Type | Notable features |
|---|---|---|
| **CVAT** | open source, self-hostable | boxes/polygons/tracks, interpolation in video, model-assisted labelling |
| **Label Studio** | open source | flexible templates, ML backend for pre-annotation |
| **X-AnyLabeling** | open source desktop | many built-in models (YOLO, SAM, Grounding DINO, DEIMv2) for one-click pre-labels |
| **Roboflow Annotate** | hosted | auto-label with foundation models, dataset versioning, format export |
| **FiftyOne** | open source | not a labeller: dataset visualisation, error mining, duplicate finding |

---

## Foundation models changed the cost curve

Pre-labelling with a model and correcting by hand is now standard. The options, from cheapest to most
flexible:

1. **Your own previous model**: the classic model-in-the-loop.
2. **Open-vocabulary detectors** (YOLOE, YOLO-World, Grounding DINO, DINO-X; Chapter 20): text-prompt
   "forklift", get boxes, correct them.
3. **Promptable segmenters** (SAM 2/3): click once, get a mask, derive a tight box. SAM 3 takes a
   noun phrase and returns all instances.
4. **MLLMs** (Qwen3-VL, Rex-Omni; Chapter 21): handle descriptive prompts ("damaged boxes on the
   bottom shelf") that closed vocabularies cannot express.

The resulting labels are then used to train a small, fast detector, a pattern called **auto-labelling
→ distillation**. Chapter 49 runs it end to end and measures how much accuracy is lost relative to
human labels.

**Active learning** decides *which* images to send for labelling: the model's most uncertain images,
images where two models disagree, or images from deployment where the system was later found to have
failed. In practice, mining recent deployment failures beats uncertainty sampling, because it
targets the distribution you actually care about.

**Synthetic data** (rendered scenes, copy-paste composites, diffusion-generated images) helps with rare
classes and rare conditions. It needs some real data to bridge the domain gap, and its validation
must be on real images.

---

## Licenses

COCO annotations are CC BY 4.0, while its images carry their original Flickr licenses. Open Images
annotations are CC BY 4.0. Objects365 is released for research. LVIS annotations are CC BY 4.0.
**Pre-training on a research-only dataset and shipping the weights commercially is a legal question,
not a technical one.** It is one of several licensing layers, together with code and weight licenses,
that Chapter 35 untangles.

---

## Key Takeaways

- COCO is the comparison benchmark. Objects365 is the de facto detection pre-training set. LVIS and
  ODinW test the long tail and transfer. RF100-VL tests fine-tuning across domains, where rankings
  differ from COCO.
- COCO JSON is xywh top-left with sparse category ids and crowd flags. YOLO txt is normalised cxcywh
  with dense zero-based classes and no ignore mechanism.
- A written annotation guideline (occlusion, truncation, minimum size, crowds, tightness) is the
  cheapest accuracy gain available. Measure inter-annotator IoU before trusting AP$_{75+}$.
- Missing labels are the most damaging noise for dense detectors. They train real objects as
  background and turn correct detections into false positives.
- Split by source (video, site, day), deduplicate, and freeze a test set.
- Foundation models make pre-labelling cheap. The emerging workflow is auto-label → human correction
  → distil into a small model.

## Check Yourself

<details class="check"><summary>You split a traffic-camera dataset randomly by frame and get 0.82 AP on validation but 0.61 in deployment. What is the most likely cause?</summary>
Leakage through near-duplicate frames. Validation frames are almost identical to training frames
from the same cameras and times, so validation measures memorisation. Re-split by camera or by day
and expect validation AP much closer to deployment.</details>

<details class="check"><summary>Why can a YOLO-format dataset not represent COCO's crowd annotations, and what is the consequence?</summary>
YOLO txt has no field for ignore or crowd regions. Converters either drop crowd boxes, so detections
inside crowds become false positives in training, or convert them to ordinary boxes, so a whole crowd
becomes one target. Training on COCO in YOLO format therefore handles crowds differently from
pycocotools evaluation, which ignores detections in crowd regions.</details>

<details class="check"><summary>Your model's top false positives are mostly real, unlabelled objects. What do you do?</summary>
Fix the labels, not the model. The model is being penalised, and worse, trained, to suppress real
objects. Run a label audit on high-confidence false positives across the training set, add the missing
boxes, and re-train. Evaluation AP will rise from the corrected validation labels alone.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| The PASCAL VOC Challenge | Everingham et al., 2010 | IJCV 2010 | VOC statistics, XML format |
| Microsoft COCO | Lin et al., 2014 | arXiv:1405.0312 | JSON format, crowd, statistics |
| Objects365 | Shao et al., 2019 | ICCV 2019 | Scale of detection pre-training |
| Open Images V4 | Kuznetsova et al., 2020 | arXiv:1811.00982 | Boxes, hierarchy |
| LVIS | Gupta et al., 2019 | arXiv:1908.03195 | Long tail, federated labels |
| CrowdHuman | Shao et al., 2018 | arXiv:1805.00123 | Crowd statistics |
| WIDER FACE | Yang et al., 2016 | arXiv:1511.06523 | Face statistics |
| DOTA | Xia et al., 2018; Ding et al., 2021 | arXiv:1711.10398; 2102.12219 | Aerial OBB statistics |
| SKU-110K | Goldman et al., 2019 | arXiv:1904.00853 | Dense retail |
| RF100-VL | Robicheaux et al. (Roboflow), 2025 | arXiv:2505.20612 | Cross-domain fine-tuning benchmark |
| RF-DETR README benchmark table | Roboflow, 2026 | github.com/roboflow/rf-detr | COCO vs RF100-VL rankings |
| YOLO26 training recipe | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Objects365v1 → COCO for all sizes |
| D-FINE README | Peng et al., 2024 | github.com/Peterande/D-FINE | 55.8 → 59.3 AP with Objects365 |
| Ultralytics "Tips for Best Training Results" | Ultralytics | docs.ultralytics.com | Images / instances per class, background fraction |
| `odlab/data.py` | this book | code/odlab | Conversions |

---

**Next:** [Chapter 10 — Augmentation](./10_augmentation.md) — given labelled data, augmentation
decides how much the model can learn from each image.
