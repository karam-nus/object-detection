---
title: "Chapter 42 — Label-Efficient Learning & Distillation"
---

[← Back to Table of Contents](./README.md)

# Chapter 42 — Label-Efficient Learning & Distillation

> *"Box labels are the most expensive input to a detector. Every method in this chapter buys accuracy with something cheaper: unlabelled images, a bigger model's opinion, or a human's attention spent only where it matters."*

## Overview

A COCO-quality box takes a trained annotator several seconds; a reviewed, consistent dataset of 10,000
images is weeks of work. This chapter covers the ways to get more accuracy per labelled image:
**semi-supervised** training on unlabelled images, **auto-labelling** with foundation and
open-vocabulary detectors, **active learning** to choose what to label, **knowledge distillation** from
a large detector into a small one, distillation from **vision foundation models**, self-supervised
pre-training on your own images, and synthetic data. For each it gives the mechanism, the evidence and
the trap.

<div class="diagram">
<div class="diagram-title">Where accuracy can come from besides new labels</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card blue"><div class="card-title">Unlabelled images</div><div class="card-desc">Semi-supervised teacher–student · self-supervised pre-training · active selection</div></div>
  <div class="diagram-card accent"><div class="card-title">Bigger models</div><div class="card-desc">Auto-labels from open-vocabulary detectors and VLMs · output and feature distillation · foundation-model features</div></div>
  <div class="diagram-card green"><div class="card-title">Generated data</div><div class="card-desc">Copy-paste · rendering · diffusion-generated images (with care)</div></div>
</div>
</div>

---

## Semi-supervised detection: teacher–student

The dominant recipe since 2021:

1. Train on the labelled set.
2. Copy the model into a **teacher** updated as an EMA of the student.
3. For each unlabelled image, the teacher predicts on a **weakly augmented** view. Predictions above a
   confidence threshold become **pseudo-labels**.
4. The **student** trains on a **strongly augmented** view of the same image against those
   pseudo-labels, plus the labelled loss.

| Method | Year | Key idea | Reported result (COCO, partial labels) |
|---|---|---|---|
| STAC | 2020 | Offline pseudo-labels + strong augmentation | — |
| **Unbiased Teacher** | 2021 | EMA teacher; focal loss to counter class imbalance in pseudo-labels | 1% COCO: **20.75 AP vs 9.05** supervised baseline |
| Soft Teacher | 2021 | Weight pseudo-negatives by teacher background score; box jittering for regression reliability | — |
| Dense Teacher | 2022 | Dense pseudo-labels (score maps) instead of thresholded boxes | — |
| **Consistent-Teacher** | 2023 | Adaptive assignment robust to noisy pseudo-boxes; GMM-based dynamic threshold | 10% COCO: **40.0 AP** with ResNet-50 |

The traps:

- **Confirmation bias**: the student learns the teacher's mistakes. Thresholds that are too low feed
  noise; too high and rare or small objects never get pseudo-labels. Dynamic thresholds help.
- **Class imbalance in pseudo-labels**: frequent classes dominate. Per-class thresholds or
  imbalance-aware losses help.
- **Box quality**: classification confidence does not certify box accuracy. Methods filter or weight
  the regression target separately.
- **Gains shrink as labels grow.** The largest improvements are at 1–10% labelled data. With a fully
  labelled, large dataset, semi-supervision adds less.

---

## Auto-labelling with foundation models

Open-vocabulary detectors (Grounding DINO, YOLOE, SAM 3, DINO-X) and VLMs (Chapter 21) can label
classes no closed-set model knows. The workflow:

```text
unlabelled images → open-vocabulary detector (text prompts) → pre-labels
→ human review (fix misses, delete false positives, tighten boxes) → train a small closed-set detector
```

How to make it work:

| Practice | Why |
|---|---|
| **Measure the auto-labeller first** on 100–200 hand-labelled images: per-class precision and recall at the chosen threshold | You need to know what review must fix |
| **Tune the prompts** ("forklift" vs "industrial forklift truck"), one threshold per class | Open-vocabulary scores are not calibrated across classes |
| **Review, do not trust**: misses on small and unusual objects are the typical failure | Missed objects become background in training |
| **Use visual prompts** (YOLOE SAVPE, T-Rex) when text fails | Domain-specific parts often have no good name |
| **Combine with SAM** for masks if you need segmentation | Box prompt → mask |

Chapter 49 measures raw auto-label quality against human labels (YOLOE-26s finds about half the
COCO128 objects at its default threshold) and gives the A/B/C protocol for measuring what reviewed and
unreviewed auto-labels cost a nano model.

---

## Active learning: choose what to label

When the labelling budget is fixed, choosing images matters. Strategies:

- **Uncertainty**: images where the detector's scores are near the threshold, or where test-time
  augmented predictions disagree.
- **Diversity**: cover the embedding space (core-set selection), so the labelled set represents the
  deployment distribution.
- **Error-driven**: images from deployment where the model failed (user reports, disagreements with a
  bigger model). In production this is the most valuable source (Chapter 47).
- **Rare-class mining**: run an open-vocabulary detector for the rare classes over the unlabelled pool.

A simple, strong baseline is half diversity, half uncertainty, plus every rare-class candidate.

---

## Knowledge distillation for detectors

Distillation trains a small **student** to match a large **teacher**. Detection makes it harder than
classification: most locations are background, outputs are sets of boxes, and teacher and student may
use different heads and assignments.

| Type | What is matched | Examples | Notes |
|---|---|---|---|
| **Output (logit)** | Class score distributions at matched locations | Classic KD adapted to dense heads | Needs location correspondence between teacher and student |
| **Localisation distillation (LD)** | Box-edge distributions (DFL bins) | LD (Zheng et al., 2022) | Natural for DFL heads; not available for YOLO26's L1 head |
| **Feature imitation** | Neck features, weighted by foreground or attention | FGD, CWD, MGD | Works across different heads; needs a projector when widths differ |
| **Relation** | Pairwise relations between features or instances | — | Less common in practice |
| **Pseudo-label distillation** | Teacher's boxes on unlabelled data | Semi-supervised teacher above | Works across any architectures |

**In Ultralytics**: `train(distill_model="teacher.pt")` hooks the neck outputs feeding both heads,
projects student features to the teacher's width with a small MLP, and adds a **score-weighted L2
feature loss** with weight `dis = 6.0` (Chapter 31). It requires the same number of detection levels in
teacher and student.

When distillation pays off:

- Teacher and student see the **same data**, and the teacher is clearly better on **your** data (a
  COCO-strong teacher that is not fine-tuned is a weak teacher).
- The student is capacity-limited (n/s), not data-limited.
- You validate the student against the same student trained without distillation, with seeds.

DAMO-YOLO and Gold-YOLO distil (or self-distil) during training. The reported gains for small models are
usually around 1–2 AP. Compare that with what a longer schedule or Objects365 pre-training gives before
adding the complexity.

---

## Distillation from vision foundation models

The 2025–2026 real-time DETRs moved the teacher from "a bigger detector" to "a self-supervised
foundation model":

| Approach | Example | What is transferred |
|---|---|---|
| Foundation backbone, fine-tuned | RF-DETR (DINOv2), DEIMv2 S+ (DINOv3) | The backbone itself, at real-time size |
| Feature distillation from a VFM | RT-DETRv4 | VFM features into the detector's backbone during training; no inference cost |
| Distilled foundation backbone | DEIMv2 small sizes | A small backbone distilled from DINOv3 |

These are the main reason the DETR curves sit above the YOLO curves in Chapter 34, and the main reason
they transfer well to small, unusual datasets.

---

## Self-supervised pre-training on your own images

If you have hundreds of thousands of unlabelled in-domain images (medical, industrial, satellite),
self-supervised pre-training (DINO-style self-distillation, MAE-style masked reconstruction) on them
before fine-tuning can beat ImageNet- or COCO-initialised backbones, especially when the domain is far
from natural photographs. It is expensive (GPU-days to weeks) and pays off mainly at large unlabelled
scale. A middle path: start from a public foundation backbone (DINOv2/v3) and continue
self-supervised training briefly on in-domain images.

---

## Synthetic data

| Source | Strength | Trap |
|---|---|---|
| **Copy-paste** of real instances | Cheap, realistic objects, more rare-class instances | Context and lighting mismatch; needs masks or careful crops |
| **Rendering** (game engines, CAD) | Perfect labels, rare scenarios, controllable | Domain gap; textures and sensor noise differ |
| **Diffusion-generated images** | Diversity on demand | Label alignment (box ↔ generated object) is imperfect; distribution artefacts |

Synthetic data works best **mixed with real data** and validated on real data only. Domain
randomisation (varying textures, lighting, backgrounds widely) narrows the gap more reliably than
photorealism.

---

## Comparing methods fairly

Always compare at **equal labelling cost**: "1,000 labelled + 20,000 unlabelled with Consistent-Teacher"
against "1,500 labelled" (the same money spent on more labels). The second option often wins at small
scales because label quality is easier to control. Count review time for auto-labels as labelling time.

---

## Key Takeaways

- Teacher–student semi-supervision (EMA teacher, weak/strong augmentation, thresholded pseudo-labels)
  gives large gains at 1–10% labels: Unbiased Teacher reports 20.75 vs 9.05 AP at 1% COCO.
  Gains shrink with more labels.
- Auto-labelling with open-vocabulary models is labelling with a fast, biased annotator. Measure it,
  tune per-class thresholds, review the output.
- Active learning: mix diversity, uncertainty and rare-class mining; production failures are the best
  source.
- Detector distillation works best when the teacher is strong on your data and the student is
  capacity-limited. Ultralytics has built-in feature distillation. Expect about 1–2 AP and measure it.
- Foundation-model backbones and VFM distillation explain much of the recent DETR advantage and transfer
  well to unusual data.
- Compare label-efficient methods at equal labelling cost, counting review time.

## Check Yourself

<details class="check"><summary>Your semi-supervised run improves mAP on common classes but lowers it on two rare ones. What happened, and what would you change?</summary>
Pseudo-labels are dominated by frequent classes, and a global confidence threshold rarely passes rare
classes' lower-scoring predictions, so rare-class objects in unlabelled images become background. Use
per-class or dynamic thresholds, mine rare-class images for labelling, or down-weight pseudo-negatives
for rare classes.</details>

<details class="check"><summary>Why can a COCO-pretrained YOLO26x be a poor distillation teacher for your YOLO26n on a factory dataset?</summary>
Unless the teacher is fine-tuned on the factory data, it is not better than the student on that
domain, and its features and outputs encode COCO classes. Distillation transfers what the teacher
knows. Fine-tune the teacher on your data first and confirm it beats the student on your validation set.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| STAC | Sohn et al., 2020 | arXiv:2005.04757 | Pseudo-labels with strong augmentation |
| Unbiased Teacher | Liu et al., 2021 | arXiv:2102.09480 | EMA teacher; 1% COCO result |
| Soft Teacher | Xu et al., 2021 | arXiv:2106.09018 | End-to-end semi-supervised detection |
| Dense Teacher | Zhou et al., 2022 | arXiv:2207.02541 | Dense pseudo-labels |
| Consistent-Teacher | Wang et al., 2023 | arXiv:2209.01589 | Robust assignment, dynamic thresholds; 10% COCO result |
| Semi-supervised object detection survey | 2023 | arXiv:2306.14106 | Overview |
| LD (localisation distillation) | Zheng et al., 2022 | arXiv:2102.12252 | Distilling box distributions |
| FGD; CWD; MGD | Yang et al., 2022; Shu et al., 2021; Yang et al., 2022 | arXiv:2111.11837; arXiv:2011.13256; arXiv:2205.01529 | Feature distillation for detectors |
| Ultralytics `nn/distill_model.py` | Ultralytics | github.com/ultralytics/ultralytics | Built-in feature distillation |
| RT-DETRv4; DEIMv2; RF-DETR | Liao et al., 2025; Huang et al., 2025; Robinson et al., 2025 | arXiv:2510.25257; arXiv:2509.20787; arXiv:2511.09554 | Foundation-model features in real-time detectors |
| DINOv2; DINOv3 | Oquab et al., 2023; Siméoni et al., 2025 | arXiv:2304.07193; arXiv:2508.10104 | Self-supervised backbones |
| Simple Copy-Paste | Ghiasi et al., 2021 | arXiv:2012.07177 | Copy-paste augmentation |
| Domain randomization | Tobin et al., 2017 | arXiv:1703.06907 | Synthetic-to-real transfer |

---

**Next:** [Chapter 43 — Oriented, 3D, Video & Domain Shift](./43_specialized_detection.md) — detection
when the box, the dimension or the distribution changes.
