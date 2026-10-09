---
title: "Chapter 0 — The Whole Field on One Page"
---

[← Back to Table of Contents](./README.md)

# Chapter 0 — The Whole Field on One Page

> *"Every detector is the same six decisions wearing a different backbone."*

## Overview

Object detection looks like a zoo: two-stage and one-stage, anchors and anchor-free, CNNs and
transformers, YOLO versions with numbers that do not mean what you think, and now language models
that write bounding boxes as text. This chapter collapses the zoo into **six independent design
decisions**. Every model in this book is a point in that six-dimensional space. Once you can place a
model in the space, you can predict most of its behaviour before you read its paper.

<div class="diagram">
<div class="diagram-title">The six decisions inside every detector</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent"><div class="card-icon">1</div><div class="card-title">Where do predictions come from?</div><div class="card-desc">Anchor boxes, anchor points, learned queries, or generated tokens. (Ch 4)</div></div>
  <div class="diagram-card green"><div class="card-icon">2</div><div class="card-title">Who is responsible for which object?</div><div class="card-desc">Label assignment: IoU thresholds, ATSS, SimOTA, TAL, Hungarian. (Ch 5)</div></div>
  <div class="diagram-card purple"><div class="card-icon">3</div><div class="card-title">What is the box, numerically?</div><div class="card-desc">Deltas to an anchor, distances to a point (scalar or distribution), normalised cxcywh, quantised tokens. (Ch 2, 6)</div></div>
  <div class="diagram-card orange"><div class="card-icon">4</div><div class="card-title">How are duplicates removed?</div><div class="card-desc">NMS and its variants, or a one-to-one training target that makes NMS unnecessary. (Ch 7)</div></div>
  <div class="diagram-card cyan"><div class="card-icon">5</div><div class="card-title">What features feed the head?</div><div class="card-desc">Backbone + neck: CNN pyramids, hybrid encoders, plain ViTs with foundation pre-training. (Ch 11–13)</div></div>
  <div class="diagram-card pink"><div class="card-icon">6</div><div class="card-title">What is it optimised for?</div><div class="card-desc">mAP on COCO, latency on a T4, SRAM on an MCU, recall on a long tail, zero-shot transfer. (Ch 8, 46)</div></div>
</div>
</div>

---

## The problem in one sentence

Given an image, output a **set** of $(b, k, s)$ triples: a box $b$, a class $k$, and a confidence
$s$. The number of elements is not known in advance, and the order does not matter. Everything
difficult about detection follows from that sentence:

- A network has a **fixed-size output**, but the answer has a **variable size**. So every detector
  over-produces candidates and then selects from them.
- Over-producing means **many candidates near each object**. Training needs a rule for which of them
  learn to fire (assignment). Inference needs a rule for which of them survive (NMS, or a
  one-to-one design).
- A box is a **continuous** quantity, but the image is a **discrete** grid at several strides.
  Small objects fall between grid points (Chapter 41).
- Background dominates. A 640×640 YOLO evaluates 8,400 locations to find perhaps 7 objects
  (Chapter 4), so the loss must not let the background drown the signal (Chapter 6).

---

## Decision 1 — where predictions come from

<div class="diagram">
<div class="diagram-title">Four generations of "candidate"</div>
<div class="flow-h">
  <div class="flow-node blue">Anchor boxes<small>Faster R-CNN, SSD, RetinaNet, YOLOv2–v5, v7</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Anchor points<small>FCOS, YOLOX, YOLOv6/8/11/26, RTMDet</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Learned queries<small>DETR, DINO, RT-DETR, D-FINE, RF-DETR, YOLO27-M/L</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink">Generated tokens<small>Kosmos-2, Qwen-VL, Florence-2, Rex-Omni</small></div>
</div>
</div>

Anchor boxes encode a prior on shape. Anchor points encode only a prior on *location*. Queries encode
no fixed prior: the model learns where to look. Tokens turn the box into language. Each step removed
hand-designed priors. Each one also moved the difficulty somewhere else: from anchor tuning to
assignment design, to slow convergence, and finally to slow decoding.

---

## Decision 2 — assignment is the hidden heart

Two detectors with identical backbones and heads can differ by 3–5 AP purely through their
assignment rule. ATSS showed this explicitly: the gap between anchor-based RetinaNet and anchor-free
FCOS disappears once both use the same sample selection (Zhang et al., 2020). The field's arc:

| Era | Rule | Example models | Property |
|---|---|---|---|
| **Static, IoU threshold** | IoU ≥ 0.5 positive, < 0.4 negative | RetinaNet, SSD, Faster R-CNN | Hyperparameters per dataset; tiny objects get few positives |
| **Static, adaptive** | per-object threshold = mean + std of candidate IoUs | ATSS | Nearly hyperparameter-free |
| **Dynamic, cost-based** | optimal-transport cost + dynamic k | OTA, SimOTA (YOLOX) | Uses the network's own predictions |
| **Dynamic, task-aligned** | top-k by $p^{\alpha} u^{\beta}$ | TOOD, YOLOv6/8/11/26, PP-YOLOE | Pushes classification and localisation to agree |
| **One-to-one** | Hungarian matching | DETR family, YOLOv10/26 one-to-one head | Removes NMS; supervision is sparse, so convergence is slower |

Chapter 5 derives all five. The whole of Part IV keeps coming back to this table.

---

## Decisions 3 and 4 — the box, and the duplicates

A box can be regressed as **offsets to an anchor** ($t_x, t_y, \log t_w, \log t_h$), as **distances
from a point** ($l, t, r, b$), as a **distribution over distances** (DFL, D-FINE's FDR), as
normalised **centre–size** refined layer by layer (DETR), or as **quantised integer tokens**
(0–999 in Rex-Omni and Qwen-VL). The loss follows from the representation (Chapter 6).

Duplicates come from one-to-many assignment: several candidates learned to fire for the same object.
Greedy NMS removes them at inference. It costs latency that scales with the number of candidates, it
needs a threshold, and it fails in crowds. One-to-one assignment removes the cause instead.
YOLOv10 popularised training both heads at once ("consistent dual assignment") so that the cheap
one-to-one head can be used at inference. YOLO26 and every DETR are NMS-free. Chapter 7 covers both.

---

## Decisions 5 and 6 — features and objective

The feature extractor sets the accuracy ceiling. In 2025–2026 the strongest real-time detectors
(RF-DETR, DEIMv2, RT-DETRv4, EdgeCrafter) all lean on **vision foundation models**: DINOv2/v3
backbones or distillation from them. The objective sets what "good" means. COCO AP rewards precise
localisation averaged over IoU 0.5–0.95. An MCU deployment rewards peak SRAM. A warehouse robot
rewards recall at a fixed precision. No model is best on all of them, which is why Chapter 51 is a
set of flowcharts rather than a leaderboard.

---

## The landscape by deployment tier

| Tier | Typical budget | Representative models | COCO AP range | Chapter |
|---|---|---|:---:|---|
| **MCU / very tiny** | < 1 M params, < 1 GFLOP, ≤ 512 KB SRAM | FOMO, Yolo-FastestV2, PicoDet-XS, DEIMv2-Atto | 13–31 | [15](./15_tiny_detectors.md) |
| **Edge** | 1–10 M params, NPU/mobile GPU | YOLO26n/s, D-FINE-N, DEIMv2-N, RTMDet-tiny, RF-DETR-N | 38–50 | [16](./16_edge_detectors.md) |
| **Real-time server** | 10–60 M params, ≤ 15 ms on T4 | YOLO26m–x, D-FINE-L/X, DEIMv2-L/X, RF-DETR-M–2XL, RT-DETRv4 | 50–60 | [19](./19_realtime_detrs_and_vit.md), [26](./26_yolo_v8_to_yolo26.md) |
| **Large / accuracy-first** | 200 M+ params, seconds per image is fine | Co-DETR ViT-L, DINO Swin-L | 63–66 | [17](./17_large_detectors.md) |
| **Open-vocabulary** | text or visual prompts, no fixed classes | Grounding DINO, YOLO-World, YOLOE, DINO-X, SAM 3 | 45–56 zero-shot | [20](./20_open_vocabulary.md) |
| **LLM-based** | 3 B+ params, tokens out | Qwen3-VL, Rex-Omni, LocateAnything, Florence-2 | not comparable (F1, ODinW) | [21](./21_llm_based_detection.md) |

All COCO numbers are val2017 and come from `assets/data/detectors.json` with per-row sources. Open
the [Detection Atlas](./atlas.md) to filter and plot them.

<div class="callout myth"><span class="callout-title">Myth</span>"YOLO is a model." YOLO is a brand
used by at least nine independent groups between 2016 and 2026. YOLOv5, v8, 11, 26 and 27 come from
Ultralytics. v4, v7 and v9 come from Academia Sinica authors. v6 from Meituan, v10 from Tsinghua, v12
from Buffalo/UCAS, v13 from Tsinghua's iMoonLab. YOLOX, PP-YOLOE, DAMO-YOLO, YOLO-NAS and Gold-YOLO
each from a different company. Versions are not monotone improvements. Chapter 23 untangles them.</div>

---

## Read these five chapters first

1. [Chapter 5 — Label Assignment](./05_label_assignment.md): the concept that explains the most
   differences between models.
2. [Chapter 8 — Metrics](./08_metrics.md): you cannot compare anything until you know what AP
   actually integrates.
3. [Chapter 7 — Post-Processing](./07_post_processing.md): where a large share of real-world latency
   and crowd failures live.
4. [Chapter 22 — The YOLO Matrix](./22_yolo_matrix.md): the map of Part IV.
5. [Chapter 46 — Measuring Latency Honestly](./46_measuring_latency.md): so the next "2× faster"
   claim you read means something to you.

---

## Key Takeaways

- Detection is set prediction. Fixed-size networks over-produce candidates; assignment (training) and
  NMS or one-to-one heads (inference) decide which candidates count.
- Six decisions place every detector: candidate type, assignment, box parameterisation, duplicate
  removal, features, objective.
- Assignment rules explain gaps that are usually attributed to architecture. ATSS closed the
  RetinaNet–FCOS gap by changing assignment alone.
- The 2024–2026 shift is twofold: NMS-free inference (YOLOv10, YOLO26, all DETRs) and
  foundation-model features (DINOv2/v3 in RF-DETR, DEIMv2, RT-DETRv4).
- "YOLO" is a brand shared by many groups; version numbers encode release order, not lineage or
  quality.
- Pick a tier first (MCU, edge, server, large, open-vocabulary, LLM). Model choice inside a tier is a
  much smaller decision.

## Check Yourself

<details class="check"><summary>Why can't a detector simply output "the" list of objects?</summary>
A network's output tensor has a fixed shape, but the number of objects varies per image. Every
detector therefore produces a fixed, larger number of candidates (anchors, points, queries, or a
token budget) and then selects among them. Selection needs a training rule (assignment) and an
inference rule (NMS or one-to-one design).</details>

<details class="check"><summary>Two detectors share a backbone, neck and head but differ by 4 AP. Where do you look first?</summary>
Label assignment, then the loss weighting and the augmentation schedule. ATSS showed that
anchor-based vs anchor-free differences on COCO were explained by the definition of positive and
negative samples, not by the anchors themselves.</details>

<details class="check"><summary>What does "NMS-free" actually remove, and what does it cost?</summary>
It removes the post-processing sort-and-suppress step and its IoU threshold. To get there the model
must learn one-to-one assignment, which gives each object one positive. That sparse supervision
usually slows convergence or costs a little accuracy. YOLO26's one-to-one head is 0.6–0.8 AP behind
its one-to-many head on COCO.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Bridging the Gap Between Anchor-based and Anchor-free Detection via ATSS | Zhang et al., 2020 | arXiv:1912.02424 | Assignment, not anchors, explains the RetinaNet/FCOS gap |
| End-to-End Object Detection with Transformers (DETR) | Carion et al., 2020 | arXiv:2005.12872 | Detection as set prediction, Hungarian matching |
| YOLOv10: Real-Time End-to-End Object Detection | Wang et al., 2024 | arXiv:2405.14458 | Consistent dual assignment, NMS-free YOLO |
| Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models | Jocher et al., 2026 | arXiv:2606.03748 | e2e vs NMS AP gap (40.9 vs 40.1 for YOLO26n) |
| RF-DETR | Roboflow, 2025 | arXiv:2511.09554 | Foundation-backbone real-time DETR |
| DEIMv2: Real-Time Object Detection Meets DINOv3 | Huang et al., 2025 | arXiv:2509.20787 | DINOv3 features in a real-time detector |
| Detect Anything via Next Point Prediction (Rex-Omni) | Jiang et al., 2025 | arXiv:2510.12798 | Quantised coordinate tokens |
| Detection Atlas data | this book | [assets/data/detectors.json](./atlas.md) | Tier AP ranges |

---

**Next:** [Chapter 1 — A Short History of Finding Things](./01_history.md) — each of the six
decisions was made, unmade and remade over 25 years. History explains why the current answers look
the way they do.
