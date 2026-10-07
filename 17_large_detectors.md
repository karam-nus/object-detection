---
title: "Chapter 17 — Large Detectors"
---

[← Back to Table of Contents](./README.md)

# Chapter 17 — Large Detectors

> *"Above 60 AP, every point costs a billion parameters, a larger pre-training set, or a slower test-time procedure."*

## Overview

"Large" detectors optimise accuracy with latency as an afterthought. They have 200 M to 7 B parameters,
use 1,000–1,500 px inputs, are pre-trained on Objects365 or web-scale self-supervision, and often add
test-time augmentation and ensembles. They matter for three reasons even if you never deploy one: they
mark the ceiling of what is learnable on COCO, they are the **teachers and auto-labellers** for small
models (Chapters 42 and 49), and their tricks trickle down to real-time models within two years. This
chapter covers the 60+ AP club, what each ingredient contributes, and what it costs.

<div class="diagram">
<div class="diagram-title">The four ingredients of a leaderboard detector</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card purple"><div class="card-title">Big backbone</div><div class="card-desc">Swin-L, ViT-L / EVA-02, InternImage-H, DINOv3-7B</div></div>
  <div class="diagram-card green"><div class="card-title">Detection pre-training</div><div class="card-desc">Objects365 (often + other sets), then COCO fine-tuning</div></div>
  <div class="diagram-card accent"><div class="card-title">Strong DETR head</div><div class="card-desc">DINO, Co-DETR hybrid assignment, Plain-DETR</div></div>
  <div class="diagram-card blue"><div class="card-title">Test-time tricks</div><div class="card-desc">multi-scale + flip TTA, WBF ensembles</div></div>
</div>
</div>

---

## The 60+ AP club

| System | Backbone | Pre-training | COCO AP | Split | Notes |
|---|---|---|:---:|---|---|
| **DINO** (5-scale) | Swin-L | IN-22k + Objects365 | 63.2 / **63.3** | val / test-dev | First DETR at the top of COCO (2022) |
| **Grounding DINO-L** (fine-tuned) | Swin-L | O365 + OI + GoldG + Cap4M | 63.0 | val | Open-vocabulary model fine-tuned on COCO |
| **Co-DETR (Co-DINO)** | Swin-L | IN-22k + Objects365 | 64.1 | val | Collaborative one-to-many auxiliary heads |
| **InternImage-H** + DINO | InternImage-H (DCNv3, 2.18 B params) | large-scale | **65.4** | test-dev | Deformable-conv foundation backbone (2023) |
| **Co-DETR (Co-DINO)** | ViT-L (EVA-02) | Objects365 | 65.9 / **66.0** | val / test-dev | Strongest open checkpoint family |
| **DINOv3 + Plain-DETR** | DINOv3 ViT-7B, **frozen** | web-scale self-supervised | **66.1** | — | Frozen backbone, lightweight decoder trained on top (2025) |

COCO's evaluation-server leaderboard also lists industry submissions at about 66.0–66.1 AP (2024–2025).
The ceiling has moved by less than one point since 2023. Chapter 52 discusses what that saturation means,
including evidence that a measurable share of the remaining "errors" are annotation errors.

<div class="callout note"><span class="callout-title">Note</span>The DINOv3 result is the most
interesting one in the table. The 7-billion-parameter backbone is <em>frozen</em>, and only the
detection adapter and decoder are trained. Detection-specific feature learning, the thing the YOLO
backbone line spent eight years optimising, is no longer needed at the top. The work moved to
pre-training. A leaderboard listing marks the 66.1 entry as using test-time augmentation.</div>

---

## Ingredient 1 — backbone scale

Across papers, moving from ResNet-50 to Swin-L under the same DETR head adds roughly 8–10 AP. DINO
goes from about 49–51 AP (R50, 12–36 epochs) to 58.5 (Swin-L, 36 epochs) with COCO-only detection
training. Moving to billion-parameter backbones adds a few more. The
backbones that work at this scale share three properties: hierarchical or adapter-built multi-scale
outputs, attention or deformable convolution restricted to windows or sampled points, and large-scale
pre-training (Chapter 11).

| Backbone | Params | Mechanism | Used in |
|---|:---:|---|---|
| Swin-L | ~197 M | shifted-window attention | DINO, Co-DETR, Grounding DINO-L |
| ViT-L / EVA-02-L | ~300 M | plain ViT, MIM + CLIP-style pre-training | Co-DETR ViT-L, ViTDet |
| InternImage-H | 2.18 B | DCNv3 deformable convolutions | InternImage + DINO |
| DINOv3 ViT-7B | ~7 B | self-supervised, frozen at detection time | DINOv3 + Plain-DETR |

---

## Ingredient 2 — detection pre-training

The single most reliable large-detector ingredient is **Objects365 pre-training** of the whole detector,
backbone *and* head, followed by COCO fine-tuning. Evidence from models in this book:

| Model | COCO-only | With Objects365 | Gain |
|---|:---:|:---:|:---:|
| D-FINE-L | 54.0 | 57.3 | +3.3 |
| D-FINE-X | 55.8 | 59.3 | +3.5 |
| RT-DETR-R50 | 53.1 | 55.3 | +2.2 |
| RT-DETR-R101 | 54.3 | 56.2 | +1.9 |
| ECDet-L (EdgeCrafter) | 57.0 | 59.0 | +2.0 |

Gains of 2–3.5 AP *at real-time scale* make one point obvious: comparisons that mix COCO-only and
Objects365-pretrained models (YOLO26, LW-DETR, RF-DETR are all O365-pretrained) are comparing training
data as much as architecture (Chapter 34).

---

## Ingredient 3 — the head: DINO, Co-DETR, Plain-DETR

**DINO** combines contrastive denoising (positive *and* negative noised queries), mixed query selection
(positional queries from the encoder's top-$k$, content queries learned), and look-forward-twice box
refinement. **Co-DETR** adds parallel **one-to-many auxiliary heads** during training: ATSS- and
Faster-R-CNN-style heads on the encoder output, whose positive coordinates also become extra decoder
queries. This gives the encoder dense supervision and the decoder more positive queries. At inference
the auxiliary heads are discarded. **Plain-DETR** removes multi-scale feature maps and decoder
complexity in favour of a plain single-scale ViT feature map with box-to-pixel relative position bias.
It suits frozen foundation backbones (Chapter 18 covers these designs).

---

## Ingredient 4 — test-time augmentation and ensembles

| Technique | Typical gain on COCO | Cost |
|---|:---:|---|
| Horizontal-flip TTA | +0.3–0.6 AP | 2× inference |
| Multi-scale TTA (e.g. 3–5 scales) | +1–2 AP | 3–5× inference |
| Ensembles of different models + WBF | +1–3 AP | N× inference, N× memory |
| Soft-NMS instead of NMS (two-stage) | +0.5–1 AP | slower NMS |

These numbers are typical ranges reported across leaderboard write-ups, not a single controlled
measurement. Treat them as an order of magnitude estimate. TTA results must be reported separately from
single-scale results (Chapter 8).

---

## What large detectors cost

| Cost | Order of magnitude | Consequence |
|---|---|---|
| Training compute | tens of thousands of GPU-hours including pre-training | Out of reach for most teams; fine-tune released checkpoints |
| Inference latency | 100 ms – several s per image on a data-centre GPU (with TTA) | Offline only |
| Memory | 10–80+ GB for training at 1,500 px | Multi-GPU even for fine-tuning |
| Engineering | Custom CUDA ops (deformable attention), mmdet/detrex stacks | Hard to export; rarely TensorRT-ready |

---

## How to use a large detector anyway

1. **As an auto-labeller.** Run a strong large or open-vocabulary detector over unlabelled images, keep
   high-confidence boxes, have humans correct them, and train a small model (Chapter 49).
2. **As a teacher for distillation.** Logit, feature or localisation distillation into a real-time
   student (Chapter 42). RT-DETRv4 and DEIMv2 distil from vision *foundation* models rather than from
   large *detectors*, and that is now usually the better teacher.
3. **As an evaluation oracle.** On a new domain, a large model's errors show which failures come from
   data ambiguity (both large and small fail) and which come from capacity (only the small one fails).
4. **Offline analytics.** Archives, satellite tiles, medical review queues, anything without a latency
   budget.

---

## Key Takeaways

- The COCO ceiling is about 66 AP (Co-DETR ViT-L 66.0 test-dev; DINOv3 + Plain-DETR 66.1). It has
  moved by less than one point since 2023.
- Four ingredients produce leaderboard numbers: a large backbone, detection pre-training (Objects365),
  a strong DETR head (DINO / Co-DETR), and TTA or ensembles.
- Objects365 pre-training alone is worth about 2–3.5 AP even for real-time models (D-FINE-X 55.8 →
  59.3). Comparisons must control for it.
- DINOv3's result with a *frozen* backbone shows that the frontier moved from detection architecture
  to pre-training.
- Large detectors are most useful as auto-labellers, teachers and oracles, not as deployed models.

## Check Yourself

<details class="check"><summary>A paper reports 60.5 AP and claims a new architecture beats YOLO26x (57.5). What three things do you check first?</summary>
(1) Pre-training data: Objects365? larger? YOLO26 also used Objects365v1, so check whether it matches.
(2) Test conditions: single-scale or TTA, input size, val or test-dev. (3) Cost: latency on stated
hardware, parameters and FLOPs. A 60.5 AP model at 1,333 px with TTA is a different product from a
real-time one.</details>

<details class="check"><summary>Why does Co-DETR's one-to-many auxiliary supervision help a one-to-one DETR, and why does it cost nothing at inference?</summary>
Hungarian matching gives few positives, so the encoder features get sparse supervision. Auxiliary ATSS
or Faster R-CNN heads assign many positives per object, giving dense gradients to the encoder. Their
positive boxes also become extra positive queries for the decoder. The heads are training-only and are
dropped at inference, so the deployed model is a normal DETR.</details>

<details class="check"><summary>What is surprising about the DINOv3 + Plain-DETR result, and what does it imply for small detectors?</summary>
The backbone is frozen, so the 66.1 AP comes from general-purpose self-supervised features plus a light
detection decoder. For small detectors this implies that distilling foundation-model features (as
RT-DETRv4, DEIMv2 and EdgeCrafter do) is a better use of effort than hand-designing new detection
backbones.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| DINO | Zhang et al., 2022 | arXiv:2203.03605 + IDEA-Research/DINO README | 63.2 / 63.3 AP |
| Grounding DINO README | IDEA, 2023 | github.com/IDEA-Research/GroundingDINO | 63.0 fine-tuned |
| DETRs with Collaborative Hybrid Assignments Training (Co-DETR) | Zong et al., 2023 | arXiv:2211.12860 + Sense-X/Co-DETR README | 64.1 (Swin-L), 65.9 / 66.0 (ViT-L) |
| InternImage | Wang et al., 2023 | arXiv:2211.05778 | 2.18 B params, 65.4 test-dev |
| DINOv3 | Siméoni et al., 2025 | arXiv:2508.10104 | Frozen backbone + Plain-DETR, 66.1 mAP |
| Plain-DETR | Lin et al., 2023 | arXiv:2308.01904 | Plain single-scale DETR |
| EVA-02 | Fang et al., 2023 | arXiv:2303.11331 | ViT-L backbone pre-training |
| D-FINE; RT-DETR; EdgeCrafter READMEs | 2023–2026 | github repos | O365 vs COCO-only gains |
| COCO evaluation-server leaderboard (via codesota mirror) | 2024–2025 | codesota.com/benchmark/coco | ~66.0–66.1 industry entries |
| Benchmarking Object Detectors with COCO: A New Path Forward | Singh et al., 2024 | arXiv:2403.18819 | Annotation errors near the ceiling |

---

**Next:** [Chapter 18 — DETR and Its Descendants](./18_detr_family.md) — most of the large systems
above are DETRs. Here is how that family works, from the 500-epoch original to DINO.
