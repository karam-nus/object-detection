---
title: "Chapter 25 — YOLOv4–v7 and the Forks"
---

[← Back to Table of Contents](./README.md)

# Chapter 25 — YOLOv4–v7 and the Forks

> *"Between 2020 and 2023 the architecture barely changed. The training pipeline changed completely, and so did the accuracy."*

## Overview

After Redmon left the field, YOLO became a contest among groups, mostly fought over the **training
pipeline**: augmentation, assignment, loss, schedule. Architecture changes were smaller and
deployment-driven: CSP, re-parameterisation, NAS for TensorRT latency. This chapter covers YOLOv4's
"bag of freebies", YOLOv5's engineering, Scaled-YOLOv4 and YOLOR, PP-YOLO/PP-YOLOE, YOLOX's anchor-free
turn, YOLOv6's re-parameterised industrial design, YOLOv7's E-ELAN, and the DAMO-YOLO, YOLO-NAS, Gold-YOLO
and RTMDet branches. Each gets a fact block or table row. The axis chapters (27–37) then compare their
choices side by side.

<div class="timeline">
  <div class="timeline-item"><div class="timeline-year">Apr 2020</div><div class="timeline-title">YOLOv4</div><div class="timeline-desc">Bag of freebies / specials, CSPDarknet53, mosaic, CIoU — 43.5 AP at ~65 FPS (V100)</div></div>
  <div class="timeline-item"><div class="timeline-year">Jun 2020</div><div class="timeline-title">YOLOv5</div><div class="timeline-desc">PyTorch engineering, autoanchor, evolution, export — no paper</div></div>
  <div class="timeline-item"><div class="timeline-year">Jul 2020</div><div class="timeline-title">PP-YOLO</div><div class="timeline-desc">YOLOv3 + 10 tricks on ResNet50-vd — 45.2 AP at 72.9 FPS</div></div>
  <div class="timeline-item"><div class="timeline-year">Nov 2020</div><div class="timeline-title">Scaled-YOLOv4</div><div class="timeline-desc">CSP scaling P5–P7; YOLOv4-large 55.5 AP</div></div>
  <div class="timeline-item"><div class="timeline-year">Jul 2021</div><div class="timeline-title">YOLOX</div><div class="timeline-desc">anchor-free, decoupled head, SimOTA</div></div>
  <div class="timeline-item"><div class="timeline-year">2022</div><div class="timeline-title">PP-YOLOE · YOLOv6 · YOLOv7 · DAMO-YOLO · RTMDet</div><div class="timeline-desc">TAL/VFL/DFL arrive; re-parameterisation; NAS; T4 TensorRT becomes the yardstick</div></div>
  <div class="timeline-item"><div class="timeline-year">2023</div><div class="timeline-title">YOLOv6 3.0 · YOLO-NAS · Gold-YOLO</div><div class="timeline-desc">anchor-aided training; quantisation-aware NAS; gather-and-distribute neck</div></div>
</div>

---

## YOLOv4 — the bag of freebies

| | |
|---|---|
| **Introduced** | Bochkovskiy, Wang, Liao, 2020 (arXiv:2004.10934) |
| **Lineage** | YOLOv3 + CSPNet + PANet + a survey of tricks |
| **Paradigm** | anchors at three scales |
| **Assignment** | multiple anchors per GT above an IoU threshold |
| **Post-processing** | DIoU-NMS |
| **COCO AP (sizes)** | 43.5 AP (65.7 AP$_{50}$) at 608 px |
| **Latency** | ~65 FPS on a Tesla V100 |
| **Pre-training** | ImageNet |
| **License** | Darknet |
| **Known failure modes** | Darknet framework; many interacting tricks |

The paper's main contribution is its **taxonomy and ablation**. A *bag of freebies* (BoF) improves
accuracy at zero inference cost. A *bag of specials* (BoS) costs a little inference time:

| Bag of freebies (training only) | Bag of specials (small inference cost) |
|---|---|
| **Mosaic**, CutMix, self-adversarial training | **Mish** activation |
| DropBlock, class label smoothing | **CSP** connections (CSPDarknet53) |
| **CIoU loss** | **SPP** block (enlarged receptive field) |
| Cross mini-batch normalisation (CmBN) | Modified SAM (spatial attention) |
| Eliminating grid sensitivity (scaled sigmoid) | **PAN** neck |
| Multiple anchors per ground truth | DIoU-NMS |
| Cosine annealing, genetic-algorithm hyperparameters, random training shapes | |

Its legacy outlived Darknet. **Mosaic, CIoU, CSP, SPP, PAN and GA-tuned hyperparameters** became
standard in every later YOLO. "Eliminate grid sensitivity" ($b_x = s\cdot\sigma(t_x) - (s-1)/2 + c_x$
with $s > 1$) is the direct ancestor of YOLOv5's $2\sigma - 0.5$.

---

## YOLOv5 — engineering as the contribution

| | |
|---|---|
| **Introduced** | Ultralytics (Jocher), June 2020; last major release v7.0 (Nov 2022); no paper |
| **Lineage** | YOLOv3/v4 ideas in a new PyTorch codebase |
| **Paradigm** | anchors at three scales (P3–P5; P6 variants) |
| **Assignment** | shape-ratio matching (both side ratios within 4×) + **cross-grid**: the object's cell and its two nearest neighbours |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | r6.1: n 28.0 · s 37.4 · m 45.4 · l 49.0 · x 50.7 (val, 640) |
| **Latency** | 159 → 83 FPS on V100 from n to x (YOLOv7 README) |
| **Pre-training** | none: COCO from scratch, 300 epochs |
| **License** | AGPL-3.0 |
| **Known failure modes** | anchor-based (custom shapes need autoanchor); coupled head; objectness × class ranking |

What YOLOv5 standardised:

- **Decode** $b_{xy} = 2\sigma(t_{xy}) - 0.5 + c_{xy}$, $b_{wh} = (2\sigma(t_{wh}))^2 \cdot a_{wh}$. The
  sigmoid bounds widths at 4× the anchor, which removed the exploding-gradient risk of $e^{t_w}$.
- **Cross-grid assignment.** Each ground truth is assigned to its own cell and to the two neighbouring
  cells closest to its centre, for every anchor passing the ratio test. That triples the positives and
  is why the decode range was widened to (−0.5, 1.5) (Chapter 30).
- **Losses**: CIoU (`box = 0.05`), BCE objectness with IoU as the target (`obj = 1.0`, per-level balance
  [4.0, 1.0, 0.4]), BCE classes (`cls = 0.5`).
- **Engineering**: autoanchor (BPR check and k-means plus GA refit), hyperparameter **evolution**, AMP,
  EMA, multi-GPU DDP, nominal-batch-64 gradient accumulation, TTA, ensembling, export to ONNX,
  TensorRT, CoreML, TFLite, OpenVINO and more, and model scaling via `depth_multiple` and `width_multiple`.

YOLOv5's architecture (C3 blocks, SPPF, PAN) is unremarkable by 2026 standards. Its pipeline (augmentation
schedule, gradient accumulation, EMA, export) is the template that YOLOv7, YOLOv9 and every Ultralytics
model since inherit.

---

## Scaled-YOLOv4 and YOLOR (Academia Sinica)

**Scaled-YOLOv4** (2020) applied CSP to the whole network and scaled it to P5, P6 and P7 variants.
YOLOv4-large reached **55.5 AP (73.4 AP$_{50}$)** at 16 FPS on a V100, and YOLOv4-tiny 22.0 AP at 443 FPS
on an RTX 2080 Ti. **YOLOR** (2021) added "implicit knowledge": learned vectors added or multiplied into
features, shared across tasks. YOLOR-CSP-X reached 53.0 / 52.7 AP (test/val). Both fed into YOLOv7.

---

## PP-YOLO → PP-YOLOE (Baidu)

**PP-YOLO** (2020) kept YOLOv3's head and stacked ten tricks on a ResNet50-vd-DCN backbone: larger
batch, EMA, DropBlock, IoU loss, IoU-aware branch, grid sensitivity, Matrix NMS, CoordConv, SPP and better
pre-training. 45.2 AP at 72.9 FPS on V100. **PP-YOLOv2** (2021) pushed this to ~49.5 AP. **PP-YOLOE** (2022)
went anchor-free with a CSPRepResNet backbone, an **efficient task-aligned head** (ET-head), TAL
assignment and VFL + DFL losses: 51.4 AP (L) at 78.1 FPS on V100. **PP-YOLOE+** added Objects365
pre-training of the backbone and reached 52.9 (L) and 54.7 (X) with only 80 COCO epochs. It was one of the
first YOLO-style models to make detection pre-training a headline ingredient.

---

## YOLOX — anchor-free YOLO

| | |
|---|---|
| **Introduced** | Ge, Liu, Wang, Li, Sun (Megvii), 2021 (arXiv:2107.08430) |
| **Lineage** | YOLOv3/v5 backbone + FCOS/OTA ideas |
| **Paradigm** | anchor-free points, decoupled head |
| **Assignment** | SimOTA (dynamic k) |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | Nano 25.8 (416) · Tiny 32.8 (416) · S 40.5 · M 46.9 · L 49.7 · X 51.1 (val) |
| **Latency** | S 9.8 ms … X 17.3 ms on V100 (FP16) |
| **Pre-training** | none |
| **License** | Apache-2.0 |
| **Known failure modes** | SimOTA's per-image cost during training; NMS still needed |

YOLOX made three changes, each ablated in the paper on a YOLOv3 baseline:

1. **Decoupled head** (separate classification and regression branches, Chapter 13): faster convergence
   and higher AP. The paper also found it essential for its end-to-end experiments.
2. **Anchor-free**: one point per cell, $(l,t,r,b)$-style regression, which cut the head's output size
   by 3× and removed anchor tuning.
3. **SimOTA**: dynamic-k, prediction-aware assignment (Chapter 5), the largest single gain.

Training used Mosaic + MixUp with **no augmentation for the final 15 epochs**, the idea that later
became `close_mosaic`. YOLOX won the 2021 Streaming Perception Challenge and is still the best-known
**Apache-2.0** YOLO.

---

## YOLOv6 — built for TensorRT

| | |
|---|---|
| **Introduced** | Li et al. (Meituan), 2022 (arXiv:2209.02976); v3.0: Li et al., 2023 (arXiv:2301.05586) |
| **Lineage** | RepVGG + YOLOX-style head + TOOD's TAL |
| **Paradigm** | anchor-free points, efficient decoupled head; v3.0 adds anchor-aided training |
| **Assignment** | ATSS warm-up, then TAL |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | v3.0: N 37.5 · S 45.0 · M 50.0 · L 52.8 (640); L6 57.2 (1280) |
| **Latency** | N 779 · S 339 · M 175 · L 98 FPS (T4 TensorRT FP16, batch 1) |
| **Pre-training** | none; self-distillation |
| **License** | GPL-3.0 |
| **Known failure modes** | re-parameterised blocks need care for INT8 (addressed with RepOptimizer / QAT) |

Meituan designed YOLOv6 for industrial GPU serving. **EfficientRep** backbones (RepVGG blocks for small
models, CSPStackRep for large ones) fold into plain 3×3 convs at inference. **Rep-PAN**, and in v3.0
**RepBiPAN with BiC** (bi-directional concatenation), forms the neck. The losses are VFL classification
and SIoU (small) or GIoU (large) box loss, plus DFL in v3.0's larger models. **Self-distillation** uses
the model's own teacher copy. **Anchor-aided training (AAT)** in v3.0 adds an anchor-based auxiliary
branch during training only. For deployment, YOLOv6 introduced **RepOptimizer** and channel-wise
distillation to make re-parameterised models quantise well. It was one of the first YOLOs designed with
INT8 in mind (Chapter 45).

---

## YOLOv7 — E-ELAN and planned re-parameterisation

| | |
|---|---|
| **Introduced** | Wang, Bochkovskiy, Liao, 2022 (arXiv:2207.02696, CVPR 2023) |
| **Lineage** | Scaled-YOLOv4 / YOLOR + YOLOv5-style PyTorch code |
| **Paradigm** | anchors at three scales (P6 variants at 1280) |
| **Assignment** | lead-head-guided coarse (auxiliary) and fine (lead) labels |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | tiny 38.7 · v7 51.2 · X 52.9 (val, 640); E6E 56.8 (test-dev, 1280) |
| **Latency** | 286 / 161 / 114 FPS (tiny / v7 / X, V100) |
| **Pre-training** | none |
| **License** | GPL-3.0 |
| **Known failure modes** | anchor-based; complex training code; superseded by anchor-free heads |

YOLOv7 claimed to beat every known real-time detector between 5 and 160 FPS on V100 at release. Its
ideas:

- **E-ELAN**: extend ELAN by expanding cardinality (group convs), shuffling and merging groups. More
  diverse features without breaking the shortest/longest gradient paths that ELAN controls.
- **Compound scaling for concatenation-based models**: scaling depth changes the channel count after
  concatenation, so width must be scaled together with it.
- **Planned re-parameterisation**: RepConv's identity branch hurts when placed after a residual or
  concatenation. YOLOv7 uses **RepConvN** (no identity) in those positions.
- **Coarse-to-fine lead-guided assignment**: an auxiliary head trained with *coarse* labels (more
  positives, relaxed constraints) generated from the lead head's predictions, and the lead head with
  *fine* labels. The auxiliary head is dropped at inference.

---

## The other forks

| Model | Org, year | Key ideas | COCO AP (as reported) | License |
|---|---|---|---|---|
| **DAMO-YOLO** | Alibaba, 2022 | MAE-NAS backbones under latency constraints; **Efficient RepGFPN**; **ZeroHead** (heavy neck, near-empty head); **AlignedOTA**; distillation | T* 43.6 · S* 47.7 · M* 50.2 · L* 51.9 at 2.78–7.95 ms (T4 FP16) | Apache-2.0 |
| **RTMDet** | OpenMMLab, 2022 | CSPNeXt with 5×5 depthwise large kernels; **dynamic soft label assigner**; shared head convs with separate BN | tiny 41.1 … x 52.8 at 2.34–18.8 ms (T4 FP16) | Apache-2.0 |
| **YOLO-NAS** | Deci, 2023 | AutoNAC NAS with quantisation-aware blocks; Objects365 pre-training; pseudo-labelled COCO; distillation; DFL | S 47.5 · M 51.55 · L 52.22; INT8 47.03 / 51.0 / 52.1 (T4) | Apache code / non-commercial weights |
| **Gold-YOLO** | Huawei, 2023 | **Gather-and-distribute** neck; MAE-style backbone pre-training | N 39.9 · S 46.4 · M 51.1 · L 53.3 | GPL-3.0 |
| **EdgeYOLO** | 2023 | YOLOX-derived, edge tricks; Jetson-focused | Tiny 41.4 … L 50.6 (val) | Apache-2.0 |

---

## What this era established

| Contribution | First prominent in | Fate |
|---|---|---|
| Mosaic + MixUp + close-mosaic | v4 / YOLOX | Standard |
| CIoU box loss | v4 | Standard in Ultralytics YOLOs |
| CSP blocks | v4 / Scaled-v4 | Evolved into C3 → C2f → C3k2 |
| Hyperparameter evolution (GA) | v4 / v5 | `model.tune()` (Chapter 32) |
| Cross-grid positives | v5 | Superseded by TAL |
| Anchor-free + decoupled head | YOLOX | Standard since v6/v8 |
| SimOTA → TAL | YOLOX → PP-YOLOE / v6 | TAL is the modern default |
| VFL + DFL | PP-YOLOE / v6 | DFL standard until YOLO26 removed it |
| Re-parameterisation | v6 / v7 | Standard in TensorRT-oriented designs |
| Quantisation-aware architecture | v6 / YOLO-NAS | Revived by YOLO26's edge focus |
| Objects365 pre-training | PP-YOLOE+ / YOLO-NAS | Standard in 2025–26 (YOLO26, DETRs) |

---

## Key Takeaways

- YOLOv4 organised detection tricks into a bag of freebies (training-only) and a bag of specials (small
  inference cost). Mosaic, CIoU, CSP, SPP and PAN came from it.
- YOLOv5's contribution was engineering: a reproducible PyTorch pipeline, autoanchor, evolution, EMA,
  gradient accumulation and export. Its pipeline outlived its architecture.
- YOLOX made YOLO anchor-free with a decoupled head and SimOTA, under Apache-2.0.
- YOLOv6 and YOLOv7 optimised for GPU inference through re-parameterisation. YOLOv6 also targeted INT8
  quantisation.
- PP-YOLOE and YOLO-NAS made Objects365 pre-training a headline ingredient. DAMO-YOLO and YOLO-NAS
  brought latency-constrained NAS.
- Most of this era's gains came from training-pipeline axes (augmentation, assignment, loss, schedule)
  rather than from the backbone.

## Check Yourself

<details class="check"><summary>Why did YOLOv5 widen the centre decode from σ(t) to 2σ(t) − 0.5?</summary>
Cross-grid assignment makes the two cells neighbouring an object's centre responsible for it as well. A
prediction from a neighbouring cell must reach a centre outside its own cell, so the offset range was
widened from (0, 1) to (−0.5, 1.5).</details>

<details class="check"><summary>Which YOLOX change gave the largest gain, and why was it more than an architecture change?</summary>
SimOTA. It changed which predictions are trained as positives, dynamically and per object, using the
network's own predictions and a cost combining classification and IoU. Like ATSS before it, it showed
that assignment, a training-time decision invisible at inference, drives much of the accuracy.</details>

<details class="check"><summary>What is the difference between YOLOv4's bag of freebies and bag of specials?</summary>
Freebies change only training (augmentation, losses, label smoothing, schedules) and cost nothing at
inference. Specials add a small inference cost (Mish, SPP, attention modules, PAN, DIoU-NMS) for a larger
accuracy gain. The split is a useful way to evaluate any proposed trick.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| YOLOv4 | Bochkovskiy, Wang, Liao, 2020 | arXiv:2004.10934 | BoF/BoS, 43.5 AP @ ~65 FPS |
| Ultralytics YOLOv5 repo and docs | Ultralytics, 2020–2022 | github.com/ultralytics/yolov5 | Decode, assignment, losses, hyperparameters, engineering |
| Scaled-YOLOv4 | Wang, Bochkovskiy, Liao, 2021 | arXiv:2011.08036 | 55.5 AP large; 22.0 AP tiny @ 443 FPS |
| YOLOR | Wang, Yeh, Liao, 2021 | arXiv:2105.04206 | Implicit knowledge |
| PP-YOLO; PP-YOLOv2; PP-YOLOE | Long et al., 2020; Huang et al., 2021; Xu et al., 2022 | arXiv:2007.12099; 2104.10419; 2203.16250 + README | Trick stacks, ET-head, numbers |
| YOLOX | Ge et al., 2021 | arXiv:2107.08430 + README | Anchor-free, decoupled, SimOTA, numbers |
| YOLOv6 / YOLOv6 v3.0 | Li et al., 2022 / 2023 | arXiv:2209.02976; 2301.05586 + README | EfficientRep, BiC, AAT, RepOptimizer, numbers |
| YOLOv7 | Wang, Bochkovskiy, Liao, 2023 | arXiv:2207.02696 + README | E-ELAN, RepConvN, lead-guided assignment, numbers, YOLOv5 r6.1 table |
| DAMO-YOLO | Xu et al., 2022 | arXiv:2211.15444 + README | MAE-NAS, RepGFPN, ZeroHead, AlignedOTA, numbers |
| RTMDet | Lyu et al., 2022 | arXiv:2212.07784 + README | CSPNeXt, numbers |
| YOLO-NAS docs | Deci / Ultralytics docs | docs/en/models/yolo-nas.md | Numbers and INT8 |
| Gold-YOLO | Wang et al., 2023 | arXiv:2309.11331 + README | GD neck, numbers |
| EdgeYOLO README | LSH9832, 2023 | github.com/LSH9832/edgeyolo | Numbers |

---

**Next:** [Chapter 26 — YOLOv8 → YOLO26 (and YOLO27)](./26_yolo_v8_to_yolo26.md) — the modern era:
anchor-free DFL heads, PGI, NMS-free dual assignment, attention, hypergraphs, and the edge-first
redesign.
