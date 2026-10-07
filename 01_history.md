---
title: "Chapter 1 — A Short History of Finding Things"
---

[← Back to Table of Contents](./README.md)

# Chapter 1 — A Short History of Finding Things

> *"Each era of detection removed one hand-designed component and paid for it somewhere else."*

## Overview

Twenty-five years of object detection can be read as a sequence of removals. Hand-crafted features
were removed by CNNs. External proposals were removed by region proposal networks. The proposal
stage itself was removed by one-stage detectors. Anchor boxes were removed by anchor-free heads. NMS
was removed by set prediction. Fixed vocabularies were removed by grounding models. Each removal moved
the difficulty into training: assignment, loss design, convergence speed, data scale. This chapter
gives the timeline and, more usefully, the *reason* each step happened.

<div class="diagram">
<div class="diagram-title">The removals</div>
<div class="flow-h">
  <div class="flow-node">hand features<small>→ CNN (2014)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">selective search<small>→ RPN (2015)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">two stages<small>→ YOLO/SSD (2016)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">anchors<small>→ FCOS/YOLOX (2019–21)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">NMS<small>→ DETR / YOLOv10 (2020–24)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">fixed classes<small>→ grounding / MLLMs (2022–26)</small></div>
</div>
</div>

---

## Era 0 — Sliding windows and hand-crafted features (2001–2012)

**Viola–Jones (2001).** The first real-time face detector. It combined Haar-like features computed in
O(1) from an *integral image*, AdaBoost to select a few thousand of them, and an *attentional
cascade* that rejects most windows after a handful of features. The cascade is the ancestor of every
"cheap first stage, expensive second stage" design, including two-stage detectors and
detector + VLM pipelines (Chapter 21).

**HOG + linear SVM (Dalal & Triggs, 2005)** made pedestrian detection work with histograms of
gradient orientations over a dense grid. **Deformable Part Models (Felzenszwalb et al., 2010)**
added parts on springs and won PASCAL VOC for several years. DPM's latent-SVM training, which
decides which root/part placement is "the" positive, is an early form of *learned assignment*.

What these methods could not fix: features were designed by hand, and the sliding window evaluated
every position and scale, so cost grew with the number of windows times the cost per window.

---

## Era 1 — CNN features with external proposals (2013–2015)

| Year | Model | What it changed | What it still had |
|---|---|---|---|
| 2014 | **R-CNN** (Girshick et al.) | ~2,000 Selective Search proposals, each warped and passed through a CNN; SVM classifiers; 53.3% mAP on VOC 2012, ~30% relative over prior best | One CNN forward pass *per proposal*; multi-stage training |
| 2014 | **SPPNet** (He et al.) | One CNN pass per image; spatial pyramid pooling on the shared feature map | Still external proposals |
| 2015 | **Fast R-CNN** | RoI pooling + a single multi-task loss (classification + smooth-L1 box) | Selective Search on the CPU dominated runtime |
| 2015 | **Faster R-CNN** (Ren et al.) | **Region Proposal Network** with **anchor boxes**; proposals became nearly free | Two stages; RoI pooling quantisation |

Faster R-CNN introduced the two ideas that still define the field: **anchors** (reference boxes tiled
over a feature map, Chapter 4) and **IoU-threshold assignment** (anchors with IoU ≥ 0.7 to a
ground truth are positives, Chapter 5). On COCO test-dev with VGG-16 it reached 21.9 AP. The COCO
dataset itself, with its AP@[.5:.95] metric that rewards precise boxes, was released in 2014
(Lin et al.) and has been the yardstick ever since (Chapter 8).

---

## Era 2 — One stage, multiple scales, and the imbalance problem (2016–2018)

**YOLOv1 (Redmon et al., 2016)** reframed detection as a single regression: divide the image into an
$S \times S$ grid ($S = 7$), and let each cell predict $B = 2$ boxes and class probabilities. It ran at
45 FPS with 63.4 mAP on VOC 2007, well below Faster R-CNN's accuracy but an order of magnitude faster.
Its weaknesses (one class per cell, coarse grid, poor small-object recall) shaped the next decade of
YOLO design (Chapter 24).

**SSD (Liu et al., 2016)** predicted from *several* feature maps with anchor ("default") boxes at each
scale: the first multi-scale one-stage detector.

**FPN (Lin et al., 2017)** added a top-down pathway so that high-resolution maps also carry semantic
information. Every modern neck (PAN, BiFPN, RepGFPN, Chapter 12) descends from it.

**RetinaNet and focal loss (Lin et al., 2017)** diagnosed why one-stage detectors lagged: about
100,000 anchors per image, almost all easy negatives, swamp the cross-entropy. Focal loss
down-weights easy examples by $(1-p_t)^{\gamma}$ and gave the first one-stage detector to beat
two-stage accuracy (39.1 AP test-dev with ResNet-101-FPN). Chapter 6 derives it.

**Mask R-CNN (He et al., 2017)** replaced RoI pooling with **RoIAlign** (bilinear sampling with no
quantisation) and added a mask head. RoIAlign matters for detection too: the quantisation error it
removed was costing box precision.

**YOLOv2 (2017)** and **YOLOv3 (2018)** brought anchors chosen by **k-means over box shapes**,
batch normalisation, multi-scale training, Darknet-53, and three-scale prediction. YOLOv3-608
reached 33.0 AP at 51 ms on a Titan X. **Cascade R-CNN (Cai & Vasconcelos, 2018)** showed that training
successive heads at rising IoU thresholds (0.5, 0.6, 0.7) gives sharply better high-IoU AP.

---

## Era 3 — Anchor-free, assignment-aware, and the engineering YOLOs (2019–2022)

| Year | Model | Contribution |
|---|---|---|
| 2019 | **FCOS** | Per-pixel $(l,t,r,b)$ regression from points + centre-ness; no anchors |
| 2019 | **CenterNet** ("Objects as Points") | Objects as heat-map peaks; box size regressed at the peak; no NMS in the original |
| 2019 | **ATSS** | Proved the anchor-based vs anchor-free gap is an *assignment* gap |
| 2019 | **EfficientDet** | BiFPN + compound scaling; D0–D7x from 34.6 to 55.1 AP |
| 2020 | **YOLOv4** (Bochkovskiy et al.) | Systematic "bag of freebies / bag of specials": mosaic, CIoU, Mish, CSP, SPP, PAN |
| 2020 | **YOLOv5** (Ultralytics, no paper) | Engineering: PyTorch, auto-anchor, hyperparameter evolution, export, n/s/m/l/x scaling |
| 2020 | **GFL** (Li et al.) | Quality focal loss + **Distribution Focal Loss**: boxes as distributions |
| 2021 | **OTA / YOLOX** | Assignment as optimal transport; SimOTA's dynamic k; decoupled head |
| 2021 | **TOOD** | **Task-aligned assignment** $t = p^\alpha u^\beta$ — later adopted by YOLOv6/8/11/26 |
| 2022 | **YOLOv6, YOLOv7, PP-YOLOE, DAMO-YOLO, RTMDet** | Re-parameterisation, E-ELAN, TAL/VFL, NAS backbones; T4 TensorRT latency becomes the standard axis |

The defining shift of this era: **detection quality was set by the training pipeline** (assignment,
loss, augmentation, schedule) at least as much as by the architecture. YOLOv4's paper is essentially
a controlled ablation of training tricks. YOLOv5 made those tricks reproducible and exportable.

---

## Era 4 — Set prediction and transformers (2020–2025)

**DETR (Carion et al., 2020)** removed anchors *and* NMS. A transformer decoder turns $N=100$ learned
queries into $N$ predictions. Hungarian matching assigns each ground truth to exactly one query.
Duplicates are penalised by the loss, so no NMS is needed. The price: 500 training epochs and weak
small-object AP.

The next four years were spent paying back that price:

<div class="timeline">
  <div class="timeline-item"><div class="timeline-year">2020</div><div class="timeline-title">Deformable DETR</div><div class="timeline-desc">Sparse multi-scale deformable attention; ~10× fewer epochs</div></div>
  <div class="timeline-item"><div class="timeline-year">2021–22</div><div class="timeline-title">Conditional / DAB / DN-DETR</div><div class="timeline-desc">Queries as anchor boxes; denoising queries stabilise matching</div></div>
  <div class="timeline-item"><div class="timeline-year">2022</div><div class="timeline-title">DINO</div><div class="timeline-desc">Contrastive denoising + mixed query selection; 63.3 AP on test-dev with Swin-L + Objects365 — the first DETR at the top of COCO</div></div>
  <div class="timeline-item"><div class="timeline-year">2023</div><div class="timeline-title">RT-DETR, Co-DETR</div><div class="timeline-desc">"DETRs beat YOLOs" in real time (Baidu); one-to-many auxiliary heads push Co-DETR to 66.0 AP</div></div>
  <div class="timeline-item"><div class="timeline-year">2024</div><div class="timeline-title">LW-DETR, D-FINE, DEIM</div><div class="timeline-desc">Fine-grained distribution refinement, dense one-to-one matching; DETRs overtake YOLO at equal latency</div></div>
  <div class="timeline-item"><div class="timeline-year">2025–26</div><div class="timeline-title">RF-DETR, DEIMv2, RT-DETRv4, EdgeCrafter</div><div class="timeline-desc">DINOv2/v3 backbones or VFM distillation; RF-DETR-2XL 60.1 AP at 17.2 ms on T4</div></div>
</div>

Meanwhile YOLO absorbed DETR's idea in the other direction. **YOLOv10 (2024)** trained a one-to-one
head next to the usual one-to-many head and dropped NMS at inference. **YOLO26 (2026)** made the
one-to-one path a first-class deployment option and removed DFL. The preview of **YOLO27** (Ultralytics
docs, October 2026, preliminary) completes the convergence: its M and L detection models are
query-based, NMS-free transformer decoders. The two lineages have met.

---

## Era 5 — Open vocabulary and language (2021–2026)

**CLIP (2021)** gave image–text embeddings, and detection followed. **OWL-ViT (2022)** and
**GLIP (2022)** turned detection into phrase grounding. **Grounding DINO (2023)** reached 52.5 AP on
COCO *zero-shot*, without COCO training data. **YOLO-World (2024)** and **YOLOE (2025)** made
open-vocabulary detection real-time. **DINO-X (2024)** reached 56.0 zero-shot COCO AP. **SAM 3
(2025)** unified concept-prompted detection, segmentation and tracking.

In parallel, multimodal LLMs learned to **write boxes as text**: Kosmos-2 and Shikra (2023), the
Qwen-VL line (normalised 0–1000 coordinates as plain numbers), Florence-2 (location tokens),
**Rex-Omni** (2025: 1,000 quantised coordinate tokens, SFT then GRPO reinforcement learning), and
**NVIDIA LocateAnything-3B** (2026: parallel box decoding, about 2.5× faster than Rex-Omni). These
models handle referring expressions and reasoning ("the cup the person is about to pick up") that
closed-set detectors cannot express. They are still slower and less complete than dedicated
detectors on dense scenes (Chapter 21).

---

## COCO accuracy over time

| Year | Model | COCO AP | Split | Notes |
|---|---|:---:|---|---|
| 2015 | Faster R-CNN, VGG-16 | 21.9 | test-dev | two-stage, anchors |
| 2017 | RetinaNet, ResNet-101-FPN | 39.1 | test-dev | focal loss |
| 2018 | YOLOv3-608 | 33.0 | test-dev | 51 ms on Titan X |
| 2020 | EfficientDet-D7x | 55.1 | val | BiFPN, compound scaling |
| 2022 | DINO, Swin-L + Objects365 | 63.3 | test-dev | first DETR at the top |
| 2023 | Co-DETR (Co-DINO), ViT-L | 66.0 | test-dev | hybrid one-to-many auxiliary heads |
| 2026 | YOLO26x | 57.5 | val | 11.8 ms on T4, Objects365 pre-training |
| 2026 | YOLO27l *(preliminary)* | 60.4 | val | 2.32 ms on RTX PRO 6000; unreleased |

The real-time frontier is now roughly 6–8 AP behind the absolute frontier. In 2018 the gap was
6 AP *and* 100× the latency. Chapter 52 asks what happens when COCO saturates.

---

## Key Takeaways

- Every era removed a hand-designed component: features, proposals, the second stage, anchors, NMS,
  the fixed vocabulary. Each removal moved complexity into training.
- Faster R-CNN's anchors and IoU-threshold assignment set the template that later work kept refining.
- Focal loss explained the one-stage accuracy gap (class imbalance), and ATSS later explained the
  anchor-free gap (assignment).
- DETR's set prediction took four years and many papers to become trainable and fast. By 2024–2025
  real-time DETRs matched or beat YOLO at equal latency.
- YOLO and DETR are converging. YOLOv10/26 adopted one-to-one heads, and YOLO27's larger models use
  query-based decoders.
- Open-vocabulary and LLM-based detection add language. They trade speed and dense-scene recall for
  flexibility.

## Check Yourself

<details class="check"><summary>What did RoIAlign fix, and why does it matter for box AP rather than only masks?</summary>
RoI pooling rounded RoI coordinates and bin boundaries to integer feature-map cells. At stride 16 that
is up to ±8 pixels of misalignment. RoIAlign samples features bilinearly at exact positions, so box
refinement sees correctly aligned features. The improvement shows most at high IoU thresholds
(AP75), which COCO AP rewards.</details>

<details class="check"><summary>Why did DETR need ~500 epochs while Faster R-CNN needed ~12–36?</summary>
One-to-one Hungarian matching gives each ground truth exactly one positive query. The matching is
unstable early in training, because queries swap targets between iterations. Global attention also
has to learn sparse locality from scratch. Deformable attention (locality), anchor-box queries
(DAB), and denoising queries (DN/DINO) each fixed part of this.</details>

<details class="check"><summary>Name two things YOLO borrowed from DETR, and one thing DETR borrowed from YOLO.</summary>
YOLO borrowed the one-to-one, NMS-free head (YOLOv10, YOLO26) and, in YOLO27-M/L (preliminary), a
query-based decoder. Real-time DETRs borrowed YOLO-style CNN multi-scale encoders and heavy
deployment engineering (RT-DETR's hybrid encoder, D-FINE/DEIM's HGNetV2 backbones and T4 TensorRT
benchmarking).</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Rapid Object Detection using a Boosted Cascade of Simple Features | Viola & Jones, 2001 | CVPR 2001 | Integral image, cascade |
| Histograms of Oriented Gradients for Human Detection | Dalal & Triggs, 2005 | CVPR 2005 | HOG + SVM |
| Object Detection with Discriminatively Trained Part-Based Models | Felzenszwalb et al., 2010 | TPAMI 2010 | DPM, latent SVM |
| Rich Feature Hierarchies (R-CNN) | Girshick et al., 2014 | arXiv:1311.2524 | 53.3% VOC 2012 |
| Faster R-CNN | Ren et al., 2015 | arXiv:1506.01497 | RPN, anchors, 21.9 AP test-dev (VGG-16) |
| You Only Look Once | Redmon et al., 2016 | arXiv:1506.02640 | Grid formulation, 45 FPS, 63.4 mAP VOC07 |
| SSD | Liu et al., 2016 | arXiv:1512.02325 | Multi-scale default boxes |
| Feature Pyramid Networks | Lin et al., 2017 | arXiv:1612.03144 | Top-down pathway |
| Focal Loss for Dense Object Detection | Lin et al., 2017 | arXiv:1708.02002 | Imbalance, 39.1 AP test-dev |
| Mask R-CNN | He et al., 2017 | arXiv:1703.06870 | RoIAlign |
| YOLOv3: An Incremental Improvement | Redmon & Farhadi, 2018 | arXiv:1804.02767 | 33.0 AP at 608 |
| Cascade R-CNN | Cai & Vasconcelos, 2018 | arXiv:1712.00726 | Rising-IoU cascades |
| FCOS | Tian et al., 2019 | arXiv:1904.01355 | Anchor-free per-pixel regression |
| ATSS | Zhang et al., 2020 | arXiv:1912.02424 | Assignment explains anchor-free gap |
| EfficientDet | Tan et al., 2020 | arXiv:1911.09070 | BiFPN, D7x 55.1 AP |
| DETR | Carion et al., 2020 | arXiv:2005.12872 | Set prediction |
| DINO | Zhang et al., 2022 | arXiv:2203.03605 | 63.3 AP test-dev |
| DETRs with Collaborative Hybrid Assignments Training | Zong et al., 2023 | arXiv:2211.12860 | 66.0 AP test-dev |
| DETRs Beat YOLOs on Real-time Object Detection | Zhao et al., 2024 | arXiv:2304.08069 | RT-DETR |
| Grounding DINO | Liu et al., 2023 | arXiv:2303.05499 | 52.5 zero-shot COCO AP |
| Ultralytics docs: YOLO26 / YOLO27 | Ultralytics, 2026 | github.com/ultralytics/ultralytics docs/en/models | YOLO26 table; YOLO27 preliminary table |
| LocateAnything | NVIDIA, 2026 | research.nvidia.com/labs/lpr/locate-anything | Parallel box decoding |

---

**Next:** [Chapter 2 — The Detection Problem, Formally](./02_the_detection_problem.md) — before any
of these models can be compared, we need exact definitions of the output, the box, and the coordinate
system.
