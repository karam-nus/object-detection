---
title: "Chapter 14 — Classic Detectors"
---

[← Back to Table of Contents](./README.md)

# Chapter 14 — Classic Detectors

> *"Every modern detector is a classic one with its hand-tuned parts replaced."*

## Overview

Before the landscape splits into tiny, edge, large, transformer, open-vocabulary and LLM-based models,
this chapter covers the designs that set the template between 2015 and 2020: the R-CNN family, SSD,
RetinaNet, FCOS, CenterNet and EfficientDet. Each gets a fact block. Then the chapter asks the
practical question: **when is a classic detector still the right choice in 2026?** More often than
benchmark tables suggest, mainly for licensing, tooling and two-stage accuracy on small objects.

<div class="diagram">
<div class="diagram-title">The classic families</div>
<div class="flow-h">
  <div class="flow-node blue">Two-stage<small>Faster / Cascade / Mask R-CNN</small></div>
  <div class="flow-node green">One-stage anchor-based<small>SSD, RetinaNet</small></div>
  <div class="flow-node accent">One-stage anchor-free<small>FCOS, CenterNet</small></div>
  <div class="flow-node purple">Scaled efficient<small>EfficientDet</small></div>
</div>
</div>

---

## Faster R-CNN (and FPN)

| | |
|---|---|
| **Introduced** | Ren, He, Girshick, Sun, 2015 (arXiv:1506.01497, NeurIPS 2015); FPN: Lin et al., 2017 |
| **Lineage** | R-CNN → SPPNet → Fast R-CNN |
| **Paradigm** | two-stage, anchor-based (RPN) + RoI head |
| **Assignment** | RPN max-IoU 0.7 / 0.3; RoI head IoU 0.5 |
| **Post-processing** | NMS after RPN (≈1,000–2,000 proposals) and after the head |
| **COCO AP (sizes)** | R50-FPN 37.0; R50-FPN v2 recipe 46.7 (torchvision) |
| **Latency** | tens of ms on a V100 at 800×1333; not a real-time model |
| **Pre-training** | ImageNet |
| **License** | torchvision BSD-3, Detectron2 Apache-2.0, MMDetection Apache-2.0 |
| **Known failure modes** | slow; RoI ops hard to export to NPUs; crowds (two NMS stages) |

The RPN slides a small network over the feature map and scores $A$ anchors per location for
objectness, with box deltas. The top proposals after NMS are pooled (RoIAlign since Mask R-CNN) to 7×7,
then classified and refined. With FPN, proposals are pooled from the level matching their size
(Chapter 12).

**The 46.7 vs 37.0 lesson.** torchvision's "v2" weights use the *same architecture family* with a
modernised recipe (longer schedule, large-scale jitter, better normalisation) and gain almost 10 AP.
Recipe improvements routinely exceed architecture improvements, a recurring theme of Part IV.

---

## Cascade R-CNN

| | |
|---|---|
| **Introduced** | Cai & Vasconcelos, 2018 (arXiv:1712.00726) |
| **Lineage** | Faster R-CNN + FPN |
| **Paradigm** | multi-stage RoI heads |
| **Assignment** | stage $i$ trained at IoU 0.5 / 0.6 / 0.7 |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | about +2–4 AP over Faster R-CNN at equal backbone, mostly at high IoU |
| **Latency** | slower than Faster R-CNN (three heads) |
| **Pre-training** | ImageNet |
| **License** | Apache-2.0 (MMDetection, Detectron2) |
| **Known failure modes** | cost; diminishing returns beyond three stages |

A single head trained at IoU 0.5 produces boxes that are "good at 0.5" and cannot be trained at 0.7,
because there are too few positives that good. A cascade feeds each stage the previous stage's refined
boxes, so the distribution of training positives improves as the threshold rises. The idea of
progressive refinement reappears in every DETR decoder.

---

## SSD

| | |
|---|---|
| **Introduced** | Liu et al., 2016 (arXiv:1512.02325) |
| **Lineage** | MultiBox, YOLOv1 |
| **Paradigm** | one-stage, anchor ("default box") based, multi-scale |
| **Assignment** | max-IoU 0.5 + best-anchor rule; hard-negative mining at 3:1 |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | SSD300-VGG16 25.1; SSDLite320-MobileNetV3-L 21.3 (torchvision) |
| **Latency** | real-time on 2016 GPUs; SSDLite was the mobile default for years |
| **Pre-training** | ImageNet |
| **License** | BSD-3 (torchvision) |
| **Known failure modes** | small objects (shallow early features); hard-negative mining hyperparameters |

SSD predicted from six feature maps of decreasing resolution with default boxes at each. Before focal
loss, it handled imbalance with **hard-negative mining**: keep only the highest-loss negatives, at
3 negatives per positive. SSDLite (with depthwise convolutions) on MobileNet powered a generation of
mobile detection (TensorFlow Lite, Coral Edge TPU examples).

---

## RetinaNet

| | |
|---|---|
| **Introduced** | Lin, Goyal, Girshick, He, Dollár, 2017 (arXiv:1708.02002) |
| **Lineage** | FPN + SSD-style dense prediction |
| **Paradigm** | one-stage, anchor-based (9 anchors/location, P3–P7) |
| **Assignment** | max-IoU 0.5 / 0.4 |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | 39.1 test-dev (R101-FPN, 800px, paper); 41.5 (R50 v2 recipe, torchvision) |
| **Latency** | ~100 ms-class on 2017 GPUs at 800 px |
| **Pre-training** | ImageNet |
| **License** | Apache-2.0 / BSD-3 |
| **Known failure modes** | ~100k anchors; anchor hyperparameters |

The contribution is the loss, not the architecture: **focal loss** (Chapter 6) made a dense one-stage
detector match two-stage accuracy for the first time. Its shared decoupled heads and class-prior bias
initialisation are still standard (Chapter 13).

---

## FCOS

| | |
|---|---|
| **Introduced** | Tian, Shen, Chen, He, 2019 (arXiv:1904.01355) |
| **Lineage** | RetinaNet minus anchors |
| **Paradigm** | one-stage, anchor-free points |
| **Assignment** | points in box + level regression ranges + centre sampling |
| **Post-processing** | NMS on cls × centre-ness |
| **COCO AP (sizes)** | 39.2 (R50-FPN, torchvision) |
| **Latency** | similar to RetinaNet with 9× fewer candidates |
| **Pre-training** | ImageNet |
| **License** | BSD-3 (torchvision), 2-clause BSD (original) |
| **Known failure modes** | hand-set level ranges; centre-ness is a separate quality head |

FCOS showed that per-pixel $(l,t,r,b)$ regression plus centre-ness matches anchor-based detectors.
ATSS then showed *why*: the gap was in sample selection (Chapter 5). Its formulation became the
template for every modern anchor-free YOLO.

---

## CenterNet ("Objects as Points")

| | |
|---|---|
| **Introduced** | Zhou, Wang, Krähenbühl, 2019 (arXiv:1904.07850) |
| **Lineage** | CornerNet, pose heat maps |
| **Paradigm** | heat-map peaks at stride 4 + size/offset regression |
| **Assignment** | Gaussian splat at each object centre (penalty-reduced focal loss) |
| **Post-processing** | 3×3 max-pool peak extraction (NMS-free in practice) |
| **COCO AP (sizes)** | 28.1 (ResNet-18, 142 FPS) to 45.1 (Hourglass-104, multi-scale) per paper |
| **Latency** | very fast at the low end |
| **Pre-training** | ImageNet |
| **License** | MIT |
| **Known failure modes** | two objects sharing a centre collide; stride-4 output is memory-heavy |

CenterNet is notable for being **effectively NMS-free in 2019**, and for extending naturally to 3D
boxes and pose. Its centroid view lives on in MCU detectors that output only object centres
(Chapter 15).

---

## EfficientDet

| | |
|---|---|
| **Introduced** | Tan, Pang, Le, 2020 (arXiv:1911.09070) |
| **Lineage** | EfficientNet backbones + RetinaNet-style head |
| **Paradigm** | one-stage, anchor-based, BiFPN neck |
| **Assignment** | max-IoU |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | D0 34.6 (3.9 M, 2.5 GFLOPs) … D7x 55.1 (77 M, 410 GFLOPs) |
| **Latency** | low FLOPs but slower on GPUs than FLOP-equivalent YOLOs (depthwise convs, many small ops) |
| **Pre-training** | ImageNet |
| **License** | Apache-2.0 |
| **Known failure modes** | FLOPs ≠ latency; long training schedules |

EfficientDet's **compound scaling** grows input resolution, BiFPN depth/width and backbone together
with one coefficient. It was the FLOP-efficiency leader in 2020. It also became the textbook example
that **FLOPs do not predict GPU latency**: depthwise convolutions and many small layers are
memory-bound, so YOLOv4/v5 at higher FLOPs ran faster on the same GPU (Chapter 46).

---

## The classics in one table

| Model | Year | COCO AP | Params | Paradigm | Choose this when |
|---|:---:|:---:|:---:|---|---|
| **Faster R-CNN R50-FPN v2** | 2015/2021 | 46.7 | 43.7 M | two-stage | You need a permissively licensed, well-understood baseline in torchvision |
| **Cascade R-CNN** | 2018 | ≈ +3 over FRCNN | — | multi-stage | High-IoU accuracy offline; competition ensembles |
| **SSDLite-MobileNetV3** | 2019 | 21.3 | 3.4 M | one-stage anchors | Legacy mobile pipelines; extremely constrained runtimes with SSD support |
| **RetinaNet R50 v2** | 2017/2021 | 41.5 | 38.2 M | one-stage anchors | Research baseline for loss/assignment studies |
| **FCOS R50** | 2019 | 39.2 | 32.3 M | anchor-free | Anchor-free baseline in torchvision |
| **CenterNet** | 2019 | 28–45 | — | heat map | Centre-point tasks, 3D/pose extensions |
| **EfficientDet D0–D7x** | 2020 | 34.6–55.1 | 3.9–77 M | BiFPN | TPU/CPU settings where FLOPs track latency |

COCO numbers are val2017 from torchvision's model zoo and the EfficientDet README (see
[Appendix C](./appendix_c_model_index.md)).

---

## When is a classic detector still right?

1. **Licensing.** torchvision (BSD-3), Detectron2 and MMDetection (Apache-2.0) are permissive.
   Ultralytics YOLOs are AGPL-3.0 unless you buy an enterprise license (Chapter 35). For a closed-source
   product without a budget for licenses, a permissive detector is a legal requirement, not a
   preference. Today, permissive *real-time* options also exist (RT-DETR, D-FINE, DEIM, RF-DETR N–L,
   RTMDet, YOLOX), so the classic choice is less forced than it was.
2. **Two-stage precision on small objects at high resolution.** RoI-pooled second stages re-examine
   each candidate at higher effective resolution. On some high-resolution inspection and medical tasks,
   Faster/Cascade R-CNN with FPN at 1,333 px remains competitive with real-time detectors run at 640.
3. **Ecosystem integration.** Detectron2 and MMDetection hold years of research code: panoptic,
   long-tail losses, semi-supervised teachers. Building on them can be the fastest route to a research
   result.
4. **Pedagogy and debugging.** The components are explicit: proposals, RoIs, deltas. When a modern
   detector misbehaves, reproducing the problem with a classic one helps separate data issues from
   model issues.

When a classic detector is **not** right: real-time edge deployment (RoI ops, NMS twice, dynamic
shapes), and anything where a 2026 real-time model gives 10+ AP more at a fraction of the latency.

---

## Key Takeaways

- Faster R-CNN set the template: anchors, IoU-threshold assignment, two-stage refinement, RoI pooling.
- Cascades fix the training distribution at high IoU thresholds. Progressive refinement lives on in
  DETR decoders.
- SSD and RetinaNet made one-stage detection viable. Focal loss, not architecture, closed the accuracy
  gap.
- FCOS and CenterNet removed anchors. ATSS explained that assignment, not anchors, was what mattered.
- EfficientDet is the canonical case of FLOPs failing to predict GPU latency.
- Classic detectors remain right for permissive licensing, some high-resolution small-object tasks,
  research ecosystems and debugging. They are wrong for real-time edge deployment.

## Check Yourself

<details class="check"><summary>Why does SSD need hard-negative mining while RetinaNet does not?</summary>
Both face extreme foreground–background imbalance. SSD handles it by sampling: keep only the
highest-loss negatives at 3:1. RetinaNet handles it in the loss: focal loss down-weights easy
negatives continuously, so all anchors can be used without sampling hyperparameters.</details>

<details class="check"><summary>EfficientDet-D2 has fewer FLOPs than YOLOv5s but runs slower on a GPU. Why?</summary>
GPU latency depends on memory traffic and kernel efficiency as well as arithmetic. EfficientDet's
depthwise convolutions and many small BiFPN operations have low arithmetic intensity: they are
memory-bound and launch many kernels. YOLOv5's dense 3×3 convolutions run near peak throughput.</details>

<details class="check"><summary>torchvision's Faster R-CNN gained about 10 AP between its v1 and v2 weights. What changed?</summary>
The training recipe, not the basic architecture: a longer schedule, large-scale jitter
augmentation, improved normalisation and initialisation, and minor head changes. Recipe gains of this
size are common, so comparisons across papers with different recipes are unreliable.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Faster R-CNN | Ren et al., 2015 | arXiv:1506.01497 | RPN, anchors, two-stage design |
| Feature Pyramid Networks | Lin et al., 2017 | arXiv:1612.03144 | FPN for two-stage |
| Cascade R-CNN | Cai & Vasconcelos, 2018 | arXiv:1712.00726 | Rising-IoU cascades |
| Mask R-CNN | He et al., 2017 | arXiv:1703.06870 | RoIAlign |
| SSD | Liu et al., 2016 | arXiv:1512.02325 | Default boxes, hard-negative mining |
| Searching for MobileNetV3 | Howard et al., 2019 | arXiv:1905.02244 | SSDLite |
| Focal Loss (RetinaNet) | Lin et al., 2017 | arXiv:1708.02002 | 39.1 AP test-dev |
| FCOS | Tian et al., 2019 | arXiv:1904.01355 | Anchor-free points |
| Objects as Points | Zhou et al., 2019 | arXiv:1904.07850 | CenterNet speed/accuracy range |
| EfficientDet | Tan et al., 2020 | arXiv:1911.09070 + google/automl README | D0–D7x numbers |
| torchvision detection model zoo | PyTorch | pytorch.org/vision/stable/models | COCO AP, params, GFLOPs of R-CNN, RetinaNet, FCOS, SSD |
| Detectron2; MMDetection | Meta 2019; OpenMMLab 2019 | github.com/facebookresearch/detectron2; open-mmlab/mmdetection | Ecosystems and licenses |

---

**Next:** [Chapter 15 — Very Tiny Detectors](./15_tiny_detectors.md) — the opposite end of the scale:
detection under one million parameters and half a megabyte of SRAM.
