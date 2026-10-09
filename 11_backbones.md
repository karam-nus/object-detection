---
title: "Chapter 11 — Backbones"
---

[← Back to Table of Contents](./README.md)

# Chapter 11 — Backbones

> *"The head decides how to read the features. The backbone decides what there is to read."*

## Overview

The backbone turns pixels into a pyramid of feature maps, conventionally $C_3, C_4, C_5$ at strides
8, 16 and 32. It holds most of a detector's parameters, sets most of its accuracy ceiling, and is
usually where the latency goes. This chapter covers what detection needs from a backbone (and how that
differs from classification), the backbone families in detection from ResNet to DINOv3, the design
patterns that keep recurring (CSP splits, re-parameterisation, depthwise convolutions, windowed
attention), and how pre-training moved from ImageNet labels to self-supervised foundation models.

<div class="diagram">
<div class="diagram-title">Backbone lineages used in detection</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card blue"><div class="card-title">Residual CNNs</div><div class="card-desc">ResNet, ResNeXt, ResNet-vd → two-stage detectors, RT-DETR-R18/R50/R101</div></div>
  <div class="diagram-card accent"><div class="card-title">YOLO CNNs</div><div class="card-desc">Darknet → CSPDarknet → C3 → ELAN / C2f → GELAN → C3k2 → A2C2f</div></div>
  <div class="diagram-card green"><div class="card-title">Mobile & NAS CNNs</div><div class="card-desc">MobileNet, ShuffleNet, GhostNet, ESNet, LCNet, HGNetV2, TinyNAS, CSPNeXt</div></div>
  <div class="diagram-card purple"><div class="card-title">Transformers & foundation models</div><div class="card-desc">Swin, PVT, ConvNeXt, InternImage, ViT/ViTDet, EVA-02, DINOv2/v3, PE</div></div>
</div>
</div>

---

## What detection needs from features

| Requirement | Why | Consequence for design |
|---|---|---|
| **Semantics at high resolution** | Small objects live on stride-8 maps, which are shallow in a plain CNN | Necks (Chapter 12) push semantics down; detection-specific backbones keep more channels early |
| **Multi-scale outputs** | Objects span 10–500 px | Hierarchical stages; plain ViTs need a synthetic pyramid |
| **Precise spatial alignment** | Box edges must be located to a few pixels | Avoid aggressive early downsampling; deformable sampling helps |
| **Cost at 640–1,333 px inputs** | FLOPs grow with pixel count | Global attention is quadratic in tokens, so detection uses windowed or deformable attention |
| **Deployability** | NPUs, INT8, static shapes | ReLU-family activations, plain convolutions, re-parameterisable blocks |

A backbone ranked by ImageNet top-1 at 224 × 224 is not ranked by any of these. That is why
detection-specific backbones (CSPDarknet, ELAN, HGNetV2, CSPNeXt) exist, and why foundation models
had to be *adapted* (ViTDet, windowed DINOv2 in RF-DETR, distilled DINOv3 in DEIMv2) before they paid
off in real time.

---

## Residual networks

**ResNet** (He et al., 2016) made depth trainable with identity shortcuts. ResNet-50 (25.6 M
parameters, about 4.1 GFLOPs at 224²) was the default detection backbone from Faster R-CNN with FPN
through 2020. **ResNet-vd** (Baidu's "Bag of Tricks" variant) moves the downsampling stride in the
residual branch from the 1×1 to the 3×3 convolution and replaces the shortcut's strided 1×1 with
average pooling + 1×1. It is used by PP-YOLO and by RT-DETR's R18/R34/R50/R101 variants.

**ResNeXt** adds grouped convolutions, and Res2Net adds hierarchical multi-scale residuals. Both
appear in older leaderboard entries.

---

## The YOLO line: CSP and its descendants

<div class="timeline">
  <div class="timeline-item"><div class="timeline-year">2017–18</div><div class="timeline-title">Darknet-19 / Darknet-53</div><div class="timeline-desc">3×3 / 1×1 alternation, BatchNorm, LeakyReLU; residual blocks in Darknet-53 (YOLOv3)</div></div>
  <div class="timeline-item"><div class="timeline-year">2019–20</div><div class="timeline-title">CSPNet → CSPDarknet-53</div><div class="timeline-desc">Split channels; send one part through the blocks; concatenate. Cuts duplicated gradient flow and ~20% of compute (YOLOv4)</div></div>
  <div class="timeline-item"><div class="timeline-year">2020</div><div class="timeline-title">YOLOv5 C3</div><div class="timeline-desc">CSP bottleneck with 3 convs; SiLU; Focus stem later replaced by a 6×6 stride-2 conv</div></div>
  <div class="timeline-item"><div class="timeline-year">2022</div><div class="timeline-title">ELAN / E-ELAN</div><div class="timeline-desc">Aggregate outputs of many stacked convs with controlled gradient paths (YOLOv7)</div></div>
  <div class="timeline-item"><div class="timeline-year">2023</div><div class="timeline-title">C2f</div><div class="timeline-desc">Keep every bottleneck output and concatenate all of them (YOLOv8)</div></div>
  <div class="timeline-item"><div class="timeline-year">2024</div><div class="timeline-title">GELAN; C3k2 + C2PSA</div><div class="timeline-desc">Generalised ELAN with any computational block (YOLOv9); C2f with smaller C3k inner blocks and a position-sensitive attention block (YOLO11, kept in YOLO26)</div></div>
  <div class="timeline-item"><div class="timeline-year">2025</div><div class="timeline-title">R-ELAN / A2C2f</div><div class="timeline-desc">Residual ELAN with area attention (YOLOv12)</div></div>
</div>

The pattern is consistent: **split, transform part of the features through a stack, keep the
intermediate results, concatenate, fuse with a 1×1 convolution.** Keeping intermediate outputs gives
many gradient paths of different lengths, which is the "gradient path planning" argument of the
CSP, ELAN and GELAN papers. It is also cheap, because half the channels skip the expensive stack.
Chapter 29 draws every one of these blocks with its tensor shapes. `odlab.model.C2f` is a 20-line
version.

**Focus / space-to-depth.** YOLOv5's original stem sliced the image into four pixel-interleaved
sub-images and stacked them as channels: $[3, 640, 640] \to [12, 320, 320]$. This halves resolution
without discarding information. It was replaced by an equivalent 6×6 stride-2 convolution, which
exports better. Space-to-depth reappears in tiny-object work (SPD-Conv).

---

## Structural re-parameterisation

**RepVGG** (Ding et al., 2021) trains a block with three parallel branches (3×3 conv + BN, 1×1 conv
+ BN, identity + BN) and **merges them into a single 3×3 convolution at inference**:

1. Fold each BN into its conv: $W' = \frac{\gamma}{\sigma} W$, $b' = \beta - \frac{\gamma \mu}{\sigma}$.
2. Pad the 1×1 kernel to 3×3 (centre tap). Write the identity as a 3×3 kernel with 1 at the centre of
   its own channel.
3. Sum the three kernels and biases.

Convolution is linear, so the merged block computes exactly the same function. Training gets the
optimisation benefits of a multi-branch network. Inference gets a plain stack of 3×3 convs, which is
the fastest pattern on GPUs and NPUs. **EfficientRep** (YOLOv6) and **RepConv** (YOLOv7, YOLOv6,
Gold-YOLO) build on it.

**Failure mode: quantisation.** The merged kernel's weight and activation distributions can be
INT8-hostile, because the branches have very different scales. **QARepVGG** (Chu et al., 2023) changes
the BN placement so that the fused block quantises well. YOLO-NAS and YOLOv6 3.0 use it. YOLO-NAS
reports INT8 losses of 0.12–0.55 AP across its S/M/L models (Chapter 45).

---

## Mobile and NAS backbones

| Backbone | Key idea | Detectors | Deployment note |
|---|---|---|---|
| **MobileNetV2 / V3** | depthwise-separable convs, inverted residuals; V3 adds SE + h-swish via NAS | SSDLite, early mobile detectors | Depthwise convs are memory-bound; h-swish needs NPU support |
| **MobileNetV4** | Universal Inverted Bottleneck; tuned for many mobile accelerators | recent mobile work | Designed for cross-hardware efficiency |
| **ShuffleNetV2** | channel split + shuffle; guidelines on memory access cost | NanoDet | Good on ARM CPUs |
| **GhostNet** | cheap linear "ghost" features from a few real ones | edge YOLO variants | — |
| **ESNet / LCNet** | Baidu's mobile backbones | PicoDet, PP-YOLOE+ tiny | Paddle Lite / ARM optimised |
| **HGNetV2** | Baidu's GPU-oriented high-performance net (light-weight stages, learnable affine blocks) | RT-DETR-L/X, D-FINE, DEIM, DEIMv2 N/Atto–Pico | Strong T4 TensorRT latency |
| **TinyNAS / MAE-NAS** | latency-constrained NAS | DAMO-YOLO | Searched per latency budget |
| **CSPNeXt** | CSP with 5×5 depthwise large kernels | RTMDet | Large kernels, efficient on GPU |
| **AutoNAC search** | hardware-aware NAS with quantisation-friendly blocks | YOLO-NAS | INT8-friendly |

Chapter 16 discusses which of these survive on which edge accelerator.

---

## Transformers and hierarchical attention backbones

**Swin Transformer** (Liu et al., 2021) restores a CNN-like hierarchy, with patch merging to strides
4–32, and computes self-attention in **shifted local windows**, so cost is linear in image size. Swin-L
became the standard backbone for COCO leaderboard entries (DINO, HTC++). **PVT** and **PVTv2** use
spatial-reduction attention. **ConvNeXt** (2022) showed that a CNN modernised with transformer-era
design choices (large kernels, LayerNorm, GELU, fewer activations) matches Swin. **InternImage** (2023)
scales deformable convolution (DCNv3) into a large backbone and briefly topped COCO.

### Plain ViTs: ViTDet

A plain ViT has a single stride-16 feature map and quadratic global attention. **ViTDet** (Li et al.,
2022) showed that a plain ViT can still be a strong detection backbone if you:

1. build a **simple feature pyramid** from the last map alone, with deconvolutions for stride 8 and 4
   and pooling for stride 32, instead of an FPN over stages (Chapter 12);
2. use **windowed attention** in most blocks and a few global blocks for propagation;
3. pre-train with **MAE** (masked autoencoding) rather than supervised ImageNet labels.

This decoupled detection from hierarchical backbone design and opened the way for foundation models.

### Foundation-model backbones: DINOv2, DINOv3, PE

Self-supervised ViTs trained on very large curated image sets produce features that transfer to
detection with little adaptation:

| Backbone | Used by | How it is adapted |
|---|---|---|
| **DINOv2** (2023) | RF-DETR N–2XL | Small ViT with windowed attention interleaved with global blocks (from LW-DETR), multi-scale projector; the paper credits the DINOv2 pre-training for strong fine-tuning on RF100-VL |
| **DINOv3** (2025) | DEIMv2 S/M/L/X | DINOv3 features with a Spatial Tuning Adapter (STA) that builds the multi-scale pyramid; smaller DEIMv2 sizes use distilled/pruned ViTs or HGNetV2 |
| **Perception Encoder (PE-Core-T)** | RF-DETR Atto/Femto/Pico | Tiny encoder from Meta's PE family |
| **VFM as teacher only** | RT-DETRv4, EdgeCrafter | Distil foundation features into a small CNN or ViT; the foundation model is not deployed |

This is the clearest trend of 2025–2026: the real-time accuracy frontier moved because of
**pre-training**, not because of new detection heads. RT-DETRv4 and EdgeCrafter show the cheaper
variant: use the foundation model as a teacher and ship a compact student (Chapter 42).

---

## Pre-training regimes

| Regime | Examples | Notes |
|---|---|---|
| **ImageNet-1k supervised** | ResNet, CSPDarknet in YOLOv4, HGNetV2 | The classic default |
| **ImageNet-22k supervised** | Swin-L, ConvNeXt-L in leaderboard detectors | More data, larger models |
| **From scratch on COCO** | Ultralytics YOLOv5–YOLO11 detection checkpoints | 300–500 epochs, strong augmentation; He et al. (2019) showed scratch training can match ImageNet init with long enough schedules |
| **Detection pre-training (Objects365)** | YOLO26, LW-DETR, D-FINE/DEIM (O365 variants), RF-DETR, DINO | Pre-trains backbone **and** neck/head; typically +2–4 AP on COCO |
| **Self-supervised (MAE, DINOv2/v3)** | ViTDet, RF-DETR, DEIMv2 | Best transfer to new domains |
| **Image–text (CLIP-style)** | OWL-ViT, YOLO-World text towers, Grounding DINO text encoder | Needed for open vocabulary (Chapter 20) |

---

## Choosing a backbone

| If you need… | Prefer | Avoid |
|---|---|---|
| INT8 on an NPU | plain or re-parameterised convs, ReLU (QARepVGG, RepVGG-style, ReLU YOLO variants) | attention with softmax on odd axes, h-swish without hardware support, depthwise-heavy nets on NPUs with poor depthwise support |
| Best accuracy per T4 millisecond | HGNetV2-, CSP- or DINO-based real-time detectors | ResNet-50 (slow per AP point) |
| Transfer to unusual domains with little data | DINOv2/v3-based (RF-DETR, DEIMv2) or Objects365-pretrained | from-scratch COCO checkpoints |
| ARM CPU | ShuffleNet / ESNet / LCNet (NanoDet, PicoDet) | large-kernel depthwise on CPUs without good kernels |
| Maximum accuracy, latency irrelevant | Swin-L, ViT-L / EVA-02, InternImage with Objects365 | — |

---

## Key Takeaways

- Detection needs semantics at high resolution, multi-scale outputs, spatial precision and
  affordable cost at large inputs. ImageNet ranking at 224² predicts none of these.
- The YOLO backbone line is one idea refined for eight years: split channels, stack blocks, keep
  intermediate outputs, concatenate, fuse.
- Re-parameterisation trains multi-branch blocks and deploys single 3×3 convs. It can hurt INT8, which
  QARepVGG addresses.
- Plain ViTs work for detection with a simple pyramid, windowed attention and MAE-style pre-training
  (ViTDet). That opened the door to DINOv2/v3 backbones.
- The 2025–2026 real-time gains (RF-DETR, DEIMv2, RT-DETRv4, EdgeCrafter) come mostly from
  foundation-model pre-training or distillation.
- Objects365 detection pre-training is worth several COCO AP points. Always check whether a
  comparison controls for it.

## Check Yourself

<details class="check"><summary>Write the merged 3×3 kernel for a RepVGG block whose 3×3 branch has kernel K₃ (after BN folding), whose 1×1 branch has kernel K₁, and whose identity branch has BN scale γ/σ.</summary>
K = K₃ + pad(K₁) + I·(γ/σ), where pad(K₁) places the 1×1 weights at the centre of a 3×3 kernel. I
is a 3×3 kernel with 1 at the centre for input channel = output channel and 0 elsewhere. The biases
(from BN folding) add likewise. Because convolution is linear, the merged block is exactly
equivalent.</details>

<details class="check"><summary>Why can't you simply drop a DINOv2 ViT-L into YOLO's backbone slot?</summary>
It produces a single stride-14 or stride-16 map, not a P3–P5 pyramid. Global attention at 640² is
expensive (about 2,000 tokens at patch 14, with quadratic attention). Its latency is far above a YOLO
budget. Detectors that use it build a pyramid from one map (ViTDet-style), use windowed attention
and a small model (RF-DETR), add an adapter (DEIMv2's STA), or use it only as a distillation teacher
(RT-DETRv4).</details>

<details class="check"><summary>A paper reports +3 AP from a "new backbone" but its baseline used ImageNet init and the new model used Objects365 pre-training. What can you conclude?</summary>
Nothing about the backbone yet. Objects365 pre-training alone is worth roughly 2–4 AP on COCO (for
example D-FINE-X goes from 55.8 to 59.3). The comparison confounds architecture with pre-training
data. Ask for both models under the same pre-training.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Deep Residual Learning | He et al., 2016 | arXiv:1512.03385 | ResNet |
| Bag of Tricks for Image Classification | He et al., 2019 | arXiv:1812.01187 | ResNet-vd |
| CSPNet | Wang et al., 2020 | arXiv:1911.11929 | Cross-stage partial design, ~20% compute reduction |
| YOLOv4; YOLOv7; YOLOv9 | Bochkovskiy 2020; Wang 2022; Wang 2024 | arXiv:2004.10934; 2207.02696; 2402.13616 | CSPDarknet, E-ELAN, GELAN |
| RepVGG | Ding et al., 2021 | arXiv:2101.03697 | Structural re-parameterisation |
| Make RepVGG Greater Again (QARepVGG) | Chu et al., 2023 | arXiv:2212.01593 | Quantisation-friendly re-param |
| MobileNetV2 / V3 / V4 | Sandler 2018; Howard 2019; Qin 2024 | arXiv:1801.04381; 1905.02244; 2404.10518 | Mobile blocks |
| ShuffleNetV2 | Ma et al., 2018 | arXiv:1807.11164 | Memory-access guidelines |
| Swin Transformer | Liu et al., 2021 | arXiv:2103.14030 | Shifted windows |
| ConvNeXt | Liu et al., 2022 | arXiv:2201.03545 | Modernised CNN |
| InternImage | Wang et al., 2023 | arXiv:2211.05778 | DCNv3 backbone |
| ViTDet | Li et al., 2022 | arXiv:2203.16527 | Plain ViT + simple pyramid |
| Rethinking ImageNet Pre-training | He et al., 2019 | arXiv:1811.08883 | Scratch training can match |
| DINOv2; DINOv3 | Oquab et al., 2023; Siméoni et al., 2025 | arXiv:2304.07193; 2508.10104 | Foundation backbones |
| RF-DETR; DEIMv2; RT-DETRv4; EdgeCrafter | Roboflow 2025; Huang 2025; Liao 2025; Intellindust 2026 | arXiv:2511.09554; 2509.20787; 2510.25257; 2603.18739 | Foundation-model use in real-time detectors |
| D-FINE README | Peng et al., 2024 | github.com/Peterande/D-FINE | 55.8 → 59.3 with Objects365 |

---

**Next:** [Chapter 12 — Necks](./12_necks.md) — the backbone produces a pyramid; the neck decides how
information flows between its levels.
