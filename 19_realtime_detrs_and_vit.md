---
title: "Chapter 19 — Real-Time DETRs & ViT Backbones"
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — Real-Time DETRs & ViT Backbones

> *"'DETRs Beat YOLOs' was a provocative title in 2023. By 2026 it is a plot."*

## Overview

In 2023 Baidu's **RT-DETR** made a DETR competitive with YOLO at equal T4 latency. Within three years a
family of real-time DETRs (RT-DETR v2–v4, LW-DETR, D-FINE, DEIM, DEIMv2, RF-DETR, EdgeCrafter) moved
ahead of CNN YOLOs on the COCO accuracy–latency plot, all NMS-free and mostly Apache-2.0. YOLO27's
preview responds by making its M and L detectors query-based (preliminary). This chapter explains each
step: what was changed, what it bought, and what it costs on hardware. The single most important trend
is the last one: **foundation-model backbones and distillation**.

<div class="diagram">
<div class="diagram-title">The real-time DETR lineage</div>
<div class="flow">
  <div class="flow-node blue wide">RT-DETR (2023) <small>hybrid encoder, IoU-aware query selection, adjustable decoder</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">RT-DETRv2 / v3 (2024) <small>selective sampling, discrete sampler; dense positive supervision</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">LW-DETR (2024) <small>plain ViT encoder (window + global attention), Objects365 pre-training</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">D-FINE (2024) → DEIM (2024) <small>distribution refinement + self-distillation; dense one-to-one + MAL</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">RF-DETR (2025) · DEIMv2 (2025) · RT-DETRv4 (2025) · EdgeCrafter (2026) <small>DINOv2/v3 backbones, NAS, VFM distillation</small></div>
</div>
</div>

<div class="lab" data-lab="atlas"></div>

---

## RT-DETR — the first real-time DETR

| | |
|---|---|
| **Introduced** | Zhao et al. (Baidu), 2023 (arXiv:2304.08069, CVPR 2024: "DETRs Beat YOLOs on Real-time Object Detection") |
| **Lineage** | DINO / Deformable DETR + PP-YOLOE-style engineering |
| **Paradigm** | query-based, NMS-free |
| **Assignment** | Hungarian; VFL classification |
| **Post-processing** | top-300 queries |
| **COCO AP (sizes)** | R18 46.5 · R34 48.9 · R50 53.1 · R101 54.3 · HGNetv2-L 53.0 · X 54.8 |
| **Latency** | 217 / 161 / 108 / 74 FPS on T4 TensorRT FP16 for R18 / R34 / R50 / R101 |
| **Pre-training** | ImageNet (+ Objects365 variants: R50 55.3, R101 56.2) |
| **License** | Apache-2.0 |
| **Known failure modes** | `grid_sample` in deformable attention (export, INT8); 72-epoch-class schedules |

Three ideas made it real-time:

1. **Efficient hybrid encoder.** Self-attention only on the stride-32 map (AIFI), with convolutional
   cross-scale fusion (CCFF) for the rest (Chapter 12). The encoder was where Deformable DETR spent its
   time.
2. **IoU-aware query selection.** The top-$k$ encoder tokens become the initial queries. Training
   pushes their class scores to reflect IoU, the same idea as TAL/VFL, so the selected tokens are the
   well-localised ones.
3. **Adjustable decoder depth.** Inference can use fewer decoder layers than training without
   retraining, trading AP for latency on the fly.

---

## RT-DETRv2 and RT-DETRv3 — bags of freebies

**RT-DETRv2** (2024) adds *selective multi-scale sampling* (a different number of deformable sampling
points per scale), an optional **discrete sampling operator** that replaces `grid_sample` for
deployment-friendliness (AP cost 0.1–0.6 by the README's numbers), dynamic data augmentation, and
scale-adaptive hyperparameters. It gains +0.1 to +1.6 AP over v1 at identical latency: S (R18) 48.1,
M 49.9/51.9, L 53.4, X 54.3.

**RT-DETRv3** (2024) attacks DETR's sparse supervision: a CNN **auxiliary branch** with dense
one-to-many supervision for the encoder, **self-attention perturbation** to diversify which queries
match each ground truth across query groups, and a **shared-weight decoder branch** for dense positive
supervision. All of it is training-only. R18 reaches 48.1 AP (+1.6 over RT-DETR) at the same latency.

---

## LW-DETR — a plain ViT for real time

| | |
|---|---|
| **Introduced** | Chen et al. (Baidu), 2024 (arXiv:2406.03459) |
| **Lineage** | ViTDet-style plain ViT + DETR decoder |
| **Paradigm** | query-based, NMS-free |
| **Assignment** | Hungarian + group DETR; IoU-aware classification loss |
| **Post-processing** | top-k |
| **COCO AP (sizes)** | tiny 42.6 · small 48.0 · medium 52.5 · large 56.1 · xlarge 58.3 |
| **Latency** | 2.0 / 2.9 / 5.6 / 8.8 / 19.1 ms (T4 TensorRT FP16) |
| **Pre-training** | ViT pre-trained (MIM), then **whole detector pre-trained on Objects365** |
| **License** | Apache-2.0 |
| **Known failure modes** | depends on O365 pre-training for its numbers |

LW-DETR is "a simple stack of a ViT encoder, a projector, and a shallow DETR decoder" (its README). Its
encoder **interleaves window attention and global attention**, and organises feature maps "window-major"
so window attention is cheap. Its main result: with Objects365 pre-training, a plain ViT DETR beats the
YOLOs of 2024 at equal latency. RF-DETR is built directly on it.

---

## D-FINE — regression as distribution refinement

| | |
|---|---|
| **Introduced** | Peng et al. (USTC), 2024 (arXiv:2410.13842, ICLR 2025 spotlight) |
| **Lineage** | RT-DETR + GFL/DFL ideas |
| **Paradigm** | query-based, NMS-free |
| **Assignment** | Hungarian |
| **Post-processing** | top-k |
| **COCO AP (sizes)** | N 42.8 · S 48.5 · M 52.3 · L 54.0 · X 55.8 (COCO); S 50.7 · M 55.1 · L 57.3 · X 59.3 (O365 → COCO) |
| **Latency** | 2.12 / 3.49 / 5.62 / 8.07 / 12.89 ms (T4 TensorRT FP16) |
| **Pre-training** | HGNetV2 (ImageNet); optional Objects365 |
| **License** | Apache-2.0 |
| **Known failure modes** | many hyperparameters for FDR bins; the original repo is research-grade |

Two contributions, both with zero inference cost:

- **Fine-grained Distribution Refinement (FDR).** Each decoder layer predicts a residual *probability
  distribution* over offsets for each box edge, relative to the previous layer's box. The bins are spaced
  non-uniformly by a weighting function, dense near zero for fine corrections and sparse far away for
  coarse ones. It is DFL (Chapter 6) turned into iterative refinement.
- **Global Optimal Localization Self-Distillation (GO-LSD).** The last layer's refined distributions
  are distilled into earlier layers through a decoupled distillation focal loss. Earlier layers learn to
  predict what the final layer will conclude, which speeds convergence and improves every layer.

D-FINE at N–X sizes became the most-used permissive real-time DETR baseline of 2025.

---

## DEIM — more positives for one-to-one training

| | |
|---|---|
| **Introduced** | Huang et al. (Intellindust), 2024 (arXiv:2412.04234, CVPR 2025) |
| **Lineage** | training framework applied to D-FINE and RT-DETRv2 |
| **Paradigm** | query-based, NMS-free |
| **Assignment** | Hungarian with **Dense O2O** |
| **Post-processing** | top-k |
| **COCO AP (sizes)** | DEIM-D-FINE: N 43.0 · S 49.0 · M 52.7 · L 54.7 · X 56.5 |
| **Latency** | unchanged from the base model (2.12 … 12.89 ms on T4) |
| **Pre-training** | as base model |
| **License** | Apache-2.0 |
| **Known failure modes** | gains are training-side; no architecture change to inspect |

DEIM's diagnosis: one-to-one matching starves DETRs of positives. **Dense O2O** keeps strict one-to-one
matching but packs more *objects* into each training image with mosaic-style composition, so more
queries get positive targets. The **matchability-aware loss (MAL)** reweights the classification target
by IoU so low-quality matches are not over-rewarded. DEIM reports that D-FINE trains to higher accuracy
in roughly half the epochs, and it lifts every size by +0.2 to +0.7 AP over D-FINE.

---

## DEIMv2 — DINOv3 features at every size

| | |
|---|---|
| **Introduced** | Huang et al. (Intellindust), 2025 (arXiv:2509.20787) |
| **Lineage** | DEIM + DINOv3 |
| **Paradigm** | query-based, NMS-free |
| **Assignment** | Dense O2O + MAL |
| **Post-processing** | top-k |
| **COCO AP (sizes)** | Atto 23.8 · Femto 31.0 · Pico 38.5 · N 43.0 · S 50.9 · M 53.0 · L 56.0 · X 57.8 |
| **Latency** | 1.10 / 1.45 / 2.13 / 2.32 / 5.78 / 8.80 / 10.47 / 13.75 ms (T4 TensorRT FP16; README notes TensorRT ≥ 10.6 needed for correct FP16) |
| **Pre-training** | DINOv3 (S–X; S/M use distilled ViTs); HGNetV2 for Atto–N |
| **License** | Apache-2.0 |
| **Known failure modes** | FP16 correctness depends on TensorRT version; larger sizes heavier than CNN peers per ms |

The **Spatial Tuning Adapter (STA)** turns DINOv3's single-scale ViT features into the multi-scale
pyramid the decoder needs, and adds fine-grained detail. DEIMv2 spans eight sizes, from 0.5 M to
50.3 M parameters. Its sub-2 M models are discussed in Chapter 15. Its S model (9.7 M) reaches 50.9 AP
with COCO-only training, about what 50 M-parameter CNNs reached in 2022.

---

## RF-DETR — a DINOv2 DETR specialised by NAS

| | |
|---|---|
| **Introduced** | Robinson et al. (Roboflow), 2025 (arXiv:2511.09554, ICLR 2026) |
| **Lineage** | LW-DETR with a DINOv2 backbone (PE-Core-T for Atto/Femto/Pico) |
| **Paradigm** | query-based, NMS-free; detection, segmentation, keypoints (preview) |
| **Assignment** | Hungarian (LW-DETR style) |
| **Post-processing** | top-k |
| **COCO AP (sizes)** | A 30.5 · F 37.8 · P 41.6 · N 48.4 · S 53.0 · M 54.7 · L 56.5 · XL 58.6 · 2XL 60.1 |
| **Latency** | 1.0 / 1.4 / 1.7 / 2.3 / 3.5 / 4.4 / 6.8 / 11.5 / 17.2 ms (T4 TensorRT FP16, Roboflow's harness) |
| **Pre-training** | DINOv2 / PE + Objects365 |
| **License** | Apache-2.0 for N–L; Roboflow PML-1.0 for A/F/P/XL/2XL |
| **Known failure modes** | 30 M+ parameters even at N (backbone-heavy); attention-dependent; license split |

**Weight-sharing NAS.** RF-DETR trains *one* network while randomly varying its configuration: input
resolution, ViT **patch size** (FlexiViT-style interpolation of the patch embedding), number of decoder
layers, number of queries, and attention-window counts. After training, thousands of configurations can
be evaluated **without retraining**, and the accuracy–latency Pareto front is read off. The published
N…2XL sizes are points on that front. This is why the input resolutions are unusual (384, 512, 576,
704, 880) and why the same procedure can be run per dataset.

**The fine-tuning claim.** RF-DETR is designed and evaluated for how well it *fine-tunes*. On RF100-VL
(100 datasets), its README shows RF-DETR-N at 57.7 AP$_{50:95}$ vs YOLO26-N at 52.0 and YOLO11-N at
55.3. The authors credit DINOv2 pre-training for transfer to small, unusual datasets. Treat it as
author-measured, with every model run in one harness, which is better than mixing published numbers.

---

## RT-DETRv4 and EdgeCrafter — use the foundation model as a teacher

Running a DINOv2/v3 backbone at inference costs parameters and attention. Two 2025–2026 works keep the
compact detector and **transfer the foundation model's knowledge during training only**:

- **RT-DETRv4** (Liao et al., PKU/Tsinghua, ECCV 2026): a **Deep Semantic Injector** aligns the
  detector's deep features with VFM representations, and **Gradient-guided Adaptive Modulation** scales
  the distillation strength by gradient-norm ratios, so distillation does not overwhelm the detection
  loss. No inference overhead: S 49.8 · M 53.7 · L 55.4 · X 57.0 AP at 3.66 / 5.91 / 8.07 / 12.90 ms on
  T4, on HGNetV2 backbones.
- **EdgeCrafter / ECDet** (Intellindust, TMLR 2026): compact ViTs distilled from a foundation model,
  with an edge-friendly encoder–decoder. ECDet-S reaches 51.7 AP at 10 M parameters with COCO labels
  only (53.6 with Objects365), and ECDet-X 57.9 (59.9 with O365) at 12.7 ms on T4. Its license is a
  custom EdgeCrafter License, so read it before use.

---

## The scoreboard at ~5 ms and ~12 ms (T4 TensorRT FP16)

| Band | Model | AP | ms | Params | Pre-training | License |
|---|---|:---:|:---:|:---:|---|---|
| **~4.5 ms** | YOLO26m | 53.1 | 4.7 | 20.4 M | O365 | AGPL-3.0 |
| | RF-DETR-M | **54.7** | 4.4 | 33.7 M | DINOv2 + O365 | Apache-2.0 |
| | YOLO11m | 51.5 | 4.7 | 20.1 M | — | AGPL-3.0 |
| | YOLOv12-M | 52.5 | 4.86 | 20.2 M | — | AGPL-3.0 |
| **~5.5–6 ms** | DEIMv2-S | 50.9 | 5.78 | 9.7 M | DINOv3 | Apache-2.0 |
| | D-FINE-M / DEIM-M | 52.3 / 52.7 | 5.62 | 19 M | — | Apache-2.0 |
| | RT-DETRv4-M | 53.7 | 5.91 | — | VFM distill | Apache-2.0 |
| **~11–13 ms** | YOLO26x | 57.5 | 11.8 | 55.7 M | O365 | AGPL-3.0 |
| | RF-DETR-XL | **58.6** | 11.5 | 126.4 M | DINOv2 + O365 | PML-1.0 |
| | D-FINE-X (O365) | **59.3** | 12.89 | 62 M | O365 | Apache-2.0 |
| | ECDet-X (O365) | **59.9** | 12.70 | 49 M | distill + O365 | EdgeCrafter |
| | RT-DETRv4-X | 57.0 | 12.90 | — | VFM distill | Apache-2.0 |

Two caveats apply to the whole table. Latency harnesses differ between groups: RF-DETR re-measures all
models in its own harness, and the others report their own. And pre-training differs row by row.
Within those limits the picture is clear: at 4–13 ms on a T4 GPU, the best permissively licensed DETRs
match or beat the best YOLOs on COCO.

---

## What still favours YOLO

- **Non-GPU targets.** Deformable attention (`grid_sample`), large attention blocks and top-k query
  selection are still poorly supported on many NPUs and in INT8. RT-DETRv2's discrete sampler exists for
  exactly this reason.
- **Parameter count.** RF-DETR-N is 30.5 M parameters, against 2.4 M for YOLO26n.
- **Training cost and ecosystem.** Ultralytics' single-command training, export to every format,
  tracking, and documentation (Chapter 35).
- **CPU inference.** YOLO26n runs at 38.9 ms with CPU ONNX in Ultralytics' table. DETRs at similar AP
  are typically slower on CPU.

---

## Key Takeaways

- RT-DETR's hybrid encoder (attention only at stride 32), IoU-aware query selection and adjustable
  decoder depth made DETRs real-time in 2023.
- The next gains were training-side and free at inference: dense positive supervision (RT-DETRv3),
  distribution refinement and self-distillation (D-FINE), dense one-to-one and MAL (DEIM).
- LW-DETR showed that a plain ViT with window/global attention and Objects365 pre-training is
  competitive. RF-DETR added DINOv2 and weight-sharing NAS, and leads at many latencies on T4.
- Foundation models enter either as backbones (RF-DETR, DEIMv2) or as training-time teachers
  (RT-DETRv4, EdgeCrafter). The teacher route keeps deployment cheap.
- On GPUs, permissive DETRs now match or beat YOLO at equal latency. On NPUs, CPUs, and when parameter
  count matters, CNN YOLOs keep their edge.

## Check Yourself

<details class="check"><summary>How can RF-DETR offer many sizes without training each one?</summary>
Weight-sharing NAS. During training, the network is run with randomly sampled configurations (resolution,
patch size via FlexiViT-style interpolation, decoder depth, query count, attention windows), so the
shared weights work for all of them. Afterwards each configuration can be evaluated directly, and the
accuracy–latency Pareto front picked without retraining.</details>

<details class="check"><summary>What does D-FINE's FDR have in common with YOLOv8's DFL, and how does it differ?</summary>
Both represent box edges as probability distributions over discrete offsets instead of single scalars.
DFL predicts one distribution per edge from a dense head, with uniform bins. FDR predicts residual
distributions at every decoder layer relative to the previous layer's box, with non-uniform bins, and
GO-LSD distils the final layer's distributions into earlier ones.</details>

<details class="check"><summary>Why would a team choose RT-DETRv4 over DEIMv2 even at slightly lower AP?</summary>
RT-DETRv4 uses the foundation model only as a training-time teacher, so the deployed detector is a
compact HGNetV2-based DETR with no ViT backbone. That is easier to export and quantise, and cheaper on
non-GPU hardware. DEIMv2 S–X run DINOv3-derived ViT backbones at inference.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| DETRs Beat YOLOs on Real-time Object Detection (RT-DETR) | Zhao et al., 2024 | arXiv:2304.08069 + lyuwenyu/RT-DETR README | Hybrid encoder, query selection, AP/FPS |
| RT-DETRv2 | Lv et al., 2024 | arXiv:2407.17140 + README | Selective sampling, discrete sampler, numbers |
| RT-DETRv3 | Wang et al., 2025 | arXiv:2409.08475 (WACV 2025) | Hierarchical dense positive supervision, R18 48.1 |
| LW-DETR | Chen et al., 2024 | arXiv:2406.03459 + README | Plain ViT, window/global attention, O365, numbers |
| D-FINE | Peng et al., 2024 | arXiv:2410.13842 + README | FDR, GO-LSD, numbers |
| DEIM | Huang et al., 2024 | arXiv:2412.04234 + README | Dense O2O, MAL, numbers |
| DEIMv2 | Huang et al., 2025 | arXiv:2509.20787 + README | STA, DINOv3, numbers, TensorRT note |
| RF-DETR: Neural Architecture Search for Real-Time Detection Transformers | Robinson et al., 2025 | arXiv:2511.09554 + README | Weight-sharing NAS, DINOv2, COCO/RF100-VL tables |
| RT-DETRv4 | Liao et al., 2025 | arXiv:2510.25257 + README | DSI, GAM, numbers |
| EdgeCrafter | Intellindust, 2026 | arXiv:2603.18739 + README | ECDet numbers and license |
| Ultralytics YOLO26 / YOLO11 / YOLO12 docs | Ultralytics, 2026 | docs/en/models | YOLO rows |
| `assets/data/detectors.json` | this book | [atlas](./atlas.md) | Scoreboard rows |

---

**Next:** [Chapter 20 — Open-Vocabulary & Grounded Detection](./20_open_vocabulary.md) — every model so
far knows a fixed list of classes. The next family takes text prompts instead.
