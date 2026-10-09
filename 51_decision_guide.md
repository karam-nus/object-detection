---
title: "Chapter 51 — The Decision Guide"
---

[← Back to Table of Contents](./README.md)

# Chapter 51 — The Decision Guide

> *"Start from the constraint that cannot move. Everything else is a trade."*

## Overview

This chapter turns the book into decisions. Work through five questions in order: **where it runs**,
**what licence you can ship**, **what labels you have**, **what your objects look like**, and **how
accuracy trades against latency**. Each path ends in a concrete starting model and recipe, with the
chapters that justify it. The recommendations reflect the state of the field in October 2026. The
reasoning outlives the model names, so each card also says *why*.

<div class="diagram">
<div class="diagram-title">The five questions, in order</div>
<div class="flow">
  <div class="flow-node red wide">1 · Where does it run? MCU · edge NPU · phone · CPU · Jetson · server GPU</div>
  <div class="flow-arrow"></div>
  <div class="flow-node orange wide">2 · Which licences can you ship? permissive only · AGPL/GPL acceptable · commercial licence budget</div>
  <div class="flow-arrow"></div>
  <div class="flow-node yellow wide">3 · What labels do you have? none · a few per class · hundreds+ per class</div>
  <div class="flow-arrow"></div>
  <div class="flow-node green wide">4 · What do the objects look like? tiny · rotated · crowded · open-ended classes</div>
  <div class="flow-arrow"></div>
  <div class="flow-node blue wide">5 · Accuracy vs latency: pick the scale on the Pareto front of your hardware</div>
</div>
</div>

---

## Question 1: where does it run?

| Target | Start with | Why | Then read |
|---|---|---|---|
| **Microcontroller** (Cortex-M, < 1 MB SRAM) | FOMO-style centroid models, TinyissimoYOLO, Yolo-Fastest-class detectors | Peak SRAM and operator support decide; full YOLOs do not fit | 15 |
| **Micro-NPU / smart sensor** (Ethos-U, STM32N6, IMX500) | Vendor-supported tiny YOLO variants at low resolution, INT8 | Toolchain op list | 15, 44 |
| **Edge NPU** (Hailo, RK3588, Edge TPU, QNN) | **YOLO26n/s** INT8 (host NMS where needed) | Pure convs, no DFL, broad exporter support | 16, 36, 44, 45 |
| **Phone** | **YOLO26n/s** via CoreML / LiteRT / QNN / ExecuTorch | Same; FP16 on GPU/ANE, INT8 on NPUs | 36, 44 |
| **x86 / ARM CPU** | **YOLO26n/s** (ONNX Runtime / OpenVINO / NCNN), `nms=False` | Fastest at n/s on CPU; m+ are slower than YOLO11 on CPU | 34, 36 |
| **Jetson** | **YOLO26s/m** TensorRT FP16/INT8; RF-DETR-N/S or D-FINE/DEIM for permissive licences | TensorRT runs attention and DETR decoders well | 16, 34 |
| **Server GPU** | **RF-DETR** (N–L Apache, XL/2XL PML) or **YOLO26 m–x**; DEIMv2 for permissive high accuracy | Best accuracy per ms on T4-class GPUs | 19, 34 |

---

## Question 2: which licence can you ship?

| Situation | Effect on the shortlist |
|---|---|
| AGPL-3.0 acceptable (open product) or enterprise licence budget | Ultralytics models (YOLO26, YOLO11, YOLOE) are available |
| Permissive only, no budget | Remove AGPL/GPL models: use RF-DETR N–L, D-FINE, DEIM/DEIMv2, RT-DETR, LW-DETR on GPUs; YOLOX, RTMDet, PP-YOLOE/PicoDet on CPUs and NPUs |
| No non-commercial weights | Remove YOLO-NAS pre-trained weights |
| Platform licence acceptable | RF-DETR XL/2XL |

Chapter 35 explains the licences. Decide this before running any benchmark.

---

## Question 3: what labels do you have?

| Labels | Path | Chapters |
|---|---|---|
| **None** | Open-vocabulary labeller (YOLOE / Grounding DINO / SAM 3) → audit → review → nano detector | 20, 42, 49 |
| **None, and the classes change often** | Deploy an open-vocabulary model (YOLOE-26 for real time) and update prompts | 20, 38 |
| **< 50 per class** | Try a prompted open-vocabulary model first; fine-tune with the small-data recipe if it falls short; several seeds | 31, 40 |
| **50–1,000 per class** | Fine-tune from COCO/Objects365 weights with the small-data recipe; auto-label to grow the set | 27, 31, 42 |
| **Thousands per class** | Default recipes; tune `scale`, resolution, schedule; consider a foundation-backbone DETR for unusual domains | 31, 32, 34 |
| **Lots of unlabelled data too** | Semi-supervised teacher–student or self-supervised pre-training | 42 |

---

## Question 4: what do the objects look like?

| Property | Adjustment | Chapters |
|---|---|---|
| **Small** (< 16 px at the input) | Higher `imgsz`; P2 head; STAL (YOLO26); SAHI tiling for very large images | 41 |
| **Rotated** (aerial, documents, parts) | YOLO26-obb (ProbIoU); permissive: RTMDet-R, Oriented R-CNN | 38, 43 |
| **Crowded / overlapping** | NMS IoU up or Soft-NMS; visible-part labels; DETR-family or one-to-one heads | 43 |
| **Need masks or keypoints** | `-seg`, `-pose` variants of the same detector | 38 |
| **Open-ended or described by text** | Open-vocabulary detector; or detector + VLM verifier cascade | 20, 21, 47 |
| **Video with identities** | Detector + ByteTrack/BoT-SORT/OC-SORT; keep low-confidence boxes | 38, 43 |
| **Domain far from COCO** (medical, thermal, satellite) | Foundation-backbone detectors; first-conv adaptation for extra channels; in-domain pre-training if data is large | 40, 42, 43 |

---

## Question 5: accuracy versus latency

1. Measure (or look up, Chapter 34) the latency of each candidate **on your hardware, end to end**
   (Chapter 46).
2. Draw your own Pareto front: AP on your validation set against latency on your device.
3. Pick the smallest model that meets the accuracy requirement **at your deployment metric** (recall at
   threshold, false alarms per hour), with margin for INT8 and domain drift.
4. If no model meets both, change the problem: input size, region of interest cropping, cascade,
   better optics, or a faster chip.

---

## Recipe cards

<div class="diagram-grid cols-2">
  <div class="diagram-card accent"><div class="card-title">A. Factory line, Jetson, AGPL acceptable</div><div class="card-desc">YOLO26s, imgsz 640 (or a crop of the belt), fine-tune from COCO weights, 100 epochs, per-class thresholds from error costs, TensorRT FP16 → INT8 with real calibration images, golden-image test. Ch. 31, 36, 45, 47.</div></div>
  <div class="diagram-card blue"><div class="card-title">B. Retail analytics, server GPU, permissive only</div><div class="card-desc">RF-DETR-S/M (Apache-2.0), fine-tuned with its recipe; ByteTrack on top; evaluate with pycocotools; TensorRT FP16; monitor counts per camera. Ch. 34, 38, 47, 48.</div></div>
  <div class="diagram-card green"><div class="card-title">C. Drone imagery, small objects</div><div class="card-desc">YOLO26s-P2 at 1024–1280 or SAHI with sliced fine-tuning at 640; STAL; label audit for tiny objects; report APs and recall by size. Ch. 41.</div></div>
  <div class="diagram-card purple"><div class="card-title">D. New domain, no labels, edge NPU</div><div class="card-desc">YOLOE-26 text prompts → audit 150 images → uncertainty review → YOLO26n trained on reviewed labels → INT8 export with host NMS if the NPU falls back. Ch. 49.</div></div>
  <div class="diagram-card orange"><div class="card-title">E. Phone app, CPU fallback, offline</div><div class="card-desc">YOLO26n, CoreML (iOS) / LiteRT (Android), FP16 on GPU/ANE, nms=False where supported; test thermal throttling; model size as a download budget. Ch. 36, 44, 46.</div></div>
  <div class="diagram-card pink"><div class="card-title">F. Microcontroller person detector</div><div class="card-desc">FOMO-style or TinyissimoYOLO at 96–160 px, INT8, peak-SRAM check before training, CMSIS-NN or vendor NPU toolchain. Ch. 15.</div></div>
</div>

---

## When the guide is wrong

The guide encodes defaults. Measure instead of following it when:

- your hardware vendor's toolchain favours a specific architecture (some NPUs are tuned for particular
  YOLO versions),
- your objects are very unlike COCO (foundation backbones may win where the guide picks a YOLO),
- the field has moved: check Chapter 34's sources and the Atlas data for newer releases.

---

## Key Takeaways

- Decide in order: hardware, licence, labels, object properties, then the accuracy–latency point.
- YOLO26 n/s is the default for CPUs, phones and NPUs if AGPL (or an enterprise licence) is acceptable.
  RF-DETR and the D-FINE/DEIM family lead on GPUs and have permissive options. YOLOX, RTMDet and
  PicoDet remain permissive choices for CPUs and NPUs.
- No labels: open-vocabulary labelling with review, then a nano detector.
- Small, rotated, crowded or open-ended objects each have a specific adjustment; apply it before
  changing architecture.
- Pick the smallest model that meets your own metric on your own hardware, with margin.

## Check Yourself

<details class="check"><summary>A closed-source product must run on a Rockchip RK3588 NPU, detecting 12 industrial classes with 400 labelled images per class. What is your shortlist and first recipe?</summary>
Licence rules out AGPL/GPL without an enterprise licence, so start with permissive CNNs that the RKNN
toolchain handles well (YOLOX-S/Tiny, RTMDet-tiny/s, PicoDet), INT8 with host NMS. If an Ultralytics
enterprise licence is possible, YOLO26n/s joins the list. Fine-tune from COCO weights with the
small-data recipe, calibrate INT8 on real images, validate the exported model, and compare on recall at
the deployment threshold.</details>

<details class="check"><summary>Your server-side system must find "any vehicle that blocks a fire exit", described in plain language by operators. Which design fits?</summary>
A cascade: a fast closed-set or open-vocabulary detector (YOLOE-26 or a fine-tuned detector) finds
vehicles and exits, tracking and simple geometry find candidates, and a VLM verifies the described
condition on candidate crops. Open-ended language belongs in the verifier, not in the real-time detector.</details>

## References

This chapter applies the evidence of Chapters 15–49. Model sources are in Appendix C and the Atlas.

---

**Next:** [Chapter 52 — The Frontier](./52_frontier.md) — where detection is going.
