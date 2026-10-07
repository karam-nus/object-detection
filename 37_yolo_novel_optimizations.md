---
title: "Chapter 37 — Novel Optimisations"
---

[← Back to Table of Contents](./README.md)

# Chapter 37 — Novel Optimisations

> *"An idea is worth adopting when its ablation survives a different codebase, a different dataset and an INT8 export."*

## Overview

Chapters 24–26 tell the YOLO story version by version. This chapter cuts across it: a catalogue of the
specific techniques that YOLO-family detectors introduced or popularised from 2022 to 2026, grouped by
axis. Each entry gives what the technique does, what it costs, where it transfers, and **how strong the
evidence is**. Most techniques are published with an ablation in the authors' own codebase and nowhere
else. A few have been measured independently or in this book. The grading makes that difference visible.

<div class="callout note"><span class="callout-title">Evidence grades used in this chapter</span>
<b>A</b>: measured in this book, or reproduced across several independent codebases.
<b>B</b>: an ablation in the authors' paper, same codebase, same data.
<b>C</b>: claimed or implied by headline results, without an isolated ablation.
A low grade does not mean an idea is bad. It means you should measure it on your own task before relying
on it.</div>

---

## Head and post-processing

| Technique | Introduced | What it does | Cost | Transfers to | Evidence |
|---|---|---|---|---|---|
| **Consistent dual assignment** | YOLOv10 (2024) | Train one-to-many (top-10) and one-to-one (top-1) heads with the same metric; deploy the one-to-one head, no NMS | 2× head at training; small AP loss (0.6–0.8 AP for YOLO26) | Any dense detector with TAL-style assignment | **A**: reproduced in YOLO26 and in this book's TinyYOLO |
| **ProgLoss** (decaying o2m weight) | YOLO26 | o2m loss weight 0.8 → 0.1 over training, o2o weight 0.2 → 0.9 | None | Any dual-head model | **C**: no isolated public ablation |
| **One-to-one top-k → top-1** | YOLO26 | Pick 7 candidates, resolve conflicts, keep 1 | None | Dual-head models | **C** |
| **Detached one-to-one features** | YOLOv10/26 code | The one-to-one head trains on `x.detach()`, so it cannot disturb the backbone | None | Any auxiliary or secondary head | **B** (design choice in code; consistent with the v10 paper) |
| **DFL removal** (`reg_max = 1`, L1 on normalised distances) | YOLO26 | Direct distance regression; box tower shrinks from 64 to 16 channels at n | Possible localisation loss vs DFL | Any DFL head, especially for INT8/NPU targets | **A**: head params 465k → 155k measured (Chapter 29); TinyYOLO results below |
| **Grouped exact top-k** | Ultralytics TensorRT export | Top-300 over 8,400 × 80 scores computed as top-k within 8 groups, then top-k over the winners | None (exact result) | Any top-k post-processing on accelerators | **C**: engineering choice in `Detect._grouped_topk` |

**Measured in this book (TinyYOLO, Chapter 39).** On the same synthetic data and schedule:

| Head | Post-processing | AP | AP50 |
|---|---|---|---|
| DFL (`reg_max=16`), one-to-many | NMS | 0.833 | 0.900 |
| L1 (`reg_max=1`), dual head, one-to-many branch | NMS | 0.810 | — |
| L1 (`reg_max=1`), dual head, one-to-one branch | none | 0.787 | 0.877 |

Both YOLO26 ideas cost accuracy on this small model and short schedule: about 2 AP for removing DFL
and another 2 AP for dropping NMS. The YOLO26 paper's comparison involves a different recipe
(Objects365 pre-training, MuSGD, STAL) and cannot isolate these effects. On your own data, validate both
heads (`nms=None` and `nms=False`) before choosing.

---

## Assignment and loss

| Technique | Introduced | What it does | Cost | Evidence |
|---|---|---|---|---|
| **TAL soft targets** (α = 0.5, β = 6) | TOOD → PP-YOLOE / YOLOv6 / YOLOv8 | Class score trained toward localisation quality; positives chosen by score^α · IoU^β | None | **A**: standard across codebases |
| **STAL** | YOLO26 | Boxes under 16 px enlarged to 16 px for candidate selection | None | **A** for the mechanism: a 6 × 6 box gets 2–5 candidates instead of 0–1 (Chapter 30). **C** for the AP gain |
| **VariFocal / MAL / quality focal losses** | VFNet, GFL, DEIM (MAL) | Asymmetric losses weighting positives by IoU | None | **B** in each paper |
| **Dense one-to-one (Dense O2O)** | DEIM (2024) | More objects per image (mosaic-style) so a one-to-one model gets more positives per batch | Data pipeline change | **B**; adopted by DEIMv2 |
| **Lead-guided auxiliary head** | YOLOv7 | Auxiliary head trained on coarse labels derived from the lead head | Training-only head | **B** |
| **PGI** (programmable gradient information) | YOLOv9 | A reversible auxiliary branch supplies reliable gradients to deep layers; removed at inference | Training-only branch; more training memory | **B** |
| **Foreground alignment branch** | YOLO27 preview | Training-only branch aimed at the o2m vs o2o gap (0.8–0.9 → 0.4 AP) | Training-only | **C**, preliminary |

The pattern since 2022: **spend training compute on supervision you delete before deployment**.
Auxiliary heads, one-to-many branches, PGI, denoising queries (in DETRs) and distillation teachers are
all instances.

---

## Architecture

| Technique | Introduced | What it does | Cost / risk | Evidence |
|---|---|---|---|---|
| **C2f → C3k2 with e = 0.25 early stages** | YOLO11 | Thin early stages; C3k inner blocks at m/l/x | None | **A**: measured compute shift away from stride 8 (Chapter 29) |
| **Partial self-attention at P5** (PSA, C2PSA) | YOLOv10 → YOLO11/26 | Attention on half the channels, only at 20 × 20 | Attention ops on NPUs | **B** |
| **Area attention + R-ELAN** | YOLOv12 | Attention within 4 strips of the map; residual scaling for stability | FlashAttention dependence for GPU speed; memory; NPU support | **B**; Ultralytics recommends YOLO11/26 for stable deployment |
| **HyperACE + FullPAD** | YOLOv13 | Learnable hypergraph correlations distributed through backbone, neck and head | Custom ops; limited accelerator support | **B** (README: +3.0 AP over YOLO11-N at n) |
| **Gather-and-distribute neck** | Gold-YOLO | Global fusion of all levels, then redistribution, instead of PAN's neighbour fusion | More neck compute | **B** |
| **SCDown, CIB, rank-guided design** | YOLOv10 | Cheap downsampling; compact blocks where stage rank analysis shows redundancy | Depthwise ops can be slow on some NPUs | **B** |
| **Structural re-parameterisation** | RepVGG → YOLOv6/v7/YOLO-NAS | Multi-branch training, single-conv inference | Weight ranges that hurt INT8 | **A**: widely reproduced; INT8 issue also reproduced, hence QA-RepVGG / RepOptimizer |
| **Residual SPPF** | YOLO26 | SPPF output added to its input | None | **C** |
| **Dual-scale prediction (drop P4)** | YOLO27 preview (n/s) | Two prediction maps instead of three, widened high-resolution stage | Untested publicly | **C**, preliminary |
| **Query-based decoder in a YOLO** | YOLO27 preview (m/l) | DETR-style decoder over object queries, NMS-free | Decoder latency on edge hardware | **C**, preliminary |

---

## Optimisation and training

| Technique | Introduced | What it does | Evidence |
|---|---|---|---|
| **MuSGD** | YOLO26 | Muon orthogonalised updates + Nesterov SGD on conv/linear weights; class head at 3× LR | **C** for detection specifically: Muon is well established for language-model training, and the YOLO26 materials attribute faster, steadier convergence to it without a public isolated ablation |
| **Objects365 pre-training for every size** | YOLO26 (also LW-DETR, D-FINE O365 variants, RF-DETR) | 150 epochs on 365 classes before COCO | **A**: consistent gains across DETR papers; YOLO26 attributes part of its COCO gain to it |
| **Evolved per-size hyperparameters** | YOLOv5 → YOLO26 | Genetic search of LR, gains and augmentation per model size | **B** (Chapter 32) |
| **Vision-foundation-model distillation** | RT-DETRv4, DEIMv2 (DINOv3) | Distil a large self-supervised backbone's features into a real-time detector | **B**, with large reported gains at the same latency |
| **Built-in feature distillation** | Ultralytics (`distill_model`) | Score-weighted L2 between projected student and teacher neck features | **C** until measured on your task (Chapter 42) |
| **QAT through NVIDIA ModelOpt** | Ultralytics (`train(quantize=8)`) | Fake-quantised training with calibrated, fixed ranges; head output convs kept in float | **C** until measured (Chapter 45) |

### Muon, briefly, and what to check

Muon replaces the raw momentum update of a weight matrix with its nearest semi-orthogonal matrix
(computed by five Newton–Schulz iterations). Every singular direction then moves by a similar amount.
In language models this has given clear speed-ups in training efficiency. For a detector, check two
things before adopting it outside Ultralytics: whether the gain survives a well-tuned SGD or AdamW
baseline at the same number of epochs, and whether it changes the INT8 behaviour of the final weights.
Neither has public evidence yet.

---

## Search and scaling

| Technique | Introduced | What it does | Evidence |
|---|---|---|---|
| **AutoNAC** (hardware-aware NAS) | YOLO-NAS (Deci) | Searches block types and widths for a target device, with quantisation-aware blocks | **B**; weights non-commercial |
| **MAE-NAS** | DAMO-YOLO | Zero-shot NAS by maximum entropy under latency constraints | **B** |
| **Weight-sharing NAS over resolution, patch size and decoder depth** | RF-DETR | Train one super-network, read off the Pareto front without retraining | **B**; the published sizes are points on that front |
| **Compound scaling with `max_channels`** | Ultralytics YAMLs | Width capped before multiplying; m/l share widths | **A** (Chapter 29) |

---

## How to evaluate a "new optimisation" claim

1. **Was it ablated in isolation?** A table where one row adds only this technique, with everything
   else fixed. YOLO release notes often bundle several changes with new pre-training.
2. **Same pre-training and schedule?** Objects365 or a foundation-model backbone can explain several AP
   on its own.
3. **Does it survive export?** Attention, custom ops and DFL can fall off the accelerator. INT8 can
   erase a 0.5 AP gain.
4. **Does it help your object sizes?** Many techniques move APs or APl, not both.
5. **Is it reproduced elsewhere?** A technique adopted by a second, unrelated codebase (TAL, re-param,
   dual assignment) has earned more trust than one that has not.

---

## Key Takeaways

- The most reproduced YOLO-era optimisations are TAL soft targets, structural re-parameterisation,
  dual (one-to-many + one-to-one) assignment and Objects365-style pre-training.
- YOLO26's DFL removal and NMS-free head are deployment optimisations. In this book's small-model
  experiment each cost about 2 AP. The head shrinks threefold. Validate both heads on your data.
- Many recent ideas (ProgLoss, MuSGD for detection, residual SPPF, YOLO27's dual-scale and query designs)
  have no public isolated ablation yet. Treat them as hypotheses to measure.
- Training-only supervision (auxiliary heads, PGI, one-to-many branches, distillation) is the dominant
  trend: more training compute, same inference cost.
- Judge a claim by isolation, matched pre-training, survival through export and INT8, and independent
  reproduction.

## Check Yourself

<details class="check"><summary>A new YOLO reports +2.5 AP over its predecessor and lists five changes, including Objects365 pre-training. How much of the gain can you attribute to its new block design?</summary>
None, from the headline alone. Without an ablation that adds the block with everything else fixed (same
pre-training, schedule and head), the gain cannot be split among the changes. Objects365 pre-training on
its own can account for a large share.</details>

<details class="check"><summary>Why might a technique graded B in its paper still fail on your edge device?</summary>
The ablation was measured in FP16 or FP32 on a GPU, on COCO. Your device may not support its operators
(attention, top-k, custom ops), may run them on the CPU, or may lose the gain in INT8. Your objects may
also have a different size distribution.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| YOLOv7 | Wang, Bochkovskiy, Liao, 2022 | arXiv:2207.02696 | Lead-guided assignment, E-ELAN |
| YOLOv9 | Wang, Yeh, Liao, 2024 | arXiv:2402.13616 | PGI, GELAN |
| YOLOv10 | Wang et al., 2024 | arXiv:2405.14458 | Dual assignment, SCDown, CIB, PSA |
| YOLOv12 | Tian, Ye, Doermann, 2025 | arXiv:2502.12524 | Area attention, R-ELAN |
| YOLOv13 | Lei et al., 2025 | arXiv:2506.17733 | HyperACE, FullPAD |
| YOLO26 paper and docs | Ultralytics, 2026 | arXiv:2606.03748; docs/en/models/yolo26.md | DFL removal, ProgLoss, STAL, MuSGD, Objects365 |
| YOLO27 preview docs | Ultralytics, 2026 | docs.ultralytics.com | Dual-scale, query-based designs (preliminary) |
| Gold-YOLO | Wang et al., 2023 | arXiv:2309.11331 | Gather-and-distribute neck |
| YOLO-NAS / SuperGradients | Deci AI, 2023 | docs.deci.ai | AutoNAC, QA-RepVGG |
| DAMO-YOLO | Xu et al., 2022 | arXiv:2211.15444 | MAE-NAS, distillation |
| RepVGG; RepOptimizer | Ding et al., 2021; 2022 | arXiv:2101.03697; arXiv:2205.15242 | Re-parameterisation and its INT8 fix |
| DEIM; RT-DETRv4; DEIMv2 | Huang et al., 2024; Liao et al., 2025; Huang et al., 2025 | arXiv:2412.04234; arXiv:2510.25257; arXiv:2509.20787 | Dense O2O, MAL, VFM distillation |
| RF-DETR | Robinson et al., 2025 | arXiv:2511.09554 | Weight-sharing NAS |
| Muon | Jordan et al., 2024 | kellerjordan.github.io/posts/muon | Orthogonalised updates |
| Ultralytics `nn/modules/head.py`, `utils/loss.py`, `utils/tal.py`, `optim/muon.py`, `utils/torch_utils.py` | Ultralytics | github.com/ultralytics/ultralytics | Implementations of the YOLO26-era techniques |
| `odlab` TinyYOLO runs | this book | code/odlab | DFL vs L1 vs one-to-one measurements |

---

**Next:** [Chapter 38 — YOLO Beyond Boxes](./38_yolo_beyond_boxes.md) — segmentation, pose, OBB, tracking
and open vocabulary on the same backbone.
