---
title: "Chapter 22 — The YOLO Matrix"
---

[← Back to Table of Contents](./README.md)

# Chapter 22 — The YOLO Matrix

> *"YOLO is not a model. It is a decade-long experiment in which every variable was changed by someone, usually several at once."*

## Overview

Most YOLO explanations are timelines: v1 did this, v2 did that. Timelines hide the structure. Every
version changed several *independent* things at once (data pipeline, blocks, assignment, loss, schedule,
pre-training), and its headline number mixes all of them. This part of the book treats YOLO as a
**matrix**. The rows are versions. The columns are **eleven orthogonal axes**, each of which can be
studied, tuned and swapped on its own. Chapters 23–26 walk the rows once (lineage and history).
Chapters 27–37 each take one column across all versions. Chapters 38–39 extend YOLO beyond boxes and
build one from scratch.

This chapter is the map: the axes, the version × axis matrix, and how to read a YOLO release.

---

## The eleven axes

<div class="axis-grid">
  <div class="axis-cell"><span class="axis-num">AXIS 1</span><a href="./27_yolo_data"><span class="axis-name">Data</span></a><span class="axis-q">What goes in, in which format, how much, and pre-trained on what?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 2</span><a href="./28_yolo_preprocessing_augmentation"><span class="axis-name">Preprocessing & augmentation</span></a><span class="axis-q">What happens to a pixel before the network sees it?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 3</span><a href="./29_yolo_architecture"><span class="axis-name">Architecture</span></a><span class="axis-q">Which blocks, at what scale, with which tensor shapes?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 4</span><a href="./30_yolo_assignment_and_loss"><span class="axis-name">Assignment & loss</span></a><span class="axis-q">Which prediction learns which object, against what target?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 5</span><a href="./31_yolo_training_recipe"><span class="axis-name">Training recipe</span></a><span class="axis-q">Optimizer, schedule, epochs, EMA, precision, pre-training.</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 6</span><a href="./32_yolo_hyperparameter_tuning"><span class="axis-name">Hyperparameter tuning</span></a><span class="axis-q">How are ~30 knobs set, and which matter?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 7</span><a href="./33_yolo_metrics_and_validation"><span class="axis-name">Metrics & validation</span></a><span class="axis-q">What number is reported, and how is it computed?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 8</span><a href="./34_yolo_family_performance"><span class="axis-name">Family performance</span></a><span class="axis-q">Accuracy vs latency vs size, like for like.</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 9</span><a href="./35_yolo_ease_of_use_and_licensing"><span class="axis-name">Ease of use & licensing</span></a><span class="axis-q">How fast can a team ship it — and may they?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 10</span><a href="./36_yolo_export_and_deployment"><span class="axis-name">Export & deployment</span></a><span class="axis-q">What survives conversion to each runtime?</span></div>
  <div class="axis-cell"><span class="axis-num">AXIS 11</span><a href="./37_yolo_novel_optimizations"><span class="axis-name">Novel optimizations</span></a><span class="axis-q">What was genuinely new in each version, and did it last?</span></div>
</div>

Axis chapters: [1 Data](./27_yolo_data.md) · [2 Preprocessing & augmentation](./28_yolo_preprocessing_augmentation.md) ·
[3 Architecture](./29_yolo_architecture.md) · [4 Assignment & loss](./30_yolo_assignment_and_loss.md) ·
[5 Training recipe](./31_yolo_training_recipe.md) · [6 Tuning](./32_yolo_hyperparameter_tuning.md) ·
[7 Metrics](./33_yolo_metrics_and_validation.md) · [8 Performance](./34_yolo_family_performance.md) ·
[9 Ease of use & licensing](./35_yolo_ease_of_use_and_licensing.md) · [10 Export](./36_yolo_export_and_deployment.md) ·
[11 Novel optimizations](./37_yolo_novel_optimizations.md)

**Why "orthogonal".** Each axis can be changed without touching the others. You can train YOLO11's
architecture with YOLOv5's assignment, or YOLO26's recipe on YOLOv8's blocks. Papers rarely do this, so
their improvements are entangled. Throughout Part IV, every claimed gain gets attributed to an axis
where the source allows it, and flagged as entangled where it does not.

---

## The versions (rows)

| Version | Year | Authors / Org | Paper | Code license |
|---|:---:|---|---|---|
| **YOLOv1** | 2016 | Redmon, Divvala, Girshick, Farhadi | arXiv:1506.02640 | Darknet (permissive) |
| **YOLOv2 / YOLO9000** | 2017 | Redmon, Farhadi | arXiv:1612.08242 | Darknet |
| **YOLOv3** | 2018 | Redmon, Farhadi | arXiv:1804.02767 | Darknet |
| **YOLOv4** | 2020 | Bochkovskiy, Wang, Liao | arXiv:2004.10934 | Darknet (AlexeyAB) |
| **YOLOv5** | 2020 | Ultralytics (Jocher) | no paper | AGPL-3.0 |
| **Scaled-YOLOv4 / YOLOR** | 2020–21 | Wang, Bochkovskiy, Liao; Wang, Yeh, Liao | arXiv:2011.08036; 2105.04206 | GPL-3.0 |
| **YOLOX** | 2021 | Megvii (Ge et al.) | arXiv:2107.08430 | Apache-2.0 |
| **PP-YOLOE** | 2022 | Baidu | arXiv:2203.16250 | Apache-2.0 |
| **YOLOv6 (→ v3.0)** | 2022–23 | Meituan | arXiv:2209.02976; 2301.05586 | GPL-3.0 |
| **YOLOv7** | 2022 | Wang, Bochkovskiy, Liao | arXiv:2207.02696 | GPL-3.0 |
| **DAMO-YOLO** | 2022 | Alibaba | arXiv:2211.15444 | Apache-2.0 |
| **YOLOv8** | 2023 | Ultralytics | docs only | AGPL-3.0 |
| **YOLO-NAS** | 2023 | Deci | no paper | Apache code; non-commercial weights |
| **Gold-YOLO** | 2023 | Huawei Noah's Ark | arXiv:2309.11331 | GPL-3.0 |
| **YOLOv9** | 2024 | Wang, Yeh, Liao | arXiv:2402.13616 | GPL-3.0 |
| **YOLOv10** | 2024 | Tsinghua (THU-MIG) | arXiv:2405.14458 | AGPL-3.0 |
| **YOLO11** | 2024 | Ultralytics | docs only | AGPL-3.0 |
| **YOLOv12** | 2025 | Tian, Ye, Doermann (Buffalo / UCAS) | arXiv:2502.12524 | AGPL-3.0 |
| **YOLOv13** | 2025 | Lei et al. (Tsinghua iMoonLab) | arXiv:2506.17733 | AGPL-3.0 |
| **YOLO26** | 2026 | Ultralytics | arXiv:2606.03748 | AGPL-3.0 / Enterprise |
| **YOLO27** <span class="badge preview">preview</span> | 2026 | Ultralytics | not released | — |

---

## The matrix, part 1: architecture axes

| Version | Backbone block | Neck | Head | Candidates | Activation |
|---|---|---|---|---|---|
| **v1** | 24 conv + 2 FC | — | FC → 7×7×30 | 7×7 cells × 2 boxes | Leaky ReLU |
| **v2** | Darknet-19 | passthrough (reorg) | conv | 13×13 × 5 anchors | Leaky ReLU |
| **v3** | Darknet-53 (residual) | FPN-style, 3 scales | coupled | 9 anchors (3/scale) | Leaky ReLU |
| **v4** | CSPDarknet-53 + SPP | PANet | v3 head | anchors | Mish / Leaky |
| **v5** | C3 + SPPF | PAN (C3) | coupled | anchors (autoanchor) | SiLU |
| **YOLOX** | CSPDarknet | PAN | **decoupled** | **points** | SiLU |
| **v6 3.0** | EfficientRep (RepVGG) / CSPStackRep | Rep(Bi)PAN | efficient decoupled | points (+ anchor-aided training) | ReLU (N/S) / SiLU |
| **v7** | **E-ELAN** + RepConv | ELAN-PAN, SPPCSPC | coupled + auxiliary head | anchors | SiLU |
| **v8** | **C2f** + SPPF | PAN (C2f) | decoupled, **DFL** ($R$=16) | points | SiLU |
| **v9** | **GELAN** + PGI branch (train only) | GELAN-PAN | v8-style | points | SiLU |
| **v10** | C2f + CIB + **PSA**, SCDown | PAN | **dual (o2m + o2o)** | points | SiLU |
| **11** | **C3k2** + SPPF + **C2PSA** | PAN (C3k2) | decoupled, DWConv class tower | points | SiLU |
| **v12** | **R-ELAN / A2C2f (area attention)** | A2C2f PAN | as 11 | points | SiLU |
| **v13** | DS-C3k2 + **HyperACE** | **FullPAD** | as 11 | points | SiLU |
| **26** | C3k2 + SPPF + C2PSA | PAN (C3k2) | dual optional, **no DFL** ($R$=1) | points | SiLU |
| **27** *(preview)* | N/S: widened early stage; L: UltraViT | N/S: **2 scales** | N/S: dense + alignment branch; **M/L: query decoder** | N/S points; M/L queries | — |

## The matrix, part 2: training axes

| Version | Assignment | Cls loss | Box loss | NMS | Optimizer (default) | Epochs (COCO) | Pre-training |
|---|---|---|---|:---:|---|:---:|---|
| **v1** | responsible cell + best box | SSE | SSE on $\sqrt{w}, \sqrt{h}$ | yes | SGD | 135 (VOC) | ImageNet |
| **v3** | best anchor per GT; ignore IoU > 0.5 | BCE | SSE / BCE on offsets | yes | SGD | — | ImageNet |
| **v4** | all anchors above IoU threshold | BCE + label smoothing | **CIoU** | DIoU-NMS | SGD | — | ImageNet |
| **v5** | shape ratio < 4, **3 cells** | BCE | CIoU | yes | SGD 0.01 | 300 | none |
| **YOLOX** | **SimOTA** | BCE (IoU-aware obj) | IoU | yes | SGD | 300 | none |
| **v6 3.0** | **TAL** (ATSS warm-up) | VFL | SIoU / GIoU + DFL | yes | SGD | 300–400 | none (+ self-distillation) |
| **v7** | lead-guided coarse/fine | BCE | CIoU | yes | SGD | 300 | none |
| **v8** | **TAL** k = 10, α = 0.5, β = 6 | BCE (soft) | CIoU + DFL | yes | SGD 0.01 | 500 | none |
| **v9** | TAL | BCE | CIoU + DFL | yes | SGD | 500 | none |
| **v10** | **dual TAL** (k = 10 / k = 1) | BCE | CIoU + DFL | **no** | SGD | 500 | none |
| **11** | TAL | BCE | CIoU + DFL | yes | SGD (auto) | not documented | none |
| **v12** | TAL | BCE | CIoU + DFL | yes | SGD | 600 | none |
| **26** | TAL + **STAL**; o2o top-k → 1; **ProgLoss** | BCE | CIoU + **L1** | **optional** | **MuSGD** | 150 (O365) + 40–245 | **Objects365v1** |

"Not documented" means not documented: YOLO11's COCO schedule is recorded in its checkpoints'
`train_args`, which you can inspect (Chapter 31).

---

## How to read a YOLO release

Ask these questions of every new YOLO, in this order:

1. **Which axes changed?** Architecture only? Assignment? Recipe? Pre-training? Write it down per axis.
2. **Is pre-training the same as the baseline?** YOLO26 uses Objects365. YOLO11 and v8 trained COCO from
   scratch. A 2–3 AP gap can come entirely from this (Chapter 17).
3. **Is the epoch count the same?** 300 vs 500 vs 600 epochs changes small-model AP by around a point.
4. **What exactly is the latency?** Hardware, precision, batch, whether NMS and pre/post-processing are
   included, TensorRT version (Chapter 46).
5. **Which head is the reported number from?** YOLO26's headline AP (40.9 for n) is the one-to-many
   head with NMS. The NMS-free one-to-one head scores 40.1, while the headline speed (1.7 ms) is measured
   with `nms=False`.
6. **Can I reproduce it?** Weights, configs, training code, logs. YOLO26 embeds its full `train_args`,
   per-epoch results and code commit in each checkpoint. That is the best reproducibility practice in
   the family.
7. **Can I use it?** License for code and weights (Chapter 35).

<div class="callout myth"><span class="callout-title">Myth</span>"Higher version number = better
model." YOLOv6 (Meituan) appeared after YOLOv5 and before YOLOv7 (a different group). YOLOv10 and
YOLOv12 are academic models by groups unrelated to YOLO11 and YOLO26. YOLOv13's README reports +3.0 AP
over YOLO11-N, but it requires a hypergraph neck that few accelerators support. Version numbers
encode release order and branding, not lineage or deployability.</div>

---

## A one-line summary per version

| Version | The idea it is remembered for |
|---|---|
| **v1** | Detection as one regression over a grid |
| **v2** | k-means anchors, BatchNorm, multi-scale training |
| **v3** | Darknet-53, three scales, multi-label BCE |
| **v4** | Systematic bag of freebies/specials; CSP; mosaic; CIoU |
| **v5** | Engineering: PyTorch, autoanchor, hyperparameter evolution, export, n–x scaling |
| **YOLOX** | Anchor-free + decoupled head + SimOTA |
| **v6** | Re-parameterised backbones for TensorRT; TAL; quantisation-aware design |
| **v7** | E-ELAN, planned re-parameterisation, lead-guided assignment |
| **v8** | C2f, anchor-free DFL head, TAL; the modern Ultralytics API |
| **v9** | PGI (auxiliary reversible branch) + GELAN |
| **v10** | NMS-free consistent dual assignment |
| **11** | C3k2 + C2PSA: fewer parameters, higher AP than v8 |
| **v12** | Area attention made fast enough for a real-time YOLO |
| **v13** | Hypergraph correlation (HyperACE) + full-pipeline distribution (FullPAD) |
| **26** | Edge-first: no DFL, NMS-free option, MuSGD, ProgLoss, STAL, Objects365 pre-training |
| **27** *(preview)* | Two-scale compact models; query-based M/L |

---

## Key Takeaways

- YOLO is best studied as a matrix: versions × eleven independent axes. Most version-to-version gains
  mix several axes.
- Architecture moved from Darknet to CSP to ELAN/C2f to C3k2 with attention. Candidates moved from cells
  to anchors to points (and to queries in YOLO27-M/L, preliminary).
- Assignment moved from responsible cells to IoU anchors to cross-grid ratios to SimOTA to TAL to dual
  TAL. YOLO26 adds STAL and progressive loss weighting.
- Training moved from ImageNet-pretrained backbones (v1–v4) to from-scratch COCO training (v5–v12) to
  Objects365 pre-training (YOLO26).
- Read every release by asking which axes changed, whether pre-training and epochs match, what the
  latency includes, which head produced the number, and whether you may use it.

## Check Yourself

<details class="check"><summary>YOLO26n reports 40.9 AP and 1.7 ms. YOLO11n reports 39.5 AP and 1.5 ms. Name three axes on which these differ, and why that matters for the comparison.</summary>
Pre-training: YOLO26 used Objects365v1, YOLO11 trained COCO from scratch. Assignment and loss: STAL,
dual heads with ProgLoss, L1 instead of DFL. Optimizer: MuSGD vs SGD. Head used: 40.9 is the NMS head,
while 1.7 ms is measured with the NMS-free head (40.1 AP). The +1.4 AP cannot be attributed to
architecture alone.</details>

<details class="check"><summary>Which YOLO versions are NMS-free by default, and which optionally?</summary>
YOLOv10 is NMS-free by design: only the one-to-one head is used at inference. YOLO26 trains both heads.
Ultralytics' default prediction, validation and export use the one-to-many head with NMS, and nms=False
selects the NMS-free head. YOLO27's M/L query-based models are NMS-free (preliminary).</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| YOLOv1–v3 papers | Redmon et al., 2016–2018 | arXiv:1506.02640; 1612.08242; 1804.02767 | Rows v1–v3 |
| YOLOv4; Scaled-YOLOv4; YOLOR; YOLOv7; YOLOv9 | Wang, Bochkovskiy, Liao, Yeh et al. | arXiv:2004.10934; 2011.08036; 2105.04206; 2207.02696; 2402.13616 | Rows |
| YOLOX; PP-YOLOE; YOLOv6; DAMO-YOLO; Gold-YOLO | Megvii; Baidu; Meituan; Alibaba; Huawei | arXiv:2107.08430; 2203.16250; 2209.02976 + 2301.05586; 2211.15444; 2309.11331 | Rows |
| YOLOv10; YOLOv12; YOLOv13 | THU-MIG; Tian et al.; Lei et al. | arXiv:2405.14458; 2502.12524; 2506.17733 + READMEs | Rows, epochs (500, 600) |
| Ultralytics docs (YOLOv5, v8, 11, 26, 27) and `cfg/models/*.yaml` | Ultralytics, 2020–2026 | github.com/ultralytics/ultralytics | Rows, architecture, defaults |
| YOLO26 paper and training recipe | Jocher et al., 2026 | arXiv:2606.03748; docs/en/guides/yolo26-training-recipe.md | Row 26 |

---

**Next:** [Chapter 23 — Family Tree & Lineage](./23_yolo_family_tree.md) — who built which YOLO, and how
the names, forks and licenses relate.
