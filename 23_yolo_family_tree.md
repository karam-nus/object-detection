---
title: "Chapter 23 — YOLO Family Tree & Lineage"
---

[← Back to Table of Contents](./README.md)

# Chapter 23 — YOLO Family Tree & Lineage

> *"Follow the code, not the version number. The codebase decides the license, the API, and most of the training recipe."*

## Overview

Ten groups have published models called YOLO. They share an idea (one-stage, grid-based, real-time) and,
increasingly, code. This chapter maps who built what, which codebase each model lives in, how licenses
propagate through those codebases, and what the naming conventions mean. It is short on equations and
long on provenance, because provenance answers practical questions: *Which repo do I train this in?
Which recipe produced this number? Can I ship it?*

<div class="diagram">
<div class="diagram-title">Four lineages</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card blue"><div class="card-title">Darknet → Academia Sinica</div><div class="card-desc">Redmon (v1–v3) → Bochkovskiy, Wang, Liao (v4, Scaled-v4, YOLOR, v7, v9). C/Darknet, then PyTorch. GPL-3.0.</div></div>
  <div class="diagram-card accent"><div class="card-title">Ultralytics</div><div class="card-desc">YOLOv3 PyTorch port → v5 → v8 → 11 → 26 → 27 (preview). One codebase that also hosts most third-party YOLOs. AGPL-3.0 / Enterprise.</div></div>
  <div class="diagram-card green"><div class="card-title">Industry labs</div><div class="card-desc">Megvii YOLOX, Baidu PP-YOLO/E, Meituan YOLOv6, Alibaba DAMO-YOLO, Deci YOLO-NAS, Huawei Gold-YOLO, OpenMMLab RTMDet. Mostly Apache-2.0.</div></div>
  <div class="diagram-card purple"><div class="card-title">Academic forks of Ultralytics</div><div class="card-desc">Tsinghua YOLOv10 and YOLOE, Buffalo YOLOv12, iMoonLab YOLOv13, Tencent YOLO-World. They inherit AGPL-3.0 (or GPL-3.0).</div></div>
</div>
</div>

---

## The original line: Darknet (2015–2018)

**Joseph Redmon** (University of Washington, with Santosh Divvala, Ross Girshick and Ali Farhadi on v1)
wrote YOLOv1–v3 in **Darknet**, his own C/CUDA framework. The papers set the template (Chapter 24).
YOLOv3's paper is famously informal ("YOLOv3: An Incremental Improvement"). In February 2020 Redmon
announced he had stopped computer-vision research, citing concerns about military and privacy uses.
The name was then free for others.

## The Academia Sinica line (2020–2024)

**Alexey Bochkovskiy** maintained the most active Darknet fork ("AlexeyAB/darknet"). With **Chien-Yao
Wang** and **Hong-Yuan Mark Liao** (Academia Sinica, Taiwan) he published **YOLOv4** (April 2020), a
systematic study of training and architecture tricks. The same group continued with:

| Model | Year | Contribution | Code |
|---|:---:|---|---|
| **Scaled-YOLOv4** | 2020 | CSP-ised YOLOv4 scaling (P5–P7) | PyTorch, GPL-3.0 |
| **YOLOR** | 2021 | "implicit knowledge" representations | PyTorch, GPL-3.0 |
| **YOLOv7** | 2022 | E-ELAN, planned re-parameterisation, lead-guided assignment | PyTorch (YOLOv5-derived code), GPL-3.0 |
| **YOLOv9** | 2024 | PGI + GELAN | PyTorch, GPL-3.0 |

YOLOv7's and YOLOv9's repositories grew from YOLOv5-style PyTorch code, which shows how quickly the
lineages cross-pollinated.

## The Ultralytics line (2018–2026)

**Ultralytics** (Glenn Jocher) began with a PyTorch port of YOLOv3, then released **YOLOv5** in June
2020, without a paper. The name caused a public controversy: it came weeks after YOLOv4, from a different
group. What YOLOv5 offered was *engineering*: a clean PyTorch codebase, autoanchor, hyperparameter
evolution, AMP, EMA, n/s/m/l/x scaling, and export to ONNX, TensorRT, CoreML and TFLite. It became the
most widely deployed YOLO for years. The line continued:

| Release | Date | Package | Notes |
|---|---|---|---|
| **YOLOv5** | Jun 2020 | `yolov5` repo | anchor-based; later "u" variants retrained anchor-free in the new package |
| **YOLOv8** | Jan 2023 | `ultralytics` | new unified package and API (`YOLO("yolov8n.pt")`), tasks: detect/seg/pose/cls/OBB |
| **YOLO11** | Sep 2024 | `ultralytics` | the "v" was dropped from the name |
| **YOLO26** | Jan 2026 (paper June 2026) | `ultralytics` | edge-first; first Ultralytics YOLO with a paper |
| **YOLO27** | preview, Oct 2026 | not released | "final R&D", no launch date |

The `ultralytics` package also **hosts other groups' models** behind the same API: YOLOv3u, YOLOv5u,
YOLOv6, YOLOv9, YOLOv10, YOLO12, RT-DETR, YOLO-NAS (inference), YOLO-World, YOLOE, SAM/SAM 2/SAM 3,
FastSAM and MobileSAM. A model trained in the `ultralytics` package uses the Ultralytics trainer,
augmentations, loss conventions and evaluator, which matters when comparing numbers (Chapter 33).

## Industry labs (2020–2023)

| Model | Org | Year | Why it mattered | License |
|---|---|:---:|---|---|
| **PP-YOLO / v2 / PP-YOLOE(+)** | Baidu | 2020–22 | Bag of tricks on ResNet-vd; then anchor-free CSPRepResNet + TAL + VFL; ET-head | Apache-2.0 |
| **YOLOX** | Megvii | 2021 | Anchor-free, decoupled head, SimOTA: brought the FCOS/OTA ideas into YOLO | Apache-2.0 |
| **YOLOv6** | Meituan | 2022–23 | Re-parameterised, TensorRT-oriented; TAL; quantisation-aware (RepOpt); v3.0 adds BiC, AAT, DFL | GPL-3.0 |
| **DAMO-YOLO** | Alibaba | 2022 | MAE-NAS backbones, RepGFPN, ZeroHead, AlignedOTA, distillation | Apache-2.0 |
| **RTMDet** | OpenMMLab | 2022 | CSPNeXt large-kernel depthwise; dynamic soft labels; strong permissive baseline | Apache-2.0 |
| **YOLO-NAS** | Deci | 2023 | AutoNAC search with QARepVGG blocks; INT8-friendly | code Apache-2.0; **weights non-commercial** |
| **Gold-YOLO** | Huawei | 2023 | Gather-and-distribute neck | GPL-3.0 |

## Academic models in the Ultralytics codebase (2024–2025)

| Model | Group | Year | Idea | Codebase / license |
|---|---|:---:|---|---|
| **YOLO-World** | Tencent AI Lab | 2024 | open-vocabulary YOLO | MMYOLO (GPL-3.0); Ultralytics port (AGPL-3.0) |
| **YOLOv10** | Tsinghua THU-MIG | 2024 | consistent dual assignment, NMS-free | Ultralytics fork, AGPL-3.0 |
| **YOLOv12** | Univ. at Buffalo / UCAS | 2025 | area attention, R-ELAN | Ultralytics fork, AGPL-3.0 |
| **YOLOE** | Tsinghua THU-MIG | 2025 | text/visual/prompt-free OV YOLO | Ultralytics fork, AGPL-3.0 |
| **YOLOv13** | Tsinghua iMoonLab | 2025 | HyperACE, FullPAD | Ultralytics fork, AGPL-3.0 |

Because these are forks of `ultralytics`, they **inherit AGPL-3.0**, even though their authors are
unrelated to Ultralytics. This surprises teams who assume "academic = permissive".

---

## License propagation

<div class="diagram">
<div class="diagram-title">Licenses follow code</div>
<div class="flow">
  <div class="flow-node blue wide">Darknet (v1–v4): permissive custom license</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Academia Sinica PyTorch repos (Scaled-v4, YOLOR, v7, v9): GPL-3.0</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red wide">Ultralytics (v5, v8, 11, 26) and every fork of it (v10, v12, v13, YOLOE): AGPL-3.0, or a paid Enterprise license</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Independent industry codebases (YOLOX, PP-YOLOE, DAMO-YOLO, RTMDet): Apache-2.0</div>
</div>
</div>

Three layers can each carry a license: the **code** (training and inference), the **weights**
(pre-trained checkpoints), and the **data** they were trained on. YOLO-NAS is the cleanest example of a
split: Apache-2.0 code, non-commercial weights. RF-DETR splits by model size (Apache-2.0 for N–L, PML-1.0
for others). Chapter 35 discusses what AGPL-3.0 obliges you to do and the usual compliance options.

---

## Naming conventions

| Pattern | Meaning | Examples |
|---|---|---|
| **n / s / m / l / x** | compound scaling (depth, width, max channels) | yolo26n … yolo26x |
| **t / s / m / c / e** | YOLOv9's sizes (tiny … compact, extended) | yolov9c |
| **b** | YOLOv10's "balanced" size between m and l | yolov10b |
| **6 suffix / -p6** | extra P6 output, 1280-px input | yolov5x6u, yolo26-p6.yaml |
| **-p2** | extra P2 output for small objects (YAML only) | yolo26-p2.yaml |
| **u suffix** | Ultralytics retrain of an older architecture with the anchor-free v8-style head | yolov5nu, yolov3u |
| **-seg, -pose, -obb, -cls, -sem, -depth** | task heads (Chapter 38) | yolo26n-seg |
| **-world, -worldv2, yoloe-** | open-vocabulary variants | yolov8s-worldv2, yoloe-26s |
| **-objv1-150** | Objects365v1 pre-training checkpoint (150 epochs) | yolo26s-objv1-150.pt |
| **YOLO12 vs YOLOv12** | Ultralytics' spelling vs the authors' spelling of the same model | — |

---

## How to establish a checkpoint's provenance

```python
import torch
ckpt = torch.load("yolo26n.pt", map_location="cpu", weights_only=False)
print(ckpt.get("train_args", {}).get("epochs"), ckpt.get("train_args", {}).get("optimizer"))
print(ckpt.get("git"))       # YOLO26 checkpoints record the training branch and commit
print(ckpt.get("date"), ckpt.get("version"), ckpt.get("license"))
```

Ultralytics checkpoints carry their training arguments, metrics, date, package version and, since
YOLO26, the code commit and per-epoch results. That is enough to reproduce or audit how a number was
produced. Checkpoints from other repositories vary: read the config file shipped next to the weights.

---

## Key Takeaways

- Four lineages share the YOLO name: Darknet → Academia Sinica (v4, v7, v9), Ultralytics (v5, v8, 11, 26,
  27), industry labs (YOLOX, PP-YOLOE, v6, DAMO, NAS, Gold), and academic forks of Ultralytics (v10, v12,
  v13, YOLOE).
- The codebase decides the license, the trainer, the augmentation pipeline and the evaluator. Follow the
  code, not the version number.
- Every fork of `ultralytics` inherits AGPL-3.0. Academia Sinica repos are GPL-3.0. YOLOX, PP-YOLOE,
  DAMO-YOLO and RTMDet are Apache-2.0.
- Code, weights and data can carry different licenses (YOLO-NAS, RF-DETR).
- Ultralytics checkpoints embed training arguments and, since YOLO26, the code commit. Use them to
  establish provenance.

## Check Yourself

<details class="check"><summary>A team picks YOLOv12 "because it is academic, not Ultralytics", to avoid AGPL. Are they right?</summary>
No. The official YOLOv12 code is a fork of the Ultralytics package and is released under AGPL-3.0. The
license follows the code. A permissive alternative needs a permissively licensed codebase and weights,
for example YOLOX, RTMDet, D-FINE/DEIM or RF-DETR N–L.</details>

<details class="check"><summary>What does the "u" in yolov5nu.pt mean, and why does it matter for comparisons with original YOLOv5n?</summary>
It is Ultralytics' retrain of the YOLOv5 architecture with the anchor-free, DFL-based head and loss of
YOLOv8, in the new package (34.3 AP for v5nu vs 28.0 for the original v5n r6.1 at 640). Comparing v5u with
the original v5 measures a head and loss change, not two releases of the same model.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| YOLOv1–v3 | Redmon et al. | arXiv:1506.02640; 1612.08242; 1804.02767 | Original line |
| YOLOv4, Scaled-YOLOv4, YOLOR, YOLOv7, YOLOv9 | Wang, Bochkovskiy, Liao, Yeh | arXiv:2004.10934; 2011.08036; 2105.04206; 2207.02696; 2402.13616 | Academia Sinica line |
| Ultralytics docs: models index, YOLOv5/v8/11/26/27 pages | Ultralytics | github.com/ultralytics/ultralytics docs/en/models | Release history, hosted models, naming |
| YOLOX; PP-YOLOE; YOLOv6; DAMO-YOLO; RTMDet; Gold-YOLO | various | arXiv:2107.08430; 2203.16250; 2301.05586; 2211.15444; 2212.07784; 2309.11331 | Industry rows |
| YOLO-NAS docs | Deci / Ultralytics docs | docs/en/models/yolo-nas.md | Weight license split |
| YOLOv10, YOLOv12, YOLOv13, YOLOE READMEs | THU-MIG; sunsmarterjie; iMoonLab | github repos | Codebase and license |
| YOLOv7 README (v5 r6.1 numbers) | WongKinYiu | github.com/WongKinYiu/yolov7 | YOLOv5-N r6.1 = 28.0 AP |
| YOLO26 training recipe | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Checkpoint provenance fields |

---

**Next:** [Chapter 24 — YOLOv1–v3: The Founding Ideas](./24_yolo_v1_to_v3.md) — the three papers that
defined the grid formulation, with their original equations.
