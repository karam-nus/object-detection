---
title: "Chapter 26 — YOLOv8 → YOLO26 (and YOLO27)"
---

[← Back to Table of Contents](./README.md)

# Chapter 26 — YOLOv8 → YOLO26 (and YOLO27)

> *"From 2023 to 2026 the YOLO head lost its anchors, then its objectness, then its NMS, then its distribution. Each removal was a deployment decision dressed as a research result."*

## Overview

The modern YOLO era starts with **YOLOv8** (January 2023), whose anchor-free DFL head, C2f blocks and TAL
assignment became the base for most YOLOs that followed, including several from other groups. This
chapter walks the seven modern releases: v8, v9, v10, 11, v12, v13 and 26, plus the YOLO27 preview.
For each it states what changed, on which axis, what it measured, and how it holds up for deployment.
Chapter 34 puts all their numbers on one plot.

<div class="diagram">
<div class="diagram-title">The modern line and its side branches</div>
<div class="flow">
  <div class="flow-node accent wide">YOLOv8 (Ultralytics, 2023): C2f · anchor-free decoupled DFL head · TAL</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-h">
    <div class="flow-node blue">YOLOv9 (2024)<small>PGI + GELAN</small></div>
    <div class="flow-node green">YOLOv10 (2024)<small>NMS-free dual assignment</small></div>
    <div class="flow-node accent">YOLO11 (2024)<small>C3k2 + C2PSA</small></div>
  </div>
  <div class="flow-arrow accent"></div>
  <div class="flow-h">
    <div class="flow-node purple">YOLOv12 (2025)<small>area attention</small></div>
    <div class="flow-node pink">YOLOv13 (2025)<small>hypergraph HyperACE</small></div>
    <div class="flow-node accent">YOLO26 (2026)<small>no DFL · e2e · MuSGD · STAL</small></div>
  </div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">YOLO27 (preview): two-scale compact models · query-based M/L</div>
</div>
</div>

---

## YOLOv8

| | |
|---|---|
| **Introduced** | Ultralytics, January 2023 (documentation; no paper) |
| **Lineage** | YOLOv5 codebase + YOLOX/PP-YOLOE/YOLOv6 head and assignment ideas |
| **Paradigm** | anchor-free points, decoupled head, **no objectness** |
| **Assignment** | TAL: top-10, α = 0.5, β = 6.0 |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | n 37.3 · s 44.9 · m 50.2 · l 52.9 · x 53.9 |
| **Latency** | 1.77 / 2.33 / 5.09 / 8.06 / 12.83 ms forward on T4 TRT FP16 (YOLOv10 paper); 6.16–16.86 ms with NMS |
| **Pre-training** | none: COCO from scratch, 500 epochs |
| **License** | AGPL-3.0 |
| **Known failure modes** | DFL in the head complicates some NPUs and INT8; NMS cost on small models |

What changed, by axis:

- **Architecture**: **C2f** replaces C3 (every bottleneck output is concatenated, Chapter 29). The
  stem is a 3×3 stride-2 conv. The neck drops YOLOv5's 1×1 lateral convs.
- **Head**: decoupled box (4 × 16 DFL logits) and class towers. No objectness branch: the class score
  alone ranks boxes.
- **Assignment & loss**: TAL with soft targets; BCE + CIoU + DFL with gains 0.5 / 7.5 / 1.5 (Chapter 30).
- **Recipe**: `close_mosaic = 10`; SGD or AdamW chosen automatically by dataset size (Chapter 31).
- **Ecosystem**: the `ultralytics` package and its one-line API (`YOLO("yolov8n.pt").train(...)`)
  for detection, segmentation, pose, classification and OBB, with one export path. This, more than any
  AP gain, made YOLOv8 the default detector of 2023–2024.

---

## YOLOv9 — programmable gradient information

| | |
|---|---|
| **Introduced** | Wang, Yeh, Liao, 2024 (arXiv:2402.13616, ECCV 2024) |
| **Lineage** | YOLOv7 (ELAN) + YOLOv8-style head |
| **Paradigm** | anchor-free DFL head |
| **Assignment** | TAL |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | T 38.3 · S 46.8 · M 51.4 · C 53.0 · E 55.6 |
| **Latency** | not reported on T4 in the README; params 2.0–57.3 M, 7.7–189 GFLOPs |
| **Pre-training** | none; 500 epochs |
| **License** | GPL-3.0 (original); AGPL-3.0 in Ultralytics |
| **Known failure modes** | training-time auxiliary branch roughly doubles training memory; GPL |

**The idea.** Deep networks lose information about the input as it passes through layers, the
"information bottleneck". Gradients computed from such features can be unreliable for the objective.
**PGI (Programmable Gradient Information)** adds an *auxiliary reversible branch* during training that
preserves complete input information. Its gradients guide the main branch, and multi-level auxiliary
information helps each pyramid level. The branch is removed at inference. **GELAN** (Generalized ELAN)
generalises ELAN so that any computational block (CSP, residual, …) can be stacked inside it.

**Attribution.** The README lists GELAN-C at 52.3 AP and YOLOv9-C (GELAN-C + PGI) at 53.0. That is
+0.7 AP from PGI at identical inference cost, a clean axis attribution that few YOLO releases provide.

---

## YOLOv10 — NMS-free by design

| | |
|---|---|
| **Introduced** | Wang, Chen, Liu, et al. (Tsinghua), 2024 (arXiv:2405.14458, NeurIPS 2024) |
| **Lineage** | YOLOv8 codebase |
| **Paradigm** | anchor-free, **dual heads** (one-to-many for training, one-to-one for inference) |
| **Assignment** | consistent dual assignment: TAL top-10 and top-1 with the same metric |
| **Post-processing** | **none** (top-k) |
| **COCO AP (sizes)** | N 38.5 · S 46.3 · M 51.1 · B 52.5 · L 53.2 · X 54.4 |
| **Latency** | 1.84 / 2.49 / 4.74 / 5.74 / 7.28 / 10.70 ms end to end (T4 TRT FP16) |
| **Pre-training** | none; 500 epochs, batch 256 |
| **License** | AGPL-3.0 |
| **Known failure modes** | one-to-one head trails one-to-many slightly; duplicates under distribution shift |

Two contributions:

1. **Consistent dual assignment** (Chapter 5). Train both heads, and run only the one-to-one head at
   inference. Using the same matching metric for both makes the one-to-one positive the top one-to-many
   positive, so supervision is consistent.
2. **Holistic efficiency–accuracy design**:
   - a **lightweight classification head** (depthwise separable convs; YOLO11 kept this),
   - **spatial–channel decoupled downsampling** (SCDown: a 1×1 conv for channels, then a stride-2
     depthwise conv for space),
   - **rank-guided block design**: measure each stage's intrinsic rank and replace redundant stages
     with a compact inverted block (CIB),
   - **large-kernel depthwise convs** in deep stages of small models,
   - **partial self-attention (PSA)**: attention on half the channels at the lowest resolution.
     YOLO11's C2PSA descends from it.

The paper reports YOLOv10-S at 1.8× the speed of RT-DETR-R18 at similar AP, and YOLOv10-B at 46% less
latency and 25% fewer parameters than YOLOv9-C at the same accuracy.

---

## YOLO11

| | |
|---|---|
| **Introduced** | Ultralytics, September 2024 (documentation) |
| **Lineage** | YOLOv8 + YOLOv10's lightweight class head and PSA idea |
| **Paradigm** | anchor-free DFL head |
| **Assignment** | TAL |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | n 39.5 · s 47.0 · m 51.5 · l 53.4 · x 54.7 |
| **Latency** | 1.5 / 2.5 / 4.7 / 6.2 / 11.3 ms (T4 TensorRT10); 56.1–462.8 ms CPU ONNX |
| **Pre-training** | none (COCO schedule recorded in checkpoint `train_args`) |
| **License** | AGPL-3.0 |
| **Known failure modes** | as v8 (DFL, NMS) |

**C3k2** blocks (a C2f variant whose inner blocks can be small C3 blocks with two convs, "C3k") replace
C2f. A **C2PSA** block (CSP-wrapped position-sensitive attention) follows SPPF at stride 32. The class
tower uses depthwise separable convs. Ultralytics reports YOLO11m at higher COCO AP than YOLOv8m with
**22% fewer parameters**. For two years YOLO11 was the default Ultralytics detector, and YOLO26 keeps its
backbone almost unchanged (Chapter 29).

---

## YOLOv12 — attention made fast enough

| | |
|---|---|
| **Introduced** | Tian, Ye, Doermann, 2025 (arXiv:2502.12524) |
| **Lineage** | YOLO11 codebase |
| **Paradigm** | anchor-free DFL head, attention-centric backbone/neck |
| **Assignment** | TAL |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | N 40.6 · S 48.0 · M 52.5 · L 53.7 · X 55.2 (v1.0); "turbo": 40.4 … 55.4 at lower latency |
| **Latency** | 1.64 / 2.61 / 4.86 / 6.77 / 11.79 ms (T4 TRT FP16, v1.0) |
| **Pre-training** | none; 600 epochs |
| **License** | AGPL-3.0 |
| **Known failure modes** | memory and stability; Ultralytics labels it a "community model… for benchmarking and research" |

**Area attention (A²).** Global self-attention over a stride-16 map is too expensive. Window attention
adds partition overhead. YOLOv12 instead splits the feature map into $l$ (default 4) equal horizontal or
vertical **areas** and runs attention within each. That is a reshape, not a window partition, and it
cuts cost by about $l\times$ while keeping a large receptive field. **R-ELAN** adds block-level residuals
with scaling, so that large attention-heavy models train stably. Further details: FlashAttention kernels
(optional in Ultralytics' implementation), no positional encoding (a 7×7 separable "position perceiver"
conv instead), MLP ratio 1.2–2 instead of 4.

**Deployment note.** Ultralytics' docs state plainly: "If you need stable training, predictable memory
usage, and optimized CPU inference, choose YOLO11 or YOLO26 for deployment." Attention blocks also
narrow NPU support (Chapter 16).

---

## YOLOv13 — hypergraph correlations

| | |
|---|---|
| **Introduced** | Lei et al. (Tsinghua iMoonLab), 2025 (arXiv:2506.17733) |
| **Lineage** | YOLO11/12 codebase + Hyper-YOLO's hypergraph ideas |
| **Paradigm** | anchor-free DFL head |
| **Assignment** | TAL |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | N 41.6 · S 48.0 · L 53.4 · X 54.8 |
| **Latency** | 1.97 / 2.98 / 8.63 / 14.67 ms (GPU per README; not directly comparable with T4 rows) |
| **Pre-training** | none |
| **License** | AGPL-3.0 |
| **Known failure modes** | custom hypergraph ops; limited accelerator support |

**HyperACE** treats multi-scale feature pixels as hypergraph vertices with *learnable* hyperedges, each
connecting many vertices. It aggregates high-order correlations with linear-complexity message passing.
**FullPAD** distributes the correlation-enhanced features to three places: backbone→neck, inside the neck,
and neck→head. Depthwise-separable blocks keep parameters low. The README reports +3.0 AP for N over
YOLO11-N and +1.5 over YOLOv12-N, with fewer parameters and FLOPs.

---

## YOLO26 — edge-first

| | |
|---|---|
| **Introduced** | Jocher, Qiu, Liu, Lyu, Akyon, Kalfaoglu (Ultralytics), released January 2026; paper June 2026 (arXiv:2606.03748) |
| **Lineage** | YOLO11 backbone/neck + YOLOv10 dual heads |
| **Paradigm** | anchor-free, `reg_max = 1` (no DFL), one-to-many + one-to-one heads |
| **Assignment** | TAL + **STAL**; one-to-one top-k → top-1; **ProgLoss** weighting |
| **Post-processing** | NMS (default head) or none (`nms=False`) |
| **COCO AP (sizes)** | n 40.9 · s 48.6 · m 53.1 · l 55.0 · x 57.5 (e2e: 40.1 · 47.8 · 52.5 · 54.4 · 56.9) |
| **Latency** | 1.7 / 2.5 / 4.7 / 6.2 / 11.8 ms (T4 TensorRT10, `nms=False`); CPU ONNX 38.9 ms (n) |
| **Pre-training** | **Objects365v1, 150 epochs**, then COCO 40–245 epochs |
| **License** | AGPL-3.0 / Enterprise |
| **Known failure modes** | gains partly from pre-training (not like for like with YOLO11); one-to-one head 0.6–0.8 AP lower |

YOLO26 is the first Ultralytics YOLO whose changes are mostly **deployment-driven**:

| Change | Axis | What it does | Why |
|---|---|---|---|
| **DFL removed** (`reg_max = 1`) | architecture / loss | direct distances; the `dfl` gain now weights an L1 loss on image-normalised distances | removes softmax + fixed conv from the head; "unconstrained regression range"; simpler export and INT8 |
| **Dual head, NMS optional** | head / post-processing | one-to-many (default, NMS) + one-to-one (`nms=False`, output `(N, 300, 6)`) | removes NMS where deployment needs it |
| **ProgLoss** | loss schedule | one-to-many weight decays linearly 0.8 → 0.1 over training | ends training on the deployed head |
| **STAL** | assignment | GT sides < 16 px enlarged to 16 px for candidate selection | tiny objects always get positives |
| **MuSGD** | optimizer | SGD + Muon-style orthogonalised updates on conv/linear weights | faster, smoother convergence (Chapter 31) |
| **Objects365 pre-training** | data | all sizes pre-trained 150 epochs | +AP, and COCO stage is short |
| **Task heads** | other tasks | semantic-seg loss + multi-scale proto (seg); RLE (pose); angle loss (OBB) | +2.5 box / +3.7 mask AP (seg), +7.2 AP (pose), +3.4 mAP (DOTA OBB) over YOLO11, per the paper |
| **Reproducibility** | ecosystem | checkpoints embed `train_args`, per-epoch logs and the git commit | auditable recipe |

**CPU speed.** The paper reports **up to 43% faster CPU ONNX inference** for YOLO26n vs YOLO11n on an
Intel Xeon @ 2.00 GHz: 38.9 vs 56.1 ms in the docs' table, a 1.44× speed-up. On a Raspberry Pi 5 the docs
report about +15% FPS (ONNX). Part of the speed-up comes from removing DFL and NMS overheads, which matter
more on CPUs than on GPUs.

**YOLOE-26** extends YOLO26 to open-vocabulary detection and segmentation (Chapter 20): 40.6 LVIS AP with
text prompts (x, non-e2e).

---

## YOLO27 — the preview <span class="badge preview">preliminary</span>

As of October 2026, YOLO27 is announced in Ultralytics' documentation as "undergoing final R&D". Models,
configs and code are not released and no launch date is set. Every number below is labelled
preliminary by Ultralytics.

| Model | COCO AP | RTX PRO 6000 TRT11 FP16 (ms) | Params (M) | GFLOPs |
|---|:---:|:---:|:---:|:---:|
| YOLO27n | 42.3 | 0.62 | 3.0 | 7.2 |
| YOLO27s | 49.6 | 0.79 | 11.8 | 28.2 |
| YOLO27m | 55.8 | 1.39 | 22.8 | 65.0 |
| YOLO27l | 60.4 (61.2 at 800 px) | 2.32 | 72.3 | 165.3 |

The design splits by size:

- **N and S (CNN)**: **dual-scale detection**. The medium (P4) prediction map is dropped, and only a fine
  and a coarse map remain, "with a fixed scaling on the fused features keeping the two scales balanced".
  The early high-resolution stage is **widened** for small-object detail. A training-only **foreground
  alignment branch** cuts the one-to-many vs one-to-one gap from 0.9/0.8 AP (YOLO26n/s) to 0.4.
- **M and L (query-based)**: a transformer **decoder over object queries**, NMS-free. M keeps a
  YOLO26-style CNN backbone. L uses an **UltraViT** backbone with self-attention in its deepest stage.

YOLO27l would be the first Ultralytics model above 60 COCO AP. Its latencies are on a different GPU from
every T4 number in this book and cannot be compared with them. The architectural message is clear,
though: at the accuracy end, the YOLO line is adopting the DETR design (Chapter 19).

---

## The modern era on one table

| Model | n-size AP | x-size AP | n params (M) | Head | NMS | Pre-training | Main idea |
|---|:---:|:---:|:---:|---|:---:|---|---|
| **YOLOv8** | 37.3 | 53.9 | 3.2 | DFL | yes | — | C2f + anchor-free DFL + TAL |
| **YOLOv9** | 38.3 (T) | 55.6 (E) | 2.0 | DFL | yes | — | PGI + GELAN |
| **YOLOv10** | 38.5 | 54.4 | 2.3 | DFL, dual | no | — | NMS-free dual assignment |
| **YOLO11** | 39.5 | 54.7 | 2.6 | DFL | yes | — | C3k2 + C2PSA |
| **YOLOv12** | 40.6 | 55.2 | 2.6 | DFL | yes | — | area attention |
| **YOLOv13** | 41.6 | 54.8 | 2.5 | DFL | yes | — | hypergraph HyperACE |
| **YOLO26** | 40.9 | 57.5 | 2.4 | L1, dual | optional | Objects365 | edge-first: no DFL, e2e, MuSGD, STAL |
| **YOLO27** *(prelim.)* | 42.3 | 60.4 (L) | 3.0 | dense (N/S) / queries (M/L) | optional / no | — | two-scale; query-based M/L |

---

## Key Takeaways

- YOLOv8 set the modern template: C2f, an anchor-free decoupled DFL head with no objectness, TAL, and the
  `ultralytics` API.
- YOLOv9's PGI is a training-only auxiliary branch (+0.7 AP for GELAN-C → YOLOv9-C). YOLOv10 made YOLO
  NMS-free through consistent dual assignment.
- YOLO11 refined blocks (C3k2, C2PSA, depthwise class head) for fewer parameters at higher AP. YOLOv12 and
  v13 added attention and hypergraph modules with higher accuracy and narrower deployability.
- YOLO26's changes are deployment-driven: no DFL, optional NMS-free head, ProgLoss, STAL, MuSGD,
  Objects365 pre-training, and reproducible checkpoints.
- The YOLO27 preview (preliminary) drops a pyramid level in compact models and moves M/L to query-based
  decoders. The YOLO and DETR lines are converging.

## Check Yourself

<details class="check"><summary>YOLO26n's docs list 40.9 AP and 1.7 ms. Why is that pairing slightly optimistic, and what are the matched numbers?</summary>
The speed is measured with the NMS-free one-to-one head (nms=False). The 40.9 AP is from the default
one-to-many head, which also needs NMS. Matched pairs: 40.1 AP at about 1.7 ms (no NMS), or 40.9 AP at
1.7 ms plus NMS time.</details>

<details class="check"><summary>Which YOLO26 change would you expect to help most on a CPU, and why?</summary>
Removing NMS (one-to-one head) and DFL. On CPUs, post-processing and small ops (softmax over 16 bins,
reshapes, sorting thousands of boxes) take a much larger fraction of runtime than on GPUs, where the conv
layers dominate. The docs report a 1.44× CPU ONNX speed-up for n over YOLO11n.</details>

<details class="check"><summary>YOLOv13-N reports +3.0 AP over YOLO11-N. What would you check before adopting it for a Hailo-8 deployment?</summary>
Whether its HyperACE and FullPAD operators (hypergraph construction, message passing) compile and run in
INT8 on the Hailo toolchain without CPU fallback. Then measure end-to-end latency on the device and the
accuracy after quantisation on your data. Check the license (AGPL-3.0) too.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics YOLOv8 / YOLO11 / YOLO12 / YOLO26 / YOLO27 docs | Ultralytics, 2023–2026 | github.com/ultralytics/ultralytics docs/en/models | Architecture, numbers, caveats, preview |
| YOLOv10 paper and docs table | Wang et al., 2024 | arXiv:2405.14458 | Dual assignment, efficiency design, YOLOv8 latencies with/without NMS |
| YOLOv9 | Wang, Yeh, Liao, 2024 | arXiv:2402.13616 + README | PGI, GELAN, GELAN-C vs YOLOv9-C |
| YOLOv12 | Tian, Ye, Doermann, 2025 | arXiv:2502.12524 + README | Area attention, R-ELAN, numbers, 600 epochs |
| YOLOv13 | Lei et al., 2025 | arXiv:2506.17733 + README | HyperACE, FullPAD, numbers |
| Ultralytics YOLO26 | Jocher et al., 2026 | arXiv:2606.03748 + docs + training recipe | All YOLO26 changes and numbers |
| Ultralytics `nn/modules/head.py`, `utils/loss.py`, `utils/tal.py`, `optim/muon.py` | Ultralytics | github.com/ultralytics/ultralytics | Implementation details |
| Ultralytics Raspberry Pi guide | Ultralytics, 2026 | docs/en/guides/raspberry-pi.md | Pi 5 FPS comparison |

---

**Next:** [Chapter 27 — Axis 1: Data](./27_yolo_data.md) — the lineage is mapped; now the axes. First,
the data that goes in.
