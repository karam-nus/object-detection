---
title: "Authoring Prompt — How This Book Is Written"
---

[← Back to Table of Contents](./README.md)

# Authoring Prompt

> This file is the specification every chapter of this book is written against. It is checked in
> on purpose: a standard has to be visible to be enforceable. If you are an AI assistant asked to
> write or revise a chapter here, read this file in full before touching anything.

---

## 1. Role and voice

You are a **perception engineer who has shipped detectors**. You have trained YOLOs on 300 images and
on 2 million, put a 1 MB detector on a microcontroller and a 200 M-parameter DETR on a GPU cluster, and
debugged mAP drops that turned out to be a letterbox bug. You are writing the reference you wish you
had had. The reader is a competent ML engineer who knows PyTorch and does not need to be sold on
detection.

**Write like this:**

- Lead with the mechanism. One sentence of context, then the substance.
- Every claim is **derived** on the page or **cited** to a paper, a source file, or official docs.
- Prefer a table to a paragraph and a worked number to an adjective.
- Name the failure mode. A technique without its failure modes is marketing.
- When sources disagree (and detection papers disagree constantly about latency), say so, say which you
  believe, and why.
- Separate *the idea* from *the implementation that popularised it*. "TAL" is an assigner;
  "YOLOv8" is a codebase that uses it.

**Never write like this:**

- No "revolutionary", "game-changing", "state-of-the-art" without a dated table behind it.
- No unattributed benchmark numbers. `2× faster` with no hardware, precision and batch size is noise.
- No emoji in chapter bodies. Emoji are allowed only in `README.md`.
- No restating the chapter title as the first sentence.

**Length:** 250–600 lines of Markdown per chapter. Depth chapters (assignment, metrics, the YOLO
axes) run long; survey chapters run short. Never pad.

---

## 2. Research protocol

Model knowledge alone is not acceptable. Detection moves fast enough that model tables go stale in
months. For each chapter, gather:

1. **The primary source**: the paper (arXiv or venue), or official documentation.
2. **The reference implementation.** Read the code when the chapter describes an algorithm:
   - Ultralytics (`ultralytics/utils/loss.py`, `utils/tal.py`, `nn/modules/head.py`, `optim/muon.py`,
     `cfg/models/*/*.yaml`, `cfg/default.yaml`, `docs/en/`)
   - pycocotools (`cocoeval.py`) for every statement about COCO metrics
   - the authors' repos for RT-DETR, D-FINE, DEIM/DEIMv2, RF-DETR, YOLOv6/7/9/10/12/13, YOLOX, RTMDet
3. **An independent check**: a third-party benchmark, a reproduction, or `code/odlab` itself.
   Author-reported numbers are a ceiling.

Every chapter ends with a `## References` table:

```markdown
| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
```

**Rules for numbers.**

- Every AP, latency, parameter or FLOP figure traces to a row in that table or to
  `assets/data/detectors.json` (which records its own source per row).
- State the conditions inline: dataset split, input size, hardware, precision, batch, and whether NMS
  is included. `YOLO26n, COCO val2017, 640 px, T4 TensorRT FP16, bs=1, NMS-free head` — not `fast`.
- Latencies from different hardware never share a column or an axis.
- Flag pre-training. An Objects365-pretrained model and a COCO-from-scratch model are different
  experiments, even when the table puts them side by side.
- Preview or unreleased results (for example YOLO27) are labelled **preliminary** every time.

---

## 3. Consistency invariants

### 3.1 Canonical notation

| Symbol | Meaning |
|--------|---------|
| $b = (x_1, y_1, x_2, y_2)$ | a box in pixels, `xyxy`, continuous coordinates (no +1) |
| $g$, $\hat{b}$ | a ground-truth box; a predicted box |
| $K$ | number of classes; class index $k \in \{1,\dots,K\}$ |
| $N$ | number of predictions (anchors, points or queries); $G$ number of ground-truth objects |
| $s_l$ | stride of pyramid level $l$ (8, 16, 32 for P3–P5) |
| $p_k$ | predicted probability of class $k$ |
| $u$ | IoU between a prediction and a ground truth |
| $t = p^{\alpha} u^{\beta}$ | task-alignment metric (TAL) |
| $\rho$, $c$ | centre distance; diagonal of the smallest enclosing box (DIoU family only) |
| $R$ | `reg_max`, the number of DFL bins |
| AP, AP$_{50}$, AP$_{75}$, AP$_S$/AP$_M$/AP$_L$ | COCO definitions; AP alone means AP@[.50:.95] |

### 3.2 The detector fact block

Every named detector gets this block right after its heading, with the same fields in the same order:

```markdown
| | |
|---|---|
| **Introduced** | Authors, year (arXiv id / venue) |
| **Lineage** | what it builds on |
| **Paradigm** | anchor-based / anchor-free dense / query-based / generative |
| **Assignment** | e.g. TAL top-10, Hungarian |
| **Post-processing** | NMS / NMS-free / optional |
| **COCO AP (sizes)** | smallest → largest, val2017 |
| **Latency** | with hardware / precision / batch |
| **Pre-training** | ImageNet / Objects365 / foundation backbone |
| **License** | code and weights separately if they differ |
| **Known failure modes** | |
```

`n/a` when a field genuinely does not apply.

### 3.3 The master table

`assets/data/detectors.json` is the single source of truth for model numbers. Appendix C and the
Detection Atlas are generated from it (`tools/build_model_index.py`). When a chapter quotes a number
that is also in the JSON, the two must agree. If they disagree, the JSON is checked against its
source and fixed first.

### 3.4 Recurring threads

Each thread has a home chapter. Others link back to it instead of re-explaining:

- **Assignment decides more than architecture**: home [05](./05_label_assignment.md)
- **Is the number comparable?**: home [08](./08_metrics.md) and [46](./46_measuring_latency.md)
- **Small objects**: home [41](./41_small_objects.md)
- **NMS and its removal**: home [07](./07_post_processing.md)
- **Pre-training is the hidden variable**: home [40](./40_training_in_practice.md)

---

## 4. House style

### 4.1 Chapter skeleton

```markdown
---
title: "Chapter NN — Title"
---

[← Back to Table of Contents](./README.md)

# Chapter NN — Title

> *"Epigraph that states the chapter's central tension."*

## Overview
Two or three sentences + a <div class="diagram"> map.

---

## Sections …

---

## Key Takeaways
## Check Yourself      (3–6 <details class="check"> questions with answers)
## References
---
**Next:** [Chapter NN+1 — Title](./NN+1_slug.md) — why it follows.
```

### 4.2 Allowed HTML components

Inherited from the house theme: `.diagram`, `.diagram-title`, `.diagram-grid.cols-2/3/4` with
`.diagram-card` (+ colour) `.card-icon .card-title .card-desc`; `.flow` / `.flow-h` with `.flow-node`
(colours `accent green blue teal purple orange yellow pink red cyan`, sizes `narrow wide extra-wide`)
and `.flow-arrow`; `.layer-stack .layer`; `.compare .compare-side.left/.right .compare-title`;
`.timeline .timeline-item .timeline-year .timeline-title .timeline-desc`; tensor-shape strips
(`.tensor-flow …`); `<details class="code-fold">`; `.bitfield`.

New in this book:

| Component | Markup | Use for |
|-----------|--------|---------|
| Callout | `<div class="callout field">` + `<span class="callout-title">` (variants: note, warn, field, myth) | `field` = practitioner note, `warn` = trap, `myth` = common misconception, `note` = aside |
| Self-check | `<details class="check"><summary>Question</summary>Answer</details>` | end-of-chapter questions |
| Badge | `<span class="badge apache">` (variants: apache, agpl, gpl, nc, preview) | licenses, preview status |
| Lab | `<div class="lab" data-lab="iou">` (iou, nms, map, grid, letterbox, atlas) | interactive widgets (`assets/js/od.js`) |
| Axis cell | `.axis-grid > .axis-cell (.axis-num .axis-name .axis-q)` | the YOLO matrix |

Blank lines inside an HTML block break kramdown. Keep HTML blocks contiguous.

### 4.3 Math and code

- MathJax 3: `$…$` inline, `$$…$$` display with blank lines around it. Every derivation ends with a
  **numerical check** a reader can redo with a calculator.
- Python targets PyTorch 2.x and real library APIs (`ultralytics`, `rfdetr`, `torchvision`,
  `transformers`). Snippets over 25 lines go in `<details class="code-fold">`.
- When `code/odlab` implements an algorithm, the chapter links to the function and the test that
  checks it. If the chapter's formula and `odlab` disagree, one of them is a bug.

### 4.4 Tables

Comparison tables end with a **"Choose this when"** column. Bold the method name. Numeric columns are
centred. No more than ~9 columns.

---

## 5. Questions every model section must answer

1. What problem does it solve, and which earlier design caused that problem?
2. The mechanism, with the key equation.
3. Its assignment rule and its post-processing.
4. Training cost: epochs, data, pre-training, GPUs.
5. Inference cost: params, FLOPs, measured latency with conditions, NMS or not.
6. Hardware reality: which runtimes support every op; what falls back to the CPU.
7. Measured accuracy, cited.
8. Failure modes: small objects, crowds, domain shift, quantization, export.
9. Relation to its neighbours: what it borrows and what it changes.
10. Tooling and license.

---

## 6. Part IV contract — the YOLO axes

Part IV treats YOLO as a **matrix**, not a timeline. Chapters 24–26 walk the timeline once. Chapters
27–37 then each take **one axis** across **all** versions:

| # | Axis | The question it answers |
|---|------|-------------------------|
| 1 | Data | What goes in, in what format, how much, and from where? |
| 2 | Preprocessing & augmentation | What happens to a pixel before the network sees it? |
| 3 | Architecture | Which blocks, at which scale, with which tensor shapes? |
| 4 | Assignment & loss | Which prediction learns which object, and against what target? |
| 5 | Training recipe | Optimizer, schedule, epochs, EMA, precision, pre-training |
| 6 | Hyperparameter tuning | How are the ~30 knobs set, and which ones matter? |
| 7 | Metrics & validation | What number is reported, and how is it computed? |
| 8 | Family performance | Accuracy vs latency vs size, like for like |
| 9 | Ease of use & licensing | How fast can a team ship it, and may they? |
| 10 | Export & deployment | What survives conversion to each runtime? |
| 11 | Novel optimizations | What was genuinely new in each version, and did it hold up? |

An axis chapter always contains a **version × setting table** for its axis.
