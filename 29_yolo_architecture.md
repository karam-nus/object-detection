---
title: "Chapter 29 — YOLO Axis 3: Architecture Anatomy"
---

[← Back to Table of Contents](./README.md)

# Chapter 29 — YOLO Axis 3: Architecture Anatomy

> *"A YOLO is a YAML file that a parser turns into a list of layers. Read the YAML and you have read the model."*

## Overview

This chapter follows a 640×640 image through a modern YOLO, layer by layer. It covers the atom every
YOLO is built from (Conv–BN–SiLU), how the stem changed, every major block family (C3, C2f, ELAN,
GELAN, C3k2, CIB, A2C2f, RepConv), the context modules at the bottom of the backbone (SPPF and its
YOLO26 residual form, C2PSA), the PAN neck, the decoupled head and its channel rules, and compound
scaling (how one YAML yields n/s/m/l/x). All parameter counts, shapes and FLOP splits here were measured
by building the models from the Ultralytics YAMLs. Nothing is copied from a summary table.

<div class="diagram">
<div class="diagram-title">YOLO26n at 640 × 640: backbone → neck → head (measured shapes)</div>
<div class="flow">
  <div class="flow-node">image 3 × 640 × 640</div>
  <div class="flow-arrow"></div>
  <div class="flow-node blue wide">Backbone: Conv s2 → Conv s2 → C3k2 → Conv s2 → <b>C3k2 (P3: 128 × 80 × 80)</b> → Conv s2 → <b>C3k2 (P4: 128 × 40 × 40)</b> → Conv s2 → C3k2 → SPPF (residual) → <b>C2PSA (P5: 256 × 20 × 20)</b></div>
  <div class="flow-arrow"></div>
  <div class="flow-node purple wide">PAN neck: top-down (upsample + concat + C3k2) → bottom-up (Conv s2 + concat + C3k2) → outputs 64 × 80², 128 × 40², 256 × 20²</div>
  <div class="flow-arrow"></div>
  <div class="flow-node accent wide">Decoupled head per level: box tower (16 ch → 4 distances) · class tower (DW-separable, 80 ch → 80 logits) · ×2 (one-to-many + one-to-one)</div>
  <div class="flow-arrow"></div>
  <div class="flow-node green">8,400 points → top-300 (e2e) or NMS (o2m)</div>
</div>
</div>

---

## The atom: `Conv` = Conv2d + BatchNorm + SiLU

Nearly every layer in an Ultralytics YOLO is `Conv(c1, c2, k, s)`:

```python
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)            # eps=1e-3, momentum=0.03 set at init
        self.act = self.default_act if act is True else ...
```

Four details matter in practice:

1. **No conv bias**: BatchNorm supplies it. At export, `fuse()` folds BN into the conv weights and bias,
   so the deployed layer is a plain conv with bias followed by SiLU.
2. **"Same" padding by default** (`autopad` gives `k // 2`). A stride-2 layer exactly halves an even
   input, which is why input sizes must be multiples of 32.
3. **SiLU** (`x·σ(x)`) has been the YOLO activation since YOLOv5 v4.0. It is smooth and slightly more
   accurate than ReLU-family activations, but some NPUs and INT8 pipelines handle it poorly or approximate
   it. A YAML can override it for the whole model with one line, `activation: nn.ReLU()` (the parser
   resets `Conv.default_act`). Chapters 37 and 45 cover the accuracy cost of that swap.
4. **BN hyperparameters** (`eps=1e-3`, `momentum=0.03`) are set at initialisation, not by PyTorch's
   defaults (`1e-5`, `0.1`). This makes BN statistics change more slowly between batches.

---

## The stem: three generations

| Generation | Stem | Why it changed |
|---|---|---|
| Darknet (v1–v3) | 3×3 conv stride 1, then 3×3 stride 2 | — |
| YOLOv5 ≤ v5.0 | **Focus**: space-to-depth (four pixel-shuffled slices concatenated → 12 channels), then 3×3 conv | Fewer FLOPs at full resolution |
| YOLOv5 ≥ v6.0 | **6×6 conv, stride 2, padding 2** | Mathematically equivalent to Focus, but a single standard op that every exporter handles well |
| YOLOv8 → YOLO26 | **3×3 conv s2 → 3×3 conv s2** (P1/2, P2/4) | Simple, quantisation-friendly, cheap: 464 + 4,672 params at n scale |

The stem accounts for under 2% of a nano model's compute but runs at the highest resolution. That
makes it the first place memory bandwidth shows up on edge hardware (Chapter 44).

---

## Block families

Every YOLO since v4 has been built on one idea: **split the channels, transform one part, concatenate
everything, fuse with a 1×1 conv**. This is the CSP (Cross-Stage Partial) principle (Chapter 11). The
blocks differ in what they concatenate and what the transformed part contains.

<div class="diagram">
<div class="diagram-title">C3 vs C2f vs C3k2</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card blue"><div class="card-title">C3 (YOLOv5)</div><div class="card-desc">Two 1×1 branches. One goes through n bottlenecks. Concatenate the <b>two branch ends</b>, then 1×1. Gradient sees 2 paths.</div></div>
  <div class="diagram-card accent"><div class="card-title">C2f (YOLOv8)</div><div class="card-desc">One 1×1 conv, split into 2 halves. Bottlenecks chain off the second half. Concatenate <b>both halves + every bottleneck output</b> ((2+n)·c channels), then 1×1.</div></div>
  <div class="diagram-card green"><div class="card-title">C3k2 (YOLO11/26)</div><div class="card-desc">C2f skeleton whose inner block is a Bottleneck (small sizes), a <b>C3k</b> (a mini-C3 with two bottlenecks; m/l/x), or a <b>Bottleneck + PSABlock</b> (attention variant).</div></div>
</div>
</div>

### Bottleneck

`Bottleneck(c1, c2, shortcut, g, k=(3, 3), e=0.5)`: conv k₁ → conv k₂, with a residual add when
`shortcut and c1 == c2`. Inside C2f and C3k the expansion is `e=1.0`, so both convs keep the full
channel count. In the backbone, the shortcut is on. In the neck it is off (`C2f(..., shortcut=False)`),
because neck blocks fuse features from different sources rather than refining one.

### C3 (YOLOv5)

```text
x ─ cv1 (1×1, c_) ─ n × Bottleneck ─┐
x ─ cv2 (1×1, c_) ──────────────────┴─ concat (2c_) ─ cv3 (1×1, c2)
```

### C2f (YOLOv8, YOLOv10)

```python
def forward(self, x):
    y = list(self.cv1(x).chunk(2, 1))           # cv1: 1×1 to 2c, split into two c halves
    y.extend(m(y[-1]) for m in self.m)          # each bottleneck feeds on the previous output
    return self.cv2(torch.cat(y, 1))            # concat (2 + n)·c channels, 1×1 to c2
```

C2f keeps every intermediate output, which is the ELAN idea (YOLOv7) expressed with less code. The
gradient reaches the input through n + 2 paths of different depths. The cost is the final 1×1 conv over
(2 + n)·c channels and a wider concat: more memory traffic per FLOP than C3.

### ELAN, E-ELAN and GELAN (YOLOv7, YOLOv9)

**ELAN** stacks 3×3 convs in a chain and concatenates selected intermediate outputs. The design rule
in the YOLOv7 paper is to control the shortest and longest gradient path. **E-ELAN** adds
expand–shuffle–merge cardinality for the largest models. **GELAN** (YOLOv9's `RepNCSPELAN4`) generalises
ELAN: split after a 1×1 conv, then run two stages, each a `RepCSP` (CSP block with RepConv bottlenecks)
followed by a 3×3 conv. Concatenate all four pieces and fuse with a 1×1 conv. Any computational block can
replace RepCSP. GELAN is the backbone of YOLOv9 and of the later WongKinYiu designs.

### C3k2 (YOLO11, YOLO26)

`C3k2(c1, c2, n, c3k=False, e=0.5, attn=False, g=1, shortcut=True)` keeps C2f's split-and-concatenate
skeleton and chooses the inner block:

| Inner block | When | Effect |
|---|---|---|
| `Bottleneck` (two 3×3 convs) | `c3k=False`: YOLO11/26 **n and s**, early backbone stages | Same as C2f |
| `C3k` (a C3 with **two** 3×3 bottlenecks) | `c3k=True`: deep stages, and **every** C3k2 at m/l/x (the parser forces it) | More depth per repeat, so the depth multiplier can stay at 0.5 |
| `Bottleneck` + `PSABlock` | `attn=True`: YOLO26's last neck block (P5 output) | Global context in the P5 output at 400 tokens |

Two YOLO-specific rules hide in the parser (`parse_model` in `nn/tasks.py`):

- At scales **m/l/x**, every C3k2 is built with `c3k=True`, whatever the YAML says.
- The YOLO11/26 YAMLs use **`e=0.25`** for the first two backbone C3k2 blocks. The hidden width there is
  a quarter of the output width, which keeps the 160×160 and 80×80 stages cheap.

### CIB and C2fCIB (YOLOv10)

The **Compact Inverted Block** is DW 3×3 → PW 1×1 (expand) → DW 3×3 (or a large-kernel RepVGG-DW, `lk`)
→ PW 1×1 → DW 3×3, with a residual add. YOLOv10's rank-guided design puts it only in the stages where
the intrinsic rank analysis showed redundancy (deep stages, larger models), inside a C2f skeleton
(`C2fCIB`).

### A2C2f (YOLOv12)

**R-ELAN with area attention.** Unlike C2f there is no split: one 1×1 conv, then n inner blocks
chained, all outputs concatenated ((1 + n)·c) and fused. Each inner block is a pair of `ABlock`s:
`AAttn` (multi-head attention with `c // 32` heads, computed within `area` strips of the feature map:
4 at P4, 1 at P5, plus a 7×7 depthwise positional conv) followed by an MLP (ratio 2.0, or 1.2 at l/x).
At l/x scale the block also adds a residual scaled by a learnable `gamma` initialised to 0.01. Splitting
the map into 4 areas cuts the attention cost by 4×. The memory traffic of attention remains, which is
why YOLOv12 depends on FlashAttention for its GPU speed (Chapter 26). The neck's A2C2f blocks are built
with `a2=False`: their inner blocks are plain C3k, with no attention.

### RepConv (YOLOv6, YOLOv7, YOLO-NAS)

**Structural re-parameterisation** (RepVGG): train with parallel 3×3 + 1×1 + identity branches, each
with BN, then fold them algebraically into one 3×3 conv for inference. It gives multi-branch accuracy
at single-branch latency. The trap: the folded weights can have wide ranges that hurt INT8. YOLO-NAS's
QA-RepVGG blocks and YOLOv6's RepOptimizer exist to fix exactly that (Chapter 45).

---

## Downsampling

| Op | Used in | What it is | Trade-off |
|---|---|---|---|
| `Conv(k=3, s=2)` | v5, v8, 11, 12, **26** | Plain strided conv | Most parameters at deep stages (the P5 downsampling conv at x scale, 768 → 768 channels, is 5.3M params) |
| `SCDown` | YOLOv10 | PW 1×1 (change channels) → DW 3×3 s2 | Far fewer params and FLOPs; DW convs can be slow on some NPUs |
| `ADown` | YOLOv9 | avg-pool 2×2 s1 → split: half 3×3 s2 conv, half maxpool 3×3 s2 + 1×1 | Cheap and keeps both smooth and peak information |
| MaxPool | v1–v3 tiny | — | Parameter-free, loses information |

YOLO26 keeps the plain 3×3 s2 conv. In the nano model, the two deepest downsampling convs (layers 5 and
7) hold 443k of the 2.4M parameters.

---

## Context at P5: SPP → SPPF → residual SPPF, and C2PSA

**SPP** (YOLOv3-SPP, YOLOv4) runs max pools of 5, 9 and 13 in parallel and concatenates the results,
mixing context at several receptive fields. **SPPF** (YOLOv5 v6.0+) runs **three sequential 5×5 max
pools**. Two 5×5 pools in a row equal one 9×9 pool, and three equal a 13×13, so SPPF computes exactly
SPP's output at lower cost. The Ultralytics docstring states the equivalence.

```python
def forward(self, x):                       # SPPF(c1, c2, k=5, n=3, shortcut=False)
    y = [self.cv1(x)]                       # 1×1 to c1/2, act=False in the current code
    y.extend(self.m(y[-1]) for _ in range(self.n))
    y = self.cv2(torch.cat(y, 1))           # 4 × c1/2 → c2
    return y + x if self.add else y         # YOLO26: SPPF[1024, 5, 3, True] → residual
```

YOLO26 sets `shortcut=True`, so SPPF becomes a residual block. The input passes through unchanged and
the pooled context is added on top.

**C2PSA** (YOLO11, YOLO26) follows SPPF: a CSP split where one half goes through `PSABlock`s. Each block
is multi-head self-attention (`num_heads = c // 64`, key dim = half the head dim, a 3×3 depthwise
positional encoding on V) plus a 1×1 FFN (c → 2c → c), both residual. Attention sits **only at P5**
because cost scales with the square of the token count: 20 × 20 = 400 tokens at P5 versus 6,400 at P3.
At 400 tokens attention is cheap and gives the deepest features a global view of the image.

<div class="callout note"><span class="callout-title">Why YOLO adds attention so late, and so little</span>
At 640 input, one attention layer at P3 (6,400 tokens) builds a 6,400 × 6,400 score matrix per head:
41M entries, roughly 160 MB in FP32 for a single head. The same layer at P5 builds 160k entries. YOLO11
and YOLO26 put attention where it is nearly free. YOLOv12 puts it at P4 and P5 with area splitting.
RT-DETR runs it only on P5 (AIFI). The constraint is the same in every case.</div>

---

## The neck: PAN with concatenation

All modern Ultralytics YOLOs use the same PAN topology (Chapter 12):

1. **Top-down**: upsample P5 ×2 (nearest), concat with backbone P4, C3k2 → N4. Upsample N4, concat
   with backbone P3, C3k2 → **out P3** (stride 8).
2. **Bottom-up**: 3×3 s2 conv on out P3, concat with N4, C3k2 → **out P4**. 3×3 s2 conv, concat with the
   P5 context (C2PSA output), C3k2 → **out P5**.

YOLOv5 had 1×1 lateral convs before each upsample. YOLOv8 removed them and concatenates directly. Neck
fusion is always **concat**, not FPN's add: concat lets the next block weight the two sources, at the
cost of wider tensors. YOLO26's neck differs from YOLO11's in two places. The top-down C3k2 blocks use
`c3k=True` even at n/s scale, and the final P5 block uses `attn=True`.

---

## The head: decoupled, and smaller every generation

Per pyramid level, the `Detect` head has two towers:

```python
c2 = max(16, ch[0] // 4, 4 * reg_max)       # box tower width
c3 = max(ch[0], min(nc, 100))               # class tower width
box: Conv(x, c2, 3) → Conv(c2, c2, 3) → Conv2d(c2, 4·reg_max, 1)
cls (YOLO11+):  DWConv(x, x, 3) → Conv(x, c3, 1) → DWConv(c3, c3, 3) → Conv(c3, c3, 1) → Conv2d(c3, nc, 1)
cls (legacy v3/v5/v8/v9): Conv(x, c3, 3) → Conv(c3, c3, 3) → Conv2d(c3, nc, 1)
```

Rules that follow from this code:

- **DFL sets the box tower width.** With `reg_max = 16`, `c2 ≥ 64`, so even YOLO11n has a 64-channel
  box tower. YOLO26's `reg_max = 1` lets `c2` fall to `max(16, 64/4) = 16` at n scale. That one change is
  most of why the YOLO26n head is a third the size of YOLO11n's.
- **The class tower is depthwise-separable since YOLO11.** The parser switches to the legacy full-conv
  version only for models whose YAML has no C3k2, A2C2f or C2fCIB (that is, v3/v5/v8/v9 checkpoints).
- **`DFL` is a fixed conv.** A 1×1 conv whose weights are `[0, 1, …, 15]` with `requires_grad=False`,
  applied after a softmax over the 16 bins: an expectation implemented as a convolution, so it exports as
  a plain op.
- **Bias initialisation.** Box biases start at 2.0. Class biases start at
  `log(5 / nc / (640 / stride)²)`, a prior of roughly 5 objects per level per 640² image, spread across
  classes and cells. Without this the first iterations are flooded with false positives (the focal-loss paper's
  prior-probability trick, Chapter 6).
- **End-to-end models carry two heads.** `one2one_cv2/cv3` are deep copies of the one-to-many towers.
  During training they receive **detached** features (`x.detach()`), so the one-to-one loss trains its
  own head but does not pull on the backbone. `model.fuse()` deletes whichever branch inference does not
  use.

| Head | Model (n scale, 80 classes) | Head params (deployed branch) |
|---|---|---|
| Legacy full-conv cls, DFL 16 | YOLOv8n | 897,664 |
| DW-separable cls, DFL 16 | YOLO11n | 464,912 |
| DW-separable cls, `reg_max=1` | **YOLO26n** | **154,828** (×2 during training) |

---

## Compound scaling: one YAML, five models

```yaml
scales:          # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
```

The parser applies two formulas to every layer:

$$
n_{\text{repeats}} = \begin{cases}\max(\operatorname{round}(n \cdot \text{depth}), 1) & n > 1\\ n & n = 1\end{cases}
\qquad
c_{\text{out}} = \operatorname{make\_divisible}\!\big(\min(c, \text{max\_channels}) \cdot \text{width},\ 8\big)
$$

`max_channels` caps the deep layers *before* the width multiplier. At **m** a nominal 1024 becomes
min(1024, 512) × 1.0 = 512. At **x** it becomes 512 × 1.5 = 768, the same as a nominal 512. So m/l/x
models are much less "pyramid-shaped" in channels than n/s.

| Nominal channels in YAML | n | s | m | l | x |
|---|---|---|---|---|---|
| 64 (stem) | 16 | 32 | 64 | 64 | 96 |
| 256 | 64 | 128 | 256 | 256 | 384 |
| 512 | 128 | 256 | 512 | 512 | 768 |
| 1024 | 256 | 512 | 512 | 512 | 768 |
| C3k2 repeats (YAML 2) | 1 | 1 | 1 | 2 | 2 |
| C3k2 inner block | Bottleneck / C3k by YAML | same | **C3k forced** | **C3k forced** | **C3k forced** |

Note that **m** and **l** differ *only* in depth: identical widths, one vs two repeats per C3k2.

### YOLO26n, layer by layer

Measured by building `yolo26n.yaml` and running a 1×3×640×640 tensor through it:

| # | from | Module | Out shape (C × H × W) | Params |
|---|---|---|---|---|
| 0 | −1 | Conv 3×3 s2 | 16 × 320 × 320 | 464 |
| 1 | −1 | Conv 3×3 s2 | 32 × 160 × 160 | 4,672 |
| 2 | −1 | C3k2 (e = 0.25) | 64 × 160 × 160 | 6,640 |
| 3 | −1 | Conv 3×3 s2 | 64 × 80 × 80 | 36,992 |
| 4 | −1 | C3k2 (e = 0.25) → **backbone P3** | 128 × 80 × 80 | 26,080 |
| 5 | −1 | Conv 3×3 s2 | 128 × 40 × 40 | 147,712 |
| 6 | −1 | C3k2 (C3k) → **backbone P4** | 128 × 40 × 40 | 87,040 |
| 7 | −1 | Conv 3×3 s2 | 256 × 20 × 20 | 295,424 |
| 8 | −1 | C3k2 (C3k) | 256 × 20 × 20 | 346,112 |
| 9 | −1 | SPPF (k 5, n 3, residual) | 256 × 20 × 20 | 164,608 |
| 10 | −1 | C2PSA → **P5 context** | 256 × 20 × 20 | 249,728 |
| 11–13 | up, cat 6, C3k2 | top-down N4 | 128 × 40 × 40 | 119,808 |
| 14–16 | up, cat 4, C3k2 | **out P3** | 64 × 80 × 80 | 34,304 |
| 17–19 | Conv s2, cat 13, C3k2 | **out P4** | 128 × 40 × 40 | 37k + 95k |
| 20–22 | Conv s2, cat 10, C3k2 (attn) | **out P5** | 256 × 20 × 20 | 148k + 463k |
| 23 | 16, 19, 22 | Detect (o2m + o2o) | 80² + 40² + 20² = **8,400** points | 309,656 |
| | | **Total (training graph)** | | **2,572,280** |
| | | **Fused (BN folded, unused head branch removed)** | | **2,408,932** |

### Where the parameters and the FLOPs live

Parameters and compute sit at opposite ends of the pyramid. Convolution MACs measured with forward hooks
(conv layers only, deployed branch):

| Stride (map) | YOLO26n % of MACs | YOLO26n % of params | YOLOv8n % of MACs | YOLOv8n % of params |
|---|---|---|---|---|
| 2 (320²) | 1.7 | 0.0 | 1.0 | 0.0 |
| 4 (160²) | 10.5 | 0.5 | 6.9 | 0.4 |
| 8 (80²) | 30.4 | 5.3 | **42.8** | 9.3 |
| 16 (40²) | 31.6 | 22.0 | 31.0 | 27.0 |
| 32 (20²) | 25.9 | **72.2** | 18.2 | 63.3 |
| **Total** | **2.68 GMACs (5.4 GFLOPs)** | 2.40M | 4.37 GMACs (8.7 GFLOPs) | 3.15M |

Three consequences:

1. **Model size (MB) is a stride-32 property, latency is a stride-8 property.** Pruning or slimming the
   deep layers shrinks the file but barely moves latency. Shrinking the P3 path moves latency.
2. **YOLO11/26 rebalanced compute away from P3** (43% → 30% of MACs relative to v8n) with the `e = 0.25`
   C3k2 blocks and a lighter head. That explains a large part of the FLOP drop from 8.7 to 5.4 GFLOPs at
   similar accuracy.
3. **Adding a P2 level is expensive.** `yolo26-p2.yaml` adds an upsample to stride 4 and a fourth Detect
   input. At 160 × 160 every conv costs 4× its P3 cost, and the point count grows from 8,400 to 34,000 (adding 160 × 160 = 25,600 points)
   (Chapter 41).

### Measured totals across scales

| Model | Params (training graph) | Params (fused) | GFLOPs (fused, 640) |
|---|---|---|---|
| YOLO26n | 2.57M | 2.41M | 5.5 |
| YOLO26s | 10.01M | 9.50M | 20.9 |
| YOLO26m | 21.90M | 20.41M | 68.4 |
| YOLO26l | 26.30M | 24.81M | 86.8 |
| YOLO26x | 58.99M | 55.73M | 194.4 |
| YOLO11n | 2.62M | 2.62M | 6.5 |
| YOLOv10n | 2.78M | 2.30M | 6.8 |
| YOLO12n | 2.60M | 2.59M | 7.5 |
| YOLOv8n | 3.16M | 3.15M | 8.7 |

GFLOPs come from Ultralytics' `get_flops` (thop) in this book's environment. They can differ by a few
percent from the figures in the official YAML comments, which come from a different counting pass
(attention matmuls are the usual source of difference). The fused-vs-unfused gap for YOLO26 and YOLOv10
is the deleted second head.

---

## Reading and editing a YOLO YAML

```yaml
# [from, repeats, module, args]
- [-1, 2, C3k2, [512, False, 0.25]]    # from previous layer; 2 repeats (× depth); out 512 (× width); c3k, e
- [[-1, 6], 1, Concat, [1]]            # concat previous layer with layer 6 along channels
- [[16, 19, 22], 1, Detect, [nc]]      # head reads layers 16, 19 and 22
```

- `from` is a layer index (−1 means the previous layer). A list means several inputs.
- `repeats` is scaled by `depth`. For repeat modules (C2f, C3k2, …) it becomes the block's `n` argument.
  For other modules the layer is stacked.
- `args[0]` is the nominal output channel count, scaled by `width` and capped by `max_channels`.
- The parser records which layer outputs are reused (`save` list). That list is how Concat finds its
  inputs at runtime.

Common edits, each one line:

| Goal | Edit |
|---|---|
| ReLU everywhere (NPU / INT8 friendliness) | add `activation: nn.ReLU()` at top level |
| Add a stride-4 output for tiny objects | start from `yolo26-p2.yaml` |
| Add a stride-64 output for 1280-px inputs with huge objects | start from `yolo26-p6.yaml` |
| Use a torchvision backbone | `TorchVision` module in the YAML (loads any torchvision model, truncated) |
| Change the class count | nothing: `nc` comes from `data.yaml` at train time |

<div class="callout warn"><span class="callout-title">Editing the YAML invalidates the checkpoint where shapes change</span>
Loading <code>yolo26n.pt</code> weights into an edited YAML (<code>YOLO("my.yaml").load("yolo26n.pt")</code>)
transfers only tensors whose names and shapes still match. Change the activation and every weight still
loads, but the network now computes a different function and must be fine-tuned. Add a P2 level and the
layer indices after the insertion point shift, so the neck and head start from scratch. Check the
"Transferred X/Y items" log line.</div>

---

## Architecture across the modern line

| | YOLOv5 (v7.0) | YOLOv8 | YOLOv10 | YOLO11 | YOLOv12 | YOLO26 |
|---|---|---|---|---|---|---|
| Stem | 6×6 s2 conv | 3×3 s2 ×2 | 3×3 s2 ×2 | 3×3 s2 ×2 | 3×3 s2 ×2 | 3×3 s2 ×2 |
| Main block | C3 | C2f | C2f + C2fCIB | C3k2 | C3k2 + **A2C2f** | C3k2 (+ attn in neck) |
| Downsample | Conv s2 | Conv s2 | **SCDown** (deep) | Conv s2 | Conv s2 | Conv s2 |
| Context | SPPF | SPPF | SPPF + PSA | SPPF + C2PSA | (area attention in backbone) | **residual SPPF** + C2PSA |
| Neck | PAN + 1×1 laterals | PAN | PAN | PAN | PAN (A2C2f, no attention) | PAN |
| Head | coupled, anchors, objectness | decoupled, DFL 16 | decoupled, DFL 16, **dual o2m/o2o** | decoupled DW cls, DFL 16 | same as 11 | DW cls, **reg_max 1**, dual |
| n params | 1.9M | 3.2M | 2.3M (fused) | 2.6M | 2.6M | 2.4M (fused) |

---

## How the book's TinyYOLO maps to this

`odlab/model.py` (Chapter 39) is a 3M-parameter YOLO built from the same atoms: `Conv`, `Bottleneck`,
`C2f`, `SPPF`, a three-level PAN and a decoupled head that uses the same `c2 = max(16, ch/4, 4·reg_max)`
rule. Switching `reg_max=16` to `reg_max=1, end2end=True` turns it from a YOLOv8-style model into a
YOLO26-style one. The training results in Chapter 39 come from exactly these two settings.

---

## Key Takeaways

- Every Ultralytics YOLO is Conv–BN–SiLU blocks wired by a YAML. `parse_model` applies
  `[depth, width, max_channels]` and turns one YAML into five model sizes.
- The block lineage C3 → ELAN → C2f → GELAN → C3k2 → A2C2f is one idea refined: split channels, transform
  part, concatenate many intermediate outputs, fuse with 1×1.
- SPPF is SPP computed by three sequential 5×5 pools. YOLO26 makes it residual. Attention (C2PSA, PSA,
  area attention) sits at P5, or P4 with area splitting, because attention cost grows with the square
  of the token count.
- The head's box tower width is tied to `reg_max`. Removing DFL (`reg_max = 1`) is what lets YOLO26n's
  deployed head be one third of YOLO11n's (155k vs 465k params).
- Parameters live at stride 32. Compute lives at strides 8–16. Optimise the part that matches your
  constraint: model size or latency.
- At m/l/x, `max_channels = 512` flattens the channel pyramid, and the parser forces C3k inner blocks.
  m and l differ only in depth.

## Check Yourself

<details class="check"><summary>Why does SPPF with three 5×5 max pools produce the same output as SPP with 5, 9 and 13?</summary>
Max pooling with stride 1 and "same" padding composes: a 5×5 max of a 5×5 max is a 9×9 max, and one
more 5×5 gives 13×13. SPPF concatenates the input and the three successive pool outputs, which are
exactly the identity, 5, 9 and 13 windows that SPP concatenates. Sharing the intermediate results makes
it cheaper.</details>

<details class="check"><summary>YOLO26n has 2.57M parameters when you count them after loading the YAML but 2.41M after fuse(). Where did 160k parameters go?</summary>
Into the deleted branch. End-to-end YOLO26 carries a one-to-many and a one-to-one copy of the head
during training (154,828 parameters each at n scale). fuse() removes the branch inference does not use,
and folds BatchNorm into the conv weights, which removes the BN parameters as separate tensors.</details>

<details class="check"><summary>You must cut YOLO26s latency on a GPU by 20% and are considering halving the channels of the stride-32 layers. Good idea?</summary>
Probably not. Stride-32 layers hold most of the parameters but a minority of the MACs (about a quarter
at n scale). Halving them shrinks the file substantially and latency little. Reducing work at stride 8
and 16 (fewer channels in the P3/P4 neck blocks, lower input resolution) or using a smaller scale moves
latency. Always confirm with a measurement (Chapter 46).</details>

<details class="check"><summary>Why do m and l YOLO26 models have the same channel widths?</summary>
Both use width 1.0 and max_channels 512, so every layer's output channels are identical. They differ only
in depth (0.5 vs 1.0), which gives l two repeats per C3k2 where m has one.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics `nn/modules/conv.py` (`Conv`, `DWConv`, `Focus`, `RepConv`) | Ultralytics | github.com/ultralytics/ultralytics | The atom, stems, re-parameterisation |
| Ultralytics `nn/modules/block.py` (`Bottleneck`, `C3`, `C2f`, `C3k`, `C3k2`, `SPPF`, `C2PSA`, `PSABlock`, `A2C2f`, `AAttn`, `CIB`, `SCDown`, `ADown`, `RepNCSPELAN4`, `DFL`, `TorchVision`) | Ultralytics | github.com/ultralytics/ultralytics | Every block's exact structure |
| Ultralytics `nn/modules/head.py` (`Detect`) | Ultralytics | github.com/ultralytics/ultralytics | Head channel rules, dual head with detached features, bias init, fuse |
| Ultralytics `nn/tasks.py` (`parse_model`) | Ultralytics | github.com/ultralytics/ultralytics | Scaling formulas, forced C3k at m/l/x, legacy head switch, activation override |
| `cfg/models/26/yolo26.yaml`, `yolo26-p2.yaml`, `yolo26-p6.yaml`, `11/yolo11.yaml`, `12/yolo12.yaml`, `v10/yolov10n.yaml`, `v8/yolov8.yaml` | Ultralytics | github.com/ultralytics/ultralytics | Layer tables; all measurements in this chapter |
| CSPNet | Wang et al., 2020 | arXiv:1911.11929 | Cross-stage partial principle |
| YOLOv7 (ELAN, E-ELAN) | Wang, Bochkovskiy, Liao, 2022 | arXiv:2207.02696 | Gradient-path design |
| YOLOv9 (GELAN) | Wang, Yeh, Liao, 2024 | arXiv:2402.13616 | GELAN |
| YOLOv10 | Wang et al., 2024 | arXiv:2405.14458 | CIB, SCDown, PSA, rank-guided design |
| YOLOv12 | Tian, Ye, Doermann, 2025 | arXiv:2502.12524 | Area attention, R-ELAN |
| RepVGG | Ding et al., 2021 | arXiv:2101.03697 | Structural re-parameterisation |
| SPP-Net | He et al., 2015 | arXiv:1406.4729 | Spatial pyramid pooling |
| Measurements in this chapter | this book | `tools/measurements/shapes.py`, `tools/measurements/flops_by_res.py` | Layer shapes, parameter and MAC splits |

---

**Next:** [Chapter 30 — Axis 4: Assignment & Loss](./30_yolo_assignment_and_loss.md) — which of the 8,400
points learns from which object, and the exact loss each one receives.
