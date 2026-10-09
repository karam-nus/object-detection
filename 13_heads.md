---
title: "Chapter 13 — Heads"
---

[← Back to Table of Contents](./README.md)

# Chapter 13 — Heads

> *"Classification wants features that ignore where an object is. Regression wants features that care about nothing else."*

## Overview

The head maps each fused feature vector (dense detectors) or each query (DETRs) to class scores and a
box. Its design encodes three earlier decisions: what a candidate is (Chapter 4), how the box is
parameterised (Chapters 2 and 6), and whether duplicates are trained away (Chapters 5 and 7). This
chapter covers coupled and decoupled dense heads, the evolution of the YOLO head from v3 to YOLO26,
one-to-one heads, DETR decoders with iterative refinement, two-stage RoI heads, and the head details
that silently matter: bias initialisation, shared vs per-level weights, and in-graph decoding.

<div class="diagram">
<div class="diagram-title">Four head types</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card blue"><div class="card-title">Coupled dense</div><div class="card-desc">one conv outputs A×(4+1+K) per cell — YOLOv3–v5, v7, SSD</div></div>
  <div class="diagram-card accent"><div class="card-title">Decoupled dense</div><div class="card-desc">separate box and class towers — RetinaNet, FCOS, YOLOX, YOLOv6/8/11/26</div></div>
  <div class="diagram-card purple"><div class="card-title">Query decoder</div><div class="card-desc">cross-attention layers refine N queries — DETR family, RT-DETR, D-FINE, RF-DETR</div></div>
  <div class="diagram-card green"><div class="card-title">RoI head</div><div class="card-desc">RoIAlign → MLP → class + box per proposal — Faster/Cascade/Mask R-CNN</div></div>
</div>
</div>

---

## Coupled vs decoupled

**Coupled** (YOLOv3–v5): one 1×1 conv per level outputs, for each of $A = 3$ anchors,
$(t_x, t_y, t_w, t_h, \text{obj}, p_1..p_K)$. A COCO YOLOv5 output layer therefore has
$3 \times (4 + 1 + 80) = 255$ channels. It is cheap, but classification and regression share every
feature up to the last layer.

**Decoupled** (RetinaNet, FCOS; YOLOX brought it to YOLO): separate convolutional towers for
classification and box regression. The motivation is the epigraph. Classification benefits from
translation-*invariant* features: a dog is a dog anywhere in the receptive field. Regression needs
translation-*sensitive* features: where exactly is the edge? Sharing them forces a compromise. The
YOLOX paper reports that the decoupled head converged much faster and improved AP, and that it was
essential for its end-to-end experiments.

| | Coupled | Decoupled |
|---|---|---|
| Parameters / FLOPs | lowest | higher (two towers per level) |
| Convergence | slower | faster |
| Accuracy | lower | higher |
| Quality branch | needs objectness | can fold quality into class score (TAL/VFL) |

---

## The YOLO head, version by version

| Version | Head | Box output per point | Score | Notes |
|---|---|---|---|---|
| **v3 / v5 / v7** | coupled, 3 anchors / cell | $(t_x, t_y, t_w, t_h)$ | obj × cls | 255 channels for COCO |
| **YOLOX** | decoupled (two 3×3 convs per branch), anchor-free | $(l,t,r,b)$-style | obj × cls | IoU-aware objectness |
| **YOLOv6** | "efficient decoupled head" (hybrid channels) | $(l,t,r,b)$; DFL in v3.0 training | cls (VFL) | — |
| **YOLOv8** | decoupled; box tower width $c_2 = \max(16, C/4, 4R)$, class tower width $c_3 = \max(C, \min(K, 100))$ | $4R$ logits, $R = 16$ (DFL) | cls only (BCE vs TAL) | No objectness |
| **YOLO11** | as v8, class tower uses **depthwise-separable** 3×3 convs (`DWConv` + 1×1) | $4R$, DFL | cls | cheaper class branch |
| **YOLOv10** | dual heads: one-to-many + one-to-one; lightweight class head | $4R$, DFL | cls | NMS-free inference |
| **YOLO26** | as YOLO11, `reg_max = 1`, optional one-to-one copy | **4 values** (direct distances) | cls | No DFL; `nms=False` uses one-to-one |
| **YOLO27-M/L** *(preliminary)* | transformer decoder over object queries | — | — | query-based, NMS-free |

The box/class width formulas come straight from Ultralytics' `Detect.__init__`
(`c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))`). `odlab.model.Head`
uses the same formulas.

### Separate weights per level

RetinaNet and FCOS **share** head weights across pyramid levels. This is a regulariser and reduces
parameters, and FCOS adds a learnable per-level scale to the regression output. YOLOv8 and later use
a **separate** head per level (`nn.ModuleList` over levels). That costs more parameters and lets each
level specialise, which suits dynamic assignment, where the level choice is learned.

---

## One-to-one heads

YOLOv10 and YOLO26 attach a **deep copy** of the box and class towers. In Ultralytics' `Detect`:

```python
if end2end:
    self.one2one_cv2 = copy.deepcopy(self.cv2)    # box tower
    self.one2one_cv3 = copy.deepcopy(self.cv3)    # class tower
...
x_detach = [xi.detach() for xi in x] if self.training else x   # one2one sees detached features
```

Two details are deliberate:

- The one-to-one head reads **detached** features. Its sparse one-positive-per-object gradient never
  reaches the backbone or neck. The shared features are shaped only by the dense one-to-many loss,
  and the one-to-one head learns to *read* them.
- At inference only one head runs. Ultralytics selects it with `nms=False`, and `odlab.model.TinyYOLO`
  selects it with `use_o2o=True`. Exported models carry only the selected head ("removal of the unused
  detection branch" in the YOLO26 docs), so parameters and FLOPs match a single-head model.

`odlab.model.TinyYOLO(end2end=True)` reproduces this structure exactly. Chapter 39 trains it and
compares the two heads.

---

## Initialisation that matters

**Class-prior bias.** At the start of training every candidate should predict "background" with high
confidence. Otherwise the 8,400 × 80 initial scores of 0.5 produce an enormous, unstable loss.
RetinaNet sets the classification bias to $b = -\log\frac{1-\pi}{\pi}$ with prior $\pi = 0.01$, giving
$b = -4.6$. Ultralytics sets it per level from an assumed object density:

$$b_l = \log\left(\frac{5}{K \cdot (640 / s_l)^2}\right)$$

"about 5 objects per image, spread over the level's cells and $K$ classes". For COCO at P3:
$\log(5 / (80 \times 6400)) = -11.5$. Box biases start at a constant (2.0 in current Ultralytics).
`odlab.model.Head.bias_init` reproduces the class formula. Removing it makes the first epochs of
TinyYOLO visibly unstable.

---

## In-graph decoding

Ultralytics exports a head that **decodes inside the graph**. It applies the DFL softmax-expectation
(Chapter 6), adds the grid offsets, multiplies by the stride, and converts $(l,t,r,b)$ to box
coordinates. The exported tensor is therefore boxes in input pixels plus sigmoid class scores
(`(B, 4 + K, 8400)`), not raw logits. This is convenient, because post-processing is just NMS. It
also places DFL's softmax and fixed convolution, and a concatenation of box coordinates (range up to
640) with probabilities (range 0–1), inside the model. That combination is awkward to quantise with a
single per-tensor scale (Chapters 36 and 45).

---

## DETR decoder heads

A DETR-family decoder layer is: query self-attention → cross-attention to image features → FFN. A
**class head** (linear) and a **box head** (3-layer MLP) are applied after *every* layer during
training (deep supervision), and after the last layer at inference.

The refinements:

- **Iterative box refinement** (Deformable DETR): each layer predicts a *delta* to the previous layer's
  box, $\hat{b}^{(l)} = \sigma\big(\sigma^{-1}(\hat{b}^{(l-1)}) + \Delta^{(l)}\big)$, and the new box
  becomes the next layer's reference.
- **Deformable cross-attention**: each query samples features at a few learned offsets around its
  reference point on every level, instead of attending to all tokens.
- **Query selection** (two-stage): initial queries and reference boxes come from the top-$k$ encoder
  tokens by class score (300 in RT-DETR, D-FINE, DEIM).
- **D-FINE's FDR head**: the box head predicts *distributions* of edge offsets that are refined layer by
  layer, with GO-LSD distilling the last layer into earlier ones (Chapter 6).
- **Lightweight decoders**: RT-DETR exposes the number of decoder layers as an inference-time
  speed knob. Dropping layers at inference trades AP for latency without re-training.

---

## Two-stage RoI heads

Faster R-CNN's second stage pools a fixed 7×7 feature for each of about 1,000 proposals with
**RoIAlign** (bilinear, no rounding), then applies two FC layers and parallel class (softmax over
$K+1$) and box (class-specific deltas) outputs. **Cascade R-CNN** chains three such heads trained at
IoU 0.5, 0.6 and 0.7. Each refines the previous one's boxes, and the cascade sharply improves
AP$_{75+}$. RoI heads are per-proposal and data-dependent, so they are hard to export and batch. That
is a main reason they disappeared from real-time work.

---

## Task heads that reuse the detection head

| Task | Extra head | Example |
|---|---|---|
| Instance segmentation | prototype masks + per-detection coefficients | YOLACT, YOLOv8/11/26-seg |
| Pose | $K_{kp} \times 3$ keypoint outputs per detection (RLE in YOLO26) | YOLO-pose, RTMO |
| Oriented boxes | extra angle channel | YOLO-OBB |
| Open vocabulary | class logits = region embedding · text embedding | YOLO-World, YOLOE |

Chapter 38 covers them, because they show how far a single dense head design can stretch.

---

## Key Takeaways

- Decoupled heads separate translation-invariant classification from translation-sensitive
  regression. They converge faster and score higher, at a modest cost.
- The YOLO head went from 255-channel coupled anchors (v3/v5) to decoupled anchor-free DFL heads (v8,
  11) to direct-distance heads with an optional one-to-one copy (YOLO26).
- One-to-one heads in YOLOv10/26 are deep copies that read detached features. Only one head ships.
- Class-bias initialisation from a prior is essential: $-4.6$ for $\pi = 0.01$, about $-11.5$ at P3
  for COCO in Ultralytics' density formula.
- Ultralytics decodes boxes inside the exported graph. That simplifies deployment and complicates
  INT8.
- DETR heads refine boxes layer by layer from selected queries. RoI heads gave way because they are
  per-proposal and hard to deploy.

## Check Yourself

<details class="check"><summary>How many output channels does a coupled YOLOv5 head have per level for a 3-class dataset? And a YOLOv8 head (box + class, before decoding)?</summary>
YOLOv5: 3 anchors × (4 + 1 + 3) = 24 channels. YOLOv8: 4 × 16 (DFL bins) + 3 = 67 channels per level
(64 from the box tower and 3 from the class tower, concatenated).</details>

<details class="check"><summary>Why does the YOLO26 one-to-one head read detached features?</summary>
Its loss gives each object only one positive, a sparse and noisier signal. If that gradient flowed
into the shared backbone and neck, it would compete with the dense one-to-many supervision that shapes
good general features. Detaching lets the shared features be learned from dense supervision while the
one-to-one head learns to select one prediction per object from them.</details>

<details class="check"><summary>What goes wrong if you remove the class-bias prior initialisation?</summary>
At initialisation every class logit is near 0, so every probability is about 0.5. Hundreds of
thousands of background (candidate, class) pairs each contribute about −ln 0.5 = 0.69 to the BCE.
The loss is huge, gradients push all scores down violently, and early training is unstable or
diverges. The prior starts the network at the correct base rate.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Focal Loss (RetinaNet) | Lin et al., 2017 | arXiv:1708.02002 | Shared heads, prior bias π = 0.01 |
| FCOS | Tian et al., 2019 | arXiv:1904.01355 | Shared head with per-level scale |
| Rethinking Classification and Localization (Double-Head) | Wu et al., 2020 | arXiv:1904.06493 | Conv vs FC heads for the two tasks |
| YOLOX | Ge et al., 2021 | arXiv:2107.08430 | Decoupled YOLO head |
| YOLOv10 | Wang et al., 2024 | arXiv:2405.14458 | Dual heads, lightweight class head |
| Ultralytics `nn/modules/head.py` (`Detect`) | Ultralytics, 2026 | github.com/ultralytics/ultralytics | Width formulas, DWConv class tower, one2one deepcopy + detach, bias init |
| Deformable DETR | Zhu et al., 2020 | arXiv:2010.04159 | Iterative refinement, deformable attention |
| RT-DETR | Zhao et al., 2024 | arXiv:2304.08069 | Query selection, adjustable decoder depth |
| D-FINE | Peng et al., 2024 | arXiv:2410.13842 | FDR head |
| Cascade R-CNN | Cai & Vasconcelos, 2018 | arXiv:1712.00726 | Multi-stage RoI heads |
| Mask R-CNN | He et al., 2017 | arXiv:1703.06870 | RoIAlign |
| `odlab/model.py` | this book | code/odlab | Head, bias init, one-to-one copy |

---

**Next:** [Chapter 14 — Classic Detectors](./14_classic_detectors.md) — Part III surveys the
landscape, starting with the detectors that set the template.
