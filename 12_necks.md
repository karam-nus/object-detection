---
title: "Chapter 12 — Necks"
---

[← Back to Table of Contents](./README.md)

# Chapter 12 — Necks

> *"High-resolution maps know where. Low-resolution maps know what. The neck makes them talk."*

## Overview

A backbone's stride-8 map is spatially precise but semantically shallow. Its stride-32 map is
semantically rich but coarse. Small objects need both on the same map. The **neck** fuses the pyramid
so that every level carries both. This chapter derives FPN and PAN, covers the weighted and searched
variants (BiFPN, NAS-FPN, RepGFPN), the attention-era designs (gather-and-distribute, RT-DETR's hybrid
encoder, YOLOv13's hypergraph aggregation), and the simple pyramid that lets plain ViTs detect. It
ends with what necks cost.

<div class="diagram">
<div class="diagram-title">FPN adds top-down; PAN adds bottom-up</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">FPN (2017)</div>
    <ul>
      <li>C5 → 1×1 → P5</li>
      <li>upsample(P5) + 1×1(C4) → P4</li>
      <li>upsample(P4) + 1×1(C3) → P3</li>
      <li>semantics flow down to fine maps</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">PAN (2018)</div>
    <ul>
      <li>FPN first, then:</li>
      <li>N3 = P3</li>
      <li>N4 = fuse(downsample(N3), P4)</li>
      <li>N5 = fuse(downsample(N4), P5)</li>
      <li>localisation detail flows back up in few hops</li>
    </ul>
  </div>
</div>
</div>

---

## FPN: semantics flow down

Lin et al. (2017) added a top-down pathway to the backbone's pyramid:

$$P_5 = \text{conv}_{1\times1}(C_5), \qquad P_l = \text{up}_2(P_{l+1}) + \text{conv}_{1\times1}(C_l), \qquad P_l \leftarrow \text{conv}_{3\times3}(P_l)$$

The 1×1 *lateral* convolutions project every level to a common width (256 channels in the original).
The 3×3 conv smooths upsampling artefacts. Each output level now combines its own spatial detail with
semantics from all coarser levels. FPN gave Faster R-CNN about +2 AP overall and much more on small
objects, and every later neck descends from it.

**Level assignment for RoI-based heads.** FPN maps an RoI of size $w \times h$ (at 224-pixel
canonical scale) to level $k = \lfloor 4 + \log_2(\sqrt{wh}/224) \rfloor$. A 112-pixel RoI goes to P3
and a 448-pixel one to P5.

---

## PAN: localisation flows back up

In FPN, the path from a precise low-level feature to the top level runs through the whole backbone,
tens to hundreds of layers. **PANet** (Liu et al., 2018) adds a **bottom-up** path after the top-down
one, so low-level localisation detail reaches the coarse levels within a few layers.

Every YOLO since v4 uses this FPN + PAN structure ("PAFPN"), with two changes from the originals:

1. **Concatenate instead of add.** Concatenation keeps both feature sets and lets the following
   block learn the mix, at the cost of more channels.
2. **CSP blocks at each fusion point** (C3 in v5, C2f in v8, C3k2 in YOLO11 and YOLO26, A2C2f in
   YOLOv12) instead of a single 3×3 conv.

`odlab.model.TinyYOLO.features` is a complete PAN in eight lines: two top-down fusions (`td4`, `td3`)
and two bottom-up fusions (`bu4`, `bu5`), with nearest-neighbour upsampling and stride-2 convs for
downsampling. Chapter 29 traces the same structure through `yolo26.yaml`, layer by layer.

---

## Weighted and searched fusion

**NAS-FPN** (Ghiasi et al., 2019) searched for a cross-scale topology with reinforcement learning.
It was accurate but irregular, and hard to adapt.

**BiFPN** (Tan et al., EfficientDet, 2020) simplified the search result into a repeatable
bidirectional block with three ideas:

- drop nodes that have a single input (they contribute little fusion),
- add a skip edge from input to output at each level,
- **learn fusion weights** with "fast normalised fusion":

$$O = \sum_i \frac{w_i}{\epsilon + \sum_j w_j} I_i, \qquad w_i = \text{ReLU}(\text{learned}_i)$$

Different input resolutions contribute unequally, and the network learns how much each one should.
BiFPN blocks are stacked 3–8 times depending on the EfficientDet size.

**RepGFPN** (DAMO-YOLO, 2022) makes a generalised FPN efficient on GPUs. It uses different channel
widths per level, "queen-fusion" connections across adjacent levels, and re-parameterisable blocks
(Chapter 11).

**ASFF** (2019) learns per-pixel spatial weights for fusing levels. An object can then draw its
features mostly from the level that suits it, at each location.

---

## Attention-era necks

### Gather-and-distribute (Gold-YOLO)

FPN/PAN pass information only between *adjacent* levels, so P3 hears about P5 only through P4. Gold-YOLO
(2023) **gathers** all levels into one global representation, processes it (with convolutions in its
"low" branch and attention in its "high" branch), and **distributes** it back to every level by
attention-based injection. In its paper this gave about +1–2 AP over YOLOv6's neck at similar latency.

### RT-DETR's hybrid encoder

Deformable DETR's encoder runs multi-scale deformable attention over *all* tokens of *all* levels, and
was its most expensive part. RT-DETR (2023) decouples the job:

- **AIFI** (attention-based intra-scale feature interaction): standard self-attention on **S5 only**.
  At stride 32 a 640 image has 400 tokens, so global attention is cheap there, and that is where
  object-level semantics live.
- **CCFF** (CNN-based cross-scale feature fusion): a PAN-like convolutional fusion of S3–S5 with
  re-parameterisable fusion blocks.

The design reflects an observation: attention pays off where tokens are few and semantic. Convolution
is cheaper where tokens are many and local. D-FINE, DEIM and DEIMv2 keep this hybrid encoder (DEIMv2's
smallest variants use a lighter "LiteEncoder").

### FullPAD and hypergraphs (YOLOv13)

YOLOv13 (2025) gathers multi-scale backbone features into a **HyperACE** module. Pixels become
vertices of a hypergraph. Learnable hyperedges connect many vertices at once and model high-order
correlations, with linear-complexity message passing. Its **FullPAD** paradigm then distributes the
enhanced features through three tunnels: backbone→neck, inside the neck, and neck→head. The authors
report +3.0 AP for YOLOv13-N over YOLO11-N. It is a neck-level idea, and the hardest YOLO neck so far
to deploy on accelerators without good support for its custom ops.

---

## Pyramids for plain ViTs

A plain ViT has one stride-16 map. **ViTDet's simple feature pyramid** builds P2–P6 from that map
alone: two transposed convolutions for stride 4, one for stride 8, identity for 16, and max-pooling for
32 and 64. It has no top-down or bottom-up paths, and works because the ViT's global (or windowed +
global) attention has already mixed information across scales. Li et al. (2022) found it as good as
an FPN built on intermediate ViT blocks. RF-DETR's multi-scale projector and DEIMv2's Spatial Tuning
Adapter are descendants: light modules that turn one foundation-model map into the pyramid a detector
head expects.

---

## How many levels?

| Configuration | Levels | Used by | Trade-off |
|---|---|---|---|
| **P3–P7** | 5 | RetinaNet, FCOS, ATSS | Coverage of very large objects at 800–1333 px inputs |
| **P3–P5** | 3 | YOLOv3 → YOLO26, RT-DETR family | The real-time default at 640 |
| **P2–P5** | 4 | `yolo26-p2.yaml`, aerial/small-object models | +25,600 candidates at 640; better AP$_S$, slower head and NMS |
| **P3–P6** | 4 | `-p6` YOLOs at 1280 | Large inputs, large objects |
| **P3 + P5 only** | 2 | YOLO27-N/S *(preliminary)* | Drops the middle scale in compact models; "fixed scaling on the fused features keeps the two scales balanced" |

The YOLO27 preview is worth watching: removing the medium scale challenges a ten-year-old default.
Its claim is that a widened early stage plus the fine map covers small objects, the coarse map covers
large ones, and the middle level was mostly costing head compute.

---

## What a neck costs

- **FLOPs.** PAN necks are a substantial share of a YOLO's compute because they run at stride 8 with
  wide channels. In YOLO26n the head section (neck + detect) of the YAML has as many C3k2 blocks as the
  backbone.
- **Memory traffic.** Upsample + concat operations move whole feature maps through memory. On
  bandwidth-limited edge chips they can cost more than their FLOPs suggest (Chapter 44).
- **Export.** Nearest-neighbour upsampling and concatenation are universally supported. Learned
  weights (BiFPN), attention (AIFI, gather-and-distribute) and custom ops (hypergraph message passing)
  narrow the set of runtimes that run the model fast.

---

## Key Takeaways

- FPN's top-down path gives fine maps semantics. PAN's bottom-up path gives coarse maps localisation
  in a few hops. YOLO's PAFPN combines both, with concatenation and CSP blocks.
- BiFPN learns how much each resolution contributes. RepGFPN optimises the generalised FPN for GPUs.
- Attention belongs where tokens are few and semantic. RT-DETR's hybrid encoder runs self-attention
  only on stride 32 and convolutional fusion elsewhere.
- Gather-and-distribute and FullPAD mix all levels globally. They are more accurate and harder to
  deploy.
- Plain ViTs need only a simple pyramid built from their last map. Foundation-model detectors use
  light adapters in the same spirit.
- Level count is a trade-off. P2 helps small objects at a large candidate cost, and YOLO27's
  (preliminary) two-level compact models question whether P4 is needed at all.

## Check Yourself

<details class="check"><summary>Why does RT-DETR apply self-attention only to the stride-32 map?</summary>
At 640 px, stride 32 gives 20 × 20 = 400 tokens, so global self-attention is cheap. Stride 8 gives
6,400 tokens, and attention over all levels (as in Deformable DETR's encoder) dominated the cost. The
coarse level also holds the most semantic features, where long-range interaction helps most. Local,
fine-grained fusion is done well by convolutions (CCFF).</details>

<details class="check"><summary>With FPN's level-assignment formula, which level handles a 300 × 150 RoI?</summary>
√(300·150) = 212.1, and log₂(212.1/224) = −0.08, so k = ⌊4 − 0.08⌋ = ⌊3.92⌋ = 3. The RoI is pooled
from P3 (stride 8). Because of the floor, anything just under the 224-pixel canonical size drops a
level. FPN also clamps k to the available levels, [2, 5].</details>

<details class="check"><summary>What does concatenation give a YOLO neck that FPN's addition does not, and what does it cost?</summary>
Concatenation keeps both inputs' channels intact, and the following CSP block learns how to combine
them, a richer fusion than a fixed sum. It doubles the channels entering the fusion block, which costs
FLOPs and memory traffic, and is why YOLO necks are a large share of total compute.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Feature Pyramid Networks | Lin et al., 2017 | arXiv:1612.03144 | Top-down + lateral, RoI level formula |
| Path Aggregation Network (PANet) | Liu et al., 2018 | arXiv:1803.01534 | Bottom-up augmentation |
| NAS-FPN | Ghiasi et al., 2019 | arXiv:1904.07392 | Searched topology |
| EfficientDet (BiFPN) | Tan et al., 2020 | arXiv:1911.09070 | Fast normalised fusion |
| Learning Spatial Fusion (ASFF) | Liu et al., 2019 | arXiv:1911.09516 | Per-pixel level weights |
| DAMO-YOLO (RepGFPN) | Xu et al., 2022 | arXiv:2211.15444 | Efficient generalised FPN |
| Gold-YOLO | Wang et al., 2023 | arXiv:2309.11331 | Gather-and-distribute |
| RT-DETR | Zhao et al., 2024 | arXiv:2304.08069 | AIFI + CCFF hybrid encoder |
| YOLOv13 | Lei et al., 2025 | arXiv:2506.17733 | HyperACE, FullPAD, +3.0 AP over YOLO11-N |
| ViTDet | Li et al., 2022 | arXiv:2203.16527 | Simple feature pyramid |
| Ultralytics `cfg/models/26/yolo26.yaml`, `yolo26-p2.yaml` | Ultralytics, 2026 | github.com/ultralytics/ultralytics | PAN structure, P2 variant |
| Ultralytics YOLO27 preview | Ultralytics, 2026 | docs/en/models/yolo27.md | Dual-scale N/S detection (preliminary) |

---

**Next:** [Chapter 13 — Heads](./13_heads.md) — fused features in hand, the head turns each location
or query into a class score and a box.
