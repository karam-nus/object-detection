---
title: "Chapter 4 — Anchors, Points and Queries"
---

[← Back to Table of Contents](./README.md)

# Chapter 4 — Anchors, Points and Queries

> *"A detector can only find an object that some candidate was in a position to see."*

## Overview

Before a detector can classify or regress anything, it has to decide **where candidates live**.
There are four answers: anchor boxes tiled over a grid, anchor points (grid-cell centres), learned
queries that attend wherever they like, and token budgets in generative models. This chapter covers
the geometry: strides, grid sizes, the famous 8,400, k-means anchors, scale ranges and query
selection. It ends with an interactive look at the small-object problem. Which candidate becomes
*responsible* for an object is assignment, covered in Chapter 5.

<div class="diagram">
<div class="diagram-title">Candidate counts for a single image</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card blue"><div class="card-title">~10⁵ anchors</div><div class="card-desc">RetinaNet: 9 anchors × every P3–P7 location (≈100k quoted in the paper)</div></div>
  <div class="diagram-card green"><div class="card-title">25,200 anchors</div><div class="card-desc">YOLOv5 at 640: 3 anchors × 8,400 cells</div></div>
  <div class="diagram-card accent"><div class="card-title">8,400 points</div><div class="card-desc">YOLOv8 / 11 / 26 at 640: one point per cell</div></div>
  <div class="diagram-card purple"><div class="card-title">100–300 queries</div><div class="card-desc">DETR 100; RT-DETR, D-FINE, DEIM, YOLO26-e2e 300</div></div>
</div>
</div>

---

## Strides and the feature pyramid

A backbone halves the spatial resolution five times. The level whose stride is $s_l = 2^l$ is
called $P_l$. A $640 \times 640$ input gives:

| Level | Stride $s_l$ | Grid | Cells | Typical object size it handles |
|---|:---:|:---:|:---:|---|
| P2 | 4 | 160 × 160 | 25,600 | tiny (< 16 px); optional `-p2` models |
| **P3** | 8 | 80 × 80 | 6,400 | small (≈ 8–64 px) |
| **P4** | 16 | 40 × 40 | 1,600 | medium (≈ 32–128 px) |
| **P5** | 32 | 20 × 20 | 400 | large (≈ 96 px +) |
| P6 | 64 | 10 × 10 | 100 | very large; 1280-input `-p6` models |

$6{,}400 + 1{,}600 + 400 = 8{,}400$. That is the length of every YOLOv8/11/26 output, and
`make_grid_points` reproduces it (`test_grid_points_count_8400`). With a P2 head the count jumps to
34,000, which is why P2 heads cost latency in the head and in NMS.

**Rule of thumb.** An object should cover at least about 2×2 cells at the level that predicts it.
Below that, the object's evidence is blurred into a single feature vector whose receptive field is
dominated by background. In theory the receptive field of P5 in a modern YOLO covers the entire 640
input. The *effective* receptive field (Luo et al., 2016) is much smaller and roughly Gaussian, which
is why stride, not theoretical receptive field, governs small-object recall.

---

## Anchor boxes

An **anchor** is a reference box at a grid location. The network predicts offsets from it
(Chapter 2's $t_x, t_y, t_w, t_h$). Anchors encode a prior on *shape*: "objects here tend to be
tall and thin, or wide and short."

### Tiling

Faster R-CNN's RPN uses 3 scales × 3 aspect ratios = 9 anchors per location on one feature map.
RetinaNet uses 9 per location on each of P3–P7. `odlab.anchors.make_anchor_boxes(feature_size,
stride, sizes, ratios)` tiles them: area $\text{size}^2$, aspect $h/w = \text{ratio}$, centred at
$(j+0.5)s, (i+0.5)s$.

### Choosing anchors with k-means (YOLOv2)

Redmon and Farhadi noticed that hand-picked anchors were a poor fit for VOC and COCO. They clustered
ground-truth $(w, h)$ pairs with k-means using the distance

$$d(\text{box}, \text{centroid}) = 1 - \text{IoU}(\text{box}, \text{centroid})$$

where both are centred at the origin, so only width and height matter. Euclidean distance on
$(w,h)$ would let large boxes dominate. The IoU distance treats a 10-pixel error on a 20-pixel box as
worse than on a 300-pixel box, which is what matching needs. YOLOv3's nine COCO anchors (in pixels at
416, three per level) became famous defaults:

| Level | Anchors (w, h) |
|---|---|
| P3 /8 | (10, 13), (16, 30), (33, 23) |
| P4 /16 | (30, 61), (62, 45), (59, 119) |
| P5 /32 | (116, 90), (156, 198), (373, 326) |

`odlab.anchors.kmeans_anchors(wh, k)` implements the IoU-distance k-means with median updates
(YOLOv5's autoanchor additionally refines the result with a genetic algorithm).

### Checking anchors: best possible recall

YOLOv5 matches a ground truth to an anchor if both side ratios are within a factor of
`anchor_t = 4`. Before training it computes the **best possible recall (BPR)**: the fraction of
ground truths that at least one anchor *could* match. If BPR < 0.98 it re-fits anchors.
`odlab.anchors.best_possible_recall` implements the same check. On a custom dataset of tall
narrow objects (poles, bottles, text lines), the COCO anchors can leave a third of objects
unmatchable. The model then cannot learn them at all, whatever else you tune.

<div class="callout field"><span class="callout-title">Field note</span>Most "my YOLOv5/v7 model
never detects class X" bugs on custom data come down to one of two causes. Either BPR is low because
of unusual aspect ratios, or class X is very small at the strides used. Check BPR and the size
histogram before touching hyperparameters.</div>

---

## Anchor points (anchor-free)

**FCOS (Tian et al., 2019)** dropped anchor boxes. Every location $(x, y)$ on a feature map is a
candidate. If it falls inside a ground truth it regresses the four distances $(l, t, r, b)$ to the
box edges. Two questions remain that anchors used to answer implicitly:

1. **Which level handles which object?** FCOS assigns by regression range. A location on level $l$
   is a positive only if $\max(l, t, r, b)$ lies in that level's range: P3 $[0, 64]$, P4
   $[64, 128]$, P5 $[128, 256]$, P6 $[256, 512]$, P7 $[512, \infty)$.
2. **Which locations inside the box?** Locations near the border produce poor boxes. FCOS predicts a
   **centre-ness** score $\sqrt{\frac{\min(l,r)}{\max(l,r)} \cdot \frac{\min(t,b)}{\max(t,b)}}$ and
   multiplies it into the class score. Later versions add *centre sampling*: only points within a
   radius of the centre are candidates.

Modern YOLOs (v6, v8, 11, 26), YOLOX, PP-YOLOE and RTMDet are all anchor-free in this sense: one point
per cell per level, 8,400 at 640. They do **not** use fixed scale ranges. Their dynamic assigners
(SimOTA, TAL) compare every candidate on every level against every ground truth and pick the best by
a cost, so the level choice is learned (Chapter 5).

### Why anchor-free won

| Aspect | Anchor boxes | Anchor points |
|---|---|---|
| Candidates per cell | 3 (YOLO) – 9 (RetinaNet) | 1 |
| Dataset-specific hyperparameters | sizes, ratios, IoU thresholds | essentially none |
| Head channels | $A \times (4 + K)$ | $4 + K$ (or $4R + K$ with DFL) |
| Export / decode | anchor table baked into the graph | grid + stride only |
| Accuracy | equal, once assignment is matched (ATSS) | equal |

ATSS (Zhang et al., 2020) showed that one anchor per location with adaptive assignment performs as
well as nine. The remaining benefit of anchors, a shape prior, is supplied in practice by good
assignment and enough data.

### Points as heat-map peaks

**CornerNet** (2018) predicts heat maps of top-left and bottom-right corners and groups them with
embeddings. **CenterNet** ("Objects as Points", 2019) predicts a heat map of object centres at stride 4
($128 \times 128$ for a 512 input) and regresses size at each peak. A 3×3 max-pool picks the peaks,
which replaces NMS. Heat-map detectors are close relatives of the dense point detectors. They live
on in pose estimation and in FOMO-style MCU detectors that output only centroids (Chapter 15).

---

## Learned queries

**DETR** replaces the grid with $N = 100$ learned **object queries**: embeddings that a transformer
decoder turns into boxes by cross-attending to image features. Queries have no fixed location. At
the start of training they are interchangeable, and Hungarian matching gradually makes them
specialise (one query to "large objects in the centre", another to "small objects at the left").
DETR's slow convergence is partly the cost of discovering, by gradient descent, the spatial prior
that a grid gives for free.

The DETR descendants (Chapter 18) put the prior back in stages:

| Model | What a query is | Effect |
|---|---|---|
| **DETR** (2020) | content embedding only | 500 epochs |
| **Deformable DETR** (2020) | + a reference point; attends to a few sampled locations | ~50 epochs; multi-scale |
| **DAB-DETR** (2022) | an explicit 4-D anchor box $(x, y, w, h)$ refined layer by layer | faster, interpretable |
| **DN-DETR / DINO** (2022) | + noisy ground-truth "denoising" queries during training | stable matching |
| **Two-stage / query selection** (Deformable, DINO, RT-DETR) | initial queries = top-$k$ encoder features by class score | queries start at likely objects |

Real-time DETRs (RT-DETR, D-FINE, DEIM, RF-DETR) use **300 queries** selected from the encoder's
dense features. In effect they are dense detectors for the first stage and query-based for the
second. The number of queries caps the number of detections. That is fine for COCO, where
evaluation keeps 100 detections per image (`maxDets=100`), and too few for a shelf of 500 products
(SKU-110K averages about 147 objects per image).

---

## Token budgets

A generative MLLM emits boxes as text: four coordinate numbers or tokens per object. Its "candidate
count" is the generation budget. Output cost is linear in the number of objects: about four
coordinate tokens plus a label per box, generated sequentially. Rex-Omni reports about 5 boxes per
second of decoding, and NVIDIA's LocateAnything about 12.7 boxes per second with parallel box
decoding. A dense detector produces 300 boxes in one forward pass. Chapter 21 develops this cost model.

---

## Interactive: who can see this object?

<div class="lab" data-lab="grid"></div>

Things to try: shrink the box to about 20 × 20 at stride 32 and move it between grid centres. With
the "centre inside box" rule it gets **zero** candidates. Switch to SimOTA's 2.5-stride centre
radius and it gets several. Tick STAL (YOLO26's small-target-aware rule, which enlarges sub-16-pixel
boxes to 16 pixels when selecting candidates) and it is rescued at stride 16. The same experiment is
`test_tal_positives_lie_inside_gt_and_stal_rescues_tiny` in the companion code.

---

## Key Takeaways

- The stride sets the finest detail a level can localise. A 640 input with P3–P5 gives 8,400 grid
  cells, and a P2 head adds 25,600 more.
- Anchor boxes encode shape priors. Fit them with IoU-distance k-means and verify best possible
  recall on custom data.
- Anchor-free points need explicit or learned level selection: FCOS regression ranges, or dynamic
  assigners that search all levels.
- With equal assignment, anchor-free matches anchor-based accuracy, and it is simpler to export and
  tune.
- Queries are candidates without a fixed location. The DETR lineage re-added locality step by step:
  reference points, anchor-box queries, denoising, query selection.
- Generative detectors pay sequential decoding cost per object, which limits dense scenes.

## Check Yourself

<details class="check"><summary>How many candidates does a 1280×1280 YOLO with a P6 head (strides 8–64) produce?</summary>
160² + 80² + 40² + 20² = 25,600 + 6,400 + 1,600 + 400 = 34,000.</details>

<details class="check"><summary>Why does k-means for anchors use 1 − IoU instead of Euclidean distance on (w, h)?</summary>
Euclidean distance weights absolute pixel errors equally, so large boxes dominate the clustering.
Matching quality depends on IoU, which is scale-relative. With 1 − IoU, centroids are placed to
maximise the overlap any box can achieve with its closest anchor.</details>

<details class="check"><summary>A DETR with 100 queries is evaluated on images containing 150 objects. What happens to recall, and how do real-time DETRs avoid it?</summary>
At most 100 objects can be detected, so recall is capped at 100/150 ≈ 0.67 for that image no matter
how good the model is. Real-time DETRs use 300 queries, and dense-retail models raise the count
further. COCO evaluation itself only keeps 100 detections per image, so COCO AP hides the problem.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Faster R-CNN | Ren et al., 2015 | arXiv:1506.01497 | Anchor boxes, 3×3 per location |
| Focal Loss (RetinaNet) | Lin et al., 2017 | arXiv:1708.02002 | 9 anchors on P3–P7, ~100k anchors |
| YOLO9000 | Redmon & Farhadi, 2017 | arXiv:1612.08242 | IoU-distance k-means |
| YOLOv3 | Redmon & Farhadi, 2018 | arXiv:1804.02767 | The nine COCO anchors |
| Ultralytics YOLOv5 `autoanchor` | Ultralytics, 2020 | github.com/ultralytics/yolov5 | BPR check, anchor_t = 4 |
| FCOS | Tian et al., 2019 | arXiv:1904.01355 | Points, regression ranges, centre-ness |
| ATSS | Zhang et al., 2020 | arXiv:1912.02424 | One anchor per location suffices |
| Objects as Points (CenterNet) | Zhou et al., 2019 | arXiv:1904.07850 | Heat-map centres at stride 4 |
| CornerNet | Law & Deng, 2018 | arXiv:1808.01244 | Corner heat maps |
| Understanding the Effective Receptive Field | Luo et al., 2016 | arXiv:1701.04128 | ERF is Gaussian and small |
| DETR / Deformable DETR / DAB-DETR / DINO | Carion 2020; Zhu 2020; Liu 2022; Zhang 2022 | arXiv:2005.12872, 2010.04159, 2201.12329, 2203.03605 | Query evolution |
| SKU-110K | Goldman et al., 2019 | arXiv:1904.00853 | ~147 objects per image |
| Rex-Omni; LocateAnything | IDEA 2025; NVIDIA 2026 | arXiv:2510.12798; research.nvidia.com | Boxes-per-second decoding rates |

---

**Next:** [Chapter 5 — Label Assignment](./05_label_assignment.md) — candidates exist; now we decide
which of them should learn to fire for which object. It is the most consequential decision in the
book.
