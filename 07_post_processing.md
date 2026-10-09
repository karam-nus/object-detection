---
title: "Chapter 7 — Post-Processing & NMS"
---

[← Back to Table of Contents](./README.md)

# Chapter 7 — Post-Processing & NMS

> *"Non-maximum suppression is a 20-line function that cost the field a decade of latency and every crowded scene it ever saw."*

## Overview

A one-to-many detector emits several confident boxes per object, because assignment trained several
candidates to fire (Chapter 5). Post-processing reduces them to one. The classic tool is greedy
**non-maximum suppression**: sort by score, keep the best, delete everything that overlaps it too
much, repeat. This chapter covers the full decode → threshold → NMS pipeline. It covers the five NMS
variants worth knowing and Weighted Boxes Fusion for ensembles. It shows what NMS costs on real
hardware, and how one-to-one heads make the step disappear. Implementations are in
[`odlab/nms.py`](https://github.com/karam-nus/object-detection/blob/main/code/odlab/nms.py), tested
against `torchvision.ops.nms`.

<div class="diagram">
<div class="diagram-title">Two post-processing contracts</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">One-to-many head (YOLOv3–v13, YOLO26 default)</div>
    <ul>
      <li>decode 8,400 boxes (+ DFL if present)</li>
      <li>score threshold (0.25 deploy / 0.001 eval)</li>
      <li>pre-NMS top-k (≤ 30,000 in Ultralytics)</li>
      <li>class-aware greedy NMS, IoU 0.7</li>
      <li>keep ≤ 300 (max_det)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">One-to-one head (DETRs, YOLOv10, YOLO26 nms=False)</div>
    <ul>
      <li>sigmoid scores [N, K]</li>
      <li>top-k over all (candidate, class) pairs</li>
      <li>score threshold</li>
      <li>done — no IoU threshold, no loop</li>
      <li>output shape fixed: [B, 300, 6]</li>
    </ul>
  </div>
</div>
</div>

---

## Greedy NMS

```python
def nms(boxes, scores, iou_thr):                     # odlab/nms.py
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]; keep.append(i)
        ious = pairwise_iou(boxes[i:i+1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]           # drop everything overlapping the winner
    return torch.stack(keep)
```

`test_nms_matches_torchvision` checks that this returns *exactly* `torchvision.ops.nms`'s indices on
random inputs. The worst case is $O(n^2)$ IoU evaluations, $n(n-1)/2$ when nothing is suppressed. GPU
implementations compute the full IoU matrix in parallel and run the sequential selection as a bitmask
scan.

### Class-aware NMS in one call

Boxes of different classes should not suppress each other. Rather than running $K$ separate NMS calls,
**shift each class into its own region of coordinate space**:

$$b' = b + k \cdot M, \qquad M > \text{max coordinate}$$

Then a single class-agnostic NMS does the job, because boxes of different classes can never overlap.
`batched_nms` in `odlab`, `torchvision.ops.batched_nms` and Ultralytics' NMS (`max_wh = 7680`) all use
this trick. **Agnostic NMS** (`agnostic_nms=True` in Ultralytics) skips the shift, so a
"truck" box can suppress a "car" box on the same vehicle. That is useful when classes are visually
ambiguous and you want one box per physical object.

### The two thresholds and their defaults

| Setting | Evaluation (Ultralytics `val`) | Deployment (Ultralytics `predict`) | torchvision Faster R-CNN |
|---|:---:|:---:|:---:|
| Score threshold `conf` | **0.001** | 0.25 | 0.05 |
| NMS IoU `iou` | 0.7 | 0.7 | 0.5 |
| Max detections | 300 | 300 | 100 |

Evaluation uses a near-zero score threshold because AP integrates over *all* operating points
(Chapter 8). Cutting low-confidence boxes truncates the PR curve and lowers AP. Deployment picks one
operating point. Benchmarking a model's speed at `conf=0.001` and reporting its accuracy at
`conf=0.25` (or the reverse) is one of the most common ways speed–accuracy plots mislead
(Chapter 46).

---

## Interactive: NMS step by step

<div class="lab" data-lab="nms"></div>

Try the two-person cluster on the left. At IoU threshold 0.3, greedy NMS deletes the second person.
Soft-NMS keeps them with a lower score. DIoU-NMS keeps them because their centres are far apart.

---

## The variants

### Soft-NMS — decay instead of delete

Bodla et al. (2017) replace "delete if IoU > τ" with a score decay:

$$s_j \leftarrow s_j \cdot e^{-\text{IoU}(b_i, b_j)^2 / \sigma} \quad \text{(Gaussian)}, \qquad s_j \leftarrow s_j \cdot (1 - \text{IoU}) \text{ if IoU} > \tau \quad \text{(linear)}$$

An overlapped true object survives with a lower score instead of vanishing, which raises recall in
crowds. It typically adds ~1 AP for two-stage detectors on COCO, and less for modern dense detectors.
It is sequential and score-dependent, so it is slower than greedy NMS.
`test_soft_nms_keeps_more_than_hard_nms`.

### DIoU-NMS — use the centre distance

Zheng et al. (2020): suppress $b_j$ only if $\text{IoU} - \rho^2/c^2 > \tau$. Two pedestrians side by
side, one partly behind the other, can have high IoU but distant centres, and DIoU-NMS keeps both.
`test_diou_nms_spares_far_centres` builds exactly that case: IoU = 1/3, DIoU = 0.233.

### Matrix NMS — fully parallel

SOLOv2's Matrix NMS (Wang et al., 2020) decays every box's score by its worst overlap with any
higher-scoring box. It compensates by how suppressed that higher box is itself:

$$\text{decay}_j = \min_{i:\, s_i > s_j} \frac{f(\text{IoU}_{ij})}{f(\text{IoU}_{\cdot,i})}, \qquad f(x) = e^{-\sigma x^2}$$

where $\text{IoU}_{\cdot,i} = \max_{k: s_k > s_i} \text{IoU}_{ki}$. One matrix operation, no loop. It
suits hardware that dislikes data-dependent control flow. PP-YOLOE and PicoDet use it.
`test_matrix_nms_decays_duplicates`.

### Cluster-NMS and Fast NMS

**Fast NMS** (YOLACT) computes the upper-triangular IoU matrix and suppresses a box if *any*
higher-scoring box overlaps it. It is parallel, but it over-suppresses, because a box that should have
been deleted can still delete others. **Cluster-NMS** (Zheng et al., 2021) iterates Fast NMS until it
converges, which recovers exact greedy NMS results in a few parallel iterations. It also composes with
DIoU and weighted coordinates.

### Weighted Boxes Fusion — for ensembles and TTA

When combining several models or test-time augmentations, NMS throws information away. **WBF**
(Solovyev et al., 2019) clusters overlapping boxes across models and averages their coordinates
weighted by score. Fused confidence is the mean score times
$\min(\text{cluster size}, \#\text{models}) / \#\text{models}$, so a box only one model found is
down-weighted. WBF is a staple of Kaggle-style ensembles and of the large-detector chapter
(Chapter 17). `test_wbf_averages` fuses `[0,0,10,10]` and `[2,2,12,12]` into `[1,1,11,11]`.

| Method | Parallel? | Keeps overlapped objects? | Extra hyperparameters | Choose this when |
|---|---|---|---|---|
| **Greedy NMS** | partly (IoU matrix) | no | IoU τ | Default for one-to-many heads |
| **Soft-NMS** | no | yes, with decayed score | σ or τ | Crowds with two-stage detectors; offline |
| **DIoU-NMS** | partly | sometimes | τ | Side-by-side objects (pedestrians, shelves) |
| **Matrix NMS** | yes | yes, with decayed score | σ, final threshold | NPUs and GPUs that dislike loops |
| **Cluster-NMS** | yes (few iters) | as greedy, or better with DIoU | τ | Exact greedy results in parallel |
| **WBF** | no | merges | IoU τ, skip τ | Ensembles, TTA, offline accuracy |
| **Top-k only** | yes | n/a (no duplicates by training) | k | One-to-one heads |

---

## What NMS costs in practice

The YOLOv10 paper measured YOLOv8 on a T4 with TensorRT FP16, with and without post-processing:

| Model | Forward only (ms) | With NMS (ms) | NMS share |
|---|:---:|:---:|:---:|
| YOLOv8n | 1.77 | 6.16 | 71% |
| YOLOv8s | 2.33 | 7.07 | 67% |
| YOLOv8m | 5.09 | 9.50 | 46% |
| YOLOv8l | 8.06 | 12.39 | 35% |
| YOLOv8x | 12.83 | 16.86 | 24% |
| YOLOv10-N (no NMS) | — | 1.84 | 0% |

The absolute NMS cost is roughly constant per image (here 4–4.7 ms) because it depends on the number
of candidates, not on model size. So it dominates small models. The exact number depends on the score
threshold and the NMS implementation. Low thresholds leave thousands of boxes for NMS. Treat these
figures as the paper's measurement conditions, not universal constants. The *direction* is not in
dispute, and it is why "end-to-end" became a headline feature for YOLOv10 and YOLO26.

### NMS on edge hardware

- **Data-dependent shapes.** The number of kept boxes varies per image. Many NPU compilers require
  static shapes, so NMS runs on the host CPU after copying 8,400 × (4 + K) floats out of the
  accelerator. On a small ARM CPU that copy-and-sort can exceed the network's own latency.
- **Plugins.** TensorRT provides `EfficientNMS` and `BatchedNMS` plugins. ONNX has a
  `NonMaxSuppression` operator, but support varies by runtime. Ultralytics can embed NMS in the
  exported graph for some formats (`nms=True` at export).
- **The fix is upstream.** A one-to-one head turns post-processing into a top-k, which every runtime
  supports with static shapes. This was an explicit motivation for YOLO26's edge-first design
  (Chapter 36).

---

## Crowds: where NMS fails by design

On CrowdHuman (about 23 persons per image, many heavily overlapped), any single IoU threshold is
wrong. A low threshold deletes real people; a high threshold keeps duplicates. Research responses
include adaptive thresholds based on predicted density (Adaptive-NMS), set-level NMS for multiple
predictions per proposal (CrowdDet), and visible-box/full-box pairs. Set prediction sidesteps the
problem. A one-to-one detector never needed to tell duplicates from neighbours with a threshold, so
DETR-style models degrade more gracefully in crowds. Chapter 43 returns to crowded detection.

---

## One-to-one post-processing

For a one-to-one head with scores $P \in [0,1]^{N \times K}$:

```python
idx, labels, scores = topk_select(P, k=300)   # odlab/nms.py: top-k over flattened N*K
keep = scores > conf
boxes = decoded[idx][keep]
```

That is the whole post-processor for DETR, RT-DETR, D-FINE, DEIM, RF-DETR, YOLOv10 and YOLO26 with
`nms=False`. It is static-shape and branch-free. Its only knob, `k`, is a capacity limit rather than a
correctness threshold.

<div class="callout myth"><span class="callout-title">Myth</span>"NMS-free models never output
duplicates." They are <em>trained</em> not to, and on COCO they mostly don't. Under distribution
shift (unusual scales, heavy blur, objects larger than anything in training) the one-to-one head's
suppression can fail, and you will see occasional double boxes. A light, high-threshold NMS
(IoU 0.8–0.9) as a safety net costs little and is worth testing on your data.</div>

---

## Key Takeaways

- Greedy NMS is sort → keep best → drop overlaps → repeat. Class-aware NMS is a coordinate-offset
  trick around one agnostic call.
- Use `conf ≈ 0.001` for evaluation and a task-specific threshold for deployment. Mixing them makes
  speed or accuracy claims meaningless.
- Soft-NMS and DIoU-NMS help in crowds. Matrix and Cluster NMS are parallel. WBF fuses ensembles.
- NMS cost is roughly constant per image, so it dominates small models. In the YOLOv10 paper's
  measurements it was 71% of YOLOv8n's end-to-end latency.
- On NPUs, NMS often falls back to the CPU because of data-dependent shapes. One-to-one heads replace
  it with a static top-k.
- NMS-free heads remove the step, not every duplicate. Keep a cheap safety net when the deployment
  domain differs from training.

## Check Yourself

<details class="check"><summary>Why is AP computed with conf = 0.001 rather than the 0.25 used in deployment?</summary>
AP integrates precision over the whole recall range. Detections between 0.001 and 0.25 add the
low-confidence tail of the PR curve, extending recall. Cutting them removes that tail and lowers AP,
sometimes by several points for small models.</details>

<details class="check"><summary>Explain the coordinate-offset trick for class-aware NMS and the condition the offset must satisfy.</summary>
Add k·M to every coordinate of each box of class k. If M exceeds the largest coordinate, the boxes of
different classes occupy disjoint regions and can never have non-zero IoU. One class-agnostic NMS
therefore performs per-class suppression. Ultralytics uses M = 7680 (max_wh); torchvision uses
max(boxes) + 1.</details>

<details class="check"><summary>A YOLOv8n runs at 1.8 ms on your Jetson's GPU but the end-to-end pipeline takes 9 ms. Where would you look first?</summary>
Post-processing and transfers. NMS is probably running on the CPU over thousands of low-confidence
boxes. Raise the confidence threshold and use pre-NMS top-k, move NMS onto the GPU (TensorRT
EfficientNMS), or switch to an NMS-free model (YOLO26 with nms=False, YOLOv10, a DETR). Also check
preprocessing (letterbox on the CPU) and device-to-host copies (Chapter 46).</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Soft-NMS | Bodla et al., 2017 | arXiv:1704.04503 | Gaussian / linear decay |
| Distance-IoU Loss (DIoU-NMS) | Zheng et al., 2020 | arXiv:1911.08287 | DIoU suppression criterion |
| SOLOv2 (Matrix NMS) | Wang et al., 2020 | arXiv:2003.10152 | Parallel decay formula |
| YOLACT (Fast NMS) | Bolya et al., 2019 | arXiv:1904.02689 | Upper-triangular parallel NMS |
| Enhancing Geometric Factors (Cluster-NMS) | Zheng et al., 2021 | arXiv:2005.03572 | Iterative parallel NMS |
| Weighted Boxes Fusion | Solovyev et al., 2019 | arXiv:1910.13302 | Ensemble fusion |
| YOLOv10 | Wang et al., 2024 | arXiv:2405.14458 | YOLOv8 latency with vs without NMS on T4 |
| CrowdHuman | Shao et al., 2018 | arXiv:1805.00123 | Crowd statistics |
| Adaptive NMS; CrowdDet | Liu et al., 2019; Chu et al., 2020 | arXiv:1904.03629; 2003.09163 | Crowd-aware suppression |
| Ultralytics `utils/nms.py`, `cfg/default.yaml` | Ultralytics | github.com/ultralytics/ultralytics | max_wh, conf/iou defaults, max_det |
| torchvision.ops.nms / batched_nms | PyTorch | pytorch.org/vision | Reference for tests |
| `odlab/nms.py` + tests | this book | code/odlab | All implementations |

---

**Next:** [Chapter 8 — Metrics](./08_metrics.md) — we have a final set of detections. How good is
it? The answer, AP, is subtler than its fame suggests.
