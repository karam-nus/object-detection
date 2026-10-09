---
title: "Chapter 18 — DETR and Its Descendants"
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — DETR and Its Descendants

> *"DETR removed every hand-designed component of detection, and then spent four years putting smarter versions of them back."*

## Overview

DETR (2020) recast detection as **direct set prediction**. A transformer decoder turns $N$ learned
queries into $N$ predictions, a bipartite matching assigns each ground truth to exactly one prediction,
and there are no anchors and no NMS. It was elegant and very slow to train: 500 epochs, with weak
small-object accuracy. This chapter follows the lineage that fixed it: Deformable DETR, Conditional and
DAB-DETR, DN-DETR, DINO, the hybrid-matching family (H-DETR, Group-DETR, Co-DETR), Stable-DINO and
Plain-DETR. Chapter 19 covers the real-time descendants (RT-DETR to RF-DETR), and Chapter 20 the
open-vocabulary ones (Grounding DINO, DINO-X).

<div class="diagram">
<div class="diagram-title">DETR's pipeline</div>
<div class="flow-h">
  <div class="flow-node">CNN backbone<small>C5, stride 32</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">transformer encoder<small>global self-attention + positional enc.</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">decoder<small>N = 100 queries × 6 layers</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">FFN heads<small>class (K+1), box (cx,cy,w,h)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Hungarian loss<small>one-to-one</small></div>
</div>
</div>

---

## DETR

| | |
|---|---|
| **Introduced** | Carion et al. (Meta AI), 2020 (arXiv:2005.12872, ECCV 2020) |
| **Lineage** | transformers + set prediction losses |
| **Paradigm** | query-based, one-to-one |
| **Assignment** | Hungarian matching on class + L1 + GIoU cost |
| **Post-processing** | none (take queries whose class ≠ "no object") |
| **COCO AP (sizes)** | 42.0 (R50, 500 epochs); 43.5 (R101) |
| **Latency** | comparable to Faster R-CNN-FPN at similar FLOPs |
| **Pre-training** | ImageNet |
| **License** | Apache-2.0 |
| **Known failure modes** | ~500 epochs to converge; AP$_S$ well below Faster R-CNN; single-scale C5 features |

**Set loss.** After Hungarian matching $\hat{\sigma}$ (Chapter 5), the loss is

$$\mathcal{L} = \sum_{i=1}^{N}\Big[-\log \hat{p}_{\hat\sigma(i)}(c_i) + \mathbb{1}_{c_i \ne \varnothing}\big(\lambda_{L1}\|b_i - \hat{b}_{\hat\sigma(i)}\|_1 + \lambda_{giou}\,\mathcal{L}_{GIoU}\big)\Big]$$

with the "no object" class down-weighted (by 0.1) to handle imbalance. **Auxiliary losses** at every
decoder layer, each with its own matching, were essential. **Why it converged slowly:** global
cross-attention starts uniform over all pixels and must learn to focus. Queries carry no positional
prior. Matching is unstable early on, so a query's target changes from step to step.

---

## Deformable DETR — attend to a few points

| | |
|---|---|
| **Introduced** | Zhu et al., 2020 (arXiv:2010.04159, ICLR 2021) |
| **Lineage** | DETR + deformable convolution |
| **Paradigm** | query-based, multi-scale |
| **Assignment** | Hungarian (focal-loss class cost) |
| **Post-processing** | top-k over queries × classes |
| **COCO AP (sizes)** | ~46–49 with R50 at 50 epochs (two-stage + box refinement variant highest) |
| **Latency** | — |
| **Pre-training** | ImageNet |
| **License** | Apache-2.0 |
| **Known failure modes** | custom CUDA op (`MSDeformAttn`) complicates export; encoder cost at P3 |

**Multi-scale deformable attention.** Each query has a reference point $p_q$. On each of $L$ feature
levels it samples $K$ (typically 4) points at learned offsets $\Delta p_{lqk}$ and sums them with learned
weights $A_{lqk}$:

$$\text{MSDeformAttn}(z_q, p_q, \{x^l\}) = \sum_{m=1}^{M} W_m \left[\sum_{l=1}^{L}\sum_{k=1}^{K} A_{mlqk}\, W'_m\, x^l\big(\phi_l(p_q) + \Delta p_{mlqk}\big)\right]$$

Cost is linear in the number of queries and independent of image size. This gives multi-scale features
(and therefore small objects) and about 10× faster convergence (50 epochs vs 500). It also introduced
**iterative box refinement** and the **two-stage** variant, where encoder tokens propose the initial
queries. Both became standard.

---

## Making queries mean something: Conditional, Anchor DETR, DAB-DETR

DETR's queries are opaque embeddings. A series of papers gave them explicit spatial meaning:

- **Conditional DETR** (2021) separates content and spatial parts of the cross-attention, so each query
  attends near a spatial reference it predicts.
- **Anchor DETR** (2021) makes queries anchor *points*.
- **DAB-DETR** (2022) makes queries explicit **4-D anchor boxes** $(x, y, w, h)$, refined layer by layer,
  and uses $(w, h)$ to modulate the positional attention so it adapts to object scale.

The lesson is the same one the anchor-free YOLOs learned in reverse: a spatial prior helps optimisation,
as long as it is updated by the network rather than hand-tuned.

---

## Stabilising matching: DN-DETR and DINO

**DN-DETR** (2022) diagnosed the slow convergence as *bipartite matching instability*: early on, a
query's matched target flips between iterations. Its fix is to add **denoising queries** during
training. Ground-truth boxes are jittered (noised positions and sizes, flipped labels) and fed as extra
queries whose target is *known*, the original ground truth. They bypass matching entirely and teach
the decoder to reconstruct boxes from nearby anchors. An attention mask stops them from leaking answers
to the matching queries. At inference they are absent.

**DINO** (2022) combined and extended everything:

| | |
|---|---|
| **Introduced** | Zhang et al. (IDEA), 2022 (arXiv:2203.03605, ICLR 2023) |
| **Lineage** | Deformable DETR + DAB-DETR + DN-DETR |
| **Paradigm** | query-based, multi-scale deformable |
| **Assignment** | Hungarian + contrastive denoising groups |
| **Post-processing** | top-k |
| **COCO AP (sizes)** | 49.0 (R50, 12 epochs, 4-scale); 58.5 (Swin-L, 36 ep); 63.2 val / 63.3 test-dev (Swin-L + Objects365) |
| **Latency** | not real-time |
| **Pre-training** | ImageNet / IN-22k / Objects365 |
| **License** | Apache-2.0 |
| **Known failure modes** | heavy encoder; custom ops; long-tail classes |

Its three additions:

1. **Contrastive denoising (CDN).** Each ground truth gets *two* noised queries: a small-noise positive
   (reconstruct the box) and a larger-noise **negative** (predict "no object"). Negatives teach the model
   to reject near-duplicates, which is NMS's job learned inside the network.
2. **Mixed query selection.** Positional queries (anchor boxes) are initialised from the top-$k$
   encoder tokens. Content queries stay learned. The model starts by looking at likely objects.
3. **Look forward twice.** Box refinement at layer $l$ also receives gradient from layer $l+1$'s
   prediction, which improves early layers.

DINO's 12-epoch R50 result (49.0 AP) roughly matched what DETR needed 500 epochs to approach, and its
Swin-L + Objects365 result put a DETR at the top of COCO for the first time.

---

## More positives without losing one-to-one: hybrid matching

One-to-one matching gives each object one positive query. That is necessary for NMS-free inference, and
it leaves the model starved of positive gradients. Three 2022–2023 papers add **one-to-many supervision
during training only**:

| Method | How | Inference |
|---|---|---|
| **H-DETR** (hybrid matching) | an extra query group trained with one-to-many matching (each GT repeated k times) | drop that group |
| **Group-DETR** | several independent query groups, each with its own one-to-one matching | keep one group |
| **Co-DETR** | parallel auxiliary *dense* heads (ATSS, Faster R-CNN) on the encoder; their positives also become customised decoder queries | drop auxiliaries |

All three add AP at no inference cost. Co-DETR became the strongest of them (66.0 AP test-dev with
ViT-L). The same idea in YOLO form is YOLOv10/YOLO26's dual-head training (Chapter 5), and DEIM's Dense
O2O is the data-side version (more objects per image instead of more queries per object).

---

## Fixing what the loss rewards: Stable-DINO, MS-DETR, DEIM's MAL

- **Stable-DINO** (2023) observed that the classification target and the matching cost can disagree,
  so a query can be "positive" with a poor box. It uses position-supervised losses whose class targets
  depend on IoU, the DETR analogue of VFL/TAL soft targets.
- **MS-DETR** (2024) adds one-to-many supervision on the *decoder* queries themselves (mixed supervision).
- **DEIM's matchability-aware loss (MAL)** (2024) re-weights the classification loss by matching quality,
  so dense one-to-one training does not over-reward low-IoU matches.

These echo the dense-detector story in Chapters 5 and 6. Once assignment is fixed, the next gains come
from making the classification target reflect localisation quality.

---

## Plain-DETR — when the backbone is good enough

**Plain-DETR** (Lin et al., 2023) asks whether DETR needs multi-scale features or locality-biased
decoders at all. With a strong plain ViT backbone (MIM pre-trained), a single-scale stride-16 feature
map, and a **box-to-pixel relative position bias** in the decoder, it matches multi-scale DETRs. DINOv3
(2025) used a Plain-DETR decoder on a *frozen* DINOv3 ViT-7B to reach 66.1 COCO AP (Chapter 17). As
backbones grow, the decoder simplifies.

---

## The DETR family at a glance

| Model | Year | Key idea | Epochs (R50) | COCO AP | Choose this when |
|---|:---:|---|:---:|:---:|---|
| **DETR** | 2020 | set prediction, Hungarian | 500 | 42.0 | Teaching; conceptual baseline |
| **Deformable DETR** | 2020 | sparse multi-scale attention, refinement, two-stage | 50 | ~46–49 | Foundation of most later DETRs |
| **DAB-DETR** | 2022 | queries as 4-D anchor boxes | 50 | ~42–47 (single- vs multi-scale variant) | Studying query design |
| **DN-DETR** | 2022 | denoising queries | 12–50 | ~43–49 (variant, schedule) | Faster convergence |
| **DINO** | 2022 | contrastive denoising, mixed query selection | 12 | 49.0 | Strong research baseline (detrex) |
| **H-DETR / Group-DETR** | 2022–23 | one-to-many groups at train time | 12–36 | +1–2 over base | Cheap AP boost for any DETR |
| **Co-DETR** | 2023 | auxiliary dense heads | 12–36 | 52.1 (R50 12ep) → 66.0 (ViT-L) | Maximum accuracy |
| **Stable-DINO** | 2023 | IoU-aware matching/losses | 12–24 | ~50–51 | Better high-IoU AP |
| **Plain-DETR** | 2023 | single-scale, big ViT | — | competitive with multi-scale DETRs at equal backbone | Frozen or huge foundation backbones |

Exact numbers vary by schedule and implementation. The [Model Index](./appendix_c_model_index.md) records
sourced figures for the models used elsewhere in the book.

---

## Key Takeaways

- DETR made detection a set-prediction problem: Hungarian matching, no anchors, no NMS. Its weaknesses
  were 500-epoch convergence and poor small-object AP.
- Deformable attention (sparse, multi-scale, linear-cost) fixed most of the cost and small-object
  problems, and introduced box refinement and two-stage query proposals.
- Giving queries spatial meaning (DAB anchor boxes) and stabilising matching with denoising queries
  (DN-DETR, DINO's contrastive denoising) brought convergence to 12 epochs.
- Hybrid one-to-many supervision at train time (H-DETR, Group-DETR, Co-DETR) adds AP at zero inference
  cost. It is the same idea as YOLOv10/26 dual heads.
- With very strong backbones, the decoder simplifies again (Plain-DETR). DINOv3 reached 66.1 AP with a
  frozen backbone.

## Check Yourself

<details class="check"><summary>What problem do DN-DETR's denoising queries solve, and why can't they be used at inference?</summary>
They solve matching instability: early in training a query's Hungarian target changes between steps,
which gives inconsistent gradients. Denoising queries are noised ground-truth boxes whose targets are
known, so they provide a stable reconstruction signal without matching. At inference there is no ground
truth to noise, and they were only ever a training scaffold.</details>

<details class="check"><summary>Why does DINO's contrastive denoising act like a learned NMS?</summary>
The negative denoising queries sit near a ground truth (larger noise) and are trained to predict "no
object". The model learns that a box near an object that is not the best one should be rejected. That
is the duplicate-suppression behaviour NMS provides, learned inside the network.</details>

<details class="check"><summary>Why is deformable attention's cost independent of image size, while DETR's encoder attention is quadratic?</summary>
DETR's encoder attends from every token to every token: O(T²) for T = HW/32² tokens. Deformable attention
lets each query sample a fixed number of points (M heads × L levels × K points), so cost is O(T·M·L·K):
linear in the number of queries and independent of how many pixels each level has.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| End-to-End Object Detection with Transformers (DETR) | Carion et al., 2020 | arXiv:2005.12872 | Set loss, 42.0 / 43.5 AP, 500 epochs |
| Deformable DETR | Zhu et al., 2020 | arXiv:2010.04159 | MSDeformAttn, refinement, two-stage |
| Conditional DETR | Meng et al., 2021 | arXiv:2108.06152 | Conditional spatial queries |
| Anchor DETR | Wang et al., 2022 | arXiv:2109.07107 | Anchor-point queries |
| DAB-DETR | Liu et al., 2022 | arXiv:2201.12329 | 4-D anchor-box queries |
| DN-DETR | Li et al., 2022 | arXiv:2203.01305 | Denoising queries |
| DINO | Zhang et al., 2022 | arXiv:2203.03605 + README | CDN, mixed query selection, look-forward-twice; AP figures |
| H-DETR | Jia et al., 2023 | arXiv:2207.13080 | Hybrid matching |
| Group DETR | Chen et al., 2023 | arXiv:2207.13085 | Group-wise one-to-one |
| Co-DETR | Zong et al., 2023 | arXiv:2211.12860 + README | Auxiliary heads, 52.1 (R50 12ep) to 66.0 |
| Detection Transformer with Stable Matching (Stable-DINO) | Liu et al., 2023 | arXiv:2304.04742 | Position-supervised loss |
| MS-DETR | Zhao et al., 2024 | arXiv:2401.03989 | Mixed supervision |
| DEIM | Huang et al., 2024 | arXiv:2412.04234 | Dense O2O, MAL |
| DETR Doesn't Need Multi-Scale or Locality Design (Plain-DETR) | Lin et al., 2023 | arXiv:2308.01904 | Plain DETR |
| detrex | Ren et al., 2023 | arXiv:2306.07265 + github.com/IDEA-Research/detrex | Unified benchmarks of DETR variants |

---

**Next:** [Chapter 19 — Real-Time DETRs & ViT Backbones](./19_realtime_detrs_and_vit.md) — how this
family went from research curiosity to beating YOLO at equal latency.
