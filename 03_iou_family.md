---
title: "Chapter 3 — IoU and Its Family"
---

[← Back to Table of Contents](./README.md)

# Chapter 3 — IoU and Its Family

> *"IoU is the metric everybody agrees on and the loss nobody can use directly."*

## Overview

Intersection-over-Union is the single similarity function underneath almost everything in detection.
Assigners threshold it (Chapter 5). Losses maximise it (Chapter 6). NMS suppresses by it
(Chapter 7). AP is defined at thresholds of it (Chapter 8). As a *loss*, though, plain IoU has two
defects: it gives zero gradient when boxes do not overlap, and it is blind to *how* boxes disagree.
GIoU, DIoU, CIoU, EIoU and SIoU each patch one of these defects. NWD and ProbIoU change the model of
a box altogether. This chapter derives each variant, checks it numerically against `odlab` and
torchvision, and records which detectors use which.

<div class="diagram">
<div class="diagram-title">What each variant adds</div>
<div class="flow">
  <div class="flow-node accent wide">IoU = |A∩B| / |A∪B| <small>scale-invariant overlap; zero gradient when disjoint</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">GIoU <small>+ penalty on empty space in the enclosing box → gradient when disjoint</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">DIoU <small>+ normalised centre distance → faster convergence, better NMS</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">CIoU <small>+ aspect-ratio consistency → the YOLOv4–YOLO26 default</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">EIoU / SIoU / MPDIoU <small>direct width/height, angle-aware or corner-distance terms</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">NWD / ProbIoU <small>boxes as Gaussians → smooth for tiny and rotated objects</small></div>
</div>
</div>

<div class="lab" data-lab="iou"></div>

---

## IoU

For boxes $A, B$ in `xyxy`:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

with intersection width $\max(0, \min(x_2^A, x_2^B) - \max(x_1^A, x_1^B))$, and the same for height.

**Properties.** It lies in $[0,1]$. It is symmetric and invariant to scaling both boxes together.
Unlike the L1 loss it does not care how large the box is: a 4-pixel error means nothing on a
400-pixel object and everything on an 8-pixel one. $1 - \text{IoU}$ is a metric (Jaccard distance).

**Numerical check (also `test_worked_example_from_chapter_03`).** $A = [0,0,4,4]$, $B = [2,2,6,6]$:
intersection $2 \times 2 = 4$, union $16 + 16 - 4 = 28$, IoU $= 1/7 \approx 0.143$.

### IoU is harsh on small objects

The same one-pixel shift costs a small box far more IoU:

| Box size | Shift | Intersection | Union | IoU |
|---|:---:|:---:|:---:|:---:|
| 8 × 8 | 1 px in x | 7 × 8 = 56 | 72 | **0.778** |
| 32 × 32 | 1 px in x | 31 × 32 = 992 | 1,056 | **0.939** |
| 64 × 64 | 1 px in x | 63 × 64 = 4,032 | 4,160 | **0.969** |

At COCO's 0.75 and 0.9 thresholds an 8-pixel object is barely matchable under ordinary annotation
noise. That is one reason AP$_S$ is always the lowest number in a COCO table, and the reason NWD
exists (below and Chapter 41).

### The gradient problem

If $A \cap B = \emptyset$ then IoU $= 0$ wherever the predicted box moves, as long as the boxes stay
disjoint. The gradient is exactly zero (`test_disjoint_boxes_have_gradient_under_giou_not_iou`). A
dense detector rarely starts with disjoint *assigned* pairs, because assignment usually requires the
anchor point to lie inside the ground truth. Early-training predictions and DETR queries do start
far away, and those need a signal.

---

## GIoU — penalising empty space

Rezatofighi et al. (2019): let $C$ be the smallest axis-aligned box enclosing both $A$ and $B$.

$$\text{GIoU} = \text{IoU} - \frac{|C| - |A \cup B|}{|C|}$$

The penalty is the fraction of $C$ not covered by either box. It is non-zero and has a gradient even
when the boxes are disjoint, and it shrinks as the boxes move toward each other. Range $(-1, 1]$.

**Numerical check.** Same $A, B$: $C = [0,0,6,6]$, $|C| = 36$, $|A \cup B| = 28$.
GIoU $= 1/7 - 8/36 = 0.143 - 0.222 = -0.079$.

**Failure mode.** When one box contains the other, $C = A \cup B$ and GIoU reduces to IoU. The
penalty vanishes, and convergence in that regime is as slow as plain IoU (Zheng et al., 2020 show this
with simulation experiments).

**Used by:** DETR and almost every DETR descendant (with L1), PP-YOLOE, YOLOv6-M/L, RTMDet,
DAMO-YOLO.

---

## DIoU — penalising centre distance

Zheng et al. (2020): with $\rho$ the distance between box centres and $c$ the diagonal of $C$:

$$\text{DIoU} = \text{IoU} - \frac{\rho^2(A, B)}{c^2}$$

Minimising $\rho^2/c^2$ directly pulls the centres together, including when one box contains the
other. In the paper's simulations it converged several times faster than GIoU. DIoU is also a better
NMS criterion: two heavily overlapping boxes whose centres are far apart are probably two objects
(Chapter 7).

**Numerical check.** Centres $(2,2)$ and $(4,4)$: $\rho^2 = 8$, $c^2 = 6^2 + 6^2 = 72$.
DIoU $= 0.143 - 0.111 = 0.032$.

---

## CIoU — adding aspect ratio

The same paper adds an aspect-ratio consistency term:

$$\text{CIoU} = \text{DIoU} - \alpha v, \qquad v = \frac{4}{\pi^2}\left(\arctan\frac{w^{g}}{h^{g}} - \arctan\frac{w}{h}\right)^2, \qquad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

$v \in [0, 1)$ measures the angular difference between the boxes' diagonal slopes. $\alpha$ is a
trade-off weight: when IoU is low the term is suppressed, so overlap is fixed first. In the reference
implementation, and in torchvision, `odlab` and Ultralytics, **$\alpha$ is computed under
`no_grad`**. It acts as a weight, not as a term to differentiate.

**Numerical check.** $A$ and $B$ are both squares, so $v = 0$ and CIoU = DIoU = 0.032
(`test_worked_example_from_chapter_03` asserts this).

**Used by:** YOLOv4, YOLOv5, YOLOv7, YOLOv8, YOLO11, YOLO26, as both the box loss *and* the IoU
inside the task-aligned assigner (Ultralytics `TaskAlignedAssigner.iou_calculation` uses CIoU clamped
at 0).

<div class="callout warn"><span class="callout-title">Trap</span>CIoU can be negative, and the
assigner clamps it at zero. Code that takes <code>u ** beta</code> with <code>beta = 6</code> on an
unclamped negative CIoU produces positive values for terrible boxes. Ultralytics clamps;
<code>odlab.assign.task_aligned_assign</code> clamps too.</div>

---

## EIoU, SIoU, MPDIoU — three more refinements

**EIoU** (Zhang et al., 2021) replaces CIoU's coupled aspect-ratio term with separate width and height
penalties normalised by the enclosing box:

$$\text{EIoU} = \text{IoU} - \frac{\rho^2}{c^2} - \frac{(w - w^g)^2}{C_w^2} - \frac{(h - h^g)^2}{C_h^2}$$

CIoU's $v$ stays zero whenever the aspect ratios match, even if both sides are wrong by the same
factor. EIoU penalises the sizes directly. The paper pairs it with a "Focal-EIoU" re-weighting.

**SIoU** (Gevorgyan, 2022) adds an *angle cost*. When the vector between the centres is close to
45°, the loss first pushes the prediction toward the nearer axis, and then shrinks the
remaining one-dimensional distance. It combines angle, distance and shape costs:
$\text{SIoU} = \text{IoU} - (\Delta + \Omega)/2$. YOLOv6 v3.0 uses SIoU for its N and T models.

**MPDIoU** (Ma & Xu, 2023) penalises the distances between the top-left corners and between the
bottom-right corners, normalised by the image size. It is simple and appears in many YOLO-derivative
papers. Evidence beyond those papers is thin.

`odlab.boxes` implements EIoU and SIoU. Both equal 1 for identical boxes
(`test_siou_and_eiou_are_one_for_identical_boxes`).

<div class="callout field"><span class="callout-title">Field note</span>Swapping CIoU for one of
these on a mature YOLO recipe rarely moves COCO AP by more than a few tenths. The large
reported gains usually come from weak baselines or small datasets. Change the IoU variant last.
Assignment, augmentation and schedule matter more.</div>

---

## Boxes as Gaussians: NWD and ProbIoU

For tiny objects, any overlap-based measure is brittle. Wang et al. (2021) model each box as a 2-D
Gaussian $\mathcal{N}\big((c_x, c_y),\ \mathrm{diag}(w^2/4, h^2/4)\big)$. The squared 2-Wasserstein
distance between two such Gaussians has the closed form

$$W_2^2 = \left\| \left(c_x^A, c_y^A, \tfrac{w^A}{2}, \tfrac{h^A}{2}\right) - \left(c_x^B, c_y^B, \tfrac{w^B}{2}, \tfrac{h^B}{2}\right) \right\|_2^2$$

and the **Normalized Wasserstein Distance** is $\text{NWD} = \exp(-\sqrt{W_2^2}/C)$, where $C$ is a
dataset constant (12.8 on AI-TOD). NWD is smooth, non-zero for disjoint boxes, and **depends only on
absolute pixel offsets**. A 2-pixel shift costs a 6-pixel box exactly as much as a 60-pixel box.
IoU, by contrast, collapses for the small box (`test_nwd_scale_behaviour`). NWD can replace IoU in
assignment, NMS and the loss for tiny-object datasets (Chapter 41).

**ProbIoU** (Llerena et al., 2021) uses the Bhattacharyya distance between box Gaussians. Because a
rotated box is just a Gaussian with a rotated covariance, it has no angle-wrap discontinuity.
Ultralytics' OBB models use it for both the loss and rotated NMS (Chapter 43).

---

## Which detectors use what

| IoU variant | Used for | Detectors | Choose this when |
|---|---|---|---|
| **IoU** | metric, NMS, simple loss | everything (metric), YOLOX loss, FCOS | Evaluation and NMS; as a loss only with good initialisation |
| **GIoU** | loss (+ L1) | DETR, DINO, RT-DETR, D-FINE, DEIM, RF-DETR, PP-YOLOE, RTMDet, DAMO-YOLO | Query-based detectors whose initial boxes may not overlap |
| **DIoU** | NMS, loss | YOLOv4 DIoU-NMS option, Cluster-NMS | Crowded scenes with side-by-side objects |
| **CIoU** | loss, TAL metric | YOLOv4/5/7/8/11/26 | Dense YOLO-style heads; the safe default |
| **EIoU / SIoU** | loss | YOLOv6-N/T (SIoU), many derivatives | Small models where a slightly different gradient helps; verify on your data |
| **NWD** | assignment, NMS, loss | tiny-object research (AI-TOD), aerial | Most objects under ~16 px |
| **ProbIoU** | loss, rotated NMS | YOLOv8/11/26-OBB | Oriented boxes |

---

## Gradients, briefly

For a predicted box with $x_2$ inside the ground truth, IoU's gradient with respect to $x_2$ is
positive while the box is too small and negative once it extends past the ground truth. It is
discontinuous at the border. The magnitude scales as $1/|A \cup B|$, so large boxes get small
gradients. **This is a feature.** It is the scale invariance that L1 lacks: L1 gives a 4-pixel
error the same gradient on every box size, and its loss for large boxes swamps small ones. Modern
recipes combine both. DETRs use L1 + GIoU. Ultralytics uses CIoU plus DFL, and YOLO26 replaced DFL
with an L1 term normalised by image size. The IoU term supplies scale invariance; the L1 or
distribution term supplies a well-conditioned gradient near the optimum, where IoU flattens.
Appendix B derives $\partial \text{IoU} / \partial x_2$ explicitly.

---

## Key Takeaways

- IoU is scale-invariant and that is why it is the metric. It is also harsh on small objects: a
  1-pixel shift drops an 8×8 box to 0.78 IoU.
- As a loss, plain IoU has zero gradient for disjoint boxes. GIoU fixes this but degenerates when one
  box contains the other.
- DIoU adds a centre-distance penalty, which converges faster and makes a better NMS criterion. CIoU
  adds aspect ratio and is the YOLO default for both loss and assignment.
- EIoU, SIoU and MPDIoU are second-order refinements. Measure them on your data and do not expect
  large COCO gains on strong baselines.
- Gaussian-based measures (NWD, ProbIoU) fix tiny-object brittleness and rotated-box discontinuities.
- `odlab.boxes` matches torchvision's IoU, GIoU, DIoU and CIoU to 1e-5 on random boxes.

## Check Yourself

<details class="check"><summary>Prediction [10,10,20,20] lies entirely inside ground truth [0,0,40,40]. Compute IoU and GIoU, and explain why GIoU gives no extra help here.</summary>
Intersection = 100, union = 1600, IoU = 0.0625. The enclosing box C is the ground truth itself, so
|C| = |A ∪ B| = 1600 and the GIoU penalty is 0: GIoU = IoU. The only gradient comes from IoU,
which is exactly the slow regime DIoU was designed for. DIoU here adds ρ²/c² with ρ² = (15−20)²·2
= 50 and c² = 3200, so it also pulls the centre.</details>

<details class="check"><summary>Why is CIoU's α treated as a constant during backpropagation?</summary>
α = v / ((1 − IoU) + v) is a balancing weight. Differentiating through it creates gradients that
change the trade-off between IoU and aspect ratio in unintended directions, and the reference
implementation detaches it. torchvision, Ultralytics and odlab all compute α under no_grad.</details>

<details class="check"><summary>Your dataset is drone imagery where 70% of objects are under 12 pixels. Which IoU-related changes would you try first?</summary>
Replace IoU with NWD (or blend them) in the assigner, so tiny objects get positives. Use NWD-based or
lower-threshold NMS. Consider NWD in the box loss. Before any of that, increase input resolution or
add a P2 head (Chapter 41). A metric change cannot recover information that the stride already
discarded.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Generalized Intersection over Union | Rezatofighi et al., 2019 | arXiv:1902.09630 | GIoU definition, disjoint-gradient argument |
| Distance-IoU Loss | Zheng et al., 2020 | arXiv:1911.08287 | DIoU, CIoU, DIoU-NMS, α treated as constant |
| Focal and Efficient IOU Loss | Zhang et al., 2021 | arXiv:2101.08158 | EIoU |
| SIoU Loss | Gevorgyan, 2022 | arXiv:2205.12740 | Angle, distance, shape costs |
| MPDIoU | Ma & Xu, 2023 | arXiv:2307.07662 | Corner-distance IoU |
| A Normalized Gaussian Wasserstein Distance for Tiny Object Detection | Wang et al., 2021 | arXiv:2110.13389 | NWD, C = 12.8 on AI-TOD |
| Gaussian Bounding Boxes and Probabilistic IoU | Llerena et al., 2021 | arXiv:2106.06072 | ProbIoU |
| YOLOv6 v3.0 | Li et al., 2023 | arXiv:2301.05586 | SIoU for small models, GIoU for larger |
| torchvision.ops box_iou / generalized / distance / complete | PyTorch | pytorch.org/vision | Reference values for tests |
| Ultralytics `utils/metrics.py bbox_iou`, `utils/tal.py` | Ultralytics | github.com/ultralytics/ultralytics | CIoU in loss and TAL, clamp at 0 |
| `odlab/boxes.py` + `tests/test_boxes_nms.py` | this book | code/odlab | Implementations and checks |

---

**Next:** [Chapter 4 — Anchors, Points and Queries](./04_anchors_points_queries.md) — IoU tells us
how similar two boxes are. The next question is which candidate boxes exist in the first place.
