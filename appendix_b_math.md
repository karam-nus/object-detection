---
title: "Appendix B — Math Reference"
---

[← Back to Table of Contents](./README.md)

# Appendix B — Math Reference

The formulas of the book in one place, with short derivations where they explain a behaviour seen in
practice.

---

## B.1 IoU and why it needs help as a loss

For boxes $A$ (prediction) and $B$ (target) with intersection area $I$ and union $U = |A| + |B| - I$:

$$\text{IoU} = \frac{I}{U}, \qquad \frac{\partial\, \text{IoU}}{\partial I} = \frac{U + I}{U^2}, \qquad \frac{\partial\, \text{IoU}}{\partial |A|} = -\frac{I}{U^2}.$$

When the boxes do not overlap, $I = 0$ and $I$ does not change under small moves of $A$, so IoU gives
**zero gradient**: a non-overlapping prediction is not pulled toward its target. The IoU family adds
terms that stay informative:

$$\text{GIoU} = \text{IoU} - \frac{|C| - U}{|C|}, \qquad \text{DIoU} = \text{IoU} - \frac{\rho^2(c_A, c_B)}{d^2}, \qquad \text{CIoU} = \text{DIoU} - \alpha v,$$

where $C$ is the smallest enclosing box, $\rho$ the distance between centres, $d$ the diagonal of $C$,
$v = \frac{4}{\pi^2}\left(\arctan\frac{w_B}{h_B} - \arctan\frac{w_A}{h_A}\right)^2$ and
$\alpha = \frac{v}{(1 - \text{IoU}) + v}$. Losses are $1 - (\cdot)$. (Chapter 3.)

**Shift sensitivity.** An $s \times s$ box shifted horizontally by $d$ pixels has
$\text{IoU} = \frac{s(s - d)}{2s^2 - s(s-d)} = \frac{s - d}{s + d}$. For $s = 8$, $d = 2$: 0.6. (Chapter 41.)

---

## B.2 Focal loss and its gradient

With $p = \sigma(x)$ and a positive label, $\mathcal{L} = -(1 - p)^{\gamma} \log p$. Then

$$\frac{\partial \mathcal{L}}{\partial x} = (1 - p)^{\gamma}\left(\gamma\, p \log p - (1 - p)\right).$$

For $\gamma = 0$ this reduces to the BCE gradient $p - 1$. For an easy positive ($p = 0.9$, $\gamma = 2$):
$0.01 \times (2 \cdot 0.9 \ln 0.9 - 0.1) = -0.0029$, about 34× smaller than BCE's $-0.1$. Easy examples
stop dominating the sum. The balancing weight $\alpha_t$ multiplies the whole expression. (Chapter 6.)

**Quality-aware variants** replace the hard label with a soft target $q \in [0, 1]$ (the IoU or the TAL
score): QFL $= -|q - p|^{\beta}\big(q \log p + (1 - q)\log(1 - p)\big)$; VFL keeps the positive term
weighted by $q$ and down-weights negatives by $\alpha p^{\gamma}$.

---

## B.3 DFL: the optimum and the entropy floor

A continuous edge distance $y$ lies between bins $y_l = \lfloor y \rfloor$ and $y_r = y_l + 1$, with
weights $w_l = y_r - y$ and $w_r = y - y_l$. With predicted bin probabilities $S$ (softmax over
`reg_max` bins):

$$\text{DFL}(S, y) = -\big(w_l \log S_{y_l} + w_r \log S_{y_r}\big).$$

This is a cross-entropy against the two-hot distribution $(w_l, w_r)$, so it is minimised at
$S_{y_l} = w_l$, $S_{y_r} = w_r$, where the decoded expectation $\sum_i i\, S_i = y_l w_l + y_r w_r = y$
is exact. Its minimum value is the entropy of the target:

$$\min_S \text{DFL} = H(w_l, w_r) = -w_l \ln w_l - w_r \ln w_r.$$

If the fractional part $f = y - y_l$ is uniform on $[0, 1)$:

$$\mathbb{E}[H] = \int_0^1 \big(-f \ln f - (1 - f)\ln(1 - f)\big)\, df = 2 \int_0^1 -f \ln f\, df = 2 \cdot \tfrac{1}{4} = \tfrac{1}{2}\ \text{nat}.$$

So DFL cannot fall below about 0.5 nats per edge on average. With Ultralytics' gain of 1.5, the logged
`dfl_loss` floor is about 0.75; TinyYOLO plateaued at 0.87. (Chapters 6, 39.)

---

## B.4 TAL: alignment metric and soft targets

For candidate $i$ inside object $j$ (class $c_j$):

$$t_{ij} = p_{i, c_j}^{\alpha}\, u_{ij}^{\beta}, \qquad \alpha = 0.5,\ \beta = 6,$$

with $u_{ij}$ the CIoU (clamped at 0) between the predicted box and the object. The top-$k$ candidates
by $t_{ij}$ are positives ($k = 10$). A point claimed by several objects keeps the one with the largest
$u_{ij}$. The classification target is

$$\hat t_{ij} = t_{ij} \cdot \frac{\max_{i'} u_{i'j}}{\max_{i'} t_{i'j}},$$

so the best-aligned positive of each object gets that object's best IoU, and others proportionally less.
The box losses of each positive are weighted by $\hat t$, and all terms are divided by $\sum \hat t$.
(Chapters 5, 30.)

---

## B.5 SimOTA's dynamic k

Cost for candidate $i$ and object $j$:
$C_{ij} = \text{BCE}_{\text{cls}} + 3\,(-\log u_{ij}) + \lambda\, \mathbb{1}[i \notin \text{centre region}_j]$
with a large $\lambda$. Object $j$ takes its $k_j = \max\big(1, \lfloor \sum \text{top-10}_i\, u_{ij} \rfloor\big)$
cheapest candidates; conflicts go to the cheaper object. (Chapter 5.)

---

## B.6 Hungarian matching

With $N$ predictions and $G \le N$ objects, find the injection $\sigma$ minimising
$\sum_{j=1}^{G} \mathcal{C}(\hat y_{\sigma(j)}, y_j)$. This is a linear assignment problem: the
Kuhn–Munkres algorithm solves it exactly in $O(N^3)$. In DETR-family detectors
$\mathcal{C} = \lambda_{cls} C_{cls} + \lambda_{L1} \lVert \hat b - b \rVert_1 + \lambda_{giou}(-\text{GIoU})$
with $(\lambda_{cls}, \lambda_{L1}, \lambda_{giou}) = (2, 5, 2)$ and a focal-style class cost.
(Chapters 5, 18.)

---

## B.7 AP as an integral

For one class, sort detections by score; precision and recall at rank $n$ are
$P_n = \text{TP}_n / n$ and $R_n = \text{TP}_n / G$. The interpolated precision is
$p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$, and

$$\text{AP} = \int_0^1 p_{\text{interp}}(r)\, dr.$$

| Convention | Approximation |
|---|---|
| VOC 2007 | $\frac{1}{11} \sum_{r \in \{0, 0.1, \dots, 1\}} p_{\text{interp}}(r)$ |
| VOC 2010+ ("all points") | Exact area under the step envelope |
| COCO | $\frac{1}{101} \sum_{r \in \{0, 0.01, \dots, 1\}} p_{\text{interp}}(r)$, averaged over IoU 0.50:0.05:0.95, classes and (maxDets, area) settings |
| Ultralytics built-in | Trapezoid integral of the envelope sampled at the same 101 recall points |

(Chapters 8, 33.)

---

## B.8 Box parameterisations

| Detector | Decode |
|---|---|
| YOLOv2/v3 | $b_x = \sigma(t_x) + c_x$, $b_w = p_w e^{t_w}$ |
| YOLOv5 | $b_{xy} = 2\sigma(t_{xy}) - 0.5 + c$, $b_{wh} = (2\sigma(t_{wh}))^2 p_{wh}$ |
| FCOS / YOLOv8+ | $(x_1, y_1, x_2, y_2) = (c_x - l, c_y - t, c_x + r, c_y + b) \cdot s$ |
| DFL heads | $l = \sum_{i=0}^{15} i \cdot \text{softmax}(z)_i$ (same for $t, r, b$) |
| DETR | $(c_x, c_y, w, h) = \sigma(\cdot)$, normalised to the image |

---

## B.9 Gaussian boxes: ProbIoU and NWD

A box $(c_x, c_y, w, h, \theta)$ is modelled as $\mathcal{N}(\mu, \Sigma)$ with $\mu = (c_x, c_y)$ and
$\Sigma = R_\theta\, \text{diag}\!\left(\tfrac{w^2}{12}, \tfrac{h^2}{12}\right) R_\theta^\top$.

**ProbIoU** uses the Bhattacharyya distance
$B = \tfrac{1}{8}\Delta\mu^\top \bar\Sigma^{-1} \Delta\mu + \tfrac{1}{2}\ln\frac{\det \bar\Sigma}{\sqrt{\det\Sigma_1 \det\Sigma_2}}$,
$\bar\Sigma = \tfrac{1}{2}(\Sigma_1 + \Sigma_2)$, and the Hellinger distance $H = \sqrt{1 - e^{-B}}$:
$\text{ProbIoU} = 1 - H$. It is smooth and unaffected by the 180° ambiguity. (Chapters 38, 43.)

**NWD** (axis-aligned) uses the 2-Wasserstein distance between the Gaussians,
$W_2^2 = \lVert (c_{x1}, c_{y1}, \tfrac{w_1}{2}, \tfrac{h_1}{2}) - (c_{x2}, c_{y2}, \tfrac{w_2}{2}, \tfrac{h_2}{2}) \rVert^2$, and
$\text{NWD} = \exp(-\sqrt{W_2^2} / C)$ with a dataset constant $C$ (about the mean object size; 12.8 in
`odlab`). It decays smoothly with offset regardless of box size. (Chapters 3, 41.)

---

## B.10 Training-recipe formulas

| Quantity | Formula | Ch. |
|---|---|---|
| Ultralytics accumulation | $\text{accumulate} = \max(\text{round}(64 / \text{batch}), 1)$ | 31 |
| Weight-decay scaling | $\lambda \cdot \text{batch} \cdot \text{accumulate} / 64$ | 31 |
| Linear LR schedule | $\text{lr}(e) = \text{lr}_0 \left[(1 - e/E)(1 - \text{lrf}) + \text{lrf}\right]$ | 31 |
| EMA decay | $d(n) = 0.9999\,(1 - e^{-n/2000})$ | 31 |
| AdamW auto LR | $\text{lr}_0 = 0.002 \cdot 5 / (4 + n_c)$ | 31 |
| Class-bias prior | $b = \log\!\big(5 / n_c / (640/s)^2\big)$ | 29 |
| Repeat-factor sampling | $r_c = \max(1, \sqrt{t / f_c})$, image factor $\max_{c \in I} r_c$ | 40 |
| Compound scaling | $n' = \max(\text{round}(n \cdot d), 1)$; $c' = \text{make\_divisible}(\min(c, c_{\max}) \cdot w, 8)$ | 29 |

---

## B.11 Roofline

Time $\ge \max\left(\frac{\text{FLOPs}}{\text{peak FLOP/s}},\ \frac{\text{bytes}}{\text{bandwidth}}\right)$.
Arithmetic intensity $I = \text{FLOPs}/\text{bytes}$; a layer is memory-bound when $I$ is below the
hardware's ridge point $\text{peak}/\text{bandwidth}$ (about 200 FLOP/byte for a T4 in FP16).
(Chapter 44.)
