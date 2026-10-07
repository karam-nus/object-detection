---
title: "Chapter 39 — Build a YOLO from Scratch"
---

[← Back to Table of Contents](./README.md)

# Chapter 39 — Build a YOLO from Scratch

> *"What I cannot create, I do not understand." — Richard Feynman*

## Overview

This chapter is the capstone of Part IV. The companion package `odlab` contains **TinyYOLO**, a complete
modern YOLO in about 270 lines of PyTorch (`code/odlab/model.py`), plus the assigner, losses, NMS,
COCO-exact evaluator and a training loop, all unit-tested. It trains on a laptop CPU in about ten minutes
on a synthetic dataset. Two switches turn it from a YOLOv8-style model (DFL head + NMS) into a
YOLO26-style one (L1 head + one-to-one head, no NMS). The chapter walks through the code axis by axis,
maps every piece to its Ultralytics counterpart, reports two measured training runs, and ends with
exercises that reproduce the experiments of earlier chapters.

```bash
cd code
pip install -r requirements.txt
pytest -q                                        # 32 tests: IoU family, NMS, mAP vs pycocotools, TAL, losses, model
python -m odlab.train --epochs 30                # YOLOv8-style: DFL head + NMS
python -m odlab.train --epochs 30 --reg-max 1 --end2end   # YOLO26-style: L1 head + one-to-one head
```

---

## The data: synthetic shapes with COCO-like sizes

`odlab.data.SyntheticShapes` draws 1–6 squares, circles and triangles per 256 × 256 image on a noisy
background. Radii are **log-uniform from 4 to 72 px**, so the dataset has as many small objects as large
ones in the COCO sense (area below 32², 32²–96², above 96²). Later shapes can occlude earlier ones, and
small rounded squares look like circles, so there is real class confusion. Boxes are tight to the
rendered pixels. It is not COCO, but every metric in the book (APs, APm, APl, AR) moves on it.

---

## The model, axis by axis

| Piece | TinyYOLO (`odlab/model.py`) | Ultralytics counterpart (Chapter 29) |
|---|---|---|
| Atom | `Conv`: Conv2d (no bias) → BN (eps 1e-3, momentum 0.03) → SiLU | `nn/modules/conv.py: Conv` |
| Block | `C2f`: split, chain bottlenecks, concat all, 1×1 | `C2f` (YOLOv8) |
| Context | `SPPF`: three chained 5×5 max pools | `SPPF` |
| Backbone | stem s2 → s2 → C2f; then P3, P4, P5 each = Conv s2 + C2f; SPPF at P5 | `yolov8.yaml` backbone, widths 16/32/64/128/256 |
| Neck | PAN: two top-down C2f (concat), two bottom-up (Conv s2 + concat + C2f), `shortcut=False` | `yolov8.yaml` head section |
| Head | decoupled; box tower width `max(16, ch/4, 4·reg_max)`; class tower width `max(ch, min(nc, 100))`; full 3×3 class convs | `Detect` (legacy class tower) |
| Bias init | box 1.0; class `log(5 / nc / (img/stride)²)` | `Detect.bias_init` (box 2.0) |
| One-to-one head | deep copy of the head, fed **detached** features | `one2one_cv2/cv3` with `x.detach()` |
| Decoding | grid points at cell centres; DFL expectation (`reg_max>1`) or identity; `dist2bbox` × stride | `make_anchors`, `DFL`, `dist2bbox` |
| Inference | o2m: conf filter → class max → batched NMS; o2o: top-k over all scores, no NMS | `Detect.postprocess`, `ops.non_max_suppression` |

The full forward pass:

```python
def features(self, x):
    c3 = self.p3(self.stem(x))                      # stride 8
    c4 = self.p4(c3)                                # stride 16
    c5 = self.p5(c4)                                # stride 32, ends with SPPF
    t4 = self.td4(torch.cat((F.interpolate(c5, scale_factor=2.0), c4), 1))      # top-down
    n3 = self.td3(torch.cat((F.interpolate(t4, scale_factor=2.0), c3), 1))      # → out P3
    n4 = self.bu4(torch.cat((self.down3(n3), t4), 1))                            # bottom-up → out P4
    n5 = self.bu5(torch.cat((self.down4(n4), c5), 1))                            # → out P5
    return [n3, n4, n5]
```

At 256 × 256 the three levels have 32², 16² and 8² cells: **1,344 grid points**. The DFL model has
**2.91M parameters**. The end-to-end model has **3.04M**: the box tower shrinks from 64 to 16 channels
without DFL, and the second head is added.

---

## Assignment and loss

`odlab.assign.task_aligned_assign` reproduces Ultralytics' `TaskAlignedAssigner` for one image
(Chapter 30): candidates are grid points strictly inside a box, alignment is
`score^0.5 · CIoU^6` (CIoU clamped at 0), the top-10 per object become positives, conflicts go to the
highest-IoU object, and soft targets are normalised so each object's best positive gets its best IoU.
Two options reproduce YOLO26: `topk2` (top-k then top-1) and `small_side_floor` (STAL).

`odlab.model.DetectionLoss` computes, per image, normalised by the sum of soft targets:

```python
l_cls  = BCE(scores, soft_targets).sum() / norm
l_box  = ((1 - CIoU(pred[fg], tgt[fg])) * w).sum() / norm
l_dist = DFL(raw[fg], target_ltrb) ...            # reg_max > 1
l_dist = L1((raw[fg] - target_ltrb) * stride / img_size) ...   # reg_max == 1 (YOLO26)
loss   = 7.5 * l_box + 0.5 * l_cls + 1.5 * l_dist  # Ultralytics' default gains
```

For the end-to-end model, the loss is `w · L_o2m + (1 − w) · L_o2o`, with the one-to-one branch
assigned at `topk = 1` and `w` decaying linearly from 0.8 to 0.1 over the epochs (the YOLO26 ProgLoss
schedule). The logged values are the one-to-one branch's, as in Ultralytics.

---

## The training loop

`odlab/train.py` keeps the parts of the Ultralytics recipe that matter on a CPU and drops the rest:

| Recipe element (Chapter 31) | TinyYOLO | Ultralytics default |
|---|---|---|
| Optimiser | AdamW, lr 2e-3, weight decay 5e-4 on weights only | auto (AdamW or MuSGD) |
| Schedule | linear warm-up for one epoch, then cosine to 1% | warm-up 3 epochs, linear to 1% |
| EMA | Ultralytics formula, τ = steps / 10 | decay 0.9999, τ = 2000 |
| Gradient clipping | norm 10 | norm 10 |
| Augmentation | flip + mild scale/translate | mosaic, affine, HSV, flip, mixup … |
| Batch | 16 | 16 (accumulated to 64) |
| Evaluation | `odlab.metrics.coco_evaluate` (matches pycocotools exactly) on the EMA model | built-in metric on the EMA model |

---

## Results: two runs, measured

30 epochs, 768 training images, 128 validation images, 256 × 256, CPU (about 20–23 s per epoch).

**Loss curves (logged values include the gains):**

| Epoch | DFL run: box / cls / dfl | E2E run (one-to-one branch): box / cls / l1 |
|---|---|---|
| 1 | 2.64 / 4.26 / 2.91 | 2.85 / 5.99 / 0.034 |
| 10 | 0.69 / 0.59 / 0.96 | 1.01 / 0.73 / 0.010 |
| 20 | 0.50 / 0.42 / 0.89 | 0.72 / 0.52 / 0.007 |
| 30 | 0.37 / 0.34 / **0.87** | 0.64 / 0.45 / 0.006 |

The DFL term plateaus at 0.87, which is 0.58 nats after removing the 1.5 gain: close to the ~0.5-nat
entropy floor derived in Chapter 6. It cannot fall much further, however good the boxes become.

**Validation (COCO protocol):**

| Model / head | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---|---|---|---|---|---|
| DFL, NMS | **0.833** | **0.900** | **0.875** | **0.775** | **0.868** | **0.918** | 0.854 |
| L1, one-to-many branch + NMS | 0.810 | 0.894 | 0.867 | 0.759 | 0.845 | 0.892 | 0.841 |
| L1, one-to-one branch, no NMS | 0.787 | 0.877 | 0.843 | 0.760 | 0.810 | 0.828 | **0.856** |

What the numbers say:

1. **Removing DFL cost 2.3 AP** here (0.833 → 0.810, comparing the two NMS heads). Small models on short
   schedules are where DFL's extra box precision is most likely to show.
2. **Dropping NMS cost another 2.3 AP**, concentrated on medium and large objects (APl 0.892 → 0.828).
   Large objects have many good candidate points, so a one-to-one head must learn to suppress more
   near-duplicates of itself.
3. **The one-to-one head has the highest recall** (AR100 0.856). It loses AP on ranking and duplicates,
   not on finding objects.
4. These are 30-epoch runs of a 3M-parameter model on synthetic data. YOLO26's published gap between its
   two heads is only 0.6–0.8 COCO AP after hundreds of epochs, with the softer top-7 → top-1 assignment
   and STAL. Exercise 3 below tests whether those close part of the gap here.

**Learning-rate sensitivity and seed noise** (12-epoch runs, three seeds each) are reported in
Chapter 32.

---

## Exercises

Each exercise changes one thing and reuses the measurement code. Expected direction, not exact numbers,
is given.

| # | Change | Where | What to look for |
|---|---|---|---|
| 1 | Add mosaic (Chapter 28) | `odlab/augment.py` + `light_augment` | APs up, longer convergence; turn it off for the last few epochs |
| 2 | STAL: `DetectionLoss(..., small_side_floor=8)` (stride 8 here, not 16, at 256 px) | `odlab/model.py` | APs and ARs on the smallest shapes |
| 3 | YOLO26 one-to-one assignment: `topk=7, topk2=1` for the o2o branch | `DetectionLoss.__call__` | The o2o vs o2m gap |
| 4 | TAL β = 2 instead of 6 | `DetectionLoss(beta=2.0)` | AP75 vs AP50: localisation quality in the ranking |
| 5 | Replace C2f with a C3k2-style block (C3k inner blocks) | `odlab/model.py` | Params, CPU time per epoch, AP |
| 6 | Train at 192 and 320 px | `--img` | APs vs latency (Chapter 41) |
| 7 | Export to ONNX and run with ONNX Runtime | `torch.onnx.export(model, ...)` + Chapter 36 post-processing | Golden-image test: boxes within 1 px of PyTorch |
| 8 | Evaluate the same predictions with Ultralytics' metric | Chapter 33 script | Difference from the COCO protocol |

---

## Key Takeaways

- A complete modern YOLO is small: Conv-BN-SiLU, C2f, SPPF, a PAN, a decoupled head, TAL assignment,
  BCE + CIoU + DFL (or L1), EMA, and NMS or top-k. `odlab` implements each in readable, tested code.
- The head's box width rule `max(16, ch/4, 4·reg_max)` shows why DFL removal shrinks the head.
  Detaching the one-to-one head's features protects the backbone.
- Measured on TinyYOLO: DFL → L1 cost 2.3 AP, NMS → one-to-one cost another 2.3 AP (mostly large
  objects), while the one-to-one head had the best recall.
- The DFL loss plateaus near its entropy floor. Read loss values against their floors, not against zero.
- Use the exercises to reproduce any claim of Part IV at small scale before spending GPU days on it.

## Check Yourself

<details class="check"><summary>Why does TinyYOLO's end-to-end model have more parameters (3.04M) than the DFL model (2.91M) even though its box tower is four times narrower?</summary>
Because it carries two complete heads, the one-to-many and the one-to-one copy. Each copy's box tower
shrinks from 64 to 16 channels (reg_max 1), but the second head (class towers included) adds more than
the box-tower savings remove. After deployment, only one head remains.</details>

<details class="check"><summary>The one-to-one head has the highest AR100 but the lowest AP. How can both be true?</summary>
AR measures whether each object is found by some prediction among the top 100. AP also depends on
ranking: duplicates and high-scoring false positives lower precision at every recall level. The
one-to-one head finds objects well but has not fully learned to suppress near-duplicates and to rank
well-localised boxes first, especially for large objects.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| `odlab` package and tests | this book | code/odlab, code/tests | All code and measurements |
| Ultralytics `nn/modules`, `utils/tal.py`, `utils/loss.py`, `engine/trainer.py` | Ultralytics | github.com/ultralytics/ultralytics | The reference implementation TinyYOLO mirrors |
| YOLOv8 architecture (`cfg/models/v8/yolov8.yaml`) | Ultralytics, 2023 | github.com/ultralytics/ultralytics | Backbone/neck layout |
| GFL (DFL) | Li et al., 2020 | arXiv:2006.04388 | Distribution focal loss |
| TOOD | Feng et al., 2021 | arXiv:2108.07755 | Task-aligned assignment |
| YOLOv10; YOLO26 | Wang et al., 2024; Ultralytics, 2026 | arXiv:2405.14458; arXiv:2606.03748 | Dual heads, L1 head, ProgLoss |
| pycocotools | cocodataset | github.com/cocodataset/cocoapi | Evaluation protocol `odlab.metrics` reproduces |

---

**Next:** [Chapter 40 — Training in Practice](./40_training_in_practice.md) — Part V begins: beyond YOLO,
the general craft of getting a detector to train well.
