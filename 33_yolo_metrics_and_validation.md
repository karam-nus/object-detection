---
title: "Chapter 33 — YOLO Axis 7: Metrics & Validation"
---

[← Back to Table of Contents](./README.md)

# Chapter 33 — YOLO Axis 7: Metrics & Validation

> *"`model.val()` prints seven numbers. Know which question each one answers, and which it does not."*

## Overview

Chapter 8 defined the COCO metrics. This chapter covers what Ultralytics' validator computes: which
predictions it scores, how it matches them to labels, how it integrates AP, where the printed precision
and recall come from, what the confusion matrix and speed line mean, and when the numbers differ from
pycocotools. It then measures, on identical predictions, how far the built-in metric is from the COCO
protocol, and how wide the uncertainty of a validation mAP is at realistic validation-set sizes.

<div class="diagram">
<div class="diagram-title">What `model.val()` does</div>
<div class="flow-h">
  <div class="flow-node">val images<small>rect batches, letterbox</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node blue">model<small>EMA weights</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node purple">NMS conf 0.001, IoU 0.7<small>or o2o top-300 (nms=False)</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node accent">match at 10 IoU thresholds<small>IoU-greedy</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node green">per-class AP (101-pt) · P/R at max-F1</div>
</div>
</div>

---

## The printed line, decoded

```text
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        500       4952      0.712      0.634      0.689      0.512
                person        500       2105      0.801      0.712      0.786      0.551
```

| Column | Meaning | Not to be confused with |
|---|---|---|
| `Images` | Validation images, including background images | — |
| `Instances` | Labelled boxes (per class on class rows) | Detections |
| `P` | Precision at the confidence threshold that **maximises the mean F1 over classes** | Precision at `conf=0.25` or at your deployment threshold |
| `R` | Recall at that same threshold | Recall at any fixed threshold |
| `mAP50` | Mean over classes of AP at IoU 0.5 | COCO's "AP50" (same idea, small implementation differences) |
| `mAP50-95` | Mean over classes of AP averaged over IoU 0.50:0.05:0.95. This is the **fitness** used for `best.pt` | COCO "AP" (same idea, see below) |

The threshold behind P and R is chosen *per validation run*, from the smoothed mean F1-confidence curve
(`F1_curve.png`). Two models' P and R are therefore measured at different confidences, and neither is at
the threshold you will deploy. For a deployment decision, read the curves or compute recall at your own
threshold (below).

---

## Which predictions are scored

| Setting | `val` default | Effect |
|---|---|---|
| `conf` | **0.001** (0.01 for OBB) | Keeps almost every candidate so the PR curve reaches high recall. Passing `conf=0.25` to `val` truncates the curve and lowers mAP |
| `iou` | 0.7 | NMS IoU threshold. Higher than the common 0.45–0.5 at predict time: duplicates cost little in AP, while suppressing overlapping true objects costs recall |
| `max_det` | 300 | Detections kept per image. If the dataset has images with more objects, the validator raises `max_det` to the observed maximum and warns |
| `nms` | `None` | YOLO26: one-to-many head + NMS. `nms=False` scores the one-to-one head with no suppression (top-300) |
| `rect` | True in val | Rectangular batches with minimal padding (Chapter 28): numbers can differ from a square exported model |
| `single_cls`, `agnostic_nms` | False | Class-agnostic evaluation or NMS |
| `half` | AMP during training | FP16 inference; differences are usually below 0.1 mAP |

---

## Matching and AP: the implementation

**Matching** (`DetectionValidator.match_predictions`), for each of the 10 IoU thresholds
independently:

1. IoU between every label and every prediction, set to zero where classes differ.
2. Take all pairs with IoU ≥ threshold, **sort by IoU (descending)**, keep the first occurrence of each
   prediction, then of each label.
3. Those predictions are true positives at this threshold.

COCO's protocol (Chapter 8) instead walks predictions **in descending confidence** and gives each the
best still-unmatched label. The two differ when a confident, loose box and a less confident, tight box
compete for one label: Ultralytics credits the tight box, COCO the confident one. An optional
`use_scipy=True` path does Hungarian (optimal one-to-one) matching.

**AP** (`ap_per_class` and `compute_ap`): for each class with at least one label, sort predictions by
confidence, accumulate TP and FP, build the precision envelope, then integrate precision over
**101 recall points with the trapezoid rule**. COCO takes the plain mean of the envelope at the
same 101 points (a step sum). mAP is the mean over classes **present in the validation labels**.
A class with predictions but no labels does not enter the mean.

**Not implemented in the built-in metric:** crowd regions (`iscrowd` annotations are dropped when
converting COCO to YOLO format, so a detection on a crowd counts as a false positive), the small /
medium / large breakdown, and maxDets 1/10/100.

**The published COCO numbers are not from this code.** For COCO and LVIS, the final validation writes
predictions to JSON and scores them with **faster-coco-eval** (a fast pycocotools-compatible
implementation), which gives the full twelve COCO numbers including APs/APm/APl. For a custom dataset,
`save_json=True` with a COCO-format annotation file does the same.

---

## How far apart are the two? Measured

The same synthetic predictions (200 images, 5 classes, jittered boxes, misses, duplicates,
misclassifications and background false positives) were scored by the COCO protocol (`odlab`, whose
implementation matches pycocotools exactly, Chapter 8) and by Ultralytics' `match_predictions` +
`ap_per_class`:

| Scenario | COCO AP | Ultralytics mAP50-95 | Difference | COCO AP50 | Ultralytics mAP50 |
|---|---|---|---|---|---|
| Clean (tight boxes, no duplicates, no background FPs) | 0.5943 | 0.5960 | +0.0017 | 0.7638 | 0.7664 |
| Typical (15% duplicates, up to 3 background FPs/image) | 0.4088 | 0.4075 | −0.0013 | 0.7463 | 0.7447 |
| Noisy (40% duplicates, up to 8 FPs, looser boxes) | 0.2538 | 0.2502 | −0.0036 | 0.6903 | 0.6808 |
| Crowded (up to 150 objects, up to 40 FPs/image) | 0.3938 | 0.3912 | −0.0026 | 0.7237 | 0.7190 |

On these data the built-in metric is within **0.4 AP points** of the COCO protocol, and it is not
systematically optimistic. The trapezoid rule nudges it up on clean curves. IoU-greedy matching and
the absence of a 100-detection cap move it either way. Differences of this size matter only when
comparing models across evaluators, which you should not do anyway: score every model in a comparison
with the same tool (Chapter 8).

---

## How uncertain is a validation mAP?

A validation set is a sample. Resampling its images with replacement (bootstrap) shows how much mAP
would move with a different sample of the same size. Measured on the "typical" synthetic predictions,
100 bootstrap resamples per size:

| Validation images | mAP50-95 | 95% bootstrap interval | Interval width | Std (points) |
|---|---|---|---|---|
| 50 | 0.401 | 0.372 – 0.441 | **6.8 points** | 1.7 |
| 200 | 0.404 | 0.385 – 0.424 | **3.8 points** | 1.0 |
| 1,000 | 0.402 | 0.395 – 0.410 | **1.5 points** | 0.3 |

With 200 validation images, two models whose mAP differs by 1–2 points cannot be told apart from a
single validation run. The interval narrows roughly with the square root of the number of images: four
times the images halves the width. Real datasets with many objects per image narrow faster, and datasets
dominated by a few hard images narrow more slowly.

Code to do the same on your own predictions with `odlab`:

```python
import numpy as np
from odlab.metrics import coco_evaluate                     # pycocotools-exact

def bootstrap_map(preds, gts, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(gts))
    aps = [coco_evaluate([preds[i] for i in s], [gts[i] for i in s]).stats["AP"]
           for s in (rng.choice(idx, len(idx)) for _ in range(n_boot))]
    return np.percentile(aps, [2.5, 97.5])
```

For comparing two models, bootstrap the **difference** on the same resampled images (a paired
bootstrap). It is much tighter than comparing two separate intervals, because both models see the same
easy and hard images.

---

## The confusion matrix and the plots

`model.val()` and training save a set of plots. Each answers a specific question:

| File | Question | Caveat |
|---|---|---|
| `confusion_matrix.png` / `_normalized` | Which classes are confused, and how much is missed or hallucinated as background | Built with the validator's `conf`, **0.001 by default**, and IoU 0.45. The "background" column is inflated by low-confidence predictions. Run `model.val(conf=0.25)` (or your threshold) for a deployment-relevant matrix, separately from the mAP run |
| `BoxPR_curve.png` | PR curve per class at IoU 0.5 | mAP50 only |
| `BoxF1_curve.png` | F1 vs confidence; the peak gives a good default threshold | Peak of the mean over classes; per-class peaks differ |
| `BoxP_curve.png`, `BoxR_curve.png` | Precision and recall vs confidence | Read recall at your deployment threshold here |
| `labels.jpg` | Class histogram, box centre and size distributions | Compare with the deployment data |
| `val_batch*_pred.jpg` vs `_labels.jpg` | Visual check of predictions | Shows the predictions above a plotting confidence |

---

## The speed line is not a benchmark

```text
Speed: 0.2ms preprocess, 1.8ms inference, 0.0ms loss, 1.1ms postprocess per image
```

These are wall-clock times per image during validation: batched (`batch=16` by default for a standalone `val`, twice
the training batch during training), in PyTorch, with rect batches, and with the confidence at 0.001 so NMS
processes far more boxes than at deployment. They are useful for spotting a regression between runs on
the same machine. They are not the latency of an exported model at batch 1 (Chapter 46).

---

## Metrics for the decision you actually make

mAP50-95 averages over IoU thresholds, confidence thresholds and classes. A deployment fixes all three.
Report alongside it:

| Deployment question | Metric | How with Ultralytics |
|---|---|---|
| "How many objects will we miss?" | Recall at the deployment threshold, per class | `BoxR_curve.png` or rerun `val(conf=thr)` and read R |
| "How many false alarms per image?" | False positives per image at the threshold | Confusion-matrix background row at `conf=thr`, divided by images |
| "Are boxes tight enough for measurement?" | AP75, or mean IoU of matched boxes | `metrics.box.map75`; custom script for mean IoU |
| "Does it work for small objects?" | APs (COCO definition) | `save_json=True` with COCO-format labels |
| "Is it the same as last month?" | Paired bootstrap of mAP difference on a frozen test set | Script above |
| "Why is AP low?" | TIDE error breakdown (classification, localisation, duplicate, background, missed) | Export predictions to COCO JSON, run TIDE (Chapter 8) |

Programmatic access:

```python
m = YOLO("best.pt").val(data="widgets.yaml", split="test")   # use a held-out split for the final report
m.box.map, m.box.map50, m.box.map75        # mAP50-95, mAP50, mAP75
m.box.maps                                 # per-class mAP50-95 (array indexed by class id)
m.box.p, m.box.r                           # per-class P and R at the max-F1 threshold
m.speed, m.confusion_matrix.matrix
```

---

## Validation pitfalls

| Pitfall | What you see | Fix |
|---|---|---|
| `conf=0.25` passed to `val` | mAP several points lower than during training | Leave `conf` unset for mAP; use a separate run for thresholded metrics |
| Validating at a different `imgsz` from training | Unexplained mAP change | Validate at the deployment size; report it with the number |
| Leakage (near-duplicate frames across splits) | Validation mAP far above field performance | Split by video, scene, or capture session; perceptual-hash dedup (Chapter 27) |
| Tuning on the validation split | Optimistic final number | Hold out a test split; report it once |
| Comparing PyTorch `val()` with an exported model | 0.5–1 mAP gap | Rect vs square input, FP16/INT8, NMS settings (Chapters 28, 36, 45) |
| Comparing with numbers in a paper | Different protocol | Same evaluator, same split, same `max_det`, same input size |

---

## Key Takeaways

- `val` scores everything above conf 0.001, after NMS at IoU 0.7 (or the one-to-one head with
  `nms=False`), keeping up to 300 detections (raised automatically for crowded data).
- Matching is IoU-greedy per threshold (COCO's is confidence-ordered). AP uses 101 recall points
  integrated with the trapezoid rule. mAP averages over classes that have labels.
- Printed P and R are at the max-mean-F1 confidence of that run, not at any fixed threshold.
- Measured on identical predictions, the built-in metric is within about 0.4 AP of the COCO protocol.
  Published COCO numbers come from faster-coco-eval on JSON.
- The confusion matrix uses conf 0.001 by default. Rerun at your deployment threshold for a meaningful
  one.
- Validation-set size sets the uncertainty: bootstrap the mAP, and use paired bootstraps to compare
  models.

## Check Yourself

<details class="check"><summary>Model A reports P = 0.81, R = 0.70; model B P = 0.76, R = 0.77 on the same data. Which has better recall at conf = 0.4?</summary>
You cannot tell from these numbers. Each pair is measured at the confidence that maximises that model's
mean F1, which is different for A and B and probably not 0.4. Read both recall curves at 0.4, or run
val(conf=0.4) for each and compare R.</details>

<details class="check"><summary>Your confusion matrix shows thousands of "background" false positives, but deployment at conf 0.5 looks clean. Is the matrix wrong?</summary>
No. The validator built it with conf = 0.001, the setting used for mAP, so every low-confidence
candidate that matched nothing counts as background. Re-run val with conf at the deployment threshold to
get a matrix that reflects deployment.</details>

<details class="check"><summary>A label is matched by a 0.9-confidence box at IoU 0.62 and a 0.6-confidence box at IoU 0.88. At threshold 0.75, which prediction is the TP under Ultralytics and under COCO?</summary>
At 0.75 only the 0.88 box qualifies, so both protocols make it the TP and the 0.9 box an FP. At
threshold 0.5 both qualify: Ultralytics' IoU-greedy matching credits the 0.88 box, COCO's
confidence-ordered matching credits the 0.9 box first. That changes the precision at high confidence and
hence AP slightly.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics `engine/validator.py` (`match_predictions`, defaults, speed) | Ultralytics | github.com/ultralytics/ultralytics | Matching, conf default, speed line |
| Ultralytics `models/yolo/detect/val.py` (`DetectionValidator`, `eval_json`) | Ultralytics | github.com/ultralytics/ultralytics | NMS settings, max_det auto-raise, faster-coco-eval path |
| Ultralytics `utils/metrics.py` (`ap_per_class`, `compute_ap`, `ConfusionMatrix`, `fitness`) | Ultralytics | github.com/ultralytics/ultralytics | AP integration, max-F1 P/R, confusion matrix thresholds |
| Ultralytics `data/converter.py` (`convert_coco`) | Ultralytics | github.com/ultralytics/ultralytics | Crowd annotations skipped |
| Ultralytics performance-metrics guide | Ultralytics | docs/en/guides/yolo-performance-metrics.md | Metric names and plots |
| COCO detection evaluation; pycocotools | Lin et al., 2014; cocodataset | cocodataset.org/#detection-eval | Reference protocol |
| faster-coco-eval | MiXaiLL76 | github.com/MiXaiLL76/faster_coco_eval | COCO-compatible evaluator used for published numbers |
| TIDE | Bolya et al., 2020 | arXiv:2008.08115 | Error decomposition |
| An Introduction to the Bootstrap | Efron, Tibshirani, 1993 | — | Bootstrap confidence intervals |
| `odlab/metrics.py` + measurements in this chapter | this book | code/odlab | COCO-exact evaluator; evaluator comparison; bootstrap widths |

---

**Next:** [Chapter 34 — Axis 8: Family Performance](./34_yolo_family_performance.md) — every YOLO on one
accuracy–latency plot, and what it takes for that plot to be fair.
