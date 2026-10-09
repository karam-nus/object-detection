---
title: "Chapter 8 — Metrics"
---

[← Back to Table of Contents](./README.md)

# Chapter 8 — Metrics

> *"mAP is an average of averages of an integral of an interpolated envelope. Every one of those words hides a decision."*

## Overview

"55.0 mAP" is shorthand for a specific, multi-step computation. Match detections to ground truth at
an IoU threshold. Sort by score. Build a precision–recall curve. Make it monotone. Sample it at 101
recall points. Average. Repeat for ten IoU thresholds and every class. Average again. This chapter
walks through that computation as pycocotools performs it. The companion implementation
`odlab.metrics.coco_evaluate` reproduces all twelve COCO numbers to within 1e-6, including crowd
regions (`test_coco_evaluate_matches_pycocotools`). The chapter then covers the metrics COCO does not
report (LVIS AP, TIDE errors, F1 at a threshold) and the ways a correct number can still mislead.

<div class="diagram">
<div class="diagram-title">The COCO AP pipeline</div>
<div class="flow-h">
  <div class="flow-node">per image, per class<small>sort dets, top 100</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">match at IoU t<small>greedy, each GT once</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">pool images<small>sort by score</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">PR curve<small>monotone envelope</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">101-point AP</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">mean over 10 t × K classes</div>
</div>
</div>

---

## Matching: what counts as a true positive

For one image and one class, at IoU threshold $t$:

1. Sort detections by score, descending, and keep at most `maxDets` (100 for the headline AP).
2. Each detection, in order, is matched to the **unmatched** ground truth with the highest IoU, if that
   IoU ≥ $t$. Matched: **TP**. Otherwise: **FP**.
3. Every ground truth left unmatched is a **FN** (it lowers recall).

Each ground truth can absorb **one** detection. A second detection on the same object is a FP, which
is why duplicates (from weak NMS) cost AP. Ground truths marked `iscrowd` can absorb any number of
detections, and those detections are *ignored* (neither TP nor FP). For crowd regions pycocotools
measures overlap as intersection / area(detection), not IoU (Chapter 2).

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{\#\text{ground truth}}$$

Both are computed **cumulatively** down the score-sorted list. Every prefix of the list is an operating
point: the precision and recall you would get with the score threshold set just below that detection.

---

## From a PR curve to AP: three conventions

AP is the area under the precision–recall curve. The raw curve is a saw-tooth: each FP drops
precision, each TP raises recall and precision. All conventions first replace precision with its
**monotone envelope**, $p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$, the best precision achievable at
recall $r$ or higher. They differ in how they sample it:

| Convention | Definition | Used by |
|---|---|---|
| **VOC 2007 11-point** | mean of $p_{\text{interp}}$ at $r \in \{0, 0.1, \dots, 1.0\}$ | PASCAL VOC 2007, early papers |
| **VOC 2010+ all-points** | exact area under the step envelope | PASCAL VOC 2010–2012, many custom tools |
| **COCO 101-point** | mean of $p_{\text{interp}}$ at $r \in \{0, 0.01, \dots, 1.00\}$; recall levels never reached contribute 0 | COCO, LVIS, Objects365, Ultralytics (approximately — Chapter 33) |

**Numerical check (`test_interpolation_conventions_on_textbook_curve`).** Five detections in score
order are TP, FP, TP, TP, FP, against 4 ground truths:

| Rank | TP? | cum. TP | Precision | Recall | Envelope |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | ✓ | 1 | 1.000 | 0.25 | 1.00 |
| 2 | ✗ | 1 | 0.500 | 0.25 | 0.75 |
| 3 | ✓ | 2 | 0.667 | 0.50 | 0.75 |
| 4 | ✓ | 3 | 0.750 | 0.75 | 0.75 |
| 5 | ✗ | 3 | 0.600 | 0.75 | 0.60 |

- **All-points**: $0.25 \times 1.0 + 0.25 \times 0.75 + 0.25 \times 0.75 = 0.625$.
- **11-point**: recall thresholds 0–0.2 give 1.0 (3 points), 0.3–0.7 give 0.75 (5 points), 0.8–1.0
  give 0 (3 points): $6.75 / 11 = 0.614$.
- **101-point**: $r \in [0, 0.25]$ gives 1.0 (26 points), $r \in (0.25, 0.75]$ gives 0.75 (50 points),
  the rest give 0: $(26 + 37.5)/101 = 0.629$.

The same detections score 0.614, 0.625 or 0.629 depending on convention. Differences of this size
between tools are routine, which is why "we re-evaluated everything with pycocotools" is a meaningful
sentence in a paper.

<div class="lab" data-lab="map"></div>

---

## The twelve COCO numbers

| Metric | IoU thresholds | Area | maxDets |
|---|---|---|---|
| **AP** (the headline, often called mAP) | 0.50:0.05:0.95 (10 values) | all | 100 |
| AP$_{50}$ | 0.50 | all | 100 |
| AP$_{75}$ | 0.75 | all | 100 |
| AP$_S$ / AP$_M$ / AP$_L$ | 0.50:0.95 | < 32², 32²–96², > 96² | 100 |
| AR$_1$ / AR$_{10}$ / AR$_{100}$ | 0.50:0.95 | all | 1 / 10 / 100 |
| AR$_S$ / AR$_M$ / AR$_L$ | 0.50:0.95 | as above | 100 |

Details that matter:

- **AP averages ten IoU thresholds**, so it rewards *localisation precision*. A model that improves
  AP$_{75}$ by 3 and leaves AP$_{50}$ unchanged gains about 1.5 AP. COCO's 2014 paper describes the
  0.5:0.95 average as deliberately rewarding better localisation.
- **"mAP" vs "AP".** COCO calls the class-averaged number "AP". Most papers write "mAP". They are the
  same number.
- **Area uses the annotation's `area` field.** On COCO that is the **segmentation mask area**, not the
  box area. A thin diagonal object has a large box and a small mask, so it counts as "small". A custom
  dataset converted to COCO JSON usually sets `area = w × h`. Small/medium/large splits are then not
  comparable with COCO's.
- **Size buckets on COCO**: 41% of objects are small, 34% medium, 24% large (Lin et al., 2014). AP$_S$
  is the lowest number in every table, for the reasons in Chapters 3, 4 and 41.
- **`-1`** means "no ground truth in this bucket". pycocotools excludes it from averages. A validation set with no object larger than 96² prints AP$_L$ = −1.
- **AR** is the maximum recall reached with at most $k$ detections per image, averaged over IoU
  thresholds and classes. AR$_{100}$ is a property of the *candidate set*, not of ranking. A model can
  have high AR and poor AP if its scores rank badly.

---

## Beyond COCO: other evaluation protocols

### LVIS: federated, long-tailed

LVIS has 1,203 categories, and images are **not exhaustively annotated** for every category. Each image
lists *positive* categories (all instances annotated), *negative* categories (verified absent), and
*not-exhaustive* flags. A detection of a category that is unverified for that image is neither TP nor FP.
Results are split by frequency: **AP$_r$** (rare: 1–10 training images), **AP$_c$** (common), and
**AP$_f$** (frequent). AP$_r$ is the number open-vocabulary papers chase (Chapter 20).

**AP$^{\text{fixed}}$** (Dave et al., 2021). With a per-image cap of 300 detections, a model can win
AP by spreading low-confidence guesses for rare classes across images. The fixed variant instead caps
detections **per class across the whole dataset** (10,000) and removes that degree of freedom. LVIS
results since 2021 usually report it.

### Open Images: hierarchy and group-of boxes

Open Images evaluates at IoU 0.5 only, expands labels up a class hierarchy (a "jaguar" detection also
counts for "animal" when the ground truth says so), and has **group-of** boxes: one box over many
similar objects. Detections inside a group-of box count as a single TP. Scores are not comparable
with COCO AP.

### Datasets that report AP$_{50}$ only

VOC, many industrial benchmarks and RF100-VL report AP$_{50}$ alongside AP. When the task only needs
"is it there and roughly where" (counting, alarms, coarse localisation), AP$_{50}$ is the
operationally relevant number. Do not optimise AP$_{75}$ you will never use.

---

## Diagnosing errors: TIDE

A single AP number does not say *why* a model fails. **TIDE** (Bolya et al., 2020) assigns every false
positive and every missed ground truth to one error type, then reports how much AP would rise if that
error type were fixed ($\Delta$AP):

| Error | Definition (foreground IoU 0.5, background 0.1) | Typical fix |
|---|---|---|
| **Cls** | IoU ≥ 0.5 with a GT of a *different* class | more data for confused classes, better features |
| **Loc** | right class, 0.1 ≤ IoU < 0.5 | box loss, resolution, assignment |
| **Both** | wrong class *and* poor box | usually a weak region; more data |
| **Dupe** | right class, IoU ≥ 0.5, GT already matched | NMS threshold, one-to-one head |
| **Bkg** | IoU < 0.1 with every GT | hard negatives, label completeness (often a *missing label*) |
| **Miss** | GT never matched | recall: resolution, assignment, small-object fixes |

`odlab.metrics.error_breakdown` implements the counting part of this taxonomy (not the $\Delta$AP
re-evaluation). It is enough to see, for example, that most of a model's FPs are "Bkg" and that half of
those are unlabelled objects, which is a data problem and not a model problem (Chapter 9).

---

## The number you deploy: F1 at a threshold

AP integrates over thresholds, but a deployed system runs at **one** threshold (or one per class). The
relevant metrics are precision and recall at that operating point, F1, or recall at a fixed precision
("≥ 95% precision, maximise recall").

`odlab.metrics.best_f1_threshold(preds, gts)` returns the confidence threshold that maximises micro-F1
at IoU 0.5. Ultralytics plots the same F1-vs-confidence curve after validation (`F1_curve.png`).

<div class="callout field"><span class="callout-title">Field note</span>Two models with equal AP
can differ by 10 points of recall at your required precision. The PR curves cross. Always compare
candidate models <em>at the operating point you will deploy</em>, per class, on data from the
deployment domain.</div>

---

## How metrics mislead

| Pitfall | What happens | Guard |
|---|---|---|
| **Different evaluators** | pycocotools vs Ultralytics vs torchmetrics vs custom: interpolation, maxDets, area definition, crowd handling differ by tenths of AP or more | Evaluate every model in a comparison with the same tool; prefer pycocotools on COCO-format JSON |
| **val vs test-dev** | COCO test-dev numbers are typically 0.1–0.5 AP higher or lower than val2017 | Never mix columns |
| **Data leakage** | Pre-training sets (Objects365, LVIS-derived grounding data) overlap COCO images or classes; "zero-shot" claims may not be | Read the data card; GroundingCap-style sets include COCO images (Chapter 20) |
| **TTA / multi-scale** | Test-time augmentation adds 1–2 AP at many times the cost | Report with and without |
| **Score threshold** | AP at conf 0.25 is lower than at 0.001 | Use 0.001 for AP |
| **Seed noise** | COCO AP varies by about ±0.1–0.3 between seeds for the same recipe; small datasets vary by several points | Multiple seeds, bootstrap confidence intervals on small data |
| **Small validation sets** | 100 validation images give AP with a standard error of several points | Bootstrap over images; report intervals |

---

## Key Takeaways

- A detection is a TP if it matches an unmatched ground truth of its class at IoU ≥ t, processed in
  score order. Duplicates are FPs, and crowd matches are ignored.
- AP is the area under the monotone precision envelope. VOC07, VOC10+ and COCO sample it differently.
  The same detections gave 0.614, 0.625 and 0.629 in our worked example.
- COCO AP averages ten IoU thresholds and therefore rewards precise boxes. Size buckets use mask area
  on COCO, and `-1` means an empty bucket.
- LVIS (federated labels, AP$_r$, AP$^{\text{fixed}}$) and Open Images (hierarchy, group-of) are not
  comparable with COCO AP.
- TIDE-style error decomposition says *why* AP is low. Background errors are often missing labels.
- Deployment cares about precision and recall at one threshold. Compare models there, per class.

## Check Yourself

<details class="check"><summary>A model's AP50 rises from 70 to 72 while its AP stays at 50. What changed?</summary>
It finds or ranks more objects correctly at a loose IoU, but its boxes got no more precise. Gains at
IoU 0.5 were offset by losses at higher thresholds, or there were none above 0.5. Typical causes are
better recall or classification with unchanged localisation, or a change that trades box precision for
recall (for example a lower NMS threshold).</details>

<details class="check"><summary>Why can a model have higher AR100 but lower AP than another?</summary>
AR only asks whether each ground truth is matched by some detection among the top 100. It ignores the
order of the scores. AP depends on ranking, because precision at each recall level depends on how many
false positives score above the true positives. Many candidates with poorly calibrated scores give
high AR and low AP.</details>

<details class="check"><summary>You converted a custom dataset to COCO JSON with area = w × h and report AP_S. Can you compare it with a paper's COCO AP_S?</summary>
No. COCO's area is the segmentation mask area, so the small/medium/large partition differs. The
datasets also differ, so the comparison was already weak. Report your own size definitions
explicitly.</details>

<details class="check"><summary>pycocotools reports AP_L = −1.000 on your validation set. Is the model broken?</summary>
No. −1 means there are no ground-truth objects in the large bucket (> 96² area), so the value is
undefined and excluded from averages. It is a statement about the dataset, not the model.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Microsoft COCO | Lin et al., 2014 | arXiv:1405.0312 | AP@[.5:.95], size buckets, 41/34/24% split |
| pycocotools `cocoeval.py` | COCO API | github.com/cocodataset/cocoapi | Matching, crowd handling, accumulation, 101-point interpolation |
| The PASCAL VOC Challenge | Everingham et al., 2010 | IJCV 2010 | 11-point and all-points AP |
| LVIS | Gupta et al., 2019 | arXiv:1908.03195 | Federated evaluation, AP_r/c/f |
| Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details | Dave et al., 2021 | arXiv:2102.01066 | AP-fixed |
| The Open Images Dataset V4 | Kuznetsova et al., 2020 | arXiv:1811.00982 | Hierarchy, group-of boxes |
| TIDE | Bolya et al., 2020 | arXiv:2008.08115 | Error taxonomy and ΔAP |
| Ultralytics `utils/metrics.py` | Ultralytics | github.com/ultralytics/ultralytics | F1 curves, AP implementation |
| `odlab/metrics.py` + `tests/test_metrics.py` | this book | code/odlab | pycocotools-exact re-implementation |

---

**Next:** [Chapter 9 — Datasets, Formats & Annotation](./09_datasets_and_annotation.md) — metrics are
only as good as the labels they compare against.
