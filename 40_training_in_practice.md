---
title: "Chapter 40 — Training Detectors in Practice"
---

[← Back to Table of Contents](./README.md)

# Chapter 40 — Training Detectors in Practice

> *"Before tuning anything, prove that the model can overfit one batch and that you can see what it was told to learn."*

## Overview

Part IV covered the YOLO recipe in detail. This chapter is detector-agnostic: the practical decisions
that come up in any detection project, whatever the architecture. It covers the first-day sanity checks,
pre-training versus training from scratch, the small-data regime, long-tailed classes, normalisation with
small batches, schedule length, merging datasets with different label sets, adding classes to a deployed
model, and how to read training curves. Each section ends in a concrete default.

<div class="diagram">
<div class="diagram-title">A training project in order</div>
<div class="flow-h">
  <div class="flow-node">sanity checks<small>overfit one batch · view labels</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node blue">baseline<small>pre-trained, defaults</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node purple">error analysis<small>per class · per size · TIDE</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node accent">fix data first<small>then recipe</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node green">freeze test set · report</div>
</div>
</div>

---

## Day one: sanity checks

Run these before any real training. Each takes minutes and catches the bugs that otherwise cost days.

| Check | How | Catches |
|---|---|---|
| **Draw the training labels** | Visualise a batch after augmentation (`train_batch0.jpg` in Ultralytics) | Swapped x/y, wrong normalisation, class-index offsets, boxes lost by augmentation |
| **Overfit one batch** | Train on 8–16 images with augmentation off for a few hundred steps | Loss that cannot reach near zero means a pipeline bug, not a hard problem |
| **Count positives per object** | Log how many points/anchors each object gets (Chapter 30) | Objects too small for the stride, wrong coordinate scale |
| **Predict with the untrained model** | Score distribution should be near the class prior | Wrong bias init → thousands of confident boxes on step 1 |
| **Validate the pre-trained model zero-shot** | Evaluate COCO weights on your val set for shared classes | Validation-label format errors (AP ≈ 0 on obviously detectable objects) |
| **Check for leakage** | Near-duplicate search across splits | Inflated validation numbers |

---

## Pre-training or from scratch

He, Girshick and Dollár (2019) showed that detectors trained **from scratch** can match ImageNet
pre-training on COCO, given **longer schedules** and normalisation that works with small batches
(GroupNorm or SyncBN). Pre-training mainly buys **faster convergence** and **robustness on small data**.
For practice:

| Your data | Recommendation |
|---|---|
| < 10k images | Always start from a detection checkpoint (COCO or Objects365), not just an ImageNet backbone |
| 10k–100k images | Pre-trained detection checkpoint; from scratch only if the domain is very far (medical, radar, thermal) and you can afford 3–5× longer training |
| > 100k images, distinct domain | From scratch becomes competitive; consider pre-training on your own unlabeled data (self-supervised) or a foundation backbone |
| Any size, unusual imagery | Try a foundation-model backbone (DINOv2/v3-based detectors such as RF-DETR, DEIMv2) alongside a YOLO; they often transfer better to unusual images (Chapter 34) |

Pre-training on a **detection** dataset (COCO, Objects365) transfers more than ImageNet classification
pre-training does, because the neck and head are also pre-trained. Objects365 pre-training is the main
reason several 2024–2026 detectors gained several AP on COCO (Chapter 37).

---

## The small-data regime (50–1,000 images)

Variance dominates. The goals are to not destroy the pre-trained features and to measure honestly.

1. **Freeze or slow the backbone.** Freezing the backbone (`freeze=10` in YOLO26) or using a lower LR
   for it keeps general features. Unfreeze if the domain is far from the pre-training data.
2. **Lower LR, shorter schedule, early stopping.** AdamW at 1e-3 or SGD at 1e-3–1e-2 with patience.
3. **Lighter, domain-correct augmentation.** Reduce mosaic and mixup (they create unrealistic contexts
   that a small dataset cannot average out); keep flips, scale and colour that match deployment.
4. **Use every label.** Background images (0–10%) for false-positive control; tight boxes; consistent
   rules for occlusion and truncation.
5. **Measure with uncertainty.** K-fold cross-validation or several seeds; bootstrap intervals
   (Chapter 33). With 100 validation images, a 3-point difference can be noise.
6. **Consider a different starting point**: an open-vocabulary detector with a text or visual prompt
   may beat a fine-tuned closed-set model when you have fewer than about 50 examples per class
   (Chapters 20 and 49).

---

## Long-tailed classes

Real datasets are long-tailed: a few classes have most of the instances. A detector trained with
uniform sampling and loss learns the rare classes poorly: their classifiers are pushed down by every
other class's positives acting as negatives. Methods, from simplest:

| Method | Mechanism | Where it is standard |
|---|---|---|
| **Repeat-factor sampling (RFS)** | Repeat image $i$ with factor $r_i = \max_{c \in i} \max\!\big(1, \sqrt{t / f_c}\big)$, where $f_c$ is the fraction of images containing class $c$ and $t$ a threshold (0.001 on LVIS) | LVIS baselines; easy to add to any loader |
| **Class-balanced loss weights** | Weight by inverse "effective number" of samples $\frac{1 - \beta}{1 - \beta^{n_c}}$ | Classification; Ultralytics' `cls_pw` is a simpler inverse-frequency power |
| **Equalization Loss (EQL, EQLv2)** | Ignore the suppressing gradient that rare classes receive as negatives from other classes' positives | Long-tail detection research |
| **Seesaw loss** | Rescale negative gradients by class-frequency ratio, with compensation for misclassification | MMDetection LVIS configs |
| **Federated loss** | Per image, apply negatives only for classes known to be annotated or present in the batch | LVIS (federated annotation), multi-dataset training |
| **Copy-paste** | Paste rare instances into other images | Strong on LVIS with masks |
| **Data** | Mine more rare-class images with an open-vocabulary detector (Chapter 42) | Always the best per unit of effort when possible |

**Default:** RFS plus copy-paste if you have masks; per-class thresholds at deployment; per-class AP
in every report. Look at the rare classes' recall at the deployment threshold, not just mAP.

---

## Normalisation with small batches

BatchNorm statistics need enough samples per GPU. Below about 8 images per GPU (common at 1280 px or
with large models), they become noisy and accuracy drops.

| Option | What it does | Cost |
|---|---|---|
| **Frozen BN** | Use the pre-trained statistics and affine parameters; never update them | None; standard in two-stage detectors with ImageNet backbones |
| **SyncBN** | Compute BN statistics across all GPUs | Communication per BN layer; needs multiple GPUs |
| **GroupNorm** | Normalise over channel groups per sample; independent of batch | Slightly slower; changes the architecture (pre-trained BN weights do not transfer directly) |
| **Gradient accumulation** | Larger effective batch for the optimiser | Does **not** fix BN statistics |

**Default:** keep batch ≥ 8 per GPU by lowering resolution or model size. For high-resolution
fine-tuning with small batches, freeze BN in the backbone or use SyncBN across GPUs.

---

## How long to train

| Family | Typical schedule | Why |
|---|---|---|
| Two-stage, RetinaNet (Detectron2) | 1× = 12 epochs, 3× = 36 epochs, light augmentation | Sparse, stable supervision; ImageNet-initialised |
| YOLO from scratch | 300–600 epochs, heavy augmentation | No pre-trained detector; augmentation needs time to pay off |
| YOLO fine-tune | 50–200 epochs | Pre-trained neck and head |
| DETR (original) | 500 epochs | Slow Hungarian-matching convergence |
| Modern DETRs (RT-DETR, D-FINE, DEIM) | 12–120 epochs | Denoising, better queries, Dense O2O |

Rules of thumb: train until validation mAP50-95 plateaus, not until the loss plateaus (soft targets and
the DFL floor keep the loss from reaching zero). If mAP is still rising at the end, the schedule was too
short. If it peaked early and fell, regularise or stop earlier.

---

## Merging datasets and label spaces

Combining datasets ("our 2,000 images + Open Images vehicles + COCO people") brings a subtle bug: an
image from dataset A contains objects of a class only labelled in dataset B. Trained naïvely, the model
learns those objects as **background**.

Options:

1. **Re-label** the union with the full class list (best, and an auto-labelling job: Chapter 42).
2. **Federated / partial-label loss**: for each image, compute classification negatives only for the
   classes its source dataset annotates (the approach of LVIS and of multi-dataset detectors).
3. **Per-dataset heads** with a shared backbone, merged at inference with a unified taxonomy (Zhou et
   al., 2022, "Simple multi-dataset detection").

Also check that **class definitions** match ("person" vs "pedestrian", whether a bicycle box includes the
rider).

---

## Adding classes to a deployed model

Fine-tuning on new-class data alone causes **catastrophic forgetting**: the old classes' performance
collapses, and old-class objects in the new images are learned as background. In order of reliability:

1. **Retrain on old + new data** with complete labels. Simplest when the old data is available.
2. **Pseudo-label the old classes** in the new images with the current model, then fine-tune on the
   union.
3. **Distil from the old model** (output or feature distillation on old classes) while learning the
   new ones (Shmelkov et al., 2017).
4. For many new classes or open-ended growth, use an **open-vocabulary detector** and fine-tune its
   prompts or embeddings.

---

## Reading training curves

| Pattern | Likely meaning | Action |
|---|---|---|
| Train loss ↓, val mAP ↑, both still moving at the end | Under-trained | Train longer |
| Train loss ↓, val mAP peaks then ↓ | Over-fitting | More augmentation or data, earlier stop, smaller model |
| Val mAP jumps when mosaic switches off | Expected (`close_mosaic`) | None |
| Loss flat from epoch 1 | Pipeline bug: labels, coordinates, LR far off | Sanity checks above |
| Loss NaN | AMP issue, LR too high for the optimiser, zero-area boxes | `amp=False` to isolate; check labels |
| Val mAP noisy epoch to epoch | Small validation set; EMA not yet warm | Bootstrap intervals; look at the trend |
| Class loss ↑ late in training while mAP ↑ | Soft targets rising as boxes improve (TAL) | Usually harmless |
| One class at 0 AP | Missing in val, index mismatch, or no positives (too small) | Per-class instance counts and size histogram |

---

## Key Takeaways

- Overfit one batch and draw your training labels before any experiment.
- Start from a detection checkpoint for anything under ~100k images. From-scratch training can match it
  but needs longer schedules and batch-independent normalisation.
- Small data: protect pre-trained features, lighten augmentation, and measure with seeds or folds.
- Long tail: repeat-factor sampling and copy-paste first, loss reweighting second, per-class
  thresholds at deployment. Report per-class recall.
- BatchNorm needs ≥ 8 images per GPU. Accumulation does not fix BN; frozen BN, SyncBN or GroupNorm do.
- Merging datasets or adding classes silently teaches the model that unlabelled objects are
  background. Re-label, use partial-label losses, or pseudo-label.

## Check Yourself

<details class="check"><summary>You merge your 3,000 forklift images with COCO to get a "person" class. Person AP on your own images is poor. Why?</summary>
Your 3,000 images probably contain people who were never labelled. During training those people are
negatives, so the model learns that people in your domain are background. Pseudo-label people in your
images with a COCO model (and review them), or use a loss that ignores the person class on images from
your dataset.</details>

<details class="check"><summary>Training at 1280 px with batch 4 per GPU gives worse results than 640 px with batch 32, despite small objects. What is a likely cause besides resolution?</summary>
BatchNorm statistics computed from 4 images are noisy, which hurts training and the running statistics
used at inference. Use SyncBN across GPUs, freeze BN in the backbone, or increase the per-GPU batch with a
smaller model or tiled crops at 640.</details>

<details class="check"><summary>A class with 40 training instances has 0.0 AP. List three checks in order.</summary>
(1) Is it present in the validation labels with the same class index? (2) Are its objects large enough
to receive positives at the training resolution (size histogram)? (3) Do its training labels look right
when drawn? Only then consider imbalance methods such as repeat-factor sampling.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Rethinking ImageNet Pre-training | He, Girshick, Dollár, 2019 | arXiv:1811.08883 | From-scratch vs pre-training |
| LVIS | Gupta, Dollár, Girshick, 2019 | arXiv:1908.03195 | Repeat-factor sampling, federated annotation |
| Class-Balanced Loss | Cui et al., 2019 | arXiv:1901.05555 | Effective number of samples |
| Equalization Loss; EQLv2 | Tan et al., 2020; 2021 | arXiv:2003.05176; arXiv:2012.08548 | Long-tail losses |
| Seesaw Loss | Wang et al., 2021 | arXiv:2008.10032 | Long-tail loss |
| Probabilistic two-stage detection (federated loss) | Zhou, Koltun, Krähenbühl, 2021 | arXiv:2103.07461 | Federated loss |
| Simple Copy-Paste | Ghiasi et al., 2021 | arXiv:2012.07177 | Copy-paste |
| Group Normalization | Wu, He, 2018 | arXiv:1803.08494 | Batch-independent normalisation |
| MegDet (SyncBN) | Peng et al., 2018 | arXiv:1711.07240 | Cross-GPU BN for detection |
| Simple multi-dataset detection | Zhou, Koltun, Krähenbühl, 2022 | arXiv:2102.13086 | Unified label spaces |
| Incremental learning of object detectors without catastrophic forgetting | Shmelkov, Schmid, Alahari, 2017 | arXiv:1708.06977 | Distillation for new classes |
| Detectron2 model zoo | Wu et al., 2019 | github.com/facebookresearch/detectron2 | 1×/3× schedules |

---

**Next:** [Chapter 41 — Small Objects](./41_small_objects.md) — the most common reason a detector that
looks good on COCO fails in the field.
