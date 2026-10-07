---
title: "Chapter 47 — Detection in Production"
---

[← Back to Table of Contents](./README.md)

# Chapter 47 — Detection in Production

> *"In production the model is one stage of a pipeline, its threshold is a business decision, and its test set is whatever arrives tomorrow."*

## Overview

A deployed detector is a system: cameras, decoding, the model, thresholds, tracking, business rules,
alerts, storage, monitoring, and a loop that turns failures into new training data. This chapter covers
the decisions that live outside the network: pipeline patterns, choosing and calibrating thresholds,
per-class operating points, monitoring for drift without labels, the data flywheel, model updates and
rollback, failure review, and the safety and privacy questions detection systems raise.

<div class="diagram">
<div class="diagram-title">The production loop</div>
<div class="cycle">
  <div class="cycle-step accent">deploy model vN</div><div class="cycle-arrow"></div>
  <div class="cycle-step green">monitor: scores · rates · drift · feedback</div><div class="cycle-arrow"></div>
  <div class="cycle-step purple">mine failures and uncertain cases</div><div class="cycle-arrow"></div>
  <div class="cycle-step orange">label · review</div><div class="cycle-arrow"></div>
  <div class="cycle-step accent">retrain · frozen-set gates · shadow / canary</div>
</div>
</div>

---

## Pipeline patterns

| Pattern | Structure | When |
|---|---|---|
| **Single-stage** | Detector → threshold → output | Clear, well-separated classes; tight latency |
| **Detector + tracker** | Detector → tracker → rules on tracks (dwell, line crossing, counting) | Video, counting, alarms that must not flicker |
| **Cascade** | Cheap detector (high recall) → crops → stronger classifier or detector (precision) | Rare events, fine-grained classes, edge-plus-cloud |
| **Detector + VLM verifier** | Detector proposes; a VLM checks crops against a text description | Open-ended rules ("a person not wearing a harness near the edge"); low event rates (Chapter 21) |
| **Ensemble / WBF** | Several detectors, boxes fused by weighted box fusion | Offline analytics where accuracy beats cost |

The cascade deserves emphasis: it lets a 2-ms nano model run everywhere at a low threshold, while an
expensive model sees only the few percent of frames that matter.

---

## Thresholds are product decisions

`conf=0.25` is a default, not a choice. The right threshold depends on the cost of each error:

| Application | Costly error | Operating point |
|---|---|---|
| Safety alarm (person in a danger zone) | Miss | High recall (low threshold), suppress flicker with tracking |
| Automated rejection on a production line | False reject | High precision (high threshold), human review of borderline cases |
| Counting / analytics | Bias in both directions | Threshold where expected FP ≈ expected FN, validated on counts |
| Pre-labelling for annotation | Misses (annotators fix FPs more easily) | Low threshold |

Procedure:

1. Collect a **threshold-tuning set** from deployment conditions (not the training validation set).
2. For each class, plot precision and recall against confidence (Ultralytics `P_curve`, `R_curve`).
3. Choose **per-class thresholds** from the error costs. Rare classes usually need lower thresholds.
4. Freeze them, then verify on a separate test set.
5. Re-tune after every retraining: a new model's score distribution is different.

---

## Calibration

A detector's confidence is a ranking score, not a probability. TAL-trained models (Chapter 30) learn
scores that track IoU, not the chance that the detection is correct. Calibration matters when scores are
combined with other evidence, used for risk decisions, or compared across models or classes.

| Method | How | Notes |
|---|---|---|
| **Reliability diagram / ECE** | Bin detections by score; compare mean score with precision in each bin (at a fixed IoU) | Detection-ECE must fix an IoU threshold for "correct" |
| **Temperature / Platt scaling** | Fit one or two parameters on held-out detections | Cheap; per class if data allows |
| **Isotonic regression** | Monotone mapping from score to precision | Needs more data |

Re-calibrate after every model update, and per camera type if the cameras differ a lot.

---

## Monitoring without labels

In production you rarely have labels. Signals that move before accuracy does:

| Signal | What it catches | How |
|---|---|---|
| **Detections per frame**, per class, per camera | Camera moved, occlusion, lighting change, model failure | Time series with seasonal baselines (time of day, weekday) |
| **Score distribution** (histogram of top scores) | Domain shift: scores drift down or become bimodal | Population stability index or KL divergence against a reference window |
| **Box size and position distribution** | Zoom change, mount change, new object types | Histograms per camera |
| **Embedding drift** | Appearance change (season, weather, sensor) | Distance between current and reference feature distributions (e.g. backbone pooled features) |
| **Disagreement with a reference model** | Silent regressions | Run a larger model on a small sample and compare |
| **Human feedback** | Ground truth for the cases people notice | Make "wrong detection" one click away; log the frame |
| **System health** | Latency, dropped frames, temperature | Standard observability (Chapter 46) |

Set alerts on changes relative to each camera's own baseline. A global threshold on "detections per
frame" means nothing across a parking lot and a corridor.

---

## The data flywheel

The highest-return activity after launch is turning production failures into training data:

1. **Capture**: log frames with low-margin scores, disagreements, user-flagged errors, and a random
   sample (to avoid only seeing hard cases).
2. **Prioritise**: rare classes, new conditions (night, rain), recurring false-positive patterns.
3. **Label and review**, with auto-labels from a stronger model as pre-labels (Chapter 42).
4. **Retrain** on old + new data (avoid forgetting, Chapter 40).
5. **Evaluate on frozen sets**: the original test set (no regression), a "hard cases" set that grows
   over time, and per-camera slices.
6. **Ship safely** (below), then repeat.

---

## Shipping model updates

| Step | Purpose |
|---|---|
| **Version everything** | Model weights, export artifact, thresholds, preprocessing code, training data snapshot, package versions (Chapter 35) |
| **Offline gate** | No regression beyond tolerance on frozen sets, per class and per slice |
| **Golden-image test** on the target device | Catches export and preprocessing regressions (Chapter 36) |
| **Shadow mode** | New model runs alongside the old one; compare outputs without acting on them |
| **Canary** | Roll out to a few cameras or users; watch the monitoring signals |
| **Rollback path** | Keep the previous artifact and thresholds deployable in minutes |

---

## Failure review

Review failures as a routine, with a taxonomy so they can be counted:

| Category | Example | Typical fix |
|---|---|---|
| Missed small or distant object | Person at 8 px | Resolution, P2, tiling (Chapter 41) |
| Missed under occlusion or crowding | Queue of people | NMS settings, crowd methods (Chapter 43) |
| False positive on a look-alike | Poster of a person, reflections | Hard negatives in training; background images |
| Wrong class | Van vs truck | Label policy, more examples, class merge |
| Duplicate boxes | Two boxes on one car | NMS threshold; one-to-one head; tracking |
| Localisation too loose | Box covers shadow | Label policy, `box` gain |
| Condition shift | Night, rain, lens dirt | Targeted data; augmentation; monitoring (Chapter 43) |
| Pipeline bug | Rotated phone images, BGR input | Golden-image tests |

TIDE (Chapter 8) gives the same breakdown automatically on labelled data.

---

## Safety, privacy and fairness

Detection systems often watch people. Engineering decisions here have consequences:

- **Purpose limitation and minimisation**: detect what the product needs; avoid storing raw frames
  when metadata suffices; on-sensor or on-device processing (Chapter 44) keeps images local.
- **Performance across groups and conditions**: measure recall by lighting, distance, clothing, skin
  tone, mobility aids, where the application involves people, and fix gaps with data.
- **Human oversight** for consequential decisions: detection should trigger review, not punishment.
- **Regulation**: rules on biometric identification and surveillance (for example the EU AI Act's
  restrictions on remote biometric identification) apply to some uses of person detection and tracking.
  Check them early.

---

## Key Takeaways

- Production detection is a pipeline: detector, tracker, rules, cascade or VLM verifier. The cascade
  pattern makes cheap models practical for rare events.
- Thresholds are per-class product decisions from error costs, tuned on deployment data and re-tuned
  after every model update. Detector scores are rankings; calibrate them if they feed decisions.
- Monitor without labels: detections per frame, score and size distributions, embedding drift,
  disagreement with a reference model, user feedback, all against per-camera baselines.
- The data flywheel (capture, prioritise, label, retrain, evaluate on frozen sets, ship) is the main
  source of improvement after launch.
- Ship with versioning, offline gates, golden-image tests, shadow and canary stages, and a rollback.

## Check Yourself

<details class="check"><summary>After a retrain, mAP rises by 1.5 points, but the alarm system fires twice as often. What was missed?</summary>
The thresholds were not re-tuned. The new model's score distribution differs, so the old per-class
thresholds sit at different precision–recall points. Re-tune thresholds on the deployment tuning set
(and re-calibrate) before rollout, and use shadow mode to compare alarm rates first.</details>

<details class="check"><summary>Detections per frame on one camera dropped 40% overnight, with no model change. Name three things to check, in order.</summary>
(1) The camera: moved, obstructed, refocused, or a dirty lens (look at frames). (2) Conditions: weather,
lighting, seasonal change (score and embedding drift). (3) The pipeline: decoding, resolution or
preprocessing changes after a software update (golden-image test on that camera's stream).</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| On Calibration of Modern Neural Networks | Guo et al., 2017 | arXiv:1706.04599 | Temperature scaling, ECE |
| Multivariate confidence calibration for object detection | Küppers et al., 2020 | arXiv:2004.13546 | Detection calibration |
| Weighted Boxes Fusion | Solovyev, Wang, Gabruseva, 2021 | arXiv:1910.13302 | Ensemble fusion |
| TIDE | Bolya et al., 2020 | arXiv:2008.08115 | Failure taxonomy |
| Hidden Technical Debt in Machine Learning Systems | Sculley et al., 2015 | NeurIPS 2015 | Production ML pitfalls |
| Failing Loudly: dataset shift detection | Rabanser, Günnemann, Lipton, 2019 | arXiv:1810.11953 | Drift detection without labels |
| EU AI Act | European Union, 2024 | eur-lex.europa.eu | Rules on biometric identification |

---

**Next:** [Chapter 48 — Hands-On: Fine-Tune, Evaluate, Export](./48_handson_finetune_export.md) — Part VII
begins: the whole process on a real dataset.
