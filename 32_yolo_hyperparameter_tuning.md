---
title: "Chapter 32 — YOLO Axis 6: Hyperparameter Tuning"
---

[← Back to Table of Contents](./README.md)

# Chapter 32 — YOLO Axis 6: Hyperparameter Tuning

> *"Tune the decisions before the knobs, and measure the noise before the effect."*

## Overview

Ultralytics exposes about 30 training hyperparameters and ships a genetic-algorithm tuner that searches
them. That makes "tune everything for 300 iterations" look like the default plan. It is rarely the best
use of compute. This chapter separates **decisions** (model scale, input size, starting weights,
schedule length) from **knobs** (learning rate, loss gains, augmentation strengths). It shows how the
Ultralytics tuner works, line by line, how to budget a search, and how to tell a real improvement from
seed noise, using a small measured experiment.

<div class="diagram">
<div class="diagram-title">Where accuracy comes from, in the order to spend effort</div>
<div class="layer-stack">
  <div class="layer red">1. Data: label quality, coverage of the deployment conditions, size (Chapters 9, 27)</div>
  <div class="layer orange">2. Decisions: model scale · imgsz · starting checkpoint · epochs</div>
  <div class="layer yellow">3. Domain-driven augmentation: flips, rotation, scale range, colour (Chapter 28)</div>
  <div class="layer green">4. Optimisation knobs: lr0, lrf, warm-up, weight decay, optimiser</div>
  <div class="layer blue">5. Loss gains and the rest: box / cls / dfl, mosaic, mixup, HSV …</div>
</div>
</div>

---

## Decisions before knobs

These four choices are not in the tuner's search space, and each usually moves mAP more than a tuning
run does:

| Decision | Typical effect | How to decide |
|---|---|---|
| **Model scale** (n → s → m) | +3 to +8 COCO AP per step (YOLO26: n 40.9, s 48.6, m 53.1) at 2–3× compute | Smallest model that meets accuracy at the latency budget (Chapters 34, 46) |
| **Input size** (`imgsz`) | Often the largest single lever for small-object AP; 640 → 1280 is ~4× compute | Box-size histogram at the deployment resolution (Chapter 27); P2 head as an alternative (Chapter 41) |
| **Starting weights** | From scratch vs pre-trained: a large gap on small datasets | COCO and Objects365 checkpoints, one run each (Chapter 27) |
| **Schedule length** | Under-training is the most common silent loss | Train long enough that mAP50-95 plateaus. Let `patience` stop it |

Only after these are settled do the knobs below matter enough to search.

---

## Which knobs matter

A practical ranking, from experience and the published recipes rather than a controlled study. Treat it
as a starting order, not a law:

| Tier | Knobs | Why |
|---|---|---|
| **High** | `lr0`, `lrf`, `optimizer`, `scale`, `mosaic`, `close_mosaic` | Set how far and how fast weights move, and the object-size distribution the model sees |
| **Domain** | `fliplr`, `flipud`, `degrees`, `hsv_*`, `copy_paste`, `mixup` | Usually binary questions with a correct answer for the domain (is a mirrored sign still the same sign? does the camera rotate?) |
| **Medium** | `box`, `cls`, `dfl`, `cls_pw`, `warmup_epochs`, `weight_decay` | Move the operating point (localisation vs recall); defaults are good starting values |
| **Low** | `momentum`, `warmup_momentum`, `translate`, `shear`, `perspective`, `bgr` | Small effects on most data |

The domain tier should be decided by reasoning, not search. If left-right flipping changes the meaning
of an object (text, arrows, left/right-hand parts), `fliplr` must be 0, whatever a short search
suggests. If the camera looks straight down (aerial, microscopy), `flipud=0.5` and rotation are almost
free accuracy.

---

## The Ultralytics tuner

```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")
model.tune(data="widgets.yaml", epochs=30, iterations=100, optimizer="AdamW",
           plots=False, save=False, val=False)          # val only at the final epoch of each trial
```

Each **iteration is one complete training run** with one hyperparameter set, scored by fitness
(mAP50-95). `iterations=100, epochs=30` means 100 separate 30-epoch trainings.

### The algorithm (`engine/tuner.py`)

1. **Iteration 1** trains with the current defaults (or your arguments).
2. Each later iteration builds a child from the history:
   - take the **top 9** results by fitness and normalise every hyperparameter to [0, 1] within its
     search bounds,
   - choose one parent at random, **weighted by fitness** (weight = fitness − min fitness + 10⁻⁶),
   - mutate each gene with probability 0.5 by Gaussian noise with σ = 0.2 × gain (gains below 1 for some
     parameters, e.g. `momentum` 0.3). σ shrinks by up to 20% when the best result has not improved for
     25 iterations,
   - once there are **30 or more** results, sometimes (probability 0.4 × confidence) sample the
     mutated genes from a **multivariate normal fitted to the elite** (top 20%, at most 30 runs)
     instead. This is a CMA-ES-like step that learns correlations, such as "higher `lr0` goes with higher
     `weight_decay`",
   - reflect values back into the bounds, round integer parameters, and reject exact duplicates.
3. Results go to `tune_results.ndjson`. The best set is written to `best_hyperparameters.yaml`, and
   non-best run directories are deleted to save space.

The tuner can run across **several datasets at once** (`data=[...]`, fitness averaged) and across
machines through a shared MongoDB collection. With `use_ray=True` it hands the search to **Ray Tune**,
which runs trials in parallel and can stop weak trials early with ASHA.

### The default search space

| Group | Parameter: (min, max) |
|---|---|
| Optimisation | `lr0` (1e-5, 1e-2) · `lrf` (0.01, 1.0) · `momentum` (0.7, 0.98) · `weight_decay` (0, 0.001) · `warmup_epochs` (0, 5) · `warmup_momentum` (0, 0.95) |
| Loss | `box` (1, 20) · `cls` (0.1, 4) · `cls_pw` (0, 1) · `dfl` (0.4, 12) |
| Colour | `hsv_h` (0, 0.1) · `hsv_s` (0, 0.9) · `hsv_v` (0, 0.9) · `bgr` (0, 1) |
| Geometry | `degrees` (0, 45) · `translate` (0, 0.9) · `scale` (0, 0.95) · `shear` (0, 10) · `perspective` (0, 0.001) · `flipud` (0, 1) · `fliplr` (0, 1) |
| Mixing | `mosaic` (0, 1) · `mixup` (0, 1) · `cutmix` (0, 1) · `copy_paste` (0, 1) · `close_mosaic` (0, 10) |

Twenty-six dimensions. A genetic search needs many iterations to make progress in that many dimensions,
which is why the YOLO26 recipes were searched with large compute budgets. **Pass a reduced `space`**
with the 4–8 parameters that matter for your problem:

```python
space = {"lr0": (1e-4, 1e-2), "lrf": (0.01, 1.0), "scale": (0.2, 0.9),
         "mosaic": (0.3, 1.0), "box": (4.0, 12.0), "cls": (0.3, 1.5)}
model.tune(data="widgets.yaml", epochs=30, iterations=60, space=space, optimizer="AdamW")
```

<div class="callout warn"><span class="callout-title">optimizer=auto during tuning</span>With the default
<code>optimizer=auto</code>, the trainer replaces <code>lr0</code> with its own value (Chapter 31), so
the tuner's mutations of <code>lr0</code> have no effect. Always pass an explicit optimiser when
<code>lr0</code> is in the search space.</div>

---

## Budgeting a search

Cost = iterations × epochs per trial × time per epoch.

| Dataset | Epoch time (one GPU, s model) | Trial (30 epochs) | 100 iterations |
|---|---|---|---|
| 1,000 images | ~10 s | 5 min | ~8 h |
| 10,000 images | ~1.5 min | 45 min | ~3 days |
| COCO | ~15–20 min | ~9 h | months |

Ways to cut cost, and what each risks:

| Shortcut | Risk |
|---|---|
| Fewer epochs per trial | Favours settings that converge fast (high LR, weak augmentation), which lose in a full-length run |
| `fraction=0.25` of the data | Favours less regularisation; small-data optima are not large-data optima |
| Smaller `imgsz` | Shifts optimal `scale` and augmentation for small objects |
| Smaller model (n instead of s) | Optimal augmentation strength grows with model size (the YOLO26 tables show it) |
| ASHA early stopping (Ray) | Kills slow starters, the same bias as fewer epochs |

The proxy-bias column is why **the final candidate must be retrained at full length** and compared
with the full-length baseline, ideally with several seeds.

---

## Noise first: how big is a real improvement?

Every training run is a random draw: initialisation, data order, augmentation samples and some
non-deterministic GPU kernels all vary. A tuner comparing single runs will happily select noise. This
book's TinyYOLO (Chapter 39) measures the size of that noise against a real hyperparameter effect: three
learning rates, three seeds each, with identical data. Each run is 12 epochs on 768 synthetic training
images, scored on the same 128 validation images.

| Learning rate (AdamW) | AP, seeds 0 / 1 / 2 | Mean AP | Seed std | Seed range | Mean APs |
|---|---|---|---|---|---|
| 5e-4 | 0.679 / 0.670 / 0.675 | 0.675 | 0.004 | 0.008 | 0.538 |
| **2e-3** | 0.760 / 0.760 / 0.749 | **0.756** | 0.007 | 0.012 | **0.676** |
| 8e-3 | 0.735 / 0.711 / 0.735 | 0.727 | **0.014** | **0.024** | 0.650 |

What it shows:

- **The real effect is large relative to noise here.** Moving from 5e-4 to 2e-3 gains 8 AP points,
  about ten times the seed spread. No statistics are needed to see it.
- **The 2e-3 vs 8e-3 comparison is closer.** The means differ by 2.9 points, but the 8e-3 seeds span
  2.4 points on their own. A single run of each could have shown a gap anywhere from about 1.4 to 4.9
  points. With one run per setting, a tuner could rank them correctly or not.
- **Too-high learning rates increase variance.** The seed spread at 8e-3 is twice that at 2e-3. That is a
  useful diagnostic: when repeated runs of a setting disagree, it is often near an instability.
- **Small-object AP is noisier than overall AP** (APs seed std up to 0.021), and it is the metric most
  hurt by a too-low learning rate on this short schedule.

This is a 3M-parameter model on synthetic data with a 128-image validation set. Real datasets with
fewer, harder objects per image tend to have larger seed spreads, not smaller ones.

How to use measurements like these:

1. **Estimate the seed spread on your data first.** Run the baseline 3 times. The standard deviation
   of mAP50-95 is your noise floor.
2. **Compare means, not single runs.** A change smaller than about twice the seed standard deviation,
   measured on single runs, is not evidence.
3. **Validation size sets the floor.** With a few hundred validation images, single-class AP can move
   by several points between seeds. Report overall mAP with a bootstrap confidence interval over images
   (Chapter 33) before trusting per-class gains.
4. **Tune on validation, report on test.** After 100 tuning iterations the best validation score is
   optimistically biased: you selected the luckiest run. Keep a test split the tuner never sees.

---

## Reading the YOLO26 recipe as a tuning result

The COCO-stage hyperparameters of every YOLO26 checkpoint were found by evolutionary search
(Chapter 31). They show what a large search actually changes:

| Setting | Default | N | S–X | What the search learned |
|---|---|---|---|---|
| `lr0` / `lrf` | 0.01 / 0.01 | 0.0054 / 0.0495 | 0.00038 / 0.882 | Larger models barely move from the Objects365 weights: low, almost constant LR |
| `box` / `cls` / `dfl` | 7.5 / 0.5 / 1.5 | 5.63 / 0.56 / 9.04 | 9.83 / 0.65 / 0.96 | N leans on the L1 distance term, S–X on CIoU |
| `scale` | 0.5 | 0.56 | 0.9 (S), 0.95 (M–X) | Larger models want stronger scale jitter |
| `mixup` / `copy_paste` | 0 / 0 | 0.012 / 0.075 | 0.05–0.43 / 0.30–0.40 | Regularisation grows with capacity |
| `degrees` / `shear` / `bgr` | 0 / 0 / 0 | 1.11 / 1.46 / 0.106 | ≈ 0 | Only N benefits from extra geometric and colour noise |
| `fliplr` | 0.5 | 0.61 | 0.30 | Even "obvious" defaults were moved by the search |

Two lessons transfer to your own work. First, **optimal hyperparameters depend on model size** even on
the same data, so values tuned for n should not be copied to m. Second, for fine-tuning from a strong
checkpoint, **gentle optimisation** (low `lr0`, high `lrf`) often wins. The tuner found that for COCO
with Objects365 weights, and it is a sensible first hypothesis for custom data.

---

## A manual playbook

When a full search is too expensive, change one thing at a time, in this order, with 2–3 seeds each:

| Step | Try | Keep if | Typical symptom it fixes |
|---|---|---|---|
| 1 | Train 2× longer | mAP still rising at the end | Under-training |
| 2 | `imgsz` up one step (640 → 832/960) | Small-object AP improves enough to pay for compute | Small objects missed |
| 3 | `lr0` × 0.3 and × 3 (explicit optimiser) | Clear change beyond seed spread | Fine-tune forgetting / slow convergence |
| 4 | `scale` 0.3 / 0.5 / 0.9 | — | Objects at sizes not seen in training |
| 5 | `mosaic` 1.0 vs 0.5, `close_mosaic` 10 vs 20 | — | Context-dependent objects; train/val distribution gap |
| 6 | `copy_paste` / `mixup` 0.1–0.3 (larger models) | — | Over-fitting (train loss ↓, val mAP ↓) |
| 7 | `box` 7.5 → 10–12 | AP75 ↑ without AP50 loss | Loose boxes |
| 8 | `cls_pw` 0.5 | Rare-class AP ↑ | Long-tailed classes |

---

## Key Takeaways

- Settle model scale, input size, starting weights and schedule length before searching knobs. Each
  usually matters more than a tuning run.
- Decide domain augmentations (flips, rotation, colour) by reasoning about the data, not by search.
- The Ultralytics tuner is a fitness-weighted genetic algorithm (top-9 parents, 50% gene mutation,
  σ = 0.2 of the range) with a covariance-guided step once 30 results exist. Each iteration is a full
  training run.
- Reduce the search space to a handful of parameters, set an explicit optimiser, and budget
  iterations × epochs × epoch time.
- Short or reduced proxies bias the search toward fast-converging, weakly regularised settings.
  Retrain the winner at full length against a full-length baseline.
- Measure seed variance first. On small validation sets it is often as large as the effect being tuned.

## Check Yourself

<details class="check"><summary>A 50-iteration tune with epochs=10 picks lr0=0.009 and mosaic=0.2. Retrained for 150 epochs, it loses to the defaults. Why?</summary>
Ten-epoch trials reward settings that converge quickly: a high learning rate and weak augmentation look
best early. Over 150 epochs, stronger augmentation and a gentler LR generalise better. The search
optimised a proxy (10-epoch mAP) whose ranking does not match the target (150-epoch mAP).</details>

<details class="check"><summary>Run A scores 0.512 mAP50-95 and run B 0.519 on a 300-image validation set. Should you switch to B's settings?</summary>
Not on this evidence. A 0.007 difference between single runs is within typical seed noise for a
validation set this small. Run both settings with 2–3 seeds, compare the means against the spread, and
confirm on a held-out test set.</details>

<details class="check"><summary>Why should fliplr stay out of the search space for a dataset of road signs?</summary>
A mirrored "turn left" sign is a "turn right" sign, so flipping creates wrong labels. The correct value
(0) follows from the domain. A search might still pick a non-zero value by chance on a validation set
with few such signs.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics `engine/tuner.py` | Ultralytics | github.com/ultralytics/ultralytics | Default search space, mutation algorithm, multi-dataset and MongoDB tuning |
| Ultralytics hyperparameter tuning guide | Ultralytics | docs/en/guides/hyperparameter-tuning.md | Iteration semantics, resume, outputs |
| Ultralytics Ray Tune integration | Ultralytics | docs/en/integrations/ray-tune.md | Parallel trials, ASHA |
| YOLO26 training recipe guide | Ultralytics, 2026 | docs/en/guides/yolo26-training-recipe.md | Evolved per-size hyperparameters |
| Hyperband / ASHA | Li et al., 2018; Li et al., 2020 | arXiv:1603.06560; arXiv:1810.05934 | Early stopping of trials, proxy bias |
| CMA-ES tutorial | Hansen, 2016 | arXiv:1604.00772 | Covariance-guided mutation |
| Accounting for variance in ML benchmarks | Bouthillier et al., 2021 | arXiv:2103.03098 | Seed variance, comparing means |
| `odlab` LR × seed experiment | this book | `tools/measurements/seed_lr.py` | Measured noise vs effect |

---

**Next:** [Chapter 33 — Axis 7: Metrics & Validation](./33_yolo_metrics_and_validation.md) — what
`model.val()` actually computes, and how it differs from pycocotools.
