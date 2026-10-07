---
title: "Chapter 49 — Hands-On: Zero-Shot → Edge"
---

[← Back to Table of Contents](./README.md)

# Chapter 49 — Hands-On: Zero-Shot → Edge

> *"Foundation models are excellent annotators and expensive detectors. Use them for the first job and a nano model for the second."*

## Overview

You have images, a list of classes and no labels, and the model must run on an edge device. This lab
builds the standard 2026 pipeline: an **open-vocabulary detector labels** the images from text prompts,
a human **reviews** a fraction, a **nano detector is trained** on the result, and the nano model is
**exported** to the device. It also measures what each shortcut costs. The first measurement below,
the quality of raw auto-labels against human labels, was run for this book. The rest is a protocol with
a results template.

<div class="diagram">
<div class="diagram-title">From text prompts to a 2-ms detector</div>
<div class="flow-h">
  <div class="flow-node">unlabelled images</div>
  <div class="flow-arrow"></div>
  <div class="flow-node purple">open-vocabulary labeller<small>YOLOE · Grounding DINO · SAM 3 · VLM</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node orange">human review<small>all, a sample, or uncertain only</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node accent">train YOLO26n</div>
  <div class="flow-arrow"></div>
  <div class="flow-node green">export INT8 · device</div>
</div>
</div>

---

## Step 1: choose the labeller

| Labeller | Speed | Strength | Weakness | Licence (check) |
|---|---|---|---|---|
| **YOLOE-26** (text or visual prompts) | Real time | Fast; exports as a closed-set detector after `set_classes` | Weaker on rare or abstract concepts | AGPL-3.0 |
| **Grounding DINO / MM-Grounding-DINO** | Slow (seconds per image on CPU) | Strong phrase grounding | Over-detects with long prompts; box quality varies | Apache-2.0 |
| **SAM 3** (promptable concept segmentation) | Moderate | Masks + boxes from text; exhaustive instance finding | Heavier model | SAM licence |
| **DINO-X / API models** | Service | Strong long-tail recognition | Cost, data leaves your premises | Commercial API |
| **VLMs** (Qwen-VL family, Gemini, others) | Slow | Complex descriptions ("cracked tile") | Localisation precision; hallucinated boxes (Chapter 21) | Varies |

A practical default: YOLOE for common objects, with Grounding DINO or SAM 3 as a second opinion on
classes where YOLOE's recall is poor.

---

## Step 2: measure the labeller before trusting it

Label 100–200 images by hand (the **audit set**) and score the labeller on them. Measured for this book:
YOLOE-26s with the 80 COCO class names as text prompts, on the 128 COCO128 images, scored against their
human labels with `odlab`'s COCO-exact evaluator:

<!-- AUTOLABEL_TABLE -->

These numbers are the error rate of your training labels if you skip review. Every miss becomes a
background example; every false positive becomes a wrong positive.

---

## Step 3: generate labels and review

```python
from ultralytics import YOLOE
lab = YOLOE("yoloe-26s-seg.pt")
lab.set_classes(["forklift", "pallet", "person", "safety cone"])
for r in lab.predict("images/", conf=0.30, stream=True):         # per-class thresholds from Step 2
    r.save_txt(f"labels/{r.path.stem}.txt")                        # YOLO-format pre-labels
```

Review strategies, in increasing cost:

| Strategy | What humans do | When |
|---|---|---|
| **No review** | Nothing | Only if Step 2 showed high precision and recall for every class |
| **Uncertainty review** | Check images with scores near the threshold, and images where two labellers disagree | Most projects: the best accuracy per hour |
| **Sample review** | Fix a random 10–20% and measure the error rate on it | Estimating label quality for reporting |
| **Full review** | Fix every image (much faster than drawing from scratch) | Safety-critical classes, small datasets |

Prompt tuning is part of this step: synonyms, more specific phrases ("red safety cone" vs "cone"), or a
visual prompt from a few example boxes when the class has no good name.

---

## Step 4: train the nano model

Train YOLO26n (or another edge model, Chapter 16) on the reviewed labels with the small-data recipe if
the set is small (Chapter 31): start from COCO or Objects365 weights, light augmentation, early stopping,
several seeds. Evaluate on a **human-labelled test set**, never on auto-labels.

To measure what the shortcuts cost, train the same model three times on the same images:

| Run | Training labels | Test labels |
|---|---|---|
| A | Raw auto-labels | Human |
| B | Auto-labels after uncertainty review | Human |
| C | Fully human labels (if affordable for a subset) | Human |

The A–C gap is the price of skipping review. The B–C gap is the price of partial review.

---

## Step 5: export to the device

Export the trained nano model for the target runtime and precision (Chapters 36, 44, 45). Two choices
matter here:

- The **labeller is not deployed**. The nano model is a closed-set detector with its own export path,
  so the labeller's size and licence do not constrain the device. (Its licence may still constrain how
  you use its outputs: check.)
- If you need the open-vocabulary model on the device itself, YOLOE can be re-parameterised into a
  closed-set detector for a fixed prompt list and exported like any YOLO (Chapter 38).

---

## Results template

| Field | Value |
|---|---|
| Classes and prompts (final wording) | |
| Labeller, version, per-class thresholds | |
| Audit set: precision / recall per class at those thresholds | |
| Review strategy and human hours | |
| Training images / instances per class | |
| Run A / B / C test mAP50-95 and recall at deployment threshold | |
| Exported model: format, precision, latency on device | |
| Accuracy of the exported model on the human test set | |

---

## Key Takeaways

- Open-vocabulary detectors are fast annotators with systematic errors. Measure their precision and
  recall per class on a small hand-labelled audit set before using their labels.
- Review where it pays: uncertain images and classes with poor labeller recall. Raw auto-labels teach
  the student the labeller's misses as background.
- Train a nano detector on the reviewed labels, evaluate on human labels only, and quantify the cost of
  skipping review with A/B/C runs.
- The labeller is a training-time tool. The deployed model is an ordinary closed-set detector.

## Check Yourself

<details class="check"><summary>Your auto-labeller has 0.9 precision and 0.6 recall on the audit set. What will the student model learn, and what review strategy fixes it?</summary>
The 40% of objects it misses are present in the images but unlabelled, so the student learns them as
background and inherits (and may amplify) the labeller's blind spots. Review should focus on finding
misses: lower the labeller threshold for review pre-labels, add a second labeller, and check images
where the two disagree.</details>

<details class="check"><summary>Why must the test set be human-labelled even if training labels come from the labeller?</summary>
Evaluating against auto-labels measures agreement with the labeller, not correctness. A student that
copies the labeller's mistakes would score well. Only human labels measure what the deployed model gets
right.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| YOLOE | Wang et al., 2025 | arXiv:2503.07465 | Text and visual prompts; re-parameterisation |
| Grounding DINO | Liu et al., 2023 | arXiv:2303.05499 | Open-set grounding |
| SAM 3 | Carion et al. (Meta), 2025 | arXiv:2511.16719 | Promptable concept segmentation |
| Roboflow100-VL | Robicheaux et al., 2025 | arXiv:2505.20612 | How open-vocabulary models fail on unusual domains |
| Ultralytics YOLOE docs | Ultralytics | docs.ultralytics.com/models/yoloe | `set_classes`, export |
| Chapters 20, 21, 42 | this book | — | Open-vocabulary and VLM detection; auto-labelling practice |
| Measurement in this chapter | this book | `autolabel_study.py` | Auto-label quality on COCO128 |

---

**Next:** [Chapter 50 — The Debugging Playbook](./50_debugging_playbook.md) — symptom → cause →
diagnostic → fix.
