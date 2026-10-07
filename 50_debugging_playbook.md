---
title: "Chapter 50 — The Debugging Playbook"
---

[← Back to Table of Contents](./README.md)

# Chapter 50 — The Debugging Playbook

> *"Most detection bugs do not crash. They lower a number by a few points and wait."*

## Overview

This chapter is an index of failures. Each row gives a **symptom** you can observe, the **likely
causes** in order of frequency, a **diagnostic** that distinguishes them in minutes, and the **fix**,
with a pointer to the chapter that explains it. Use it when something is wrong and you do not yet know
what. The rows are grouped by where the symptom appears: data, training, evaluation, export and
deployment, production.

<div class="callout field"><span class="callout-title">The three universal diagnostics</span>
<ol style="margin:0.3rem 0 0 1rem">
<li><b>Look at the pictures.</b> Training batches with labels drawn, predictions on validation images, the exported model's predictions on the same images.</li>
<li><b>Shrink the problem.</b> One image, one class, one batch, augmentation off, FP32, PyTorch.</li>
<li><b>Change one thing.</b> Then compare against a baseline measured the same way, with more than one seed.</li>
</ol></div>

---

## Data and labels

| Symptom | Likely causes | Diagnostic | Fix | Ch. |
|---|---|---|---|---|
| Log says "N backgrounds" ≈ all images | Labels not found: wrong `labels/` path, stem mismatch, extension | Check the images → labels path substitution on one file | Fix directory names; delete `*.cache` | 27 |
| Boxes drawn in the wrong place on `train_batch*.jpg` | x/y swapped, xyxy vs cxcywh, not normalised, wrong image size | Draw raw labels on raw images before any augmentation | Fix the converter; round-trip test | 2, 27 |
| One class never detected | Class index offset (1-based vs 0-based), class missing in val, too few or too small instances | Per-split class histogram; per-class size histogram at `imgsz` | Fix indices; add data; raise resolution | 27, 41 |
| Rotated phone photos have rotated boxes | EXIF orientation ignored by labelling tool or loader | View with EXIF-aware loader | Bake orientation into pixels before labelling | 27 |
| High val mAP, poor field performance | Leakage (near-duplicate frames across splits), val not representative | Perceptual-hash duplicates across splits; compare val vs field image statistics | Split by source; build a field test set | 9, 40 |
| Many "false positives" are real objects | Missing labels | TIDE background errors; inspect top-scoring FPs | Label audit; auto-label + review | 8, 42 |
| Class confusions between two specific classes | Inconsistent label policy | Confusion matrix at deployment threshold; review confused examples | Clarify guideline; relabel; or merge classes | 9, 33 |

---

## Training

| Symptom | Likely causes | Diagnostic | Fix | Ch. |
|---|---|---|---|---|
| Loss flat from epoch 1 | Labels wrong or empty; LR far off; frozen everything | Overfit one batch with augmentation off | Fix pipeline; check `freeze`; set LR explicitly | 40 |
| Loss NaN / inf | AMP on an incompatible device; LR too high for the optimiser; zero-area boxes | Re-run with `amp=False`; check min box size | Fix labels; lower LR; disable AMP | 31 |
| Your `lr0` seems ignored | `optimizer=auto` chose its own LR | Read the optimiser line in the log | Set `optimizer=` explicitly | 31 |
| Train loss ↓, val mAP ↓ after a peak | Overfitting | Train/val curves; compare augmentation strength | More data or augmentation; earlier stop; smaller model | 31, 40 |
| mAP still rising at the last epoch | Under-training | Curve slope at the end | More epochs; check `close_mosaic` timing | 31 |
| Small objects never detected | No grid points inside boxes; downscaling; mosaic shrinking them | Positives per object (Chapter 30 table); size histogram after resize | Higher `imgsz`; P2; STAL; tiling | 30, 41 |
| DFL loss stuck around 0.8–0.9 | Normal: entropy floor (≈ 0.5 nats × gain 1.5) | Divide by the gain and compare with the floor | Nothing; watch mAP | 6, 39 |
| Large accuracy drop when fine-tuning with a new class | Old classes unlabelled in new images → learned as background | Old-class recall on old test set | Pseudo-label old classes; retrain on union | 40 |
| Results vary by several points between runs | Small validation set; seed noise | 3 seeds; bootstrap interval | Report mean ± spread; bigger test set | 32, 33 |
| Out-of-memory mid-epoch | Images with many objects in the assigner; large `imgsz` | Log max objects per image | Smaller batch; the trainer halves batch on OOM in epoch 1 | 31 |
| Multi-GPU run worse than single-GPU | Effective batch and weight decay changed; fewer warm-up steps | Compare `accumulate` and steps per epoch | Adjust `lr0`; keep images per step at 64 | 31 |

---

## Evaluation

| Symptom | Likely causes | Diagnostic | Fix | Ch. |
|---|---|---|---|---|
| mAP much lower than during training | `conf=0.25` passed to `val`; different `imgsz`; different split | Re-run `val` with defaults on the same split | Leave `conf` unset for mAP | 33 |
| Your mAP differs from a paper's | Different evaluator, split, `max_det`, input size, head | Score both with pycocotools on the same JSON | Same protocol for every model | 8, 33 |
| P and R don't match deployment behaviour | Printed P/R are at the max-F1 threshold, not yours | Read P/R curves at your threshold | Report recall/precision at the deployment threshold | 33 |
| Confusion matrix full of background FPs | Built at conf 0.001 | Re-run with `conf=thr` | Separate thresholded run | 33 |
| AP50 fine, AP75 poor | Loose boxes: label policy, quantisation, low `box` gain, resolution | AP75 vs AP50 per size; TIDE localisation errors | Tighten labels; keep head in float; raise `box` | 8, 45 |
| One-to-one head much worse than one-to-many | Short fine-tune; crowded scenes | Validate both heads | Longer training; use NMS head if budget allows | 30, 39 |

---

## Export and deployment

| Symptom | Likely causes | Diagnostic | Fix | Ch. |
|---|---|---|---|---|
| Exported model detects nothing | Scores quantised with coordinates in one INT8 tensor; double sigmoid; wrong output parsing | FP32 export first; print score range | Keep output float; parse the actual output format | 36, 45 |
| Boxes shifted or scaled wrongly | Letterbox not undone (pad/gain order); cx-cy-w-h treated as xyxy | Golden-image test: compare with `predict()` | Fix post-processing | 36 |
| Accuracy a few points lower, no obvious error | BGR vs RGB; plain resize instead of letterbox; ImageNet normalisation | Feed the same tensor to PyTorch and the runtime | Match preprocessing exactly | 28, 36 |
| 0.5–1 mAP lower than `val()` | Square fixed input vs rect val; FP16; NMS settings | Validate the artifact with the same validator | Accept or use `rect`-like shapes; adjust NMS | 28, 36 |
| `nms=False` export still needs NMS | Format fell back to one-to-many (NPU toolchains) | Exporter warning; output shape | Host-side NMS | 36, 44 |
| NPU slower than CPU | Unsupported op → CPU fallback with copies | Compiler placement report | Remove/replace op; cut graph before head | 44 |
| INT8 accuracy collapse | Bad calibration data; sensitive layers quantised | Per-layer sensitivity; calibrate on real images | Mixed precision; QAT | 45 |
| TensorRT engine fails on another machine | Engine built for a different GPU/TRT version | Check versions | Build on target | 36, 44 |
| Latency far above the model card | Card is forward-only; you measure end to end; PyTorch vs TensorRT; no warm-up | Time each stage | Optimise the slow stage; measure correctly | 46 |

---

## Production

| Symptom | Likely causes | Diagnostic | Fix | Ch. |
|---|---|---|---|---|
| Detections per frame dropped on one camera | Camera moved/obstructed; lighting; pipeline change | Look at frames; score and embedding drift | Fix camera; targeted data; rollback | 47 |
| Alarm rate doubled after a model update | Thresholds not re-tuned for the new score distribution | Shadow-mode comparison | Re-tune per-class thresholds; canary | 47 |
| Tracks break during occlusion | Low-confidence detections filtered before the tracker | Tracker input count | Pass low-confidence boxes to the tracker | 38 |
| People in queues undercounted | NMS suppressing overlapping people | Raise NMS IoU on a sample | Soft-NMS; visible-box labels; one-to-one head | 43 |
| Performance degrades at night or in rain | Domain shift | Per-condition slices of the test set | Targeted data and augmentation | 43 |
| Latency spikes after minutes | Thermal throttling; memory growth | Sustained-load measurement | Smaller model; cooling; fixed allocations | 44, 46 |

---

## Key Takeaways

- Look at pictures, shrink the problem, change one thing: three diagnostics that resolve most cases.
- Most silent failures are in data (labels, paths, leakage) and in the pre/post-processing around an
  exported model, not in the architecture.
- Numbers disagree for reasons that are not bugs: evaluator, threshold, head, input shape, seed. Rule
  those out before hunting for one.
- Every symptom here has a chapter that explains its mechanism. Use the tables to find it.

## Check Yourself

<details class="check"><summary>A colleague reports "the ONNX model's scores are all between 0.5 and 0.73". Which row applies, and what is the fix?</summary>
Export and deployment, "exported model detects nothing / wrong scores": a sigmoid has been applied to
scores that the exported graph already passed through a sigmoid. Remove the second sigmoid.</details>

<details class="check"><summary>Training loss falls normally, but validation mAP is 0.00 from the first epoch. Which section, and the first diagnostic?</summary>
Data and labels (validation side): the validation labels are probably not found or not matching (paths,
class indices, coordinate format). Check the validation scan line ("backgrounds") and draw validation
labels on images.</details>

## References

This chapter consolidates the diagnostics of the chapters cited in each row. Primary sources are listed
in those chapters.

---

**Next:** [Chapter 51 — The Decision Guide](./51_decision_guide.md) — from constraints to a concrete
model and recipe.
