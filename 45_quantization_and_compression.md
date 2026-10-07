---
title: "Chapter 45 — Quantising & Compressing Detectors"
---

[← Back to Table of Contents](./README.md)

# Chapter 45 — Quantising & Compressing Detectors

> *"A classifier needs the right argmax. A detector needs the right argmax, at the right pixel, with a score that still ranks correctly after rounding."*

## Overview

INT8 is the native precision of most edge accelerators, and it roughly halves memory traffic again
compared with FP16. Detectors lose more accuracy in INT8 than classifiers do, for reasons specific to
their outputs and heads. This chapter explains those reasons, gives a post-training quantisation (PTQ)
recipe and a quantisation-aware training (QAT) recipe for YOLO-style models (including Ultralytics'
built-in QAT), measures PTQ on real YOLO26n and YOLO11n exports, and covers pruning and the other
compression options. The fundamentals of quantisation (number formats, scales, calibration methods) are
covered in depth in the companion book,
[Quantization — The Complete Guide](https://karam-nus.github.io/quantization/), whose detection chapter
complements this one.

<div class="diagram">
<div class="diagram-title">Where detectors lose accuracy in INT8</div>
<div class="layer-stack">
  <div class="layer red">Output tensor: box coordinates (0–640) and scores (0–1) sharing one scale</div>
  <div class="layer orange">Head: final 1×1 convs, DFL softmax over 16 bins</div>
  <div class="layer yellow">Attention blocks: softmax, LayerNorm and matmul outliers</div>
  <div class="layer green">Concats of features with very different ranges (PAN fusion)</div>
  <div class="layer blue">Re-parameterised convs with wide weight ranges; the first conv</div>
</div>
</div>

---

## Why detectors quantise worse than classifiers

1. **Regression needs precision.** A classifier tolerates noise as long as the top logit stays on top.
   A box edge off by 2 px drops IoU on small objects from 1.0 to 0.6 (Chapter 41). INT8 damage shows up
   first in **AP75 and APs**, not AP50.
2. **Scores are ranked across thousands of candidates.** AP integrates over the whole ranking.
   Quantisation noise that swaps the order of near-tied candidates costs AP even when every argmax is
   unchanged.
3. **Heterogeneous outputs.** Exported YOLOs concatenate decoded box coordinates (hundreds of pixels)
   with probabilities (0–1) in one tensor. Quantised with one scale, the scores lose all resolution
   (measured below).
4. **Distribution heads.** DFL's softmax over 16 bins followed by an expectation is sensitive to small
   logit errors. YOLO26 removed it partly for this reason (Chapter 37).
5. **Attention and normalisation.** Softmax, LayerNorm and attention matmuls have outliers and wide
   dynamic ranges (Chapter 44's operator matrix).
6. **Calibration data mismatch.** Calibrating on mosaic-augmented or random images gives ranges that do
   not match deployment activations.

---

## Measured: PTQ on YOLO26n and YOLO11n

ONNX Runtime static quantisation (QDQ format, per-channel symmetric INT8 weights, asymmetric UINT8
activations, MinMax calibration on 64 COCO128 images), evaluated with Ultralytics' validator on the
other 64 COCO128 images, on CPU. COCO128 is tiny, so read the numbers for their direction and size, not
as COCO results.

| Model and variant | mAP50-95 | mAP50 | CPU latency (ms) | File size (MB) |
|---|---|---|---|---|
| YOLO26n FP32 | 0.480 | 0.638 | 22.3 | 9.9 |
| YOLO26n INT8, **whole graph** | **0.000** | **0.000** | 17.9 | 3.1 |
| YOLO26n INT8, **head output convs + decoding kept in FP32** | 0.472 (−0.8) | 0.629 | 20.5 | 3.2 |
| YOLO11n FP32 | 0.497 | 0.654 | 30.6 | 10.7 |
| YOLO11n INT8, whole graph | **0.000** | **0.000** | 17.5 | 3.2 |
| YOLO11n INT8, head output convs + DFL + decoding kept in FP32 | 0.485 (−1.2) | 0.651 | 28.7 | 3.4 |

**Why the whole-graph INT8 models score exactly zero.** The exported one-to-many output concatenates
decoded box coordinates (up to about 660 px) with sigmoid class scores (0–1) in one tensor. Quantised with
a single 8-bit scale of about 2.6 px per step, every score rounds to the same value: on bus.jpg the FP32
model produces 5,509 distinct score values with a maximum of 0.90, the whole-graph INT8 model a single
value, 0.0, while its box coordinates survive. Nothing passes the confidence threshold. Leaving the 6
final 1×1 head convs and the post-head decoding in FP32 (93–103 graph nodes, all cheap) recovers almost
everything.

Three further observations:

- **YOLO26n lost less than YOLO11n** (0.8 vs 1.2 points), consistent with the DFL head being the more
  sensitive design. The difference is within the noise of a 64-image validation set, so treat it as a
  direction, not a measurement.
- **INT8 bought little speed on this CPU** (8% and 6%) because ONNX Runtime's QDQ execution on this
  machine does not have fast INT8 kernels for every layer. Accelerators with native INT8 show the
  1.2–1.6× speed-ups of Chapter 16's Jetson table. INT8's size reduction (3×) is the same everywhere.
- COCO128 images come from COCO train2017, which both checkpoints were trained on, so the absolute mAP
  values are optimistic. Only the FP32 → INT8 changes are meaningful here.

---

## A PTQ recipe for YOLO-style detectors

1. **Export cleanly**: static shapes, BN folded, the head option you will deploy (Chapter 36).
2. **Calibration set**: 200–1,000 **real deployment images**, preprocessed exactly as at inference
   (letterbox, RGB, /255). No mosaic. Cover lighting, scenes and object sizes. The Ultralytics TensorRT
   path uses up to about 500 images.
3. **Keep the output in float**: leave the final 1×1 convs of the box and class towers, DFL (if
   present) and all post-head decoding in FP16/FP32. The Ultralytics exporter does exactly this
   (`nodes_to_exclude` for `.2/Conv` and `dfl` in its ModelOpt path, Chapter 36). The cost in latency is
   tiny; the accuracy saving is large.
4. **Per-channel weights, per-tensor activations**: the standard, well supported by NPUs.
5. **Choose the calibration method**: MinMax is simple and robust for CNNs; percentile or entropy
   clipping helps when activations have outliers (attention, SiLU tails).
6. **Evaluate like a detector**: mAP50-95, AP75, APs, and per-class AP of the quantised artifact
   against the FP16 artifact, on the same validation images. AP50 alone hides most of the damage.
7. **If the drop is too large**: find the sensitive layers (quantise one layer at a time, or use the
   toolchain's sensitivity analysis), keep them in FP16 (mixed precision), and only then move to QAT.

---

## QAT, and Ultralytics' built-in QAT

Quantisation-aware training inserts fake-quantisation (quantise → dequantise) into the forward pass so
the weights adapt to rounding. For detectors it is the standard remedy when PTQ loses more than about
1 AP.

Current Ultralytics releases implement QAT through NVIDIA ModelOpt:

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")                 # a converged FP model
model.train(data="widgets.yaml", epochs=20, lr0=1e-4, optimizer="AdamW", quantize=8)  # QAT fine-tune
model.export(format="engine", quantize=8)                         # Q/DQ ranges carried into the engine
```

What `prepare_qat` does (`utils/torch_utils.py`):

- swaps Conv and Linear layers for ModelOpt equivalents that fake-quantise inputs and weights with
  `INT8_DEFAULT_CFG`,
- **calibrates activation and weight ranges once** from 8 training batches, then keeps them fixed (they
  are buffers, not learned), so training adapts the weights to the ranges,
- keeps BatchNorm unfused during training (folded at export, with ranges rescaled),
- **leaves the head's output layers and the DFL conv in float**,
- stores the quantisation state in the checkpoint (`modelopt` key) so resumed training and export
  restore it.

Practical QAT guidance for detectors: start from the converged FP checkpoint, fine-tune for about 5–10%
of the original schedule at a learning rate 10–100× lower, keep augmentation light (close to the last
epochs of the original schedule), and validate the exported INT8 artifact, not the fake-quantised
PyTorch model.

---

## Below INT8, and other precisions

| Precision | Where | Detector notes |
|---|---|---|
| **FP16 / BF16** | GPUs, Apple, many NPUs | Near-lossless for CNN detectors; the default on GPUs |
| **FP8** | Hopper/Blackwell GPUs, some new NPUs | Promising for transformer detectors; toolchain support still maturing |
| **INT8** | Almost every accelerator | The deployment standard; needs the care above |
| **W8A16, W4A16** | Some NPUs and CPUs (Ultralytics `quantize="w8a16"`) | Weight-only compression; less accuracy risk; less speed-up on compute-bound layers |
| **INT4 / mixed 4-8 bit** | Some edge NPUs (Hailo, MCU research) | Large accuracy loss for small detectors without QAT and careful layer selection |
| **Binary / ternary** | Research | Not practical for detection accuracy today |

---

## Pruning and other compression

| Technique | What it removes | Effect on detectors |
|---|---|---|
| **Structured channel pruning** | Whole channels/filters (by norm, BN scale, or sensitivity) | Real speed-ups on all hardware if it reduces work where the time is spent: stride 8–16 layers (Chapter 29), not the parameter-heavy stride-32 layers |
| **Unstructured sparsity** | Individual weights | Little speed-up except on hardware with sparse support |
| **2:4 semi-structured sparsity** | 2 of every 4 weights | Supported by NVIDIA Ampere+ tensor cores; needs fine-tuning |
| **Depth pruning / layer dropping** | Whole blocks | Large latency gains on GPUs where overhead dominates (Chapter 44) |
| **Knowledge distillation into a smaller scale** | The whole model | Often better than pruning a large model: train the n model with an s or m teacher (Chapter 42) |
| **Lower input resolution** | Pixels | The simplest compression; costs small-object recall (Chapter 41) |

A useful rule: **before pruning, try the next model scale down plus distillation, and a lower input
size.** The YOLO family's n/s/m/l/x scales are already a well-tuned pruning ladder.

---

## Designing for INT8 from the start

If INT8 deployment is certain, choose architectures that quantise well:

- **No DFL** (YOLO26) or a DFL head kept in float.
- **ReLU-family activations** on accelerators without good SiLU support (YAML `activation: nn.ReLU()`,
  Chapter 29), retrained, not swapped after training.
- **Quantisation-aware re-parameterisation** (QA-RepVGG, RepOptimizer) if using RepVGG-style blocks.
- **Attention only at low resolution** (C2PSA at stride 32) or none.
- **Separate output tensors** for boxes and scores, or float outputs.

---

## Key Takeaways

- Detectors lose more in INT8 than classifiers because of box precision, score ranking, heterogeneous
  outputs, distribution heads, attention and calibration mismatch. The damage shows in AP75 and APs.
- PTQ recipe: real calibration images with exact preprocessing, per-channel weights, head output and
  decoding kept in float, evaluation of the quantised artifact on mAP50-95, AP75 and APs.
- Quantising the concatenated output tensor (coordinates with scores) is the classic catastrophic
  mistake.
- QAT is the remedy when PTQ loses more than about 1 AP. Ultralytics supports it with
  `train(quantize=8)` through NVIDIA ModelOpt, with calibrated fixed ranges and a float head output.
- Prune where the time is (stride 8–16), or step down a model scale and distil. Lower input size is the
  simplest compression of all.

## Check Yourself

<details class="check"><summary>After INT8 conversion, AP50 drops by 0.3 points but AP75 by 3 points. What does this tell you, and where would you look?</summary>
Detection and classification survive, but box precision does not. Look at the box path: the final box
convs, DFL (if present) and decoding should stay in higher precision; check the calibration images and
the clipping method for the box-tower activations. Small objects will be hit hardest.</details>

<details class="check"><summary>Why might pruning 50% of the channels in the last backbone stage of a nano YOLO give a smaller file but almost no speed-up on a GPU?</summary>
The last stage (stride 32) holds most of the parameters but a minority of the compute and activation
traffic, and nano models on GPUs are dominated by memory traffic and overheads (Chapters 29 and 44).
Pruning there shrinks the model file without touching where the time goes.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Quantization — The Complete Guide (companion book) | karam-nus | karam-nus.github.io/quantization | Fundamentals; detection chapter |
| Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | Jacob et al., 2018 | arXiv:1712.05877 | INT8 scheme, QAT |
| A White Paper on Neural Network Quantization | Nagel et al., 2021 | arXiv:2106.08295 | PTQ/QAT practice |
| Integer Quantization for Deep Learning Inference | Wu et al., 2020 | arXiv:2004.09602 | Calibration methods |
| RepOptimizer; QA-RepVGG | Ding et al., 2022; Chu et al., 2022 | arXiv:2205.15242; arXiv:2212.01593 | Quantisation-friendly re-parameterisation |
| Accelerating Sparse Deep Neural Networks (2:4) | Mishra et al., 2021 | arXiv:2104.08378 | Semi-structured sparsity |
| Ultralytics `utils/torch_utils.py` (`prepare_qat`), `utils/export/engine.py` (`modelopt_quantize_onnx`) | Ultralytics | github.com/ultralytics/ultralytics | QAT and INT8 export details |
| NVIDIA ModelOpt | NVIDIA | github.com/NVIDIA/TensorRT-Model-Optimizer | Quantisation toolkit |
| ONNX Runtime quantisation | Microsoft | onnxruntime.ai/docs/performance/model-optimizations/quantization.html | Static QDQ quantisation used for the measurements |
| Measurements in this chapter | this book | `tools/measurements/int8_study.py` | PTQ results on YOLO26n/YOLO11n |

---

**Next:** [Chapter 46 — Measuring Latency Honestly](./46_measuring_latency.md) — what "1.7 ms" includes,
and how to measure what you will actually ship.
