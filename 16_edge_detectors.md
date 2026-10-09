---
title: "Chapter 16 — Edge Detectors"
---

[← Back to Table of Contents](./README.md)

# Chapter 16 — Edge Detectors

> *"A T4 millisecond is not an edge millisecond. The chip decides which operators are cheap."*

## Overview

"Edge" here means application-class devices: phones, Jetson modules, Raspberry Pi boards (alone or with
an AI accelerator), Rockchip/Hailo/Coral NPUs, smart cameras. They have megabytes of SRAM and gigabytes
of DRAM, 1–275 TOPS of INT8 compute, and a narrower, vendor-specific set of fast operators. Models here
have 1–10 M parameters (a few up to 30 M) and run at 640 px or below. This chapter defines what makes a
detector *edge-friendly*, collects measured on-device numbers, maps the 2024–2026 edge models, and gives a
selection procedure. Chapter 44 covers the hardware in depth, Chapter 45 quantisation, and Chapter 36 YOLO
export.

<div class="diagram">
<div class="diagram-title">What decides edge latency</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent"><div class="card-title">Operator support</div><div class="card-desc">Is every op on the NPU in INT8? One unsupported op means a CPU round-trip.</div></div>
  <div class="diagram-card green"><div class="card-title">Memory traffic</div><div class="card-desc">Activations at stride 8, concat/upsample, depthwise convs — bandwidth, not FLOPs.</div></div>
  <div class="diagram-card purple"><div class="card-title">Post-processing</div><div class="card-desc">NMS on a small ARM core can cost more than the network (Chapter 7).</div></div>
</div>
</div>

---

## Edge-friendliness, concretely

| Property | Why it matters on edge | Friendly | Unfriendly |
|---|---|---|---|
| **Static shapes, no data-dependent control flow** | NPU compilers require fixed graphs | dense heads, top-k | NMS in-graph, dynamic query counts |
| **NMS-free output** | NMS runs on the CPU over thousands of boxes | YOLO26 `nms=False`, YOLOv10, DETRs | one-to-many heads at low conf thresholds |
| **No DFL softmax** | Softmax over 16 bins + fixed conv + reshapes: awkward layouts, precision-sensitive in INT8 | YOLO26 (reg_max = 1) | YOLOv8/11 heads |
| **Activation function** | Many NPUs have native ReLU/ReLU6; SiLU/h-swish need LUTs or approximations | ReLU variants (PP-YOLOE+ t-relu, NanoDet), SiLU where supported | exotic activations, GELU on older NPUs |
| **Plain / re-parameterised convs** | 3×3 convs hit peak NPU throughput | RepVGG-style, QARepVGG | many tiny ops |
| **Attention** | Softmax over large matrices, transposes, layernorm: support varies by NPU generation | none, or attention only at stride 32 (RT-DETR's AIFI) | full-resolution attention, area attention with FlashAttention dependence |
| **Deformable sampling / `grid_sample`** | Gather-heavy; poorly supported on many NPUs | — | Deformable DETR decoders on older NPUs |
| **Quantisation robustness** | INT8 is the native precision | QARepVGG, BN-friendly blocks | concat of box coordinates (0–640) with scores (0–1) in one tensor |

This is why the Ultralytics YOLO26 paper frames the model as "edge-first". Removing DFL and making the
NMS-free head first-class both target the bottom four rows, not COCO AP.

---

## Measured on real edge hardware

From the Ultralytics Jetson guide: an **NVIDIA Jetson Orin Nano Super** developer kit, Ultralytics
8.4.32, 640 px, inference time excluding pre- and post-processing. The accuracy column is mAP50-95 on the
128-image `coco128` subset used by `model.benchmark()`, *not* COCO val, so read it only for relative
drops.

| Model | PyTorch (ms) | ONNX (ms) | TensorRT FP16 (ms) | TensorRT INT8 (ms) | coco128 mAP FP16 → INT8 |
|---|:---:|:---:|:---:|:---:|:---:|
| **YOLO26n** | 15.6 | 15.8 | **4.57** | **3.80** | 0.480 → 0.449 |
| **YOLO26s** | 22.8 | 26.3 | 7.17 | 5.25 | 0.565 → 0.547 |
| **YOLO26m** | 44.4 | 53.4 | 13.6 | 9.30 | 0.623 → 0.582 |
| **YOLO26l** | 61.0 | 68.1 | 17.4 | 11.8 | 0.623 → 0.592 |
| **YOLO26x** | 98.4 | 122.4 | 32.6 | 20.0 | 0.664 → 0.617 |

Three lessons generalise:

1. **The runtime matters more than the model size.** TensorRT FP16 is 3–4× faster than PyTorch or ONNX
   Runtime on the same module. Never benchmark an edge deployment in PyTorch.
2. **INT8 is not free.** It gives a 1.2–1.6× speed-up over FP16 here, at a visible accuracy cost with
   default calibration. Chapter 45 covers how to recover most of it.
3. **The ×4 parameter gap between n and s costs only about 1.6× latency.** At small sizes the
   network is not compute-bound, so the fixed overheads (memory traffic, kernel launches) dominate.

From the Raspberry Pi guide (**Raspberry Pi 5**, CPU only, FP32, 640 px): YOLO26n runs at 67 ms per
image with NCNN, 105 ms with OpenVINO, 126 ms with ONNX Runtime and 299 ms in PyTorch. The guide also
reports about +15% FPS for YOLO26n over YOLO11n with ONNX on the Pi 5 (6.79 → 7.79 FPS). On CPUs the
export format is a 4× decision.

---

## The edge leaderboard (T4 as a proxy)

T4 TensorRT FP16 latency is the most widely reported comparable number. It is a **proxy**: it ranks GPU
friendliness, not NPU friendliness. All values are COCO val2017 from
[`detectors.json`](./atlas.md) with per-row sources.

| Model | Params (M) | Input | COCO AP | T4 FP16 (ms) | NMS | Pre-training | License | Choose this when |
|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| **YOLO26n** | 2.4 | 640 | 40.9 (e2e 40.1) | 1.7 | optional | Objects365 | AGPL-3.0 | Broadest export coverage; NPUs; CPUs |
| **YOLO11n** | 2.6 | 640 | 39.5 | 1.5 | yes | COCO | AGPL-3.0 | Existing YOLO11 pipelines |
| **YOLOv12-N** | 2.6 | 640 | 40.6 | 1.64 | yes | COCO | AGPL-3.0 | GPUs with good attention kernels |
| **YOLOv13-N** | 2.5 | 640 | 41.6 | 1.97* | yes | COCO | AGPL-3.0 | GPU only; custom ops |
| **YOLOv10-N** | 2.3 | 640 | 38.5 | 1.84 (e2e) | no | COCO | AGPL-3.0 | NMS-free on a budget |
| **D-FINE-N** | 4 | 640 | 42.8 | 2.12 | no | ImageNet | Apache-2.0 | Permissive license, GPU/TensorRT |
| **DEIMv2-N** | 3.6 | 640 | 43.0 | 2.32 | no | ImageNet | Apache-2.0 | Best AP per parameter at this size |
| **RTMDet-tiny** | 4.8 | 640 | 41.1 | 2.34 | yes | ImageNet | Apache-2.0 | MMDetection ecosystem |
| **DAMO-YOLO-T\*** | 8.5 | 640 | 43.6 | 2.78 | yes | distilled | Apache-2.0 | Permissive CNN with NAS backbone |
| **LW-DETR-tiny** | 12.1 | 640 | 42.6 | 2.0 | no | Objects365 | Apache-2.0 | ViT-based tiny DETR |
| **RF-DETR-Pico** | 8.4 | 560 | 41.6 | 1.7 | no | PE + O365 | PML-1.0 | Fastest DETR at this AP (check license) |
| **RF-DETR-N** | 30.5 | 384 | **48.4** | 2.3 | no | DINOv2 + O365 | Apache-2.0 | Highest AP near 2 ms; GPU targets |
| **YOLO27n** *(preliminary)* | 3.0 | 640 | 42.3 | — | optional | — | AGPL-3.0 | Not released |

\* YOLOv13's README does not state the GPU; the value is not comparable with the T4 rows.

**Reading the table.** At roughly 2 ms on a T4, the DETR family now leads on AP: RF-DETR-N at 48.4 is
more than 7 AP above any CNN of similar latency. It does so with 30 M parameters, a DINOv2 backbone at
384 px, and attention. The YOLO26n row wins on the properties the table cannot show: 2.4 M parameters,
pure convolutions, INT8-friendly, CPU-friendly, NPU-friendly, and exportable to every format in
Chapter 36. Choose by *target*, not by T4 AP.

---

## Edge model families, briefly

- **YOLO n/s sizes (v5u, v8, 11, 26)**: the default. Mature export to TensorRT, OpenVINO, CoreML,
  LiteRT, NCNN, MNN, RKNN, Hailo, IMX500. YOLO26 removes the two worst edge pain points (DFL, NMS).
- **YOLOv10 / v12 / v13 N**: research YOLOs with interesting ideas (dual heads; area attention;
  hypergraphs). Verify that every op runs on your target before choosing them for edge.
- **PicoDet / PP-YOLOE+ (s, t)**: Baidu's edge line. ReLU variants for NPUs, Paddle Lite for ARM. The
  t-relu variant gives up 3.5 AP (39.9 → 36.4) for 38% higher T4 FPS (344.8 → 476.2), a clean
  illustration of the activation trade-off.
- **RTMDet-tiny/s**: CSPNeXt with large-kernel depthwise convs. Strong on GPUs, and also the base of
  RTMDet-Ins and RTMO (pose).
- **D-FINE-N / DEIM-N / DEIMv2-N–Pico**: NMS-free DETRs with HGNetV2 backbones, Apache-2.0, excellent AP
  per parameter. Check the decoder ops on your NPU.
- **RF-DETR Atto–N**: DINOv2/PE backbones, NAS-chosen resolutions. Highest AP per GPU millisecond in 2026.
  Atto/Femto/Pico use Roboflow's PML license.
- **EdgeYOLO**: YOLOX-derived with edge tricks. Reports Jetson AGX Xavier FPS (Appendix C).
- **YOLO-NAS-S**: quantisation-aware blocks; INT8 only about 0.5 AP below FP16. Weights are non-commercial.

---

## A selection procedure

1. **Fix the target and runtime first** (Jetson + TensorRT, Pi + NCNN, Hailo, RK3588 + RKNN, phone +
   CoreML/LiteRT/QNN). Get the vendor's operator support list.
2. **Shortlist 3–4 models** whose ops are all supported. Discard anything that falls back to the CPU.
3. **Export each one and measure on the device**: end-to-end latency at batch 1, including preprocessing,
   NMS or top-k, and copies (Chapter 46).
4. **Fine-tune all shortlisted models on your data** with the same split and evaluator. COCO rank does
   not predict fine-tuned rank (RF100-VL, Chapter 9).
5. **Quantise** (INT8 PTQ, then QAT if needed) and re-measure accuracy *on the device*.
6. Pick the best accuracy that meets the latency budget with margin. Check the license (Chapter 35).

<div class="callout warn"><span class="callout-title">Trap</span>Benchmarks at <code>conf=0.001</code>
(the evaluation setting) leave thousands of boxes for NMS. Benchmarks at <code>conf=0.25</code> leave a
handful. The same model can show 3× different end-to-end latency depending on which one a table used.
Always measure at your deployment threshold.</div>

---

## Key Takeaways

- Edge latency is decided by operator support, memory traffic and post-processing as much as by FLOPs.
- Edge-friendly means static shapes, no NMS, no DFL softmax, supported activations, plain or
  re-parameterised convs, limited attention, and INT8 robustness.
- On a Jetson Orin Nano Super, TensorRT FP16 runs YOLO26n at 4.57 ms vs 15.6 ms in PyTorch. On a
  Raspberry Pi 5, NCNN is about 4× faster than PyTorch. Export format is a first-order decision.
- At about 2 ms on a T4, DINOv2-based RF-DETR-N leads COCO AP (48.4). Compact CNN YOLOs lead on
  portability and size.
- Choose by fixing the target, filtering by operator support, measuring on the device, fine-tuning
  candidates on your own data, then quantising.

## Check Yourself

<details class="check"><summary>Why might YOLO26n be faster than a model with lower T4 latency once deployed on a Hailo or RKNN NPU?</summary>
The T4 runs attention, deformable sampling and arbitrary softmaxes efficiently in FP16. Many NPUs
accelerate only a CNN-centric INT8 operator set. A DETR's attention or a DFL softmax may fall back to
the CPU, adding transfers and slow CPU kernels. YOLO26n is all convolutions with an NMS-free top-k
output, so the whole graph stays on the NPU.</details>

<details class="check"><summary>On the Orin Nano Super, YOLO26n is 4.57 ms and YOLO26s 7.17 ms in FP16, despite ~4× the parameters. Why so close?</summary>
At this size the GPU is not saturated. Fixed costs (kernel launches, memory traffic for activations at
stride 8, layers that cannot fill the GPU) dominate. Latency grows sub-linearly with model size until
the model becomes compute-bound, which on this device happens around the m size.</details>

<details class="check"><summary>What does PP-YOLOE+'s t vs t-relu comparison teach about activation choice?</summary>
Replacing SiLU with ReLU cost 3.5 AP (39.9 → 36.4) but raised T4 TensorRT throughput from 344.8 to
476.2 FPS (+38%). On NPUs where SiLU is emulated, the speed gain is larger. Activation choice is a real
accuracy–latency knob, and the right setting depends on the target.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics NVIDIA Jetson guide | Ultralytics, 2026 | docs/en/guides/nvidia-jetson.md | Orin Nano Super YOLO26 latencies by format, coco128 mAP |
| Ultralytics Raspberry Pi guide | Ultralytics, 2026 | docs/en/guides/raspberry-pi.md | Pi 5 latencies by format; YOLO26n vs YOLO11n FPS |
| Ultralytics YOLO26 docs and paper | Jocher et al., 2026 | arXiv:2606.03748 | Edge-first design: DFL removal, NMS-free |
| RF-DETR README | Roboflow, 2026 | github.com/roboflow/rf-detr | N/Pico numbers, licenses |
| D-FINE; DEIM; DEIMv2 READMEs | 2024–2025 | github.com/Peterande/D-FINE etc. | N-size numbers |
| RTMDet README | OpenMMLab, 2022 | mmdetection/configs/rtmdet | tiny/s numbers |
| PP-YOLOE README | Baidu, 2022 | PaddleDetection/configs/ppyoloe | t vs t-relu trade-off |
| DAMO-YOLO README | Alibaba, 2022 | github.com/tinyvision/DAMO-YOLO | T* numbers |
| LW-DETR README | Chen et al., 2024 | github.com/Atten4Vis/LW-DETR | tiny numbers |
| YOLOv10 / YOLOv12 / YOLOv13 READMEs | 2024–2025 | THU-MIG/yolov10, sunsmarterjie/yolov12, iMoonLab/yolov13 | N-size numbers |
| Ultralytics YOLO27 preview | Ultralytics, 2026 | docs/en/models/yolo27.md | preliminary YOLO27n |

---

**Next:** [Chapter 17 — Large Detectors](./17_large_detectors.md) — the other end: when latency is no
object, how far can accuracy go, and at what cost?
