---
title: "Chapter 44 — Hardware & Runtimes"
---

[← Back to Table of Contents](./README.md)

# Chapter 44 — Hardware & Runtimes

> *"FLOPs predict latency only on the hardware where nothing else is the bottleneck, which is rarely the hardware you deploy on."*

## Overview

Part VI is about running detectors on real chips. This chapter gives the mental model: what limits a
detector's speed on each class of hardware (compute, memory bandwidth, fixed overheads or unsupported
operators), which runtime goes with which chip, and the operator-support matrix for the specific
operations detectors use. Chapter 15 covered microcontrollers and Chapter 16 edge measurements in depth.
This chapter connects them and adds the roofline view, measured on the YOLO26 family.

<div class="diagram">
<div class="diagram-title">Four ways a detector can be slow</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card red"><div class="card-title">Compute-bound</div><div class="card-desc">Large models on GPUs/NPUs: FLOPs set latency</div></div>
  <div class="diagram-card orange"><div class="card-title">Memory-bound</div><div class="card-desc">Activations stream through DRAM faster than math can use them: high-resolution, thin layers</div></div>
  <div class="diagram-card yellow"><div class="card-title">Overhead-bound</div><div class="card-desc">Hundreds of tiny kernels, launches, syncs: nano models on GPUs</div></div>
  <div class="diagram-card purple"><div class="card-title">Fallback-bound</div><div class="card-desc">One unsupported op runs on the CPU, with copies both ways: NPUs</div></div>
</div>
</div>

---

## The roofline view

A layer's time is bounded below by `max(FLOPs / peak_FLOPs, bytes_moved / bandwidth)`. The ratio
FLOPs / byte is the layer's **arithmetic intensity**. Where it crosses the hardware's ratio of peak
compute to bandwidth (the **ridge point**) decides which bound applies.

Measured on the fused YOLO26 models at 640 (conv layers; activation traffic counted as each conv reading
its input and writing its output in FP16, with no fusion, so this is an upper bound on traffic):

| Model | Conv GFLOPs | Activation traffic (MB) | Weights FP16 (MB) | Intensity (FLOP/byte) |
|---|---|---|---|---|
| YOLO26n | 5.4 | 81.9 | 4.8 | **62** |
| YOLO26s | 20.7 | 156.8 | 19.0 | 118 |
| YOLO26m | 68.2 | 320.5 | 40.8 | 189 |
| YOLO26x | 193.9 | 621.0 | 111.4 | **265** |

For an NVIDIA T4 (about 65 TFLOPS FP16 with tensor cores and 320 GB/s), the ridge point is about 200
FLOP/byte. By this estimate **n and s are memory-bound and x is compute-bound** on a T4. Runtimes fuse
layers and keep some activations in cache, so real traffic is lower, but the ordering holds.

The bounds also show what neither explains. YOLO26n's naive memory bound on a T4 is about 0.3 ms and its
compute bound below 0.1 ms, yet it measures 1.7 ms. The rest is **overhead**: kernel launches, small
convolutions that cannot fill the GPU, synchronisation and the head's small tensors. That is why going
from n to s (3.8× FLOPs) only costs 1.5× latency on a T4 (1.7 → 2.5 ms), and why the Jetson
measurements in Chapter 16 show a similar pattern.

**Consequences for design:**

- On GPUs, nano models gain little from fewer FLOPs. Fewer, larger, fused layers and fewer
  post-processing kernels help more.
- On CPUs, FLOPs track latency more closely (Chapter 34's CPU table), because CPUs have far less compute
  per byte of bandwidth.
- On NPUs, the decisive factors are operator support and quantisation, then memory (SRAM size versus
  activation size).

---

## Hardware classes

| Class | Examples | Precision sweet spot | Detector-relevant strengths | Typical traps |
|---|---|---|---|---|
| **Datacentre GPU** | T4, L4, A10, A100, H100, RTX PRO | FP16/BF16, INT8, FP8 | Batching across streams; any operator | Batch-1 latency underuses the GPU |
| **Embedded GPU** | Jetson Orin Nano/NX/AGX | FP16, INT8 (+ DLA) | TensorRT; DLA offload for convs | DLA supports a subset of layers; fallback to GPU |
| **x86 CPU** | Xeon, Core, EPYC | FP32, INT8 (VNNI/AMX), BF16 (AMX) | OpenVINO/ONNX Runtime; simple deployment | Large models are slow; NMS cost visible |
| **ARM CPU** | Raspberry Pi 5, phones, Graviton | FP32/FP16, INT8 | NCNN, XNNPACK (LiteRT), ONNX Runtime | Thermal throttling; few cores |
| **Integrated NPU (PC)** | Intel NPU, AMD XDNA, Qualcomm X Elite | INT8, FP16 | Low power | Operator coverage; driver versions |
| **Mobile NPU / DSP** | Apple Neural Engine, Qualcomm Hexagon, Google Tensor TPU, MediaTek APU | INT8, FP16 | Very high efficiency | Each vendor's compiler decides placement; silent CPU fallback |
| **Edge accelerators** | Hailo-8/8L/10, Rockchip RK3588 NPU, Coral Edge TPU, Axelera Metis, DEEPX | INT8 (some INT4) | High TOPS per watt | Static shapes; limited ops; host-side NMS |
| **Microcontrollers (+ micro-NPU)** | Cortex-M7, M55 + Ethos-U, STM32N6, MAX78000 | INT8 | Milliwatt power | SRAM for activations (Chapter 15) |
| **In-sensor** | Sony IMX500 | INT8 | Only metadata leaves the camera | Strict model size and op set |

---

## Runtimes

| Runtime | Targets | Notes for detectors |
|---|---|---|
| **TensorRT** | NVIDIA GPUs, Jetson (GPU + DLA) | Fastest on NVIDIA; engine tied to GPU and TensorRT version; EfficientNMS plugins; TensorRT 11 is strongly typed (precision baked into the ONNX) |
| **ONNX Runtime** | CPU, CUDA, TensorRT, OpenVINO, DirectML, CoreML, QNN execution providers | Portable; the easiest place to debug an export |
| **OpenVINO** | Intel CPU, iGPU, NPU | INT8 via NNCF; good CPU performance |
| **CoreML / Core AI** | Apple CPU, GPU, Neural Engine | Placement on the Neural Engine depends on ops and shapes |
| **LiteRT (TFLite)** + delegates | Android CPU (XNNPACK), GPU, NNAPI/vendor NPUs, Edge TPU, MCUs (LiteRT Micro) | INT8 full-integer for NPUs |
| **NCNN** | ARM/x86 CPU, Vulkan GPU | Very efficient on ARM CPUs (the fastest Raspberry Pi 5 option in Chapter 16) |
| **ExecuTorch** | Mobile and embedded, PyTorch-native | Delegates to XNNPACK, Core ML, QNN, Vulkan |
| **Vendor SDKs** | RKNN-Toolkit, Hailo Dataflow Compiler, QNN, Vela (Ethos-U), Axelera Voyager, DEEPX DX-COM, Ascend CANN | Each with its own op list and quantiser |
| **Serving layers** | Triton Inference Server, DeepStream, TorchServe | Batching, multi-stream video, pipelines around the runtime |

---

## The operator-support matrix for detectors

The operations below are the ones that most often break detector deployments. Support changes with each
SDK release; treat this as the list of things to check, with the typical situation as of 2026.

| Operation | Where detectors use it | GPU (TensorRT) | CPU runtimes | Edge NPUs (typical) | MCU / micro-NPU |
|---|---|---|---|---|---|
| Conv, depthwise conv, add, concat | Everywhere | ✅ | ✅ | ✅ | ✅ |
| SiLU / Hardswish | Activations | ✅ | ✅ | ✅ on most (LUT); some approximate | LUT or replace with ReLU |
| Nearest upsample | FPN/PAN | ✅ | ✅ | ✅ | ✅ (resize) |
| MaxPool 5×5 (SPPF) | Context block | ✅ | ✅ | ✅ (some limit kernel size) | Check kernel-size limits |
| Softmax | DFL head, attention | ✅ | ✅ | ✅ usually, precision-sensitive in INT8 | Often CPU |
| MatMul / attention, LayerNorm | C2PSA, area attention, DETR encoders/decoders | ✅ | ✅ | Varies by generation | Ethos-U85 adds MatMul/Transpose; older: CPU |
| Transpose / reshape-heavy layouts | Head flattening, attention | ✅ | ✅ | Costly or unsupported in some layouts | Costly |
| TopK, Gather/GatherElements | One-to-one head output, DETR query selection | ✅ | ✅ | Often unsupported → fallback | CPU |
| NonMaxSuppression | Embedded NMS | ✅ (plugins / ONNX op) | ✅ (ONNX Runtime) | Rarely on-NPU → host NMS | CPU |
| GridSample / deformable attention | Deformable DETR family | ✅ (plugins) | ✅ | Usually unsupported | Unsupported |
| Dynamic shapes | Variable input sizes, query counts | ✅ (profiles) | ✅ | Usually static only | Static only |

This is why the Ultralytics exporter falls back to the one-to-many head for several NPU formats
(Chapter 36), and why the DETR family, despite leading GPU benchmarks, is less common on NPUs.

---

## What breaks on which chip: field patterns

| Symptom | Usual cause | Fix |
|---|---|---|
| Exported model is slower on the NPU than on the CPU | One unsupported op (often in the head) forces CPU fallback with tensor copies | Inspect the compiler's placement report; move the op to host code; cut the graph before the head |
| INT8 model detects almost nothing | Box coordinates (0–640) and scores (0–1) concatenated in one quantised tensor share a scale | Keep output branches separate; or leave the head output in higher precision (Chapter 45) |
| Good on desktop, poor on phone | Different preprocessing (colour order, resize) in the mobile pipeline | Golden-image test on device (Chapter 36) |
| Latency spikes after minutes | Thermal throttling (phones, Pi, fanless boxes) | Measure sustained latency (Chapter 46); smaller model or duty cycling |
| TensorRT engine fails on another machine | Engine built for a different GPU or TensorRT version | Build on the target or in an identical container |
| DLA slower than expected | Unsupported layers fall back to the GPU with reformatting | Check per-layer placement; restrict DLA to the backbone |

---

## Choosing hardware for a detection product

1. **Start from the model you need**, not from TOPS numbers: model scale and input size from accuracy
   requirements (Chapters 34, 41), then the precision you can accept (Chapter 45).
2. **Check operator support** for that exact model and export path on the candidate chip.
3. **Measure sustained, end-to-end latency** on a development kit with the real pipeline (Chapter 46).
4. **Budget the host CPU** for decoding video, preprocessing and NMS: on many edge boxes the CPU, not
   the accelerator, sets the frame rate.
5. **Plan for the software lifecycle**: SDK updates, model updates, and whether the vendor toolchain
   will support next year's architectures.

---

## Key Takeaways

- Detector latency is set by compute, memory traffic, fixed overheads or operator fallback, depending on
  model size and chip. FLOPs alone predict it only for large models on accelerators.
- Measured intensity: YOLO26n about 62 FLOP/byte, YOLO26x about 265. Against a T4's ridge of about 200,
  nano and small models are memory- and overhead-bound, which is why n → s costs 1.5× latency for 3.8×
  FLOPs.
- Pick the runtime by chip: TensorRT (NVIDIA), OpenVINO (Intel), CoreML (Apple), LiteRT/NCNN/ExecuTorch
  (mobile and ARM), vendor SDKs for NPUs.
- The detector-specific operators to check are softmax (DFL), attention, TopK/Gather, NMS, grid-sample
  and dynamic shapes. One unsupported op can make an NPU slower than a CPU.
- Choose hardware after choosing the model and checking its operator support, and measure sustained,
  end-to-end latency on a real kit.

## Check Yourself

<details class="check"><summary>Halving the channels of YOLO26n's deep layers cuts its FLOPs by 20% but T4 latency barely changes. Why?</summary>
On a T4 the nano model is dominated by memory traffic and fixed overheads (kernel launches, small tensors),
not by compute in the deep layers. Those layers hold most of the parameters but little of the activation
traffic (Chapter 29). Reducing the number of layers or kernels, or the resolution, would move latency
more.</details>

<details class="check"><summary>Your Hailo deployment of a YOLO26 model needs NMS on the host, even though the model supports NMS-free output. Is the model wrong?</summary>
No. The exporter's default for Hailo produces raw tensors with host NMS; the one-to-one path is selected
with nms=False where supported. Many NPU flows keep TopK and NMS off the accelerator because those
operators are unsupported or slow there.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Roofline model | Williams, Waterman, Patterson, 2009 | doi:10.1145/1498765.1498785 | Arithmetic intensity and ridge point |
| NVIDIA T4 datasheet | NVIDIA | nvidia.com | Peak FP16 and bandwidth |
| TensorRT, ONNX Runtime, OpenVINO, CoreML, LiteRT, NCNN, ExecuTorch documentation | NVIDIA, Microsoft, Intel, Apple, Google, Tencent, Meta | vendor sites | Runtime capabilities |
| Arm Ethos-U85 product brief | Arm, 2024 | arm.com | Transformer operators on a micro-NPU |
| Ultralytics export table and end-to-end guide | Ultralytics, 2026 | docs/macros/export-table.md; docs/en/guides/end2end-detection.md | Per-format head fallbacks |
| Ultralytics Jetson and Raspberry Pi guides | Ultralytics | docs.ultralytics.com/guides | Edge measurements (Chapter 16) |
| Measurements in this chapter | this book | `intensity.py` | Arithmetic intensity of YOLO26 n/s/m/x |

---

**Next:** [Chapter 45 — Quantising & Compressing Detectors](./45_quantization_and_compression.md) — INT8
and below, and why detectors make it harder than classifiers.
