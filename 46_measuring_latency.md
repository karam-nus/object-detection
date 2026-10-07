---
title: "Chapter 46 — Measuring Latency Honestly"
---

[← Back to Table of Contents](./README.md)

# Chapter 46 — Measuring Latency Honestly

> *"'1.7 ms' is a sentence with the subject, the verb and the conditions deleted."*

## Overview

Latency numbers in papers and model cards answer a narrow question: how long one forward pass takes on
one GPU, at one precision, batch 1, after warm-up, excluding everything around it. Your product's
question is different: how long from photons (or a decoded frame) to an actionable box, sustained, on
your device, at the tail. This chapter takes the pipeline apart, measures each stage on a real exported
YOLO26n and YOLO11n, lists the methodology mistakes that make numbers wrong, separates latency from
throughput, and ends with a reporting template.

<div class="diagram">
<div class="diagram-title">What a model card's latency covers, and what it leaves out</div>
<div class="flow-h">
  <div class="flow-node">capture / decode</div>
  <div class="flow-arrow"></div>
  <div class="flow-node">preprocess<small>letterbox · normalise · copy to device</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node accent">forward pass<small>← the published number</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node">copy back · decode</div>
  <div class="flow-arrow"></div>
  <div class="flow-node">NMS / top-k</div>
  <div class="flow-arrow"></div>
  <div class="flow-node">tracking · logic · render</div>
</div>
</div>

---

## What the published numbers measure

| Source | What is timed | Precision / runtime | Includes NMS? |
|---|---|---|---|
| Ultralytics model tables ("T4 TensorRT10") | Forward pass, batch 1 | FP16, TensorRT 10 | No for one-to-many models; YOLO26 is timed with `nms=False` (no NMS needed) |
| Ultralytics "CPU ONNX" | Forward pass | FP32, ONNX Runtime | No |
| YOLOv10 paper | Forward pass, and separately end to end with NMS for baselines | FP16, TensorRT | Reports both |
| DETR-family papers (RT-DETR, D-FINE, DEIM, RF-DETR) | Forward pass (no NMS needed) | FP16, TensorRT | Not applicable |
| `model.val()` speed line | Pre/inference/post per image, batched, in PyTorch | AMP/FP32 | Yes, at conf 0.001 (Chapter 33) |

None of them includes image decoding, host–device copies under load, or tracking. All are medians or
means over many runs after warm-up, on a machine doing nothing else.

---

## Measured: a CPU pipeline, stage by stage

YOLO26n and YOLO11n exported to ONNX (opset 18), ONNX Runtime with 4 intra-op threads on an Intel Xeon
@ 2.3 GHz cloud VM, two real images, median of 30 runs. Pre- and post-processing are the NumPy/OpenCV
code from Chapter 36. NMS is torchvision's batched NMS.

<!-- MEASURED_PIPELINE -->

---

## Methodology: the mistakes that make numbers wrong

| Mistake | Effect | Do this instead |
|---|---|---|
| **No warm-up** | First runs include engine/graph initialisation, memory allocation, cache misses, clock ramp-up: 10–100× slower | Discard the first 10–50 iterations; report cold start separately if it matters |
| **No device synchronisation** (GPU) | Timing a kernel *launch*, not its execution: impossibly fast numbers | `torch.cuda.synchronize()` before stopping the clock, or CUDA events; `trtexec` does this for you |
| **Mean only** | Hides tail latency that breaks real-time loops | Median, p90, p99, max; plot the distribution |
| **Random-noise input** | Different activation sparsity and very different NMS load from real images | Real images from the deployment distribution |
| **Unfixed clocks / thermal state** | Boost clocks inflate short benchmarks; throttling deflates long ones | Lock clocks or run long enough to reach a steady state; record temperatures; on Jetson set the power mode |
| **Wrong batch** | Batch-32 throughput reported as per-image latency | State batch size; report latency at the deployment batch |
| **Different input sizes** | 640 vs 512 vs rectangular inputs | State the exact input shape |
| **Shared machine** | Other processes steal cores, cache and memory bandwidth | Dedicated machine; pin threads; repeat on another day |
| **Measuring PyTorch, deploying TensorRT** | 3–4× differences (Chapter 16) | Measure the exported artifact in the target runtime |
| **Ignoring post-processing** | Hides NMS cost, which grows with candidates | Time post-processing at the deployment confidence threshold |

A minimal correct GPU timing loop in PyTorch:

```python
import torch, time
x = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.half)
for _ in range(50): model(x)                      # warm-up
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
times = []
for _ in range(500):
    start.record(); model(x); end.record()
    torch.cuda.synchronize(); times.append(start.elapsed_time(end))
print(sorted(times)[len(times) // 2], sorted(times)[int(0.99 * len(times))])   # median, p99 in ms
```

For TensorRT engines, `trtexec --loadEngine=model.engine --warmUp=500 --iterations=1000` reports GPU
compute time and end-to-end host time (with copies) separately.

---

## Latency, throughput and pipelining

| Metric | Question | How to improve it |
|---|---|---|
| **Latency** (ms per frame, single stream) | How stale is each answer? | Smaller model, lower resolution, better runtime, fewer pipeline stages, no batching |
| **Throughput** (frames per second, all streams) | How many cameras per box? | Batching across streams, pipelining stages on different units, concurrency |
| **Tail latency** (p99) | How often does the loop miss its deadline? | Avoid contention, dynamic allocation, garbage collection; fixed shapes |

**Pipelining**: decode (hardware decoder), preprocess (CPU or GPU), inference (accelerator) and
post-process (CPU) can run concurrently on consecutive frames. Throughput is then set by the slowest
stage, while latency is still the sum of the stages. This is why a deployment can reach 30 FPS with
60 ms of latency.

**Little's law**: the average number of frames in flight equals throughput × latency. At 30 FPS and
60 ms latency, about two frames are in the pipeline at any time, and any queue beyond that only adds
delay (Chapter 43, streaming perception).

---

## Energy and memory

For battery and thermal budgets, latency alone is not enough:

- **Energy per inference** (mJ): power during inference × time. Measure with on-board power monitors
  (Jetson `tegrastats`, USB power meters, vendor tools).
- **Peak memory**: model weights + activations + runtime workspace; on MCUs the activation peak decides
  feasibility (Chapter 15).
- **Model load and cold start**: TensorRT engine deserialisation, CoreML compilation on first launch,
  and LiteRT delegate initialisation can take seconds. They matter for apps that start on demand.

---

## A reporting template

Every latency number you publish or hand to a product team should carry these fields:

| Field | Example |
|---|---|
| Model and export | YOLO26n, ONNX opset 18, `nms=False`, static 1×3×640×640 |
| Precision | FP16 (TensorRT 10.x), or INT8 with calibration set X |
| Hardware and settings | Jetson Orin Nano Super, power mode MAXN, clocks locked |
| Runtime and version | TensorRT 10.3, JetPack 6.x |
| What is timed | Forward only / + pre / + post / end to end from decoded frame |
| Input | Real images from the deployment set, letterboxed to 640 |
| Batch and concurrency | Batch 1, single stream |
| Warm-up and iterations | 50 warm-up, 1,000 measured |
| Statistics | Median, p90, p99, max |
| Post-processing conditions | conf 0.25, NMS IoU 0.7, max 300 detections |
| Accuracy of the same artifact | mAP50-95 on the validation set with the same export |

---

## Key Takeaways

- Published latencies are forward-pass, batch-1, warmed-up medians on an idle machine, usually without
  pre/post-processing. Your number must include what your product runs.
- On CPUs, pre/post-processing and NMS are a visible fraction of the total for nano models (measured
  above). NMS cost grows with the number of candidates, so time it at the deployment threshold.
- Warm up, synchronise, use real images, report median and tail, fix the clocks and state batch, input
  shape, precision and runtime.
- Latency and throughput are different targets. Pipelining raises throughput but not latency.
- Use the reporting template, and pair every latency with the accuracy of the same artifact.

## Check Yourself

<details class="check"><summary>A colleague measures 0.3 ms for YOLO26n in PyTorch on a GPU with time.time() around model(x). What is probably wrong?</summary>
GPU execution is asynchronous: without torch.cuda.synchronize() (or CUDA events), the timer stops when the
kernels are launched, not when they finish. Warm-up may also be missing. Synchronise, warm up, and repeat
many times.</details>

<details class="check"><summary>Your pipeline runs at 30 FPS but users complain the boxes lag behind moving objects. How can both be true?</summary>
Throughput is 30 FPS because stages run in parallel, but each frame's latency is the sum of decode,
preprocessing, inference, post-processing and rendering, which may be 80–150 ms. Measure end-to-end
latency per frame, remove queues, and consider a faster model or motion forecasting (Chapter 43).</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics model docs and benchmark mode | Ultralytics | docs.ultralytics.com | What published latencies include |
| YOLOv10 | Wang et al., 2024 | arXiv:2405.14458 | Latency with and without NMS |
| NVIDIA `trtexec` documentation | NVIDIA | docs.nvidia.com/deeplearning/tensorrt | GPU vs host timing |
| PyTorch CUDA semantics (asynchronous execution, events) | PyTorch | pytorch.org/docs | Correct GPU timing |
| Towards Streaming Perception | Li, Wang, Ramanan, 2020 | arXiv:2005.10420 | Latency as part of accuracy |
| Little's law | Little, 1961 | doi:10.1287/opre.9.3.383 | Frames in flight |
| Measurements in this chapter | this book | `postproc_study.py`, `export_study.py` | CPU stage timings |

---

**Next:** [Chapter 47 — Detection in Production](./47_production_systems.md) — the system around the
model: thresholds, calibration, monitoring and the data flywheel.
