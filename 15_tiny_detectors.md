---
title: "Chapter 15 — Very Tiny Detectors"
---

[← Back to Table of Contents](./README.md)

# Chapter 15 — Very Tiny Detectors

> *"On a microcontroller, the question is not how many FLOPs you can afford. It is whether your largest activation fits in SRAM."*

## Overview

At the bottom of the scale, detection runs on microcontrollers with 256 KB–4 MB of SRAM, a few MB of
flash, and between 1 GOPS (a Cortex-M7) and 600 GOPS (an MCU-class NPU). Here the binding constraint is
**peak activation memory**, then flash for weights, then compute. Models change shape accordingly:
centroids instead of boxes, 96×96 grayscale inputs, MobileNet stems at width multiplier 0.05, patch-based
execution, and full INT8. This chapter covers the budget arithmetic, the design strategies, the model
families (FOMO, Yolo-Fastest/FastestDet, TinyissimoYOLO, MCUNet, NanoDet, PicoDet, YOLOX-Nano, DEIMv2-Atto
and friends), the hardware and toolchains, and what accuracy to expect.

<div class="diagram">
<div class="diagram-title">The tiny-detection budget, in the order it bites</div>
<div class="layer-stack">
  <div class="layer red">1. Peak SRAM — the largest (input + output + scratch) of any layer must fit</div>
  <div class="layer orange">2. Flash — INT8 weights ≈ 1 byte per parameter, plus code and the runtime</div>
  <div class="layer yellow">3. Compute — MACs per frame vs ~1 GOPS (Cortex-M7) to 600 GOPS (STM32N6 NPU)</div>
  <div class="layer green">4. Operator support — what the runtime / NPU can execute in integer arithmetic</div>
  <div class="layer blue">5. Post-processing — NMS and decoding on a CPU with no vector units to spare</div>
</div>
</div>

---

## The budget arithmetic

### Peak SRAM

During inference, a layer needs its input and output activations resident at the same time (plus
scratch buffers). The **peak** over all layers must fit in SRAM next to the runtime. With INT8
activations, a tensor of shape $H \times W \times C$ occupies $HWC$ bytes.

**Numerical check.** First layer of a detector: 3×3 conv, stride 2, 16 output channels.

| Input | Input tensor | Output tensor | Peak at this layer |
|---|:---:|:---:|:---:|
| 96 × 96 × 1 (grayscale) | 9.2 KB | 48×48×16 = 36.9 KB | **46 KB** |
| 96 × 96 × 3 | 27.6 KB | 36.9 KB | **64.5 KB** |
| 160 × 160 × 3 | 76.8 KB | 80×80×16 = 102.4 KB | **179 KB** |
| 320 × 320 × 3 | 307 KB | 160×160×16 = 409.6 KB | **717 KB** |

A Cortex-M4 board with 256 KB of SRAM cannot even hold a 320×320 RGB input. **Resolution is the first
knob, and early layers set the peak.** That is why tiny detectors run at 96–224 pixels and why
MCUNetV2's *patch-based inference* executes the memory-heavy early stages one spatial patch at a time.
MCUNetV2 reports a 4–8× peak-memory reduction from patch-based inference alone.

### Flash

INT8 weights take about 1 byte per parameter: a 1 M-parameter model needs about 1 MB of flash plus the
runtime. A 0.25 M-parameter Yolo-FastestV2 fits comfortably. A 2.6 M-parameter YOLO11n (about 2.6 MB
INT8) fits only on larger parts.

### Compute

ST's documentation puts the STM32H7 (Cortex-M7) at about 1 GOPS peak for neural-network workloads and
the STM32N6's Neural-ART NPU at 600 GOPS. On the former, a 0.5 GOPs model takes at least half a
second. On the latter, YOLO-class models become feasible. Compute is real, but on CPU-only MCUs memory
usually rules a model out before compute does.

---

## Design strategies

| Strategy | What it does | Cost | Example |
|---|---|---|---|
| **Predict centroids, not boxes** | Output a low-resolution heat map of object centres; no box regression, no NMS | No sizes; nearby objects merge | FOMO |
| **Low input resolution / grayscale** | Shrinks every activation | Small objects vanish | FOMO 96×96 grey; TinyissimoYOLO 88–112 px |
| **Extreme width multipliers** | MobileNetV2 α = 0.05–0.35 | Capacity | FOMO, MCUNet |
| **Truncated backbones** | Stop at stride 8 (FOMO) | Less context | FOMO |
| **Patch-based execution** | Run early layers per spatial patch | Some recomputation at patch borders | MCUNetV2 |
| **Memory-aware NAS** | Search architectures under a peak-SRAM constraint | Search cost | MCUNet (TinyNAS) |
| **Full INT8, integer-only ops** | Required by TFLite Micro / CMSIS-NN / Vela | Quantisation error, especially in the head | All |
| **Hardware-friendly ops** | ReLU/ReLU6 instead of SiLU, no attention, no DFL softmax | Small accuracy loss on GPUs | ReLU variants of PP-YOLOE+, YOLO "relu" exports |
| **Tiling / adaptive tiling** | Run the detector on crops for small objects | More inferences | TinyissimoYOLO adaptive tiling |
| **Few classes** | Head size and difficulty shrink | Specialised model | Almost every MCU deployment |

<div class="callout field"><span class="callout-title">Field note</span>Most real MCU detection tasks
are <em>narrow</em>: one to five classes, a fixed camera, objects of predictable size (a person in a
doorway, a part on a conveyor, a bird at a feeder). Train a narrow model on that data. COCO AP says
almost nothing about how well a 0.3 M-parameter detector does on "is there a person within 3 m of this
camera?". Such models routinely reach very high F1 on narrow tasks.</div>

---

## The model families

### FOMO (Edge Impulse)

| | |
|---|---|
| **Introduced** | Edge Impulse, 2022 (product documentation and talks; no peer-reviewed paper) |
| **Lineage** | Truncated MobileNetV2 + per-cell classification (heat-map / CenterNet-like) |
| **Paradigm** | dense classification of grid cells at 1/8 input resolution; output = centroids |
| **Assignment** | a cell is positive for a class if an object centre falls in it |
| **Post-processing** | connected components / local maxima on the heat map; no NMS |
| **COCO AP (sizes)** | not applicable (centroids, no boxes) |
| **Latency** | ~30 FPS on Arduino Nicla Vision (Cortex-M7) using 245 KB RAM; smallest variant (96×96 grey, α = 0.05) < 100 KB RAM, ~10 FPS on an 80 MHz Cortex-M4F (vendor figures) |
| **Pre-training** | ImageNet MobileNetV2 |
| **License** | Edge Impulse platform |
| **Known failure modes** | no box sizes; objects closer than one cell merge; poor for large objects that span many cells |

FOMO's insight is that many tiny-device tasks need **counting and locating**, not boxes. Dropping box
regression removes the head, the decoder and NMS. Edge Impulse claims up to 30× less compute and memory
than MobileNet SSD.

### Yolo-Fastest, Yolo-FastestV2, FastestDet

A line of tiny YOLO-style detectors by dog-qiuqiu targeting ARM CPUs with NCNN.

| Model | Params | GFLOPs | Input | COCO | Reported latency |
|---|:---:|:---:|:---:|:---:|---|
| **Yolo-FastestV1.1** | 0.35 M | 0.252 | 320 | AP$_{50}$ 24.4 | 4.23 ms, Kirin 990, 4 cores |
| **Yolo-FastestV2** | 0.25 M | 0.212 | 352 | AP$_{50}$ 24.1 | 3.29 ms, Kirin 990, 4 cores |
| **FastestDet** | 0.24 M | — | 352 | AP$_{50}$ 25.3, AP 13.0 | 23.5 ms RK3568 (A55, 4 cores, NCNN); 28 ms on the RK3568 NPU |

FastestDet is anchor-free with a single detection scale and simplified post-processing. These models are
for application processors (phones, Rockchip boards), not MCUs, but they mark the floor of COCO accuracy:
about 13 AP at a quarter of a million parameters.

### TinyissimoYOLO

| | |
|---|---|
| **Introduced** | Moosmann et al. (ETH Zürich), 2023 (arXiv:2307.05999; smart-glasses follow-up arXiv:2311.01057) |
| **Lineage** | YOLOv1-style, fully quantised, MCU-sized |
| **Paradigm** | single-scale grid with boxes |
| **Assignment** | YOLOv1-style responsible cell |
| **Post-processing** | lightweight NMS on the MCU |
| **COCO AP (sizes)** | evaluated on narrow tasks / VOC subsets; not a COCO model |
| **Latency** | 2.12 ms and 150 µJ per inference on GAP9's accelerator; 112×112, 10 classes in 3.2 ms / 245 µJ (paper) |
| **Pre-training** | none / task-specific |
| **License** | research code |
| **Known failure modes** | small objects at 88–112 px input (addressed by adaptive tiling) |

The authors compare GAP9, MAX78000, STM32 and other MCUs. GAP9's accelerator was about 2× faster and
20% more energy-efficient than the MAX78000. A later smart-glasses prototype ran end to end at 18 FPS
at 1.59 mJ per inference.

### MCUNet / MCUNetV2

MIT's MCUNet co-designs the network (TinyNAS) and the inference engine (TinyEngine) under SRAM and flash
budgets. **MCUNetV2** adds patch-based inference and **receptive-field redistribution** (moving
receptive field to later stages so early patches need less halo). On Pascal VOC detection it reports a
13.2% mAP improvement at 1.9× smaller peak SRAM on a 256 KB Cortex-M4, and 16.9% higher mAP than the
previous state of the art overall.

### NanoDet, PicoDet, YOLOX-Nano

These are mobile-CPU detectors that sit just above MCU class:

| Model | Params | GFLOPs | Input | COCO AP | Latency (as reported) |
|---|:---:|:---:|:---:|:---:|---|
| **NanoDet-m** | 0.95 M | 0.72 | 320 | 20.6 | 10.2 ms, 4× Cortex-A76 (NCNN) |
| **NanoDet-Plus-m** | 1.17 M | 0.9 / 1.52 | 320 / 416 | 27.0 / 30.4 | 12.0 / 19.8 ms, 4× A76 |
| **PicoDet-XS** | 0.70 M | 0.67 | 320 | 23.5 | — |
| **PicoDet-S** | 1.18 M | 1.65 | 416 | 32.5 | — |
| **YOLOX-Nano** | 0.91 M | 1.08 | 416 | 25.8 | — |

NanoDet-Plus is an FCOS-style anchor-free detector with GFL losses (QFL + DFL) and a ShuffleNetV2
backbone. Its "Plus" version added an **Assign Guidance Module** (a temporary, larger head that guides
assignment early in training) and a **Dynamic Soft Label Assigner**, for +7 AP at almost no inference
cost. The INT8 model file is about 1 MB. PicoDet (Baidu) adds ESNet backbones and NAS. Both quantise
well because they avoid exotic ops.

### Modern sub-2 M-parameter detectors

| Model | Params | GFLOPs | Input | COCO AP | T4 TRT FP16 |
|---|:---:|:---:|:---:|:---:|:---:|
| **DEIMv2-Atto** | 0.5 M | 0.8 | 320 | **23.8** | 1.10 ms |
| **DEIMv2-Femto** | 1.0 M | 1.7 | 416 | **31.0** | 1.45 ms |
| **DEIMv2-Pico** | 1.5 M | 5.2 | 640 | **38.5** | 2.13 ms |

DEIMv2's smallest models are NMS-free DETRs with HGNetV2-derived backbones. At 1.5 M parameters, Pico
reaches 38.5 AP, about what a 3.2 M-parameter YOLOv8n reaches (37.3). This is the most striking tiny-model
result of 2025, though its T4 latency says nothing about MCU feasibility: decoder attention and
deformable sampling are not in most MCU operator sets. Integer-only DETR inference on micro-NPUs is an
active research topic in 2026. Treat it as emerging.

RF-DETR's Atto/Femto/Pico (7.4–8.4 M parameters, 30.5–41.6 AP, 1.0–1.7 ms on T4) are tiny in *latency*
but not in *parameters*. They belong to Chapter 16.

---

## Hardware and toolchains

| Target | Compute | Memory | Toolchain | Notes |
|---|---|---|---|---|
| **Cortex-M4 / M7** | ~0.1–1 GOPS | 256 KB – 1 MB SRAM | TFLite Micro (LiteRT Micro) + CMSIS-NN, Edge Impulse EON, STM32Cube.AI / ST Edge AI | FOMO-class models |
| **Cortex-M55 + Ethos-U55/U65** | up to ~0.5 TOPS (U55) | MB-class | Vela compiler (TFLite INT8) | CNN ops; unsupported ops fall back to CPU |
| **Ethos-U85** | 256 GOPS – 4 TOPS | — | Vela | Adds MATMUL, TRANSPOSE, GATHER, RESIZE_BILINEAR: transformer ops on-NPU |
| **STM32N6 (Neural-ART)** | 600 GOPS | 4.2 MB embedded RAM | ST Edge AI | YOLOv8n-class person detection demonstrated by ST |
| **GAP8 / GAP9 (GreenWaves)** | RISC-V cluster + NE16 accelerator | — | GAP flow / NNTool | Best energy in the TinyissimoYOLO study |
| **MAX78000 (ADI)** | CNN accelerator, weights in on-chip memory | ~442 KB weight memory | ai8x tools | µJ-class inference, strict layer constraints |
| **ESP32-S3** | Xtensa with vector extensions | 512 KB SRAM + PSRAM | ESP-DL | Popular for hobby/IoT vision |
| **Sony IMX500** | DSP inside the image sensor | on-sensor | Sony toolchain; Ultralytics `format=imx` export | Detection inside the camera; only metadata leaves the sensor |

The operator set decides the model. Before choosing an architecture, list the ops your toolchain runs in
INT8 on the accelerator: conv, depthwise conv, add, concat, resize (nearest/bilinear), pooling, activation
types, softmax, transpose. Any op off that list runs on the CPU, or the conversion fails.

---

## Benchmarking tiny detectors

**MCUBench** (2024) evaluated over 100 YOLO-based detectors on VOC across seven MCUs (including four
STM32 Nucleo boards), all through one ONNX → TFLite INT8 pipeline. It recorded AP, latency, RAM and
flash. Its main finding mirrors this book's: with modern heads and training recipes, even older
backbones such as YOLOv3's sit on the accuracy–latency Pareto front. **The training recipe and head
matter as much as the backbone, even at this scale.**

Report at least these, measured on the device:

1. task metric on *your* data (F1 at the deployment threshold, or AP$_{50}$),
2. peak SRAM and flash from the toolchain's memory report,
3. latency per frame *including* pre-processing (resize, colour conversion) and post-processing,
4. energy per inference, if battery-powered.

---

## Key Takeaways

- On MCUs, peak activation memory decides feasibility before FLOPs do. A 320×320×3 INT8 input alone
  (307 KB) overflows a 256 KB part.
- Resolution, early-layer width and patch-based execution (MCUNetV2: 4–8× lower peak memory) are the main
  memory knobs.
- Centroid detectors (FOMO) drop boxes and NMS entirely. They are often the right answer for counting and
  locating on Cortex-M.
- Sub-1 M-parameter COCO detectors reach 13–24 AP. DEIMv2-Pico reaches 38.5 AP at 1.5 M parameters,
  but its ops are not MCU-ready.
- Operator support on the target (TFLite Micro, CMSIS-NN, Vela, ST Edge AI) constrains architecture more
  than accuracy tables do.
- Narrow tasks (few classes, fixed camera) let tiny models perform far better than their COCO AP
  suggests. Evaluate on your own data, on the device.

## Check Yourself

<details class="check"><summary>Your MCU has 512 KB of SRAM and the runtime uses 60 KB. Can a 224×224 RGB INT8 model whose first conv outputs 112×112×24 run?</summary>
Input 224·224·3 = 150.5 KB; output 112·112·24 = 301 KB; together 451.6 KB, plus 60 KB of runtime =
511.6 KB. That is just under 512 KB, with almost no margin for scratch buffers, so it will probably
fail. Reduce the first layer to 16 channels (200.7 KB output), the input to 192 px, or use patch-based
execution for the first stage.</details>

<details class="check"><summary>When is FOMO a better choice than a tiny YOLO?</summary>
When the task needs only "how many and roughly where" (counting, presence near a location), objects are
roughly similar in size and not touching, and the device is a CPU-only MCU. FOMO removes box regression,
decoding and NMS, so it fits in about 100–250 KB of RAM at usable frame rates. When you need box sizes
or must separate touching objects, use a box detector.</details>

<details class="check"><summary>A model reaches 38 AP on COCO at 1.5 M parameters. Why might it still be unusable on a Cortex-M55 + Ethos-U55?</summary>
Parameter count is not the constraint. Its operators might be: deformable attention, grid sampling,
softmax over attention maps, top-k for query selection, and large intermediate activations. Ethos-U55
supports CNN-style operators, and anything else falls back to the CPU or fails to convert. Activation
peaks at 640 px input would also exceed MCU SRAM.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| MCUNet | Lin et al., 2020 | arXiv:2007.10319 | TinyNAS + TinyEngine co-design |
| MCUNetV2 | Lin et al., 2021 | arXiv:2110.15352 | Patch-based inference, 4–8× peak memory reduction, VOC results |
| FOMO documentation / announcement | Edge Impulse, 2022 | docs.edgeimpulse.com | Centroid design, RAM and FPS figures |
| TinyissimoYOLO | Moosmann et al., 2023 | arXiv:2307.05999 | GAP9 2.12 ms / 150 µJ; platform comparison |
| TinyissimoYOLO on smart glasses | Moosmann et al., 2023 | arXiv:2311.01057 | 17 ms, 1.59 mJ, 18 FPS end to end |
| MCUBench | Sah et al., 2024 | arXiv:2409.18866 | 100+ YOLO detectors on seven MCUs |
| Yolo-FastestV2; FastestDet | dog-qiuqiu, 2021–2022 | github.com/dog-qiuqiu | Params, AP, ARM latencies |
| NanoDet-Plus | RangiLyu, 2021 | github.com/RangiLyu/nanodet | AGM + DSLA; latencies on A76 |
| PP-PicoDet | Yu et al., 2021 | arXiv:2111.00902 + PaddleDetection README | PicoDet table |
| YOLOX | Ge et al., 2021 | arXiv:2107.08430 | YOLOX-Nano |
| DEIMv2 | Huang et al., 2025 | arXiv:2509.20787 + README | Atto/Femto/Pico numbers |
| STM32Cube.AI model performances; STM32N6 documentation | STMicroelectronics, 2025 | wiki.st.com | 600 GOPS NPU, 4.2 MB RAM, ~1 GOPS on H7 |
| Arm Ethos-U85 announcement | Arm, 2024 | newsroom.arm.com | Transformer operators on micro-NPU |

---

**Next:** [Chapter 16 — Edge Detectors](./16_edge_detectors.md) — one step up: phones, Jetsons and
NPUs with gigabytes of memory, where real-time YOLOs and DETRs compete.
