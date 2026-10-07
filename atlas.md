---
title: "Detection Atlas"
---

[← Back to Table of Contents](./README.md)

# Detection Atlas

> *"One table, every detector, every number with its source."*

The Atlas plots and tabulates every detector discussed in this book from a single data file,
[`assets/data/detectors.json`](https://github.com/karam-nus/object-detection/blob/main/assets/data/detectors.json).
Filter by tier (MCU, tiny, edge, real-time, large, open-vocabulary, MLLM, classic), by licence and by
NMS requirement; search by name, family or organisation; drag the latency slider; click a column to sort.

<div class="lab" data-lab="atlas"></div>

---

## How to read it

- **Only rows measured on an NVIDIA T4 with TensorRT FP16 are plotted together.** Rows measured on
  other hardware (V100, Jetson, RTX PRO 6000, CPUs, "paper GPU") appear in the table but never on the
  same axes. Mixing hardware is the most common error in published comparisons.
- **AP** is COCO val2017 AP50:95 unless the row's note says otherwise (LVIS, test-dev, minival).
- **Latency** is as reported by each source, not re-measured. Whether NMS is included differs by source:
  rows marked "needs NMS" usually exclude it (Chapter 34).
- **YOLO26 rows** carry the one-to-many AP (40.9 … 57.5) with latency measured on the one-to-one head;
  Chapter 34's plot uses the matched one-to-one AP.
- **Pre-training** and **licence** columns matter as much as AP (Chapters 34–35).

## The data file

Each row has: `name, family, tier, org, year, params_m, gflops, input, coco_ap, latency_ms, latency_hw,
pretrain, license, nms, source, note`. `source` names the paper, README or documentation page the
numbers were copied from. The same file generates
[Appendix C — Model Index](./appendix_c_model_index.md) (`tools/build_model_index.py`) and the
accuracy–latency figure in Chapter 34 (`tools/plot_pareto.py`).

To add a model: append a row with its source, keep `latency_hw` exactly `"T4 TRT FP16"` only if the source
measured it that way, and regenerate the appendix and figure.

If the interactive view does not load (for example on the raw GitHub view), use
[Appendix C](./appendix_c_model_index.md), which contains the same rows as a static table.
