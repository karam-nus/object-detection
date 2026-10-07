---
title: "Chapter 36 — YOLO Axis 10: Export & Deployment"
---

[← Back to Table of Contents](./README.md)

# Chapter 36 — YOLO Axis 10: Export & Deployment

> *"The model is the easy part of the deployed system. The bugs live in the 40 lines of code around it."*

## Overview

A trained `.pt` file is not a product. Deployment means choosing a runtime for the target hardware,
exporting the network into its format, choosing which head and post-processing to embed, choosing a
precision, and re-implementing the pre- and post-processing that `model.predict()` did for you. This
chapter walks through what `model.export()` changes, what comes out of each head option (measured on
real YOLO26n and YOLO11n exports), how to write correct pre/post-processing, how much it costs, which
runtime to pick for which hardware, and how to verify that the exported model still performs.

<div class="diagram">
<div class="diagram-title">From checkpoint to deployed detector</div>
<div class="flow-h">
  <div class="flow-node">best.pt</div>
  <div class="flow-arrow"></div>
  <div class="flow-node blue">fuse BN · drop unused head · export mode</div>
  <div class="flow-arrow"></div>
  <div class="flow-node purple">ONNX (opset 18, slimmed)</div>
  <div class="flow-arrow"></div>
  <div class="flow-node accent">runtime compile<small>TensorRT · OpenVINO · CoreML · LiteRT · NCNN · RKNN · QNN · Hailo …</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node green">host code<small>letterbox · decode · NMS? · rescale</small></div>
</div>
</div>

---

## What `export()` does to the network

1. **Fuse.** BatchNorm is folded into the preceding conv. For dual-head models (YOLOv10, YOLO26) the
   head branch that will not be used is deleted. Which branch is kept depends on `nms`.
2. **Export mode.** `Detect.export = True`: the head returns only the decoded tensor, not the training
   dictionary.
3. **Decode in the graph.** Grid points and strides are baked in for the export shape (static by
   default), and box distances are converted to boxes in input-pixel coordinates. Class scores leave
   the graph **after the sigmoid**.
4. **Optional post-processing in the graph**: top-k for the one-to-one head (`nms=False`), or an
   ONNX `NonMaxSuppression` block (`nms=True`).
5. **Metadata.** Class names, stride, `imgsz`, task, batch, `end2end` and the export arguments are
   written into the artifact, so `YOLO("model.onnx")` can predict without the YAML. The metadata also
   records the licence string.

### The three head options

| `nms=` | Head used | Output | Box format | You must do |
|---|---|---|---|---|
| `None` (default) | one-to-many | `(N, 4 + nc, 8400)` at 640 | cx, cy, w, h (pixels of the letterboxed input) | conf filter, class max, NMS, rescale |
| `False` (YOLO26, YOLOv10) | one-to-one | `(N, 300, 6)` | x1, y1, x2, y2, score, class | conf filter, rescale |
| `True` | one-to-many + embedded NMS | `(N, 300, 6)` | x1, y1, x2, y2, score, class | conf filter, rescale |

`(N, 300, 6)` does not tell you which head produced it. The `end2end` metadata does. YOLOv8 and YOLO11
have no one-to-one head, so `nms=False` is meaningless for them, and `nms=True` is the only way to
get boxes straight out of the graph.

### Measured: what the graphs contain

Exported with Ultralytics 8.4.174, ONNX opset 18, `imgsz=640`, static batch 1, from the official
checkpoints:

<!-- EXPORT_TABLE -->

Three practical consequences:

- The **one-to-one graph contains `TopK` and `GatherElements`**. These are cheap, but some NPU
  compilers cannot place them on the accelerator. That is why the Ultralytics exporter falls back to the
  one-to-many path for NCNN, RKNN, PaddlePaddle, ExecuTorch, IMX, Edge TPU and Qualcomm QNN.
- The **embedded-NMS graph contains ONNX `NonMaxSuppression`**, which many runtimes implement on the CPU
  or not at all. TensorRT, OpenVINO and ONNX Runtime support it. Most NPUs do not.
- **YOLO11's raw graph contains the DFL softmax.** YOLO26's does not (`reg_max = 1`); its softmaxes
  are in the attention blocks. On integer-only accelerators that difference matters (Chapter 45).

---

## Writing the pre- and post-processing yourself

Most deployment bugs are here. This is the full host code for both YOLO26 output types, as used for the
measurements below:

```python
import numpy as np, cv2

def letterbox(im, new=640, color=114):
    h, w = im.shape[:2]; r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (new - nh) // 2, (new - nw) // 2
    out = np.full((new, new, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)

def preprocess(bgr):                                      # cv2.imread gives BGR
    lb, r, pad = letterbox(bgr)
    x = lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0   # RGB, CHW, [0, 1]
    return np.ascontiguousarray(x), r, pad

def post_e2e(y, r, pad, conf=0.25):                      # y: (1, 300, 6)
    d = y[0][y[0][:, 4] > conf]
    boxes = d[:, :4].copy()
    boxes[:, [0, 2]] -= pad[0]; boxes[:, [1, 3]] -= pad[1]; boxes /= r
    return boxes, d[:, 4], d[:, 5].astype(int)

def post_raw(y, r, pad, conf=0.25, iou=0.7):             # y: (1, 4 + nc, 8400)
    p = y[0].T
    scores, cls = p[:, 4:].max(1), p[:, 4:].argmax(1)     # scores are already sigmoided
    keep = scores > conf
    b, s, c = p[keep, :4], scores[keep], cls[keep]
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
    k = batched_nms(xyxy, s, c, iou)[:300]                # per-class NMS (torchvision, cv2.dnn, or your own)
    xyxy = xyxy[k]; xyxy[:, [0, 2]] -= pad[0]; xyxy[:, [1, 3]] -= pad[1]; xyxy /= r
    return xyxy, s[k], c[k]
```

The checklist that catches nearly every pre/post bug:

| Step | Must match training | Typical bug |
|---|---|---|
| Colour order | RGB | Feeding BGR from OpenCV: mAP drops a few points, not to zero, so it goes unnoticed |
| Resize | Letterbox, aspect preserved, pad 114, centred | Plain resize to 640×640 (stretched objects) |
| Scale | `/255`, no mean/std | ImageNet normalisation copied from a classifier |
| Layout | NCHW float32 (or the precision the engine expects) | NHWC passed to an NCHW model |
| Box format | raw: cx, cy, w, h; e2e: x1, y1, x2, y2 | Treating raw output as xyxy |
| Undo the letterbox | subtract pad, divide by gain | Dividing before subtracting; forgetting the pad entirely |
| Scores | Already sigmoided in the graph | Applying sigmoid twice (scores squashed into 0.5–0.73) |
| EXIF orientation | Rotate phone images before inference | Boxes rotated 90° on portrait photos |

<div class="callout field"><span class="callout-title">Field note: the golden-image test</span>Keep
three images in the repository with the boxes that <code>model.predict()</code> produces for them
(coordinates, classes, scores). Every deployment build runs the exported model through the production
pre/post-processing on those images and fails if any box moves by more than a pixel or two, or any score
by more than about 0.01. It catches colour-order, letterbox and precision bugs before users do.</div>

---

## Measured: where the milliseconds go on a CPU

YOLO26n and YOLO11n ONNX models, ONNX Runtime on 4 cores of an Intel Xeon @ 2.3 GHz (a cloud VM, not a
tuned benchmark machine), median of 30 runs, on two real images:

<!-- POSTPROC_TABLE -->

---

## Choosing a runtime

| Target | Usual format (`format=`) | Notes |
|---|---|---|
| NVIDIA GPU, Jetson | TensorRT (`engine`) | Engine is tied to the GPU model and TensorRT version: build on the target. TensorRT 11 is strongly typed, so the exporter bakes FP16/INT8 into the ONNX with NVIDIA ModelOpt before building |
| Intel CPU, iGPU, NPU | OpenVINO (`openvino`) | INT8 via NNCF calibration with `quantize=8` and `data=` |
| Any CPU, portable | ONNX Runtime (`onnx`) | Easiest to debug; good default for servers without accelerators |
| Apple devices | CoreML (`coreml`), Core AI (`coreai`) | CoreML can embed NMS for Xcode previews (`nms=True`) |
| Android, mobile CPU/GPU | LiteRT (`litert`), NCNN (`ncnn`), ExecuTorch (`executorch`) | INT8 LiteRT falls back to the one-to-many head |
| Qualcomm Snapdragon NPU | QNN (`qnn`) | One-to-many path; NMS on the CPU |
| Rockchip NPU | RKNN (`rknn`) | One-to-many path; quantisation with calibration data |
| Hailo | `hailo` | YOLO26 exports raw tensors with host NMS by default; `nms=False` selects the one-to-one path |
| Sony IMX500 (camera with on-sensor AI) | `imx` | Embedded NMS required |
| Edge TPU | `edgetpu` | Full INT8; one-to-many path |
| Axelera, DEEPX, Huawei Ascend, AMD Xilinx | `axelera`, `deepx`, `ascend`, `xilinx` | Vendor toolchains; check per-model support |

**Precision** is set with `quantize=` (`16`/`fp16`, `8`/`int8`, or mixed schemes such as `w8a16`); it
replaces the older `half=` and `int8=` arguments. INT8 needs calibration images (`data=`, optionally
`fraction=`; the TensorRT path uses up to about 500 images). The exporter keeps the head's final 1×1
convs (and DFL, where present) in float, because they cost much of the INT8 accuracy for little of the
runtime. Chapter 45 covers INT8 and the QAT option (`train(..., quantize=8)`).

### Shapes and batch

- **Static shapes** (default) give the best performance and the widest runtime support. `dynamic=True`
  allows variable input sizes and batch at some speed and compatibility cost.
- **Rectangular inputs** suit video. `imgsz=[384, 640]` for 16:9 frames processes 40% fewer pixels than
  640×640, with the same stride-32 constraint. The model was trained on squares, but YOLOs are
  translation-equivariant, so rectangular inference works well in practice. Validate it on your data.
- **Batch** helps throughput on GPUs (several streams at once). For a single camera it adds latency.

---

## Verifying the exported model

1. **Accuracy.** `yolo val model=best.onnx data=widgets.yaml` scores the exported artifact with the same
   validator. Compare with `best.pt`. FP16 should be within a few tenths of mAP; INT8 differences
   depend on the model and calibration (Chapter 45). Remember that `val` uses rect batches for `.pt` but
   the export may be a fixed square (Chapter 28).
2. **Box-level agreement.** The golden-image test above.
3. **Speed.** `model.benchmark(data=..., imgsz=...)` exports to every format available on the machine
   and reports mAP and latency for each. For real numbers, measure on the target with the production
   pipeline (Chapter 46).
4. **Head choice.** For YOLO26, validate both heads: `val(nms=None)` and `val(nms=False)`. The published
   gap is 0.6–0.8 COCO AP. On your data it may be larger or smaller.

---

## Deployment patterns

| Pattern | When | Key detail |
|---|---|---|
| Single-camera edge loop | Robots, smart cameras | Pre/post on the same device; keep everything at batch 1; decode video on the hardware decoder |
| Multi-stream GPU server | Video analytics | Batch frames across streams; GPU decode (NVDEC) and GPU letterbox; DeepStream or Triton |
| Request–response API | Web and mobile backends | Dynamic batching (Triton, TorchServe); watch tail latency, not mean |
| On-device mobile | Phone apps | CoreML / LiteRT / QNN; model size is a download cost; thermal throttling after minutes |
| Cascade | Rare events on cheap hardware | Tiny detector everywhere; big model or VLM only on candidate crops (Chapter 47) |

---

## Key Takeaways

- `export()` folds BatchNorm, removes the unused head, decodes boxes in the graph, applies the sigmoid,
  and embeds metadata. `nms` selects the head: `None` raw one-to-many, `False` one-to-one top-300,
  `True` embedded NMS.
- Raw outputs are `(N, 4 + nc, 8400)` cx-cy-w-h. End-to-end outputs are `(N, 300, 6)` x1-y1-x2-y2.
  Write the post-processing for the format you actually exported.
- Pre/post-processing must match training exactly: RGB, letterbox with 114 padding, /255, NCHW.
  Undo the letterbox when mapping boxes back. Keep a golden-image test in CI.
- Pick the runtime by hardware. NPU toolchains often cannot run TopK or NMS, so they fall back to the
  one-to-many head with host-side NMS.
- Validate the exported artifact with the same validator, both heads for YOLO26, and measure speed on
  the target with the full pipeline.

## Check Yourself

<details class="check"><summary>Your ONNX model outputs (1, 84, 8400). You take the first four rows as x1, y1, x2, y2. What goes wrong, and how do you fix it?</summary>
The raw one-to-many output stores centre x, centre y, width and height. Treated as corners, boxes come
out shifted and wrongly sized. Convert with x1 = cx − w/2 and so on, then apply the confidence filter,
NMS, and the letterbox inverse.</details>

<details class="check"><summary>All your exported model's scores sit between 0.5 and 0.73. What happened?</summary>
A sigmoid was applied to scores that the exported graph had already passed through a sigmoid. σ(x) for
x in [0, 1] lies between 0.5 and 0.73. Remove the second sigmoid.</details>

<details class="check"><summary>You export YOLO26n with nms=False for a Rockchip NPU, but the deployed model still needs NMS. Why?</summary>
The RKNN toolchain cannot support the end-to-end output path (it needs TopK and gather operations), so
the exporter falls back to the one-to-many head and warns. The output is the raw (1, 84, 8400) tensor,
which needs host-side NMS.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics `engine/exporter.py`, `utils/export/engine.py` | Ultralytics | github.com/ultralytics/ultralytics | Export steps, metadata, TensorRT 11 / ModelOpt path, INT8 exclusions |
| Ultralytics `nn/modules/head.py` (`Detect.forward`, `postprocess`, `fuse`) | Ultralytics | github.com/ultralytics/ultralytics | Output formats, top-k head, branch removal |
| Ultralytics end-to-end detection guide; export table | Ultralytics, 2026 | docs/en/guides/end2end-detection.md; docs/macros/export-table.md | `nms` semantics, per-format fallbacks |
| Ultralytics `cfg/default.yaml` | Ultralytics | github.com/ultralytics/ultralytics | `quantize`, `dynamic`, `simplify`, `opset` |
| ONNX operator specs (`NonMaxSuppression`, `TopK`) | ONNX | onnx.ai | Graph operators |
| ONNX Runtime | Microsoft | onnxruntime.ai | CPU measurements |
| Measurements in this chapter | this book | `export_study.py`, `postproc_study.py` | Graph contents, CPU timings, head agreement |

---

**Next:** [Chapter 37 — Novel Optimisations](./37_yolo_novel_optimizations.md) — the ideas from 2024–2026
that changed how YOLOs are built and trained, and what is known about each.
