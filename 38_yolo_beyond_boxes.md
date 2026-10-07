---
title: "Chapter 38 — YOLO Beyond Boxes"
---

[← Back to Table of Contents](./README.md)

# Chapter 38 — YOLO Beyond Boxes

> *"Every YOLO task head is the detection head plus one more branch. Learn the detector and the rest are extensions."*

## Overview

A modern YOLO checkpoint family ships detection, instance segmentation, pose, oriented boxes,
classification and, with YOLO26, semantic segmentation and monocular depth. The first four are
detection with extra outputs per box. They share the backbone, neck, assignment and box loss, and add
one branch to the head and one term to the loss. This chapter shows exactly what each detection-derived
head adds, how tracking turns per-frame boxes into identities, and how open-vocabulary YOLOs replace the
fixed class layer. It stays on the detection side: segmentation and pose are covered as box extensions,
not as fields of their own.

<div class="diagram">
<div class="diagram-title">One detector, several heads</div>
<div class="flow">
  <div class="flow-node blue wide">Shared backbone + PAN neck (P3, P4, P5)</div>
  <div class="flow-arrow"></div>
  <div class="diagram-grid cols-4">
    <div class="diagram-card accent"><div class="card-title">Detect</div><div class="card-desc">box (4) + class (nc)</div></div>
    <div class="diagram-card green"><div class="card-title">Segment</div><div class="card-desc">+ 32 mask coefficients per box, + prototype masks at stride 4</div></div>
    <div class="diagram-card purple"><div class="card-title">Pose</div><div class="card-desc">+ K keypoints × (x, y, visibility) per box</div></div>
    <div class="diagram-card pink"><div class="card-title">OBB</div><div class="card-desc">+ 1 angle per box; rotated IoU</div></div>
  </div>
</div>
</div>

---

## Instance segmentation: boxes plus mask coefficients

YOLO segmentation follows **YOLACT**:

1. A **Proto** module turns the stride-8 features into `nm = 32` **prototype masks** at stride 4
   (160 × 160 for a 640 input): 3×3 conv → ×2 transposed conv → 3×3 conv → 1×1 conv to 32 channels.
2. The head adds a third tower per level that predicts **32 coefficients** per box.
3. An instance mask is the sigmoid of the linear combination of prototypes with that box's
   coefficients, **cropped to the box**.

So segmentation needs no per-instance RoI operations: one matrix multiply after NMS. The mask loss is BCE
between the cropped combined mask and the ground-truth mask, computed for the positives the detection
assigner chose. **Detection decides which masks exist**: a missed box is a missed mask.

**YOLO26-seg** (`Segment26`, `Proto26`) adds two things. The prototype branch fuses P4 and P5 into the
P3 features before producing prototypes (multi-scale proto), and during training an auxiliary
**semantic-segmentation** branch predicts per-pixel classes from the fused features. Both are
training-time or cheap additions. The YOLO26 paper reports +2.5 box AP and +3.7 mask AP over YOLO11
at comparable size.

| YOLO26-seg (COCO, end-to-end head) | Box mAP | Mask mAP | T4 TRT10 (ms) | Params (M) |
|---|---|---|---|---|
| n | 39.6 | 33.9 | 2.1 | 2.7 |
| s | 47.3 | 40.0 | 3.3 | 10.4 |
| m | 52.5 | 44.1 | 6.7 | 23.6 |
| x | 56.5 | 47.0 | 16.4 | 62.8 |

**When to use it for detection work.** If you need object extent more precise than a box (measuring
area, grasp points, pixel-accurate cut-outs), or if your boxes overlap heavily, a segmentation model
gives both. The box AP is close to that of the detection model of the same size, at somewhat higher
latency (2.1 vs 1.7 ms at n).

---

## Pose: boxes plus keypoints

The pose head adds a tower that predicts, for every box candidate, K keypoints × 3 values (x, y,
visibility) relative to the grid point. COCO person pose uses K = 17. Keypoints are learned for the same
positives as the box, with an **OKS-based loss** (object keypoint similarity, the keypoint analogue of
IoU, scaled by object area and per-keypoint sigmas) plus a BCE visibility term.

**YOLO26-pose** (`Pose26`) replaces the plain keypoint regression with **residual log-likelihood
estimation (RLE)**: a small normalising flow models the error distribution of each keypoint, which
gives better-calibrated keypoints and an uncertainty estimate. The paper reports +7.2 pose AP over
YOLO11 at n. Published COCO pose numbers (end-to-end head): n 57.2, s 63.0, m 68.8, l 70.4, x 71.6
mAP50-95.

Pose is detection where the "box" is a constellation of points. It is used well beyond people: animal
pose, hand keypoints, vehicle corners, tool tips, robotic grasp points. Any task where objects have a
fixed set of landmarks fits the same YAML with a different `kpt_shape`.

---

## Oriented boxes: boxes plus an angle

Aerial imagery, documents, shipping containers and parts on a conveyor are not axis-aligned. An OBB
model predicts **(cx, cy, w, h, θ)**. The changes from detection:

| Component | Axis-aligned detection | OBB |
|---|---|---|
| Label format | `cls cx cy w h` | `cls x1 y1 x2 y2 x3 y3 x4 y4` (four corners, normalised) |
| Head | box + class | box + class + **angle** |
| IoU in assignment and loss | CIoU | **ProbIoU** (boxes as 2-D Gaussians; differentiable, rotation-aware) |
| Candidate points | inside the box | inside the **rotated** box |
| NMS | axis-aligned IoU | rotated IoU |
| Typical input size | 640 | **1024** (DOTA objects are tiny) |

ProbIoU models each rotated box as a Gaussian and measures their overlap with the Bhattacharyya
coefficient. This avoids the discontinuities that make rotated-polygon IoU hard to differentiate, and
the angle periodicity problem (a 0° box and a 180° box are the same box). **YOLO26-obb** outputs the
angle without the sigmoid used before and adds an angle loss. The paper reports +3.4 mAP on DOTA over
YOLO11. Published DOTA-v1 test numbers at 1024: n 52.4, s 54.8, m 55.3, l 56.2, x 56.7 mAP50-95.

---

## Tracking: boxes over time

`model.track()` runs the detector on each frame and associates boxes into tracks. Six trackers are
configured in the current package:

| Tracker | Association | Strength | Weakness |
|---|---|---|---|
| **ByteTrack** | IoU; second pass on *low-confidence* boxes | Recovers occluded objects; fast | ID switches under long occlusion |
| **BoT-SORT** | ByteTrack + camera-motion compensation + optional ReID appearance | Moving cameras | Slower with ReID |
| **OC-SORT** | Observation-centric Kalman re-update | Non-linear motion, occlusion recovery | No appearance |
| **Deep OC-SORT** | OC-SORT + appearance features + camera-motion compensation | Crowds | ReID cost |
| **FastTrack** | ByteTrack-style, occlusion-aware, Kalman rollback | Speed with occlusion handling | Newer, less battle-tested |
| **TrackTrack** | Multi-cue (HMIoU), iterative assignment, track-aware initialisation (CVPR 2025) | Accuracy on MOT benchmarks | More parameters to tune |

Detection settings interact with tracking:

- **Keep low-confidence detections.** ByteTrack's second pass needs boxes below the usual display
  threshold (`track_low_thresh` 0.1). Do not filter at 0.5 before the tracker.
- **NMS matters more in video.** Duplicate boxes create duplicate tracks. A one-to-one head avoids
  duplicates but tends to drop occluded objects that ByteTrack could have recovered. Test both heads on
  your video.
- **Frame rate is an accuracy knob.** At low FPS, IoU association fails because boxes move too far
  between frames. A faster, smaller detector can give better tracking than a more accurate slow one.

Tracking is a field of its own, with its own metrics (HOTA, MOTA, IDF1). Chapter 43 covers video
detection more broadly.

---

## Open vocabulary: replacing the class layer

The last 1×1 conv of a YOLO class tower is a fixed set of nc class vectors. Open-vocabulary YOLOs
replace it with **text embeddings** computed from prompts (Chapter 20):

| Model | Class layer | Inference with a fixed vocabulary |
|---|---|---|
| **YOLO-World** | Region–text contrastive head; text embeddings from CLIP; RepVL-PAN fuses text into the neck | Re-parameterise: bake the prompt embeddings into conv weights → a plain YOLO with no text encoder at runtime |
| **YOLOE / YOLOE-26** | RepRTA (re-parameterisable text alignment), SAVPE (visual prompts), LRPC (prompt-free via a built-in vocabulary) | Same: after `set_classes([...])` the model exports as an ordinary detector |

```python
from ultralytics import YOLOE
model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(["forklift", "pallet", "safety vest"])   # text prompts → embedded class weights
model.export(format="onnx")                               # exports as a closed-set detector for these classes
```

This makes open-vocabulary YOLOs useful in two roles: **zero-shot detection** for prototypes and data
mining, and **auto-labelling** for fine-tuning a closed-set model (Chapters 27 and 49). On LVIS,
YOLOE-26x reaches 40.6 AP with text prompts (non-end-to-end head).

---

## The non-detection members, briefly

| Task | Head | Output | Why a detection engineer cares |
|---|---|---|---|
| Classification (`-cls`) | Global pool + linear | class probabilities | Second-stage verification of detected crops in a cascade (Chapter 47) |
| Semantic segmentation (`-sem`, YOLO26) | Dense per-pixel classes | class map (Cityscapes mIoU 78.3–83.6) | "Stuff" classes (road, sky) that boxes cannot represent; free-space estimation next to a detector |
| Depth (`-depth`, YOLO26) | Multi-scale fusion + upsampling | depth map (NYU δ1 0.88–0.93) | Distance to detected objects from a single camera |

---

## Key Takeaways

- Segment, pose and OBB heads are the detection head plus one branch, sharing the assigner and box
  loss. Detection quality bounds them: a missed box is a missed mask, pose or rotated box.
- Segmentation uses YOLACT-style prototypes (32 at stride 4) and per-box coefficients. YOLO26 adds
  multi-scale prototypes and a training-only semantic branch.
- Pose adds K × 3 keypoint outputs with an OKS loss. YOLO26 models keypoint error with RLE.
- OBB adds an angle and swaps CIoU for ProbIoU everywhere: assignment, loss and NMS.
- Tracking needs low-confidence detections and duplicate-free boxes. Detector FPS is a tracking
  accuracy knob.
- Open-vocabulary YOLOs replace the class layer with prompt embeddings and can be re-parameterised into
  ordinary closed-set detectors for export.

## Check Yourself

<details class="check"><summary>Your segmentation model's mask AP is low but its box AP is high. Where do you look first?</summary>
At the mask branch, since detection already works: mask annotation quality (polygons vs boxes),
prototype resolution (stride 4) relative to object size, and whether thin structures are lost in the
160 × 160 prototypes at 640 input. A higher imgsz increases prototype resolution.</details>

<details class="check"><summary>Why does an OBB model use ProbIoU instead of computing the exact intersection of two rotated rectangles?</summary>
The exact rotated-polygon IoU has non-smooth gradients (the intersection polygon changes shape
discontinuously) and handles angle periodicity poorly. ProbIoU treats boxes as Gaussians, giving a
smooth, differentiable overlap measure that is invariant to the 180° ambiguity of a rectangle.</details>

<details class="check"><summary>You run ByteTrack after filtering detections at conf 0.5 and see many broken tracks during occlusions. Why?</summary>
ByteTrack's second association stage uses low-confidence detections (above track_low_thresh, 0.1 by
default) to keep tracks alive through partial occlusion. Filtering at 0.5 removes exactly those boxes.
Pass the detector's low-confidence outputs to the tracker and apply the display threshold afterwards.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| YOLACT | Bolya et al., 2019 | arXiv:1904.02689 | Prototype masks + coefficients |
| OKS / COCO keypoints | Lin et al., 2014 | cocodataset.org/#keypoints-eval | Keypoint similarity |
| RLE (residual log-likelihood estimation) | Li et al., 2021 | arXiv:2107.11291 | YOLO26 pose regression |
| ProbIoU | Llerena et al., 2021 | arXiv:2106.06072 | Gaussian rotated-box IoU |
| DOTA | Xia et al., 2018 | arXiv:1711.10398 | OBB benchmark |
| ByteTrack; BoT-SORT; OC-SORT; Deep OC-SORT | Zhang et al., 2022; Aharon et al., 2022; Cao et al., 2023; Maggiolino et al., 2023 | arXiv:2110.06864; arXiv:2206.14651; arXiv:2203.14360; arXiv:2302.11813 | Trackers |
| FastTracker; TrackTrack | 2025 | arXiv:2508.14370; CVPR 2025 | Newer trackers in the package |
| YOLO-World; YOLOE | Cheng et al., 2024; Wang et al., 2025 | arXiv:2401.17270; arXiv:2503.07465 | Open-vocabulary heads |
| Ultralytics `nn/modules/head.py` (`Segment`, `Segment26`, `Pose`, `Pose26`, `OBB`, `OBB26`, `YOLOEDetect`), `block.py` (`Proto`, `Proto26`), `cfg/trackers/` | Ultralytics | github.com/ultralytics/ultralytics | Head structures, tracker configs |
| Ultralytics performance tables (`docs/macros/yolo-*-perf.md`) | Ultralytics, 2026 | github.com/ultralytics/ultralytics | Seg, pose, OBB, semantic, depth numbers |
| YOLO26 paper | Ultralytics, 2026 | arXiv:2606.03748 | Task-head gains over YOLO11 |

---

**Next:** [Chapter 39 — Hands-On: YOLO From Scratch](./39_yolo_from_scratch.md) — every axis of this part in
300 lines of PyTorch you can read and run.
