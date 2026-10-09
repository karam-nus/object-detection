---
title: "Chapter 43 — Oriented, 3D, Video & Domain Shift"
---

[← Back to Table of Contents](./README.md)

# Chapter 43 — Oriented, 3D, Video & Domain Shift

> *"Most detection theory assumes an axis-aligned box, a single image and a test set drawn from the training distribution. Field work breaks all three."*

## Overview

This chapter covers the main ways real problems depart from the COCO setting, at the depth a detection
engineer needs to recognise them and choose a starting point: **rotated boxes** (the angle problem),
**crowded scenes** (where NMS removes true objects), **3D detection** (LiDAR, camera BEV, monocular),
**video** (temporal context, tracking, streaming latency), and **domain shift** (when the deployment
images differ from the training images). Each section names the core difficulty, the standard
solutions, the metric and the benchmark.

---

## Rotated boxes: the angle problem

A rotated box is (cx, cy, w, h, θ). Chapter 38 showed how a YOLO OBB head adds θ. The difficulty is
that the parameterisation is not continuous:

- **Periodicity**: a rectangle rotated by 180° is the same rectangle, and a box with w and h swapped
  rotated by 90° is the same box. Regressing θ directly penalises predictions that are geometrically
  perfect but numerically far from the label.
- **Boundary discontinuity**: under the common "OpenCV" convention θ ∈ [0°, 90°), a box near 89° and
  one near 1° can be almost identical, yet the regression target jumps.
- **Square-like objects**: when w ≈ h, θ is nearly undefined (storage tanks, roundabouts).

| Solution | Idea | Used in |
|---|---|---|
| Angle as classification (CSL) | Discretise θ into bins with a circular smoothing window | CSL-based detectors |
| Gaussian box representations (GWD, KLD, KFIoU) | Model the box as a 2-D Gaussian; losses are distances between Gaussians, invariant to the ambiguities | Many DOTA methods |
| **ProbIoU** | Bhattacharyya overlap of the two Gaussians, used as IoU in assignment, loss and NMS | Ultralytics YOLO OBB |
| Oriented proposals | Two-stage with rotated RoI alignment | Oriented R-CNN, RoI Transformer |
| Rotated anchor-free dense heads | One-stage with an angle branch | RTMDet-R, YOLO-OBB |

**Benchmarks:** DOTA v1.0/v1.5/v2.0 (aerial, 15–18 classes), DIOR-R, HRSC2016 (ships). **Metric:**
mAP with rotated IoU (DOTA reports mAP50 on its test server).

---

## Crowded scenes: when NMS hurts

In crowds (pedestrians, shelves, livestock, cells), true objects overlap heavily. NMS at IoU 0.5–0.7
deletes real people standing behind each other, and the one-to-many assignment struggles to separate
neighbours.

| Fix | Mechanism |
|---|---|
| **Soft-NMS** | Decay overlapping scores instead of deleting the boxes |
| **Adaptive NMS** | Predict a per-object density and raise the NMS threshold in dense regions |
| **Set NMS / multiple predictions per proposal** | One anchor predicts a *set* of instances; suppression skips boxes from the same proposal (Chu et al., 2020) |
| **Repulsion loss** | Push each prediction away from neighbouring ground truths and predictions |
| **Visible-box annotations** | Train on visible parts (CrowdHuman provides full, visible and head boxes) |
| **One-to-one (NMS-free) heads** | No suppression step; the model learns to separate instances. DETR-style detectors handle crowds comparatively well |
| **Higher resolution / P2** | More grid points per person, fewer conflicts in assignment |

**Benchmarks:** CrowdHuman (about 23 people per image, heavy occlusion), CityPersons, SKU-110K (densely
packed retail). **Metrics:** log-average miss rate MR⁻² (lower is better) over false positives per image,
and AP. The Jaccard index (JI) measures one-to-one matching quality.

---

## 3D detection, in brief

3D detectors output a 7-DoF box (x, y, z, length, width, height, yaw), usually in a bird's-eye view
(BEV) of the scene.

| Input | Representative methods | Core idea |
|---|---|---|
| **LiDAR point clouds** | PointPillars, CenterPoint, voxel transformers | Voxelise or pillarise points → 2-D BEV pseudo-image → a 2-D detection head (often centre-based) |
| **Multi-camera** | BEVDet, BEVFormer, StreamPETR | Lift image features to BEV with depth estimates or attention; detect in BEV |
| **Monocular camera** | FCOS3D, monocular depth-aware heads | 2-D detection + depth and 3D size regression; depth is the hard part |
| **Fusion** | BEVFusion | Combine camera and LiDAR in BEV |

The 2-D ideas transfer directly: centre-based heads (CenterPoint is CenterNet in BEV), focal losses,
dense heads, NMS in BEV, and DETR-style queries (PETR, StreamPETR). **Benchmarks and metrics:** KITTI
(3D AP at IoU 0.7 for cars), nuScenes (NDS, a weighted combination of mAP and true-positive error terms
for translation, scale, orientation, velocity and attributes), Waymo Open Dataset (APH, heading-weighted
AP).

---

## Video: detection over time

### Per-frame detection plus tracking

The default architecture for video products is a per-frame detector followed by a tracker (Chapter 38:
ByteTrack, BoT-SORT, OC-SORT). Tracks smooth flicker, carry identities, and enable counting and dwell
time. Tracking quality depends on detector recall at **low** confidence and on frame rate.

### Temporal feature aggregation

Video object detection methods (FGFA, SELSA, MEGA and successors) aggregate features from neighbouring
frames to stabilise detections under motion blur, defocus and rare poses. They improve accuracy on
ImageNet VID but cost latency and memory, so most real-time systems prefer per-frame detection plus
tracking.

### Joint detection and tracking

CenterTrack and FairMOT predict detections and association cues (offsets or embeddings) in one network.
Tracking-by-query methods (MOTR) extend DETR queries across frames.

### Streaming perception

Li, Wang and Ramanan (2020) pointed out that offline video metrics ignore latency: by the time a slow
detector finishes frame t, the world has moved on. **Streaming AP** evaluates the latest available
output at every moment. The practical consequences: a faster, slightly less accurate detector can win in
real time; forecasting (extrapolating box motion by the expected latency) recovers part of the loss;
skipping frames is better than queuing them.

| Video technique | Typical cost | When to use |
|---|---|---|
| Per-frame detector + tracker | Detector cost + small tracker overhead | Default for real-time products |
| Detect every k-th frame, track in between | Detector cost / k | Low compute, slow scenes |
| Temporal aggregation | 1.5–3× | Offline analysis, blur-heavy footage |
| Score smoothing over tracks | Negligible | Reducing flicker and spurious alarms |

---

## Domain shift

A detector trained on one distribution fails in quiet ways on another: lower recall, more false
positives, worse calibration, without any error message.

| Shift | Example | Typical symptom |
|---|---|---|
| Appearance | Day → night, clear → fog, rain, snow | Recall collapse on distant objects |
| Sensor | RGB → thermal, new camera, compression, different ISP | Colour/contrast-dependent features break |
| Viewpoint and scale | Street-level → drone; mounting height changes | Object size and aspect distributions shift |
| Synthetic → real | Rendered training data | Texture and noise statistics differ |
| Label policy | Different annotation rules between datasets | Systematic box-extent or class-boundary disagreement |

**Standard benchmarks:** Cityscapes → Foggy Cityscapes, Sim10k → Cityscapes, KITTI → Cityscapes, BDD100K
weather and time-of-day splits, ACDC (adverse conditions).

**Methods**, in order of practical value:

1. **Collect and label a little target data.** A few hundred target-domain images usually beat any
   unsupervised adaptation method.
2. **Augment for the shift**: blur, noise, JPEG artefacts, colour jitter, synthetic fog and rain,
   low-light simulation, sensor-specific transforms.
3. **Unsupervised domain adaptation**: adversarial feature alignment (Domain Adaptive Faster R-CNN) and
   **mean-teacher adaptation** on unlabelled target images (Adaptive Teacher, the semi-supervised recipe
   of Chapter 42 applied across domains).
4. **Foundation backbones**: DINOv2/v3-based detectors and open-vocabulary models are usually more robust
   to appearance shift than COCO-trained CNNs. Validate on target data.
5. **Test-time adaptation**: update BN statistics or a small set of parameters on target batches. It can
   help gradual shifts, but it is risky without monitoring.

Monitoring for shift in production (score distributions, detections per frame, embedding drift) is
covered in Chapter 47.

---

## Other modalities worth recognising

| Domain | What differs | Notes |
|---|---|---|
| Thermal / infrared | Single channel, low texture, polarity | Fine-tune from RGB weights with replicated channels or the first-conv adaptation (Chapter 31) |
| Multispectral / satellite | 4–13 bands; very large images | Tiling (Chapter 41); first-conv adaptation |
| Medical imaging (lesion detection) | Few objects per image, 3-D volumes, expert labels | FROC (sensitivity at fixed false positives per image) instead of AP |
| Documents (layout detection) | Rectangles of text, tables, figures; extreme aspect ratios | DocLayNet, PubLayNet; aspect-ratio-aware training |
| Event cameras | Asynchronous brightness-change events | Convert to frames or use event-native networks |

---

## Key Takeaways

- Rotated boxes are hard because θ is periodic and discontinuous. Gaussian-based measures (ProbIoU, GWD,
  KLD) remove the ambiguity and are standard in modern OBB detectors.
- In crowds, NMS deletes true objects. Soft or adaptive NMS, set prediction, visible-box labels and
  one-to-one heads help. Report miss rate alongside AP.
- 3D detection reuses 2-D ideas in BEV: centre heads, dense heads, DETR queries. Metrics (NDS, APH)
  weight localisation and heading errors.
- In video, per-frame detection plus tracking is the default. Streaming evaluation shows why latency is
  part of accuracy.
- Domain shift is fixed first with target data and targeted augmentation, then with mean-teacher
  adaptation or more robust backbones, and always with monitoring.

## Check Yourself

<details class="check"><summary>An OBB model predicts a ship at θ = 89° when the label says 1°, and the two boxes almost coincide. Why is a plain L1 loss on θ a bad idea here?</summary>
The two boxes are nearly the same shape (the parameterisation wraps around), but the L1 loss on θ sees an
88° error and pushes the prediction across the whole range. Losses based on the box geometry, such as
ProbIoU or Gaussian distances, measure the true overlap and do not suffer from the discontinuity.</details>

<details class="check"><summary>Your people counter undercounts in queues, although each person is visible. Name two fixes that do not require a new model architecture.</summary>
Raise the NMS IoU threshold (or use Soft-NMS) so overlapping people are not suppressed, and annotate and
train on visible-part or head boxes, which overlap less. A higher input resolution or P2 head also
gives more grid points per person.</details>

<details class="check"><summary>Why can a detector with lower offline AP achieve higher streaming AP?</summary>
Streaming evaluation scores the latest output available at each moment. A slow detector's outputs
describe an older state of the scene, so its boxes are misaligned with moving objects when they are
evaluated. A faster detector's outputs are fresher, which can outweigh its lower per-frame accuracy.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| DOTA | Xia et al., 2018 | arXiv:1711.10398 | Aerial OBB benchmark |
| CSL | Yang, Yan, 2020 | arXiv:2003.05597 | Angle as classification |
| GWD; KLD; KFIoU | Yang et al., 2021; 2021; 2022 | arXiv:2101.11952; arXiv:2106.01883; arXiv:2201.12558 | Gaussian rotated-box losses |
| ProbIoU | Llerena et al., 2021 | arXiv:2106.06072 | Gaussian IoU |
| Oriented R-CNN | Xie et al., 2021 | arXiv:2108.05699 | Oriented proposals |
| Soft-NMS | Bodla et al., 2017 | arXiv:1704.04503 | Score decay |
| Adaptive NMS | Liu, Huang, Wang, 2019 | arXiv:1904.03629 | Density-aware NMS |
| Detection in Crowded Scenes: One Proposal, Multiple Predictions | Chu et al., 2020 | arXiv:2003.09163 | Set NMS |
| Repulsion Loss | Wang et al., 2018 | arXiv:1711.07752 | Crowd-aware loss |
| CrowdHuman | Shao et al., 2018 | arXiv:1805.00123 | Crowd benchmark |
| PointPillars; CenterPoint | Lang et al., 2019; Yin, Zhou, Krähenbühl, 2021 | arXiv:1812.05784; arXiv:2006.11275 | LiDAR 3D detection |
| BEVFormer; BEVFusion | Li et al., 2022; Liu et al., 2022 | arXiv:2203.17270; arXiv:2205.13542 | Camera BEV, fusion |
| nuScenes | Caesar et al., 2020 | arXiv:1903.11027 | NDS metric |
| FGFA; MEGA | Zhu et al., 2017; Chen et al., 2020 | arXiv:1703.10025; arXiv:2003.12063 | Video feature aggregation |
| CenterTrack; FairMOT; MOTR | Zhou et al., 2020; Zhang et al., 2021; Zeng et al., 2022 | arXiv:2004.01177; arXiv:2004.01888; arXiv:2105.03247 | Joint detection and tracking |
| Towards Streaming Perception | Li, Wang, Ramanan, 2020 | arXiv:2005.10420 | Streaming AP |
| Domain Adaptive Faster R-CNN | Chen et al., 2018 | arXiv:1803.03243 | Adversarial adaptation |
| Adaptive Teacher | Li et al., 2022 | arXiv:2111.13216 | Mean-teacher cross-domain adaptation |
| ACDC; BDD100K | Sakaridis et al., 2021; Yu et al., 2020 | arXiv:2104.13395; arXiv:1805.04687 | Adverse-condition and driving benchmarks |

---

**Next:** [Chapter 44 — Hardware & Runtimes](./44_hardware_and_runtimes.md) — Part VI begins: the chips
detectors run on, and what each can execute.
