---
title: "Appendix A — Glossary & Symbols"
---

[← Back to Table of Contents](./README.md)

# Appendix A — Glossary & Symbols

One canonical definition per term, with the chapter that develops it.

---

## Symbols

| Symbol | Meaning | Ch. |
|---|---|---|
| $b = (x_1, y_1, x_2, y_2)$ | Box in corner format (pixels) | 2 |
| $(c_x, c_y, w, h)$ | Box in centre–size format | 2 |
| $(l, t, r, b)$ | Distances from a grid point to the four box edges | 4, 6 |
| $s$ | Stride of a feature map (8, 16, 32 …) | 4 |
| $p_{i,k}$ | Predicted probability of class $k$ at candidate $i$ | 5 |
| $u_{ij}$ | IoU (or CIoU) between prediction $i$ and object $j$ | 3, 5 |
| $t_{ij} = p^{\alpha} u^{\beta}$ | Task-alignment metric (TAL), $\alpha = 0.5$, $\beta = 6$ in Ultralytics | 5, 30 |
| $\hat t$ | Normalised soft classification target (TAL) | 5, 30 |
| $k$ | Positives per object (top-$k$) | 5, 30 |
| $\gamma, \alpha$ | Focal-loss focusing and balancing parameters | 6 |
| `reg_max` | Number of DFL bins per box edge (16 in v8/11, 1 in YOLO26) | 6, 29 |
| $\tau$ | Confidence threshold; also EMA time constant in Chapter 31 | 7, 31 |
| AP$_{50}$, AP$_{75}$, AP | Average precision at IoU 0.5, 0.75, and averaged over 0.50:0.05:0.95 | 8 |
| AP$_S$, AP$_M$, AP$_L$ | COCO AP for areas < 32², 32²–96², > 96² pixels | 8 |
| AR$_{100}$ | Average recall with at most 100 detections per image | 8 |

---

## Terms

| Term | Definition | Ch. |
|---|---|---|
| **Anchor** | A predefined reference box at a grid location; predictions are offsets from it | 4 |
| **Anchor-free** | Predicting boxes from grid points (distances to edges) or centres without predefined box shapes | 4 |
| **AP (average precision)** | Area under the interpolated precision–recall curve for one class | 8 |
| **Area attention (A²)** | Self-attention computed within a few strips of the feature map (YOLOv12) | 26, 29 |
| **Assignment (label assignment)** | The rule deciding which predictions learn from which object, and which are background | 5, 30 |
| **ATSS** | Adaptive training sample selection: positives chosen by IoU statistics per object | 5 |
| **Augmentation** | Random transformations of training images and labels (mosaic, affine, HSV, mixup, copy-paste) | 10, 28 |
| **AutoBatch** | Ultralytics' automatic batch-size selection by GPU memory fraction | 31 |
| **Auto-labelling** | Producing training labels with a model (often open-vocabulary), usually followed by review | 42, 49 |
| **Backbone** | The feature-extraction network (CSPDarknet, C3k2 stacks, ResNet, ViT) | 11, 29 |
| **Background image** | Training image with no objects; reduces false positives | 27 |
| **BEV** | Bird's-eye view, the common output space of 3D detectors | 43 |
| **Bootstrap interval** | Confidence interval from resampling validation images with replacement | 33 |
| **C2f / C3k2** | CSP-style blocks of YOLOv8 / YOLO11–26 that concatenate intermediate outputs | 29 |
| **C2PSA** | CSP block with partial self-attention at stride 32 (YOLO11, YOLO26) | 29 |
| **Calibration (scores)** | Agreement between predicted confidence and empirical precision | 47 |
| **Calibration (quantisation)** | Estimating activation ranges for INT8 from representative images | 45 |
| **Candidate** | A prediction location eligible to become a positive (e.g. grid points inside a box) | 5, 30 |
| **CIoU** | Complete IoU: IoU minus centre-distance and aspect-ratio penalties | 3 |
| **close_mosaic** | Turning mosaic off for the final epochs | 28 |
| **COCO** | Common Objects in Context: 80 classes, the standard detection benchmark | 9 |
| **Compound scaling** | Deriving n/s/m/l/x models by scaling depth, width and channel caps | 29 |
| **Confidence threshold** | Score below which detections are discarded | 7, 47 |
| **Copy-paste** | Augmentation pasting object instances into other images | 10 |
| **CSP** | Cross-stage partial: split channels, transform part, merge | 11, 29 |
| **DETR** | Detection Transformer: object queries, Hungarian matching, no NMS | 18 |
| **DFL** | Distribution focal loss: each box edge predicted as a distribution over bins | 6 |
| **Distillation (KD)** | Training a student to match a teacher's outputs or features | 42 |
| **Domain shift** | Difference between training and deployment distributions | 43 |
| **Dual assignment** | Training one-to-many and one-to-one heads together; deploying the one-to-one head | 5, 30 |
| **Dynamic k** | Number of positives per object computed from its predictions (SimOTA) | 5 |
| **EMA (weights)** | Exponential moving average of weights used for evaluation and export | 31 |
| **End-to-end (e2e)** | Detection without NMS: the network outputs final boxes | 7, 26 |
| **Export** | Converting a trained model to a runtime format (ONNX, TensorRT, CoreML …) | 36 |
| **False positive (FP)** | A detection that matches no unmatched object of its class at the IoU threshold | 8 |
| **Fitness** | Ultralytics' checkpoint-selection metric (mAP50-95 in current releases) | 31 |
| **FPN / PAN** | Feature pyramid network (top-down fusion) / path aggregation network (+ bottom-up) | 12 |
| **Fuse** | Folding BatchNorm into convolutions (and removing unused heads) before export | 29, 36 |
| **GIoU / DIoU** | Generalised IoU (enclosing-box penalty) / Distance IoU (centre-distance penalty) | 3 |
| **Grid point** | Centre of a feature-map cell, the anchor-free prediction location | 4 |
| **Head** | The layers producing box and class outputs; decoupled heads separate the two | 13 |
| **Hungarian matching** | Optimal one-to-one assignment between predictions and objects | 5, 18 |
| **imgsz** | Network input size (pixels) | 28 |
| **INT8** | 8-bit integer inference precision | 45 |
| **IoU** | Intersection over union of two boxes | 3 |
| **Letterbox** | Resize preserving aspect ratio, then pad to the network size | 10, 28 |
| **Mosaic** | Augmentation combining four images into one | 10, 28 |
| **MuSGD** | Ultralytics' hybrid of Muon orthogonalised updates and SGD | 31 |
| **Neck** | Multi-scale feature fusion between backbone and head | 12 |
| **NMS** | Non-maximum suppression: remove lower-scoring boxes overlapping a higher-scoring one | 7 |
| **NWD** | Normalised Wasserstein distance between box Gaussians; robust for tiny objects | 3, 41 |
| **OBB** | Oriented (rotated) bounding box | 38, 43 |
| **Objectness** | Separate score for "an object is here", used up to YOLOv7 and YOLOX | 6 |
| **One-to-many (o2m) / one-to-one (o2o)** | Assignment giving several / exactly one positive per object | 5 |
| **Open-vocabulary detection** | Detecting classes specified at inference by text or visual prompts | 20 |
| **P2 … P6** | Pyramid levels at strides 4 … 64 | 12, 41 |
| **Positive** | A prediction assigned to an object during training | 5 |
| **Precision / recall** | TP / (TP + FP) and TP / (TP + FN) at a threshold | 8 |
| **ProbIoU** | IoU-like overlap between rotated boxes modelled as Gaussians | 38 |
| **ProgLoss** | YOLO26 schedule decaying the one-to-many loss weight from 0.8 to 0.1 | 30 |
| **PTQ / QAT** | Post-training quantisation / quantisation-aware training | 45 |
| **Query (object query)** | Learned or selected embedding that a DETR decoder turns into one detection | 4, 18 |
| **Rect training/validation** | Batches with per-batch rectangular shapes and minimal padding | 28 |
| **Re-parameterisation** | Training multi-branch blocks and folding them into one conv for inference | 37 |
| **SAHI** | Sliced inference over overlapping tiles with merged results | 41 |
| **SimOTA** | Simplified optimal-transport assignment with dynamic $k$ (YOLOX) | 5 |
| **Soft target** | Classification target below 1 that encodes localisation quality | 5, 6 |
| **SPPF** | Spatial pyramid pooling – fast: three chained 5×5 max pools | 29 |
| **STAL** | Small-target-aware label assignment: tiny boxes enlarged for candidate selection (YOLO26) | 30, 41 |
| **Stride** | Downsampling factor of a feature map relative to the input | 4 |
| **TAL** | Task-aligned assignment (TOOD), the YOLOv6/8/11/26 assigner | 5, 30 |
| **TIDE** | Error decomposition into classification, localisation, duplicate, background and missed errors | 8 |
| **Top-k head** | One-to-one head output selected by top-k over scores, no NMS | 7, 36 |
| **True positive (TP)** | A detection matched to an unmatched object of its class at IoU ≥ threshold | 8 |
| **VFM** | Vision foundation model (DINOv2/v3, SAM, CLIP-family) | 19, 42 |
| **YAML (model)** | Ultralytics architecture description parsed into layers | 29 |
