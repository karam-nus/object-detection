---
title: "Object Detection — The Complete Guide"
permalink: /
---

# 🎯 Object Detection — The Complete Guide

> **From IoU to YOLO26 and beyond**: how machines find, box and name things. Covers 0.5 M-parameter MCU detectors through real-time DETRs to 3-billion-parameter language models that emit coordinates as tokens. YOLO is taken apart along eleven separate axes, and every claim is derived or cited.

## Who This Is For

You know PyTorch and have probably trained a detector with `model.train(data=...)`. It either worked or it didn't, and you want to know **why**. This book treats object detection as what it is: a set-prediction problem. Six design decisions sit on top of it: how boxes are represented, which prediction is responsible for which object, how the loss is shaped, how duplicates are removed, how the model is measured, and what hardware runs it. Every named model is a point in that design space. Once you see the space, a new paper becomes a diff instead of a mystery.

The authoring standard is checked in at [PROMPT.md](./PROMPT.md).

## What Makes This Book Different

<div class="diagram">
<div class="diagram-grid cols-4">
  <div class="diagram-card accent"><div class="card-icon">🧪</div><div class="card-title">Interactive labs</div><div class="card-desc">Drag boxes and watch IoU, GIoU, DIoU and CIoU move. Step NMS one box at a time. Build a PR curve and get AP three ways. See which grid points can "see" a tiny object.</div></div>
  <div class="diagram-card green"><div class="card-icon">🗺️</div><div class="card-title">Detection Atlas</div><div class="card-desc">More than 130 detectors in one filterable dataset: AP, latency, parameters, pre-training, license, NMS. Only like-for-like latencies share an axis.</div></div>
  <div class="diagram-card purple"><div class="card-icon">🧩</div><div class="card-title">The YOLO Matrix</div><div class="card-desc">Eleven orthogonal axes: data, augmentation, architecture, assignment and loss, recipe, tuning, metrics, performance, usability and licensing, export, novel ideas. One chapter per axis, every version compared on each.</div></div>
  <div class="diagram-card cyan"><div class="card-icon">✅</div><div class="card-title">Tested companion code</div><div class="card-desc"><code>code/odlab</code>: IoU family, NMS variants, COCO mAP (identical to pycocotools to 1e-6), five label assigners, a trainable TinyYOLO with DFL or NMS-free heads, and 32 passing tests.</div></div>
</div>
</div>

## 📋 Table of Contents

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| **Orientation** | | |
| 0 | [The Whole Field on One Page](./00_grand_overview.md) | The six decisions inside every detector, the model map, and the five chapters to read first |
| 1 | [A Short History of Finding Things](./01_history.md) | Viola–Jones → DPM → R-CNN → YOLO/SSD → RetinaNet → DETR → RT-DETR, YOLO26, grounding models and MLLMs: what each era fixed |
| **Part I — Foundations** | | |
| 2 | [The Detection Problem, Formally](./02_the_detection_problem.md) | Output sets, box parameterisations, coordinate conventions, the off-by-one wars, and tensor shapes end to end |
| 3 | [IoU and Its Family](./03_iou_family.md) | IoU, GIoU, DIoU, CIoU, EIoU, SIoU, NWD, ProbIoU — derived, plotted, gradients checked |
| 4 | [Anchors, Points and Queries](./04_anchors_points_queries.md) | The three ways a network proposes boxes; strides, grids, k-means anchors, the 8,400 number |
| 5 | [Label Assignment](./05_label_assignment.md) | **The hidden heart of detection**: max-IoU, ATSS, OTA/SimOTA, TAL, Hungarian, one-to-many vs one-to-one, STAL |
| 6 | [Losses](./06_losses.md) | Focal, QFL, VFL, IoU losses, DFL, L1 — what each fixes and how they are balanced |
| 7 | [Post-Processing & NMS](./07_post_processing.md) | Greedy, batched, Soft-, DIoU-, Matrix-NMS, WBF, and why the field spent five years getting rid of NMS |
| 8 | [Metrics](./08_metrics.md) | AP the way pycocotools computes it, the three interpolation conventions, AP_S/M/L, AR, LVIS AP, TIDE, F1, and the ways mAP misleads |
| 9 | [Datasets, Formats & Annotation](./09_datasets_and_annotation.md) | COCO, Objects365, LVIS, OpenImages, RF100-VL; COCO JSON vs YOLO txt; label noise, guidelines and auto-labelling |
| 10 | [Augmentation](./10_augmentation.md) | Box-aware geometry, mosaic, mixup, copy-paste, LSJ, letterbox, TTA, and augmentation schedules |
| **Part II — Building Blocks** | | |
| 11 | [Backbones](./11_backbones.md) | ResNet → CSPDarknet → ELAN/GELAN → RepVGG → HGNetV2 → ViT/DINOv2/v3: what a detector needs from features |
| 12 | [Necks](./12_necks.md) | FPN, PAN, BiFPN, RepGFPN, gather-and-distribute, hybrid encoders, simple FPN for plain ViTs |
| 13 | [Heads](./13_heads.md) | Coupled vs decoupled, anchor-based vs free, DFL heads, one-to-one heads, DETR decoders |
| **Part III — The Detector Landscape** | | |
| 14 | [Classic Detectors](./14_classic_detectors.md) | The R-CNN family, SSD, RetinaNet, FCOS, CenterNet, EfficientDet, and when they are still the right answer |
| 15 | [Very Tiny Detectors](./15_tiny_detectors.md) | Sub-1 M-parameter and MCU detectors: FOMO, Yolo-Fastest, NanoDet, PicoDet, DEIMv2-Atto, peak-SRAM arithmetic |
| 16 | [Edge Detectors](./16_edge_detectors.md) | Phones, Jetson, NPUs: what makes a model edge-friendly, operator support, and the edge leaderboard |
| 17 | [Large Detectors](./17_large_detectors.md) | Co-DETR, DINO with Swin-L/ViT-L, Objects365 pre-training, TTA and ensembles: the 60+ AP club and its cost |
| 18 | [DETR and Its Descendants](./18_detr_family.md) | Set prediction, Hungarian matching, Deformable/Conditional/DAB/DN/DINO, hybrid matching, Co-DETR |
| 19 | [Real-Time DETRs & ViT Backbones](./19_realtime_detrs_and_vit.md) | RT-DETR v1–v4, LW-DETR, D-FINE, DEIM/v2, RF-DETR, EdgeCrafter: how transformers caught YOLO |
| 20 | [Open-Vocabulary & Grounded Detection](./20_open_vocabulary.md) | OWL-ViT/v2, GLIP, Grounding DINO, YOLO-World, YOLOE, DINO-X, T-Rex2, SAM 3 |
| 21 | [LLM-Based Detection](./21_llm_based_detection.md) | Boxes as tokens: Kosmos-2 → Qwen3-VL, Florence-2, Rex-Omni, LocateAnything, VLM-R1; why dense detection is hard for LLMs; hybrid pipelines |
| **Part IV — YOLO: The Complete Treatment** | | |
| 22 | [The YOLO Matrix](./22_yolo_matrix.md) | Eleven orthogonal axes and the version × axis grid: the map for this whole part |
| 23 | [Family Tree & Lineage](./23_yolo_family_tree.md) | Who built which YOLO, naming chaos, forks, licenses, and how to read a YOLO paper |
| 24 | [YOLOv1–v3: Founding Ideas](./24_yolo_v1_to_v3.md) | Grid cells, k-means anchors, Darknet, multi-scale heads — with the original equations |
| 25 | [YOLOv4–v7 and the Forks](./25_yolo_v4_to_v7.md) | Bag of freebies, CSP, YOLOv5's engineering, YOLOX, PP-YOLOE, YOLOv6, YOLOv7, DAMO-YOLO, YOLO-NAS, Gold-YOLO |
| 26 | [YOLOv8 → YOLO26 (and YOLO27)](./26_yolo_v8_to_yolo26.md) | Anchor-free + DFL, PGI/GELAN, NMS-free dual assignment, C3k2/C2PSA, area attention, hypergraphs, DFL removal, MuSGD |
| 27 | [Axis 1 — Data](./27_yolo_data.md) | The YOLO label format, `data.yaml`, dataset sizing, class balance, Objects365 pre-training, auto-annotation |
| 28 | [Axis 2 — Preprocessing & Augmentation](./28_yolo_preprocessing_augmentation.md) | Letterbox maths, rect batches, mosaic internals, mixup, copy-paste, `close_mosaic`, per-version defaults |
| 29 | [Axis 3 — Architecture Anatomy](./29_yolo_architecture.md) | Every block (C3, C2f, C3k2, ELAN, GELAN, SPPF, C2PSA, A2C2f, RepConv), compound scaling, YAML reading, tensor shapes through YOLO26n |
| 30 | [Axis 4 — Assignment & Loss](./30_yolo_assignment_and_loss.md) | v3 IoU, v5 cross-grid ratios, SimOTA, TAL maths, DFL derivation, dual assignment, STAL, ProgLoss — version by version |
| 31 | [Axis 5 — Training Recipe](./31_yolo_training_recipe.md) | SGD/AdamW/MuSGD, LR schedules, warm-up, EMA, nominal batch 64, AMP, epochs, Objects365 → COCO, the YOLO26 recipe table |
| 32 | [Axis 6 — Hyperparameter Tuning](./32_yolo_hyperparameter_tuning.md) | Genetic evolution, Ray Tune/Optuna, what to tune first, budgets, and the fitness function |
| 33 | [Axis 7 — Metrics & Validation](./33_yolo_metrics_and_validation.md) | How Ultralytics computes mAP vs pycocotools, val-time NMS settings, the fitness score, and reported-speed conventions |
| 34 | [Axis 8 — Family Performance](./34_yolo_family_performance.md) | v5u → YOLO26 (+ YOLO27 preview) on one table, Pareto fronts, RF100-VL, and the pre-training asterisk |
| 35 | [Axis 9 — Ease of Use, Ecosystem & Licensing](./35_yolo_ease_of_use_and_licensing.md) | APIs, CLIs, integrations, AGPL vs GPL vs Apache, and what licensing actually means for your product |
| 36 | [Axis 10 — Export & Deployment](./36_yolo_export_and_deployment.md) | ONNX, TensorRT, OpenVINO, CoreML, LiteRT, NCNN, RKNN, Hailo; NMS-in-graph vs end-to-end; INT8 pitfalls |
| 37 | [Axis 11 — Novel Optimizations](./37_yolo_novel_optimizations.md) | Every new idea since 2016, with its mechanism, measured gain and cost: re-param, PGI, dual assignment, MuSGD, STAL, area attention… |
| 38 | [YOLO Beyond Boxes](./38_yolo_beyond_boxes.md) | Segmentation, pose (RLE), OBB, classification, YOLO-World/YOLOE and tracking: how one head family grew |
| 39 | [Hands-On: YOLO From Scratch](./39_yolo_from_scratch.md) | Build, train and evaluate a modern YOLO on a CPU with `odlab`, then flip it to DFL-free and NMS-free |
| **Part V — Training & Data Strategy** | | |
| 40 | [Training Detectors in Practice](./40_training_in_practice.md) | Fine-tuning vs scratch, small data, long tail, class imbalance, BN with small batches, schedules |
| 41 | [Small Objects](./41_small_objects.md) | Why small objects fail, P2 heads, high resolution, SAHI tiling, NWD, STAL, aerial benchmarks |
| 42 | [Label-Efficient Learning & Distillation](./42_label_efficient_and_distillation.md) | Semi-supervised teachers, auto-labelling with foundation models, detector KD, VFM distillation |
| 43 | [Oriented, 3D, Video & Domain Shift](./43_specialized_detection.md) | OBB maths, crowded scenes, LiDAR/BEV in brief, tracking-by-detection, domain adaptation |
| **Part VI — Deployment** | | |
| 44 | [Hardware & Runtimes](./44_hardware_and_runtimes.md) | GPUs, CPUs, mobile and edge NPUs, MCUs; the operator-support matrix; what breaks on which chip |
| 45 | [Quantizing & Compressing Detectors](./45_quantization_and_compression.md) | Why detectors quantize worse than classifiers, sensitive layers, PTQ/QAT recipes, pruning |
| 46 | [Measuring Latency Honestly](./46_measuring_latency.md) | What is in "1.7 ms", pre/post-processing, batch, warm-up, throughput vs latency, a reporting template |
| 47 | [Detection in Production](./47_production_systems.md) | Pipelines, per-class thresholds, calibration, monitoring, data flywheels, failure review |
| **Part VII — Practice & Decisions** | | |
| 48 | [Hands-On: Fine-Tune, Evaluate, Export](./48_handson_finetune_export.md) | YOLO26 and RF-DETR on the same custom dataset, evaluated identically, exported and benchmarked |
| 49 | [Hands-On: Zero-Shot → Edge](./49_handson_zero_shot_to_edge.md) | Auto-label with YOLOE / Grounding DINO / a VLM, train a nano model, measure what was lost |
| 50 | [The Debugging Playbook](./50_debugging_playbook.md) | Symptom → cause → diagnostic → fix, for training, evaluation and deployment failures |
| 51 | [The Decision Guide](./51_decision_guide.md) | Flowcharts by constraint, each ending in a concrete model and recipe |
| 52 | [The Frontier](./52_frontier.md) | Detection as language, unified perception, test-time compute, COCO saturation, open problems |
| **Interactive** | | |
| ⚗ | [Interactive Labs](./labs.md) | All six labs on one page |
| 🗺 | [Detection Atlas](./atlas.md) | Filter, sort and plot every detector in the book |
| **Appendices** | | |
| A | [Glossary & Symbols](./appendix_a_glossary.md) | Every term and symbol, one canonical definition each |
| B | [Math Reference](./appendix_b_math.md) | IoU gradients, focal-loss derivative, DFL optimum, Hungarian matching, AP as an integral, TAL normalisation |
| C | [Model Index](./appendix_c_model_index.md) | The master table: every detector × year × params × FLOPs × AP × latency × pre-training × license |
| D | [Dataset Index](./appendix_d_datasets.md) | Every dataset × size × classes × domain × license × what it is good for |
| E | [Hardware & Export Matrix](./appendix_e_hardware_export_matrix.md) | Runtime × operator support × precision × NMS support × known traps |
| F | [Bibliography](./appendix_f_bibliography.md) | The reading list by theme, with a suggested order |
| G | [Question Bank](./appendix_g_question_bank.md) | 120 self-test and interview questions with answers, keyed to chapters |

## 🗺️ Learning Path

<div class="diagram">
<div class="diagram-title">Recommended Reading Order</div>
<div class="flow">
  <div class="flow-node accent wide">🧭 Ch 0–1: Orientation & History</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">📐 Ch 2–10: The Primitives <small>boxes, IoU, assignment, losses, NMS, metrics, data</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">🧱 Ch 11–13: Backbones, Necks, Heads</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🌍 Ch 14–21: The Landscape <small>tiny → edge → large → DETR → open-vocab → LLM</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">⚡ Ch 22–39: YOLO, Axis by Axis</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">🏋️ Ch 40–43: Training & Data Strategy</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">🚀 Ch 44–47: Deployment</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">🔬 Ch 48–52: Hands-On, Debugging, Decisions, Frontier</div>
</div>
</div>

## ⚡ Quick Start Paths

### Path A: "I need to master YOLO"

1. [22 — The YOLO Matrix](./22_yolo_matrix.md) — the map
2. [05 — Label Assignment](./05_label_assignment.md) — the concept every YOLO version re-invents
3. [26 — YOLOv8 → YOLO26](./26_yolo_v8_to_yolo26.md) — the modern lineage
4. [29 — Architecture](./29_yolo_architecture.md) → [30 — Assignment & Loss](./30_yolo_assignment_and_loss.md) → [31 — Training Recipe](./31_yolo_training_recipe.md)
5. [34 — Family Performance](./34_yolo_family_performance.md) — what the numbers really say
6. [39 — Hands-On: YOLO From Scratch](./39_yolo_from_scratch.md)

### Path B: "I ship to an edge device or a microcontroller"

1. [15 — Very Tiny Detectors](./15_tiny_detectors.md) → [16 — Edge Detectors](./16_edge_detectors.md)
2. [44 — Hardware & Runtimes](./44_hardware_and_runtimes.md) — what your chip can execute
3. [36 — YOLO Export & Deployment](./36_yolo_export_and_deployment.md)
4. [45 — Quantizing Detectors](./45_quantization_and_compression.md)
5. [46 — Measuring Latency Honestly](./46_measuring_latency.md)

### Path C: "I want the highest accuracy, or I'm choosing a transformer detector"

1. [18 — DETR and Its Descendants](./18_detr_family.md) → [19 — Real-Time DETRs & ViTs](./19_realtime_detrs_and_vit.md)
2. [17 — Large Detectors](./17_large_detectors.md)
3. [42 — Label-Efficient Learning & Distillation](./42_label_efficient_and_distillation.md)

### Path D: "I don't have labels — zero-shot, grounding, LLMs"

1. [20 — Open-Vocabulary Detection](./20_open_vocabulary.md)
2. [21 — LLM-Based Detection](./21_llm_based_detection.md)
3. [49 — Hands-On: Zero-Shot → Edge](./49_handson_zero_shot_to_edge.md)

### Path E: "My detector is broken"

1. [50 — The Debugging Playbook](./50_debugging_playbook.md) — symptom-indexed
2. [08 — Metrics](./08_metrics.md) — make sure the number itself is right
3. [05 — Label Assignment](./05_label_assignment.md) and [41 — Small Objects](./41_small_objects.md) — the two most common root causes

## 🧰 Companion Code

```bash
cd code
pip install -r requirements.txt
pytest                                   # 32 tests: odlab vs torchvision / pycocotools
python -m odlab.train --epochs 30        # TinyYOLO on synthetic shapes, CPU, ~10 minutes
python -m odlab.train --epochs 30 --reg-max 1 --end2end   # YOLO26-style: no DFL, no NMS
```

See [code/README.md](https://github.com/karam-nus/object-detection/blob/main/code/README.md) for the module ↔ chapter map.

---

*Part of a knowledge base published at [karam-nus.github.io](https://karam-nus.github.io). Companion repos: [vision](https://karam-nus.github.io/vision), [quantization](https://karam-nus.github.io/quantization), [metrics](https://karam-nus.github.io/metrics), [inference](https://karam-nus.github.io/inference), [model-optimization](https://karam-nus.github.io/model-optimization), [training](https://karam-nus.github.io/training).*
