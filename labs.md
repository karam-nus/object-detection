---
title: "Interactive Labs"
---

[← Back to Table of Contents](./README.md)

# Interactive Labs

> *"Drag the box until the number surprises you. That is where the understanding is."*

Six small browser labs, each tied to a chapter. They run entirely in your browser (`assets/js/od.js`),
need no install, and use the same definitions as the tested `odlab` code. On the raw GitHub view the
labs do not run. Open the published site instead.

---

## 1. IoU and its family

Drag or resize either box. The readout shows IoU, GIoU, DIoU and CIoU, and how each behaves when the
boxes stop overlapping. **Try:** move the boxes apart until IoU is 0 and watch GIoU and DIoU keep
changing (they still give a gradient); make one box a thin sliver and see CIoU's aspect-ratio term.

<div class="lab" data-lab="iou"></div>

Chapter: [3 — IoU and Its Family](./03_iou_family.md).

---

## 2. NMS, one step at a time

A scene of overlapping candidate boxes with scores. Step through NMS one box at a time, change the IoU
threshold, and switch between greedy NMS, Gaussian Soft-NMS and DIoU-NMS. **Try:** set the threshold to
0.3 and see a true neighbouring object suppressed; set it to 0.9 and see duplicates survive.

<div class="lab" data-lab="nms"></div>

Chapter: [7 — Post-Processing](./07_post_processing.md).

---

## 3. From detections to AP

Add detections, click rows to flip them between true and false positive, and set the number of ground
truths. The lab builds the PR curve and computes AP three ways: VOC 11-point, all-point (area under the
envelope) and COCO 101-point.
**Try:** add one high-scoring false positive and watch AP fall more than recall suggests.

<div class="lab" data-lab="map"></div>

Chapters: [8 — Metrics](./08_metrics.md), [33 — YOLO Metrics & Validation](./33_yolo_metrics_and_validation.md).

---

## 4. Which grid points can see an object?

A box over a grid of stride 8, 16 or 32. Choose the candidate rule (centre inside the box as in
FCOS/TAL, SimOTA's centre radius, or YOLOv5's cell plus two neighbours) and toggle YOLO26's STAL floor.
The lab highlights the grid points that can become positives. **Try:** at stride 32, shrink the box to
about 20 px and slide it: the candidate count drops to zero between grid centres. Then switch on STAL
or change the rule.

<div class="lab" data-lab="grid"></div>

Chapters: [4 — Anchors, Points, Queries](./04_anchors_points_queries.md), [30 — Assignment & Loss](./30_yolo_assignment_and_loss.md), [41 — Small Objects](./41_small_objects.md).

---

## 5. Letterbox and its inverse

Enter an image size and a network size. The lab computes the scale and the padding, draws the padded
canvas, and supports the "auto" minimum-rectangle mode (padding only to the next multiple of 32).
**Try:** a 3840 × 2160 frame at 640, then multiply 30 px by the scale to see how big a 30-pixel object
becomes.

<div class="lab" data-lab="letterbox"></div>

Chapters: [10 — Augmentation](./10_augmentation.md), [28 — Preprocessing & Augmentation](./28_yolo_preprocessing_augmentation.md), [36 — Export & Deployment](./36_yolo_export_and_deployment.md).

---

## 6. The Detection Atlas

Every detector in the book on one accuracy–latency plot, filterable by tier, licence and NMS. Only rows
measured on a T4 with TensorRT FP16 are plotted together. It has its own page: [Detection Atlas](./atlas.md).

<div class="lab" data-lab="atlas"></div>

Chapters: [34 — Family Performance](./34_yolo_family_performance.md), [Appendix C — Model Index](./appendix_c_model_index.md).
