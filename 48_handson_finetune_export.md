---
title: "Chapter 48 — Hands-On: Fine-Tune, Evaluate, Export"
---

[← Back to Table of Contents](./README.md)

# Chapter 48 — Hands-On: Fine-Tune, Evaluate, Export

> *"A fair comparison is one where the only thing that differs is the thing you are comparing."*

## Overview

This chapter is a lab protocol. You fine-tune two strong real-time detectors from different families,
**YOLO26** (Ultralytics, CNN, AGPL-3.0) and **RF-DETR** (Roboflow, DINOv2-based DETR, Apache-2.0 for
N–L), on the same custom dataset; evaluate both with the **same evaluator** on the **same frozen test
split**; export both; and benchmark both on the same hardware with the full pipeline. Every step uses
the tools and pitfalls of earlier chapters. The protocol needs a GPU for training (a T4-class GPU or
Colab is enough for small datasets). Results are yours to fill in: a template is at the end.

<div class="diagram">
<div class="diagram-title">The protocol</div>
<div class="flow-h">
  <div class="flow-node">1 · dataset<small>splits frozen</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node blue">2 · two formats<small>YOLO txt + COCO JSON</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node purple">3 · fine-tune both<small>default recipes</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node accent">4 · one evaluator<small>pycocotools on JSON</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node green">5 · export · benchmark<small>end to end</small></div>
</div>
</div>

---

## Step 1: choose and freeze the dataset

Use your own data, or a public dataset from RF100-VL / Roboflow Universe so others can reproduce it.
Requirements for a meaningful comparison:

- **Three splits**: train, validation (for checkpoint selection) and **test** (touched once, at the end).
- **Split by source** (video, scene, day) to avoid leakage (Chapter 40).
- **At least a few hundred test instances per class you care about**, or report bootstrap intervals
  (Chapter 33).
- Run the dataset health checklist (Chapter 27): class histogram per split, box-size histogram at the
  training resolution, background count, visual spot check.

Record a hash of the split files. Every number in the final table refers to this exact split.

---

## Step 2: one dataset, two formats

Ultralytics trains on YOLO txt labels; RF-DETR on COCO JSON (or YOLO format, which recent versions
auto-detect). Convert once, from one source of truth, and verify the round trip:

```python
# YOLO txt -> COCO JSON for one split (boxes in absolute xywh, category ids from 0 or 1, consistently)
import json, glob, os
from PIL import Image

def yolo_split_to_coco(img_dir, lbl_dir, names, out_json):
    images, anns, aid = [], [], 1
    for i, p in enumerate(sorted(glob.glob(f"{img_dir}/*"))):
        w, h = Image.open(p).size
        images.append({"id": i, "file_name": os.path.basename(p), "width": w, "height": h})
        lp = f"{lbl_dir}/{os.path.splitext(os.path.basename(p))[0]}.txt"
        for line in open(lp).read().split("\n") if os.path.exists(lp) else []:
            if not line.strip(): continue
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            anns.append({"id": aid, "image_id": i, "category_id": int(c), "bbox": [x, y, bw * w, bh * h],
                         "area": bw * w * bh * h, "iscrowd": 0}); aid += 1
    cats = [{"id": k, "name": v} for k, v in names.items()]
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(out_json, "w"))
```

Check: the number of boxes per class must be identical in both formats, and drawing ten random images
from each format must give identical boxes. `odlab.data` has tested converters for the reverse direction
(`coco_to_yolo`).

---

## Step 3: fine-tune both with their default recipes

Use each framework's **recommended fine-tuning recipe**, not a recipe tuned for one of them. Match what
can be matched: input resolution class (≈ 640 or the model's native size), the same epoch budget or
"until convergence with early stopping", and the same starting point type (COCO-pretrained weights).

```python
# YOLO26 (Ultralytics)
from ultralytics import YOLO
yolo = YOLO("yolo26s.pt")
yolo.train(data="data.yaml", epochs=100, imgsz=640, patience=30, seed=0)   # optimizer=auto (Chapter 31)
```

```python
# RF-DETR (pip install rfdetr)
from rfdetr import RFDETRSmall
det = RFDETRSmall()
det.train(dataset_dir="dataset_coco/", epochs=100, batch_size=4, grad_accum_steps=4, output_dir="rfdetr_s")
```

RF-DETR's documentation recommends keeping the effective batch at 16 (for example `batch_size=4,
grad_accum_steps=4` on a T4). Note the model sizes you compare. YOLO26s (9.5M params, 2.5 ms on a T4)
and RF-DETR-S (32.1M params, 3.5 ms) are close in latency, not in parameters (Chapter 34). Pick pairs
by your constraint.

For small datasets run **2–3 seeds** per model. Training time is a result too; record it.

---

## Step 4: evaluate both with one evaluator

Do not compare `yolo.val()` with RF-DETR's internal metric. Produce COCO-format predictions from both
on the **test split** and score them with pycocotools (or `odlab.metrics.coco_evaluate`, which matches
it exactly):

```python
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def score(gt_json, dets):                     # dets: [{"image_id", "category_id", "bbox": [x,y,w,h], "score"}]
    gt = COCO(gt_json); dt = gt.loadRes(dets)
    e = COCOeval(gt, dt, "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    return e.stats                            # AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl
```

Generate the predictions at **conf 0.001** with each model's standard post-processing (NMS or one-to-one),
up to 100 detections per image (COCO's maxDets). Map class indices to the same `category_id`s. Then
report, per model:

- the twelve COCO numbers, with bootstrap intervals if the test set is small,
- per-class AP and per-class recall at the deployment threshold,
- a TIDE-style error breakdown (`odlab.metrics.error_breakdown` gives classification, localisation,
  duplicate, background and missed counts at a threshold),
- for YOLO26, both heads (`nms=None` and `nms=False`).

---

## Step 5: export and benchmark end to end

```python
yolo.export(format="onnx", nms=False)          # or format="engine" on the target GPU (Chapter 36)
det.export()                                   # RF-DETR ONNX export (pip install "rfdetr[onnx]")
```

Then, on the deployment hardware and runtime:

1. **Validate the exported artifacts** on the test split with the same evaluator (accuracy can move in
   export, especially at FP16/INT8).
2. **Measure end-to-end latency** with your production pre/post-processing: decode → letterbox (or the
   model's resize) → inference → post-processing, median and p99, after warm-up (Chapter 46).
3. Run the **golden-image test** (Chapter 36) for each artifact.
4. If deploying INT8, quantise both with the same tool and calibration set, and re-validate
   (Chapter 45).

---

## Results template

| Field | YOLO26-? | RF-DETR-? |
|---|---|---|
| Dataset, split hash, # test images / instances | | |
| Starting weights | | |
| Training: epochs, batch (effective), GPU, wall-clock time, seeds | | |
| Test AP / AP50 / AP75 (± bootstrap 95% CI) | | |
| APs / APm / APl | | |
| Per-class AP (worst three classes) | | |
| Recall at the deployment threshold (per class) | | |
| TIDE: cls / loc / dup / bkg / miss | | |
| Export format, precision, head | | |
| Exported-artifact AP (same evaluator) | | |
| End-to-end latency median / p99 on target | | |
| Parameters / model file size | | |
| Licence of code and weights | | |

What to look for when reading your table:

- **Where the AP difference comes from**: AP75 vs AP50 (localisation), APs (small objects), particular
  classes (data issues), TIDE background errors (missing labels).
- **Whether the difference survives export** to your runtime and precision.
- **Whether it is larger than seed variation.**
- **What it costs**: latency on your hardware, memory, licence.

---

## Key Takeaways

- Freeze the splits, convert once from one source of truth, and verify the conversion.
- Fine-tune each model with its own recommended recipe; match resolution class, epoch budget and
  starting-point type.
- Score both with pycocotools on the same test JSON, at conf 0.001 and maxDets 100, with bootstrap
  intervals and per-class results.
- Benchmark the exported artifacts end to end on the target, and validate their accuracy after export.
- Fill the template. A comparison without training time, latency on target, licence and seed variation
  is incomplete.

## Check Yourself

<details class="check"><summary>Why not simply compare the mAP that each framework prints at the end of training?</summary>
The frameworks use different evaluators (matching, interpolation, maxDets, crowd handling), may evaluate
on the validation split used for checkpoint selection, and may use different confidence thresholds or
post-processing. Scoring both models' predictions on the same frozen test JSON with one evaluator removes
those differences.</details>

<details class="check"><summary>RF-DETR-S beats YOLO26s by 4 AP on your test set, but runs at 60% of the speed on your Jetson. How do you decide?</summary>
Check whether the YOLO26 size that matches RF-DETR-S's Jetson latency (perhaps YOLO26m) closes the gap,
compare at your deployment metric (recall at threshold, false alarms) rather than AP, and include
licence, export support on the device and INT8 behaviour. Decide on the constraint that binds.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics train / val / export docs | Ultralytics | docs.ultralytics.com/modes | YOLO26 workflow |
| RF-DETR documentation (training, export) | Roboflow, 2025–2026 | rfdetr.roboflow.com | Dataset format, batch guidance, ONNX export |
| RF100-VL | Robicheaux et al., 2025 | arXiv:2505.20612 | Multi-domain fine-tuning benchmark |
| pycocotools | cocodataset | github.com/cocodataset/cocoapi | Reference evaluator |
| `odlab.metrics` (`coco_evaluate`, `error_breakdown`), `odlab.data` | this book | code/odlab | Evaluator and converters |

---

**Next:** [Chapter 49 — Hands-On: Zero-Shot → Edge](./49_handson_zero_shot_to_edge.md) — from no labels
to a nano model on a device.
