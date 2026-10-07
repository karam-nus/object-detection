---
title: "Chapter 35 — YOLO Axis 9: Ease of Use, Ecosystem & Licensing"
---

[← Back to Table of Contents](./README.md)

# Chapter 35 — YOLO Axis 9: Ease of Use, Ecosystem & Licensing

> *"More projects pick a detector for its `pip install` and its licence than for its AP."*

## Overview

Two detectors with the same accuracy can differ by weeks of engineering in how long it takes to train,
export, deploy and maintain them, and by a legal review in whether you may ship them at all. This
chapter treats ease of use as a measurable axis (install, API, tasks, export targets, documentation,
maintenance, reproducibility), compares the main codebases on it, and explains what the licences in the
YOLO world require. It is practical guidance for engineers. It is not legal advice.

<div class="diagram">
<div class="diagram-title">The two questions this axis answers</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card blue"><div class="card-title">Can my team ship it?</div><div class="card-desc">Install · one-line training · tasks beyond boxes · export to <em>my</em> hardware · tracking · docs · active maintenance · reproducible checkpoints</div></div>
  <div class="diagram-card red"><div class="card-title">May my company ship it?</div><div class="card-desc">Code licence · weights licence · what triggers obligations (distribution, network use) · commercial licence options · dataset terms behind the weights</div></div>
</div>
</div>

---

## The Ultralytics workflow in twelve lines

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")                                    # or .yaml to build from scratch
model.train(data="widgets.yaml", epochs=100, imgsz=640)       # Chapters 27–32
metrics = model.val(data="widgets.yaml", split="test")         # Chapter 33
for r in model.predict("video.mp4", stream=True, conf=0.4):   # generator: constant memory
    boxes = r.boxes.xyxy, r.boxes.conf, r.boxes.cls            # tensors; r.plot() draws them
model.track("video.mp4", tracker="bytetrack.yaml")            # detection + tracking (Chapter 43)
model.export(format="onnx", nms=False)                         # Chapter 36
model.benchmark(data="widgets.yaml", imgsz=640)                # speed + accuracy per export format
YOLO("yolo26s.onnx").predict("img.jpg")                        # exported models load the same way
print(model.ckpt["train_args"])                                # every argument the checkpoint was trained with
```

The same five verbs work for detection, segmentation, pose, OBB, classification, semantic segmentation
and depth (`yolo26n-seg.pt`, `-pose`, `-obb`, `-cls`, `-sem`, `-depth`), and for RT-DETR, YOLO-World,
YOLOE and SAM models loaded through their own classes. The CLI mirrors the API
(`yolo detect train data=... model=...`). Passing a **list of datasets** to `train` fine-tunes the same
base weights on each in turn and writes a comparison (`MultiTrainer`). This is a quick way to benchmark
a model across several of your datasets.

### What the convenience hides

| Hidden by default | Where it bites | Where this book opens it |
|---|---|---|
| Optimiser and `lr0` chosen automatically (`optimizer=auto`) | Your `lr0` is ignored | Chapter 31 |
| Assignment, loss gains, STAL, ProgLoss | "Loss is flat" debugging | Chapter 30 |
| Built-in metric ≠ pycocotools | Comparing with papers | Chapter 33 |
| YOLO26 default head is one-to-many + NMS | Expecting NMS-free output from a default export | Chapter 36 |
| Defaults change between releases (optimiser, fitness, `nms`, export args such as `quantize` replacing `half`/`int8`) | A re-run months later gives different numbers | Pin the version |

**Pin the package version** in every project (`ultralytics==X.Y.Z`) and record it with the results.
Checkpoints store their `train_args` and the git commit that produced them (`ckpt["git"]`), so an old
run can be reproduced exactly, but only with the code it was trained on.

---

## Codebases compared

| Codebase | Install | Detectors | Tasks | Export | Licence (code / weights) | Activity (2026) |
|---|---|---|---|---|---|---|
| **Ultralytics** (`ultralytics`) | `pip` | YOLOv3u–v12, YOLO26, RT-DETR, YOLO-World, YOLOE | detect, seg, pose, OBB, cls, semantic, depth, tracking | ~20 formats incl. TensorRT, OpenVINO, CoreML, LiteRT, NCNN, RKNN, Hailo, QNN, ExecuTorch | AGPL-3.0 / AGPL-3.0, enterprise licence available | Very active |
| **YOLOv5** repo | git clone | YOLOv5 | detect, seg, cls | ONNX, TensorRT, CoreML, TFLite, … | AGPL-3.0 | Maintenance; superseded by the `ultralytics` package (`yolov5nu` etc.) |
| **WongKinYiu** (YOLOv7, YOLOv9) | git clone | v7, v9 | detect, seg (v7 pose) | ONNX via scripts | GPL-3.0 | Low |
| **Meituan YOLOv6** | git clone | v6 | detect, seg | ONNX, TensorRT, OpenVINO | GPL-3.0 | Low |
| **YOLOX** | git clone / pip | YOLOX | detect | ONNX, TensorRT, NCNN, OpenVINO | Apache-2.0 | Low |
| **PaddleDetection** | pip / clone | PP-YOLOE(+), RT-DETR, PicoDet | detect, seg, keypoints, MOT | Paddle Inference, ONNX | Apache-2.0 | Moderate |
| **MMDetection / MMYOLO** | pip | very many (RTMDet, YOLO variants, DINO, …) | detect, seg | MMDeploy | Apache-2.0 | No MMDetection release since v3.3.0 (January 2024) |
| **SuperGradients** | pip | YOLO-NAS | detect, pose | ONNX, TensorRT | Apache-2.0 code / **non-commercial weights** | Little development since Deci was acquired by NVIDIA (2024) |
| **RF-DETR** (`rfdetr`) | `pip` | RF-DETR N…2XL | detect, seg | ONNX, TensorRT | Apache-2.0 (N–L); **PML-1.0** for XL/2XL | Very active |
| **D-FINE / DEIM / DEIMv2** repos | git clone | DETR family | detect | ONNX, TensorRT | Apache-2.0 | Active (research cadence) |
| **Hugging Face `transformers`** | `pip` | DETR, Deformable/Conditional DETR, RT-DETR(v2), D-FINE, Grounding DINO, OWLv2, … | detect (+ zero-shot) | ONNX via Optimum | Apache-2.0 (weights vary) | Very active |

Supporting libraries most teams end up using regardless of detector: **supervision** (Roboflow, MIT:
drawing, tracking glue, zone counting, metrics), **SAHI** (MIT, sliced inference for small objects,
Chapter 41), **CVAT / Label Studio** (annotation), **FiftyOne** (dataset curation and error analysis).

---

## Ease of use as a checklist

Score candidates for your project rather than in general:

| Question | Why it matters | How to check in an afternoon |
|---|---|---|
| Does it export to **my** runtime with post-processing included? | The most common late surprise | Export a pretrained model, run it on the target, compare boxes with PyTorch |
| Does it train on my format without conversion scripts? | Conversion bugs are silent label bugs | Train 5 epochs on 50 images; visualise `train_batch0` |
| Can I resume, change the class list and fine-tune from a checkpoint? | Continual updates | Fine-tune with a new class added |
| Are hyperparameters and code version recorded in the checkpoint? | Reproducibility, audits | Inspect the checkpoint |
| Is someone fixing issues this year? | Security fixes, new runtimes, new PyTorch versions | Release history, open issues answered |
| Is there a tracking / counting / ROI pipeline? | Most products need more than boxes | Run the tracker on a sample video |

---

## Licensing

### The licences you will meet

| Licence | Key obligation | Trigger | Typical YOLO-world use |
|---|---|---|---|
| **Apache-2.0** | Keep notices; state changes; includes a patent licence | Distribution | YOLOX, RTMDet, PP-YOLOE, RT-DETR, D-FINE, DEIM, LW-DETR, RF-DETR N–L, `transformers` |
| **MIT / BSD** | Keep the notice | Distribution | supervision, SAHI, many utilities |
| **GPL-3.0** | Release the source of the whole combined work under GPL-3.0 | **Distribution** of the software (shipping a binary, a device, an app) | YOLOv6, YOLOv7, YOLOv9, Gold-YOLO |
| **AGPL-3.0** | As GPL-3.0, **plus** users who interact with it **over a network** must be offered the source | Distribution **or network use** (SaaS, API) | YOLOv3u/v5/v8/11/26 (Ultralytics), YOLOv10, YOLOv12, YOLOv13 |
| **Non-commercial weights** | No commercial use of the weights | Any commercial use | YOLO-NAS pre-trained weights |
| **PML-1.0** (Roboflow Platform Model License) | Platform terms; requires a Roboflow account | Use of those models | RF-DETR XL and 2XL |

### What Ultralytics says about AGPL-3.0

Ultralytics states that its code **and its models** are AGPL-3.0, and that using Ultralytics YOLO code,
architectures, training pipelines or trained and fine-tuned models in a product requires either
releasing the whole project under AGPL-3.0 or buying an **Ultralytics Enterprise License**. Its licence
page says this applies even when you train from scratch without the pre-trained weights, use the model
only internally, or deploy it behind an API. That last reading (that weights *you* trained fall under
the licence of the training code) is Ultralytics' position. Whether it holds in a given jurisdiction is
a question for your lawyer. In practice, companies either open-source their application, buy the
enterprise licence, or choose an Apache-2.0 detector.

### Practical guidance

1. **Decide the licence constraint before benchmarking.** If the product cannot be AGPL and there is
   no budget for a commercial licence, remove those models from the shortlist on day one.
2. **Weights and code can have different licences.** YOLO-NAS (Apache-2.0 code, non-commercial weights)
   and RF-DETR (Apache-2.0 for N–L, PML-1.0 for XL/2XL) are the clearest cases.
3. **Training data has terms too.** Pre-trained weights inherit questions from their training data
   (COCO's images carry individual Flickr licences; large web-scraped corpora and research-only datasets
   have their own terms). Check the dataset terms for the checkpoint you fine-tune from if your legal
   team cares about provenance.
4. **Re-implementing an architecture** in an Apache-2.0 codebase is a common route (MMYOLO,
   PaddleDetection and others contain YOLO-style models). The architecture idea is not what a copyright
   licence covers. Copied code and weights are.
5. **Keep a bill of materials**: model, version, code licence, weights licence, dataset licences. It is
   the first thing a customer's security or legal review asks for.

<div class="callout warn"><span class="callout-title">Not legal advice</span>This section summarises
public licence texts and vendors' own statements to help you ask the right questions. Licence
interpretation depends on the jurisdiction and the facts of your deployment. Ask a lawyer before you
ship.</div>

---

## Choosing on this axis

| Situation | Usually the best fit |
|---|---|
| Prototype fast, open-source product, or enterprise licence budget | Ultralytics (YOLO26 / YOLO11) |
| Closed-source product, no licence budget, GPU or capable NPU | RF-DETR N–L, D-FINE / DEIM / DEIMv2, RT-DETR (Apache-2.0) |
| Closed-source, CPU or small NPU, permissive licence required | YOLOX, RTMDet, PP-YOLOE / PicoDet, or an Apache-2.0 re-implementation of a YOLO architecture |
| Research on detector internals | MMDetection-style modular codebases (accepting their maintenance status), or the original research repos |
| Many tasks in one pipeline (detect + seg + pose + track) | Ultralytics |
| Hugging Face-centric stack | `transformers` DETR-family models |

---

## Migration notes

- **YOLOv5 repo → `ultralytics` package.** `yolov5nu.pt` … `yolov5xu.pt` are YOLOv5 backbones with the
  anchor-free v8 head, trained in the new package. The old `yolov5s.pt` checkpoints do not load in the new
  package's training path. Dataset YAMLs and label files carry over unchanged.
- **YOLOv8 / YOLO11 → YOLO26.** Same API and dataset format. Check three things: the export output
  (one-to-many + NMS by default, `nms=False` for NMS-free `(N, 300, 6)`), the loss columns
  (`l1_loss` instead of `dfl_loss`), and CPU speed at m/l/x (Chapter 34).
- **Ultralytics → a permissive codebase.** Convert labels (YOLO txt → COCO JSON), re-tune the recipe
  (Chapter 31 is Ultralytics-specific), and re-validate with pycocotools so the numbers are comparable.

---

## Key Takeaways

- Ease of use is concrete: export to your runtime, train on your format, reproducible checkpoints and
  active maintenance. Check each in an afternoon on your own data and hardware.
- The Ultralytics package gives the shortest path from data to deployed model across many tasks and
  export targets. It hides choices (optimiser, head, metric) that you should know about, and its
  defaults change between releases. Pin versions.
- Licences: Apache/MIT are permissive; GPL-3.0 triggers on distribution; AGPL-3.0 also on network use.
  Ultralytics applies AGPL-3.0 to code and models and sells an enterprise licence.
- Weights can be licensed differently from code (YOLO-NAS, RF-DETR XL/2XL). Datasets behind weights
  have terms too.
- Decide the licence constraint before benchmarking. It removes candidates faster than any experiment.

## Check Yourself

<details class="check"><summary>Your startup offers a paid web API that runs a fine-tuned YOLO11 model on customer images. The application code is closed. Which licence clause matters, and what are the options?</summary>
AGPL-3.0's network clause: users interacting with the software over a network must be offered its
source, and Ultralytics states this covers its models and trained derivatives. The options are to release
the application under AGPL-3.0, buy an Ultralytics Enterprise License, or switch to an Apache-2.0
detector. Have a lawyer confirm the reading for your case.</details>

<details class="check"><summary>YOLO-NAS-S has Apache-2.0 code. Can you ship the pre-trained YOLO-NAS-S weights in a commercial product?</summary>
No. The YOLO-NAS pre-trained weights are released under a separate non-commercial licence. Training
your own weights with the Apache-2.0 code from a permissively licensed initialisation is a different
question, worth checking against the licence terms.</details>

<details class="check"><summary>A colleague upgrades the ultralytics package and the same training command gives 1 mAP less. Name three release-level changes that could explain it without any bug.</summary>
The automatic optimiser choice (for example SGD replaced by MuSGD), changed defaults (augmentation,
fitness definition used to pick best.pt, loss details such as STAL), and changed export or validation
defaults (the YOLO26 head selection, the `quantize` argument). Pin the version and diff the train_args
stored in the two checkpoints.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Ultralytics licensing page | Ultralytics | ultralytics.com/license | AGPL-3.0 scope, enterprise licence, position on trained models |
| Ultralytics docs: modes, export table, end-to-end guide, tasks | Ultralytics | docs.ultralytics.com | API, tasks, export formats, `nms` and `quantize` arguments |
| GNU GPL-3.0 and AGPL-3.0 texts | Free Software Foundation, 2007 | gnu.org/licenses | Distribution and network-use clauses |
| Apache License 2.0 | Apache Software Foundation, 2004 | apache.org/licenses/LICENSE-2.0 | Permissive terms, patent grant |
| RF-DETR README / docs | Roboflow, 2025–2026 | github.com/roboflow/rf-detr; rfdetr.roboflow.com | Apache-2.0 vs PML-1.0 split |
| YOLO-NAS / SuperGradients docs | Deci AI, 2023 | docs.deci.ai/super-gradients/YOLONAS.html | Weights licence |
| MMDetection releases | OpenMMLab | github.com/open-mmlab/mmdetection | Last release v3.3.0 (Jan 2024) |
| "Why AGPL-3.0 is a risk for computer vision teams" | Roboflow blog | blog.roboflow.com | A competitor's reading of AGPL for CV (read critically) |
| supervision; SAHI | Roboflow; Akyon et al., 2022 | github.com/roboflow/supervision; github.com/obss/sahi | Supporting libraries |

---

**Next:** [Chapter 36 — Axis 10: Export & Deployment](./36_yolo_export_and_deployment.md) — from `.pt`
to the runtime your product actually uses.
