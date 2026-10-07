# odlab — the book's companion code

`odlab` is a small, tested Python package that implements the core of modern object detection in readable
PyTorch: box maths and the IoU family, NMS variants, a COCO-exact evaluator, anchor tools, every major
label-assignment strategy, the detection losses, YOLO-style augmentation, label-format conversion, and
**TinyYOLO**, a complete modern YOLO that trains on a laptop CPU in about ten minutes.

```bash
cd code
pip install -r requirements.txt
pytest -q                                                   # 32 tests
python -m odlab.train --epochs 30                           # YOLOv8-style: DFL head + NMS
python -m odlab.train --epochs 30 --reg-max 1 --end2end     # YOLO26-style: L1 head + one-to-one head
```

Requirements: Python 3.10+, PyTorch, NumPy, SciPy; `pycocotools` and `torchvision` are used by the tests
to check `odlab` against the reference implementations.

## Module ↔ chapter map

| Module | What it contains | Chapters |
|---|---|---|
| `odlab/boxes.py` | Box formats and conversions; IoU, GIoU, DIoU, CIoU, EIoU, SIoU, NWD; pairwise and element-wise | 2, 3 |
| `odlab/anchors.py` | Grid points, anchor boxes, k-means anchors, best possible recall | 4 |
| `odlab/assign.py` | Max-IoU, ATSS, SimOTA, TAL (with YOLO26 `topk2` and STAL options), Hungarian matching and DETR costs | 5, 30 |
| `odlab/losses.py` | Sigmoid focal, QFL, VFL, soft-target BCE, IoU losses, DFL encode/decode/loss, L1 distance loss | 6 |
| `odlab/nms.py` | Greedy, batched, Soft-NMS, DIoU-NMS, Matrix NMS, weighted box fusion, top-k selection | 7 |
| `odlab/metrics.py` | COCO evaluation (matches pycocotools), PR curves, VOC AP, best-F1 threshold, TIDE-style error breakdown | 8, 33 |
| `odlab/augment.py` | Letterbox and its inverse, flips, HSV, random affine with box filtering, mosaic, mixup | 10, 28 |
| `odlab/data.py` | `SyntheticShapes` dataset; YOLO txt and COCO JSON conversion | 27, 39 |
| `odlab/model.py` | TinyYOLO (Conv, C2f, SPPF, PAN, decoupled head, optional one-to-one head) and its loss | 29, 30, 39 |
| `odlab/train.py` | Training loop with warm-up, cosine schedule, EMA, gradient clipping, evaluation | 31, 39 |

## Tests

| Test file | What it checks |
|---|---|
| `tests/test_boxes_nms.py` | IoU family against torchvision and the Chapter 3 worked example; GIoU gradient on disjoint boxes; NWD scale behaviour; NMS against `torchvision.ops.nms`; Soft-NMS, DIoU-NMS, Matrix NMS, WBF, top-k; 8,400 grid points at 640; anchors and k-means |
| `tests/test_metrics.py` | `coco_evaluate` against pycocotools on random predictions with and without crowd regions; perfect predictions give AP 1; VOC/COCO interpolation on a textbook curve; VOC mAP, best-F1 and error breakdown |
| `tests/test_assign_losses_model.py` | Assigner properties (max-IoU coverage, ATSS and TAL positives inside boxes, STAL rescue of tiny boxes, SimOTA dynamic k, one-to-one Hungarian matching); focal loss reduces to BCE; DFL expectation; letterbox and YOLO-txt round trips; mosaic boxes; TinyYOLO shapes and a short overfitting run |

## Results reproduced in the book

| Run | Command | Result | Chapter |
|---|---|---|---|
| DFL + NMS, 30 epochs | `python -m odlab.train --epochs 30` | AP 0.833, AP50 0.900 | 39 |
| L1 + one-to-one, 30 epochs | `python -m odlab.train --epochs 30 --reg-max 1 --end2end` | one-to-one AP 0.787; one-to-many + NMS 0.810 | 37, 39 |
| Learning rate × seed sweep, 12 epochs | see Chapter 32 | 2e-3 best; seed spread up to 2.4 AP at 8e-3 | 32 |

The book's other measurements (Ultralytics model anatomy, ONNX export, CPU pipeline timings, INT8
post-training quantisation, auto-label quality) use the Ultralytics package and ONNX Runtime; the scripts
are described in the chapters that report them (29, 33, 36, 44, 45, 46, 49).
