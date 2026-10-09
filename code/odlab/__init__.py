"""odlab — readable reference implementations for the Object Detection guide.

Modules map to chapters:
    boxes    Ch 02-03  box formats, IoU / GIoU / DIoU / CIoU / EIoU / SIoU / NWD
    anchors  Ch 04     grid points, anchor boxes, k-means anchors, best possible recall
    assign   Ch 05     max-IoU, ATSS, SimOTA, Task-Aligned (+STAL, topk2), Hungarian
    losses   Ch 06     focal, QFL, VFL, IoU losses, DFL, L1 distances
    nms      Ch 07     greedy / batched / soft / DIoU / matrix NMS, WBF, top-k (NMS-free)
    metrics  Ch 08     COCO AP/AR (pycocotools-exact), VOC AP, PR, F1, error breakdown
    augment  Ch 10,28  letterbox, HSV, affine, mosaic, mixup
    data     Ch 09     synthetic dataset, YOLO/COCO label conversion
    model    Ch 39     TinyYOLO + DetectionLoss (DFL or L1 head, optional NMS-free head)
    train    Ch 39     end-to-end training/eval loop on CPU
"""

__version__ = "1.0.0"
