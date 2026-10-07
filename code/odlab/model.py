"""TinyYOLO: a ~300-line modern YOLO you can read in one sitting and train on a CPU.

Architecture (YOLOv8/11-style, scaled down):
    backbone  Conv-BN-SiLU stem, stride-2 convs, CSP blocks, SPPF   -> P3 (/8), P4 (/16), P5 (/32)
    neck      PAN: top-down (upsample + concat + CSP) then bottom-up (stride-2 conv + concat + CSP)
    head      decoupled, anchor-free: per level a box branch (4 * reg_max) and a class branch (C)

Switches that reproduce the main design choices discussed in the book:
    reg_max=16  -> DFL box head (YOLOv8/11);   reg_max=1 -> direct distances + L1 (YOLO26)
    end2end=True -> extra one-to-one head on detached features, trained with topk=1 assignment;
                    inference without NMS (YOLOv10 / YOLO26 e2e)
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import make_grid_points
from .assign import task_aligned_assign
from .boxes import ciou
from .losses import bbox2dist, dfl_decode, dfl_loss, dist2bbox
from .nms import batched_nms, topk_select

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv(nn.Module):
    """Conv2d -> BatchNorm -> SiLU. The atom of every YOLO since v5."""

    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1, self.cv2 = Conv(c, c, 3), Conv(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP block with all intermediate outputs concatenated (YOLOv8's C2f).

    Split channels in two; push one half through n bottlenecks, keeping every intermediate
    map; concatenate everything and fuse with a 1x1 conv. More gradient paths, cheap.
    """

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: three chained 5x5 max-pools == pools of 5, 9, 13."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1, self.cv2 = Conv(c1, c_, 1), Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Head(nn.Module):
    """Decoupled anchor-free head: separate box and class towers per pyramid level."""

    def __init__(self, nc, chs, reg_max, img_size):
        super().__init__()
        self.nc, self.reg_max = nc, reg_max
        c2 = max(16, chs[0] // 4, 4 * reg_max)
        c3 = max(chs[0], min(nc, 100))
        self.box = nn.ModuleList(nn.Sequential(Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for c in chs)
        self.cls = nn.ModuleList(nn.Sequential(Conv(c, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for c in chs)
        self.img_size = img_size

    def bias_init(self, strides):
        # class prior: ~5 objects per image spread over (img/stride)^2 cells  (Ultralytics' init)
        for b, c, s in zip(self.box, self.cls, strides):
            b[-1].bias.data[:] = 1.0
            c[-1].bias.data[: self.nc] = math.log(5 / self.nc / (self.img_size / s) ** 2)

    def forward(self, feats):
        boxes = torch.cat([b(f).flatten(2) for b, f in zip(self.box, feats)], 2)  # [B, 4*reg_max, A]
        scores = torch.cat([c(f).flatten(2) for c, f in zip(self.cls, feats)], 2)  # [B, nc, A]
        return boxes.transpose(1, 2), scores.transpose(1, 2)


class TinyYOLO(nn.Module):
    def __init__(self, nc=3, width=(16, 32, 64, 128, 256), depth=1, reg_max=16, end2end=False, img_size=256):
        super().__init__()
        w0, w1, w2, w3, w4 = width
        self.nc, self.reg_max, self.end2end = nc, reg_max, end2end
        self.strides = (8, 16, 32)
        self.stem = nn.Sequential(Conv(3, w0, 3, 2), Conv(w0, w1, 3, 2), C2f(w1, w1, depth))  # /4
        self.p3 = nn.Sequential(Conv(w1, w2, 3, 2), C2f(w2, w2, depth))  # /8
        self.p4 = nn.Sequential(Conv(w2, w3, 3, 2), C2f(w3, w3, depth))  # /16
        self.p5 = nn.Sequential(Conv(w3, w4, 3, 2), C2f(w4, w4, depth), SPPF(w4, w4))  # /32
        self.td4 = C2f(w4 + w3, w3, depth, shortcut=False)
        self.td3 = C2f(w3 + w2, w2, depth, shortcut=False)
        self.down3 = Conv(w2, w2, 3, 2)
        self.bu4 = C2f(w2 + w3, w3, depth, shortcut=False)
        self.down4 = Conv(w3, w3, 3, 2)
        self.bu5 = C2f(w3 + w4, w4, depth, shortcut=False)
        self.head = Head(nc, (w2, w3, w4), reg_max, img_size)
        self.head.bias_init(self.strides)
        self.head_o2o = copy.deepcopy(self.head) if end2end else None

    def features(self, x):
        c3 = self.p3(self.stem(x))
        c4 = self.p4(c3)
        c5 = self.p5(c4)
        t4 = self.td4(torch.cat((F.interpolate(c5, scale_factor=2.0, mode="nearest"), c4), 1))
        n3 = self.td3(torch.cat((F.interpolate(t4, scale_factor=2.0, mode="nearest"), c3), 1))
        n4 = self.bu4(torch.cat((self.down3(n3), t4), 1))
        n5 = self.bu5(torch.cat((self.down4(n4), c5), 1))
        return [n3, n4, n5]

    def forward(self, x):
        feats = self.features(x)
        boxes, scores = self.head(feats)
        out = {"feats": [f.shape[-2:] for f in feats], "boxes": boxes, "scores": scores}
        if self.head_o2o is not None:
            # one-to-one head sees detached features: its gradients must not reshape the backbone
            b2, s2 = self.head_o2o([f.detach() for f in feats])
            out["o2o"] = {"boxes": b2, "scores": s2}
        return out

    # ------------------------------------------------------------------ decoding

    def points(self, out):
        return make_grid_points(out["feats"], self.strides)

    def decode_boxes(self, raw, points, strides):
        """Raw head output -> xyxy pixels. DFL: expectation over bins; reg_max=1: identity."""
        dist = dfl_decode(raw, self.reg_max) if self.reg_max > 1 else raw
        return dist2bbox(points[None], dist * strides[None])

    @torch.no_grad()
    def predict(self, x, conf=0.25, iou=0.7, max_det=300, use_o2o=None):
        """Images -> list of dicts(boxes, scores, labels). NMS-free when the one-to-one head is used."""
        out = self.forward(x)
        pts, st = self.points(out)
        use_o2o = self.end2end if use_o2o is None else use_o2o
        branch = out["o2o"] if use_o2o else out
        boxes = self.decode_boxes(branch["boxes"], pts, st)
        probs = branch["scores"].sigmoid()
        results = []
        for b, p in zip(boxes, probs):
            if use_o2o:
                idx, lab, sc = topk_select(p, max_det)
                keep = sc > conf
                results.append({"boxes": b[idx][keep], "scores": sc[keep], "labels": lab[keep]})
                continue
            sc, lab = p.max(1)
            keep = sc > conf
            b, sc, lab = b[keep], sc[keep], lab[keep]
            k = batched_nms(b, sc, lab, iou)[:max_det]
            results.append({"boxes": b[k], "scores": sc[k], "labels": lab[k]})
        return results


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class DetectionLoss:
    """YOLOv8-style loss: BCE(cls, TAL soft targets) + CIoU + DFL (or L1 when reg_max == 1).

    Every term is normalised by the sum of the soft targets, so the loss scale does not
    depend on how many positives the assigner happened to pick.
    Gains default to Ultralytics' box=7.5, cls=0.5, dfl=1.5.
    """

    def __init__(self, model: TinyYOLO, topk=10, topk2=None, alpha=0.5, beta=6.0, box=7.5, cls=0.5, dfl=1.5, small_side_floor=None):
        self.m = model
        self.topk, self.topk2, self.alpha, self.beta = topk, topk2, alpha, beta
        self.gain = (box, cls, dfl)
        self.floor = small_side_floor

    def branch_loss(self, branch, out, gt_boxes, gt_labels):
        pts, st = self.m.points(out)
        pred_boxes = self.m.decode_boxes(branch["boxes"], pts, st)
        probs = branch["scores"].sigmoid()
        tot = torch.zeros(3)
        n_images = len(gt_boxes)
        for i in range(n_images):
            a = task_aligned_assign(
                probs[i].detach(),
                pred_boxes[i].detach(),
                pts,
                gt_boxes[i],
                gt_labels[i],
                topk=self.topk,
                alpha=self.alpha,
                beta=self.beta,
                topk2=self.topk2,
                small_side_floor=self.floor,
            )
            ts = a["target_scores"]
            norm = ts.sum().clamp(min=1)
            l_cls = F.binary_cross_entropy_with_logits(branch["scores"][i], ts, reduction="sum") / norm
            fg = a["fg"]
            if fg.any():
                w = ts[fg].sum(-1)
                l_box = ((1 - ciou(pred_boxes[i][fg], a["target_boxes"][fg])) * w).sum() / norm
                tgt = bbox2dist(pts[fg], a["target_boxes"][fg]) / st[fg]
                if self.m.reg_max > 1:
                    l_dist = (dfl_loss(branch["boxes"][i][fg], tgt.clamp(max=self.m.reg_max - 1.01), self.m.reg_max) * w).sum() / norm
                else:  # YOLO26-style: L1 on distances normalised by image size
                    img = float(self.m.head.img_size)
                    l_dist = (((branch["boxes"][i][fg] - tgt) * st[fg] / img).abs().mean(-1) * w).sum() / norm
            else:
                l_box = l_dist = branch["scores"][i].sum() * 0
            tot = tot + torch.stack((l_box, l_cls, l_dist))
        return tot * torch.tensor(self.gain) / n_images

    def __call__(self, out, gt_boxes, gt_labels, o2m_weight=0.8):
        """Returns (scalar loss, per-term tensor[box, cls, dist]).

        With an end-to-end model the one-to-many (topk=self.topk) and one-to-one (topk=1)
        branch losses are mixed: total = w * L_o2m + (1 - w) * L_o2o. Ultralytics' YOLO26
        decays w linearly from 0.8 to 0.1 over training ("ProgLoss").
        """
        l_o2m = self.branch_loss(out, out, gt_boxes, gt_labels)
        if "o2o" not in out:
            return l_o2m.sum(), l_o2m.detach()
        saved = self.topk, self.topk2
        self.topk, self.topk2 = 1, None
        l_o2o = self.branch_loss(out["o2o"], out, gt_boxes, gt_labels)
        self.topk, self.topk2 = saved
        total = o2m_weight * l_o2m + (1 - o2m_weight) * l_o2o
        return total.sum(), l_o2o.detach()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
