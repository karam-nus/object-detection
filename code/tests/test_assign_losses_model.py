import numpy as np
import torch

from odlab.anchors import make_anchor_boxes, make_grid_points
from odlab.assign import atss_assign, detr_match_cost, hungarian_assign, max_iou_assign, simota_assign, task_aligned_assign
from odlab.augment import letterbox, letterbox_boxes, mosaic4, unletterbox_boxes
from odlab.data import SyntheticShapes, collate, from_yolo_lines, make_sample, to_yolo_lines
from odlab.losses import dfl_decode, dfl_loss, sigmoid_focal_loss
from odlab.model import DetectionLoss, TinyYOLO


def test_max_iou_every_gt_gets_an_anchor():
    anchors = make_anchor_boxes((8, 8), 32, sizes=(64,), ratios=(1.0,))
    gt = torch.tensor([[10.0, 10, 22, 22], [100, 100, 220, 180]])  # first is tiny: max IoU << 0.5
    idx = max_iou_assign(anchors, gt)
    assert (idx == 0).sum() >= 1 and (idx == 1).sum() >= 1


def test_atss_assigns_inside_boxes():
    a3 = make_anchor_boxes((16, 16), 8, sizes=(32,), ratios=(1.0,))
    a4 = make_anchor_boxes((8, 8), 16, sizes=(64,), ratios=(1.0,))
    anchors = torch.cat((a3, a4))
    gt = torch.tensor([[20.0, 20, 60, 60]])
    idx = atss_assign(anchors, [len(a3), len(a4)], gt)
    pos = anchors[idx == 0]
    c = (pos[:, :2] + pos[:, 2:]) / 2
    assert len(pos) > 0 and ((c > 20) & (c < 60)).all()


def test_tal_positives_lie_inside_gt_and_stal_rescues_tiny():
    pts, st = make_grid_points([(32, 32), (16, 16), (8, 8)], [8, 16, 32])
    n = len(pts)
    gt = torch.tensor([[40.0, 40, 120, 120]])
    pred_boxes = torch.cat((pts - 30, pts + 30), 1)
    scores = torch.full((n, 3), 0.5)
    a = task_aligned_assign(scores, pred_boxes, pts, gt, torch.tensor([1]), topk=10)
    p = pts[a["fg"]]
    assert 0 < a["fg"].sum() <= 10 and ((p > 40) & (p < 120)).all()
    assert (a["target_scores"][a["fg"], 1] > 0).all() and a["target_scores"][:, [0, 2]].sum() == 0
    # a 3x3 object sitting between grid centres has no point inside it ...
    tiny = torch.tensor([[9.0, 9, 12, 12]])
    a = task_aligned_assign(scores, pred_boxes, pts, tiny, torch.tensor([0]))
    assert a["fg"].sum() == 0
    # ... unless STAL enlarges it to the stride for candidate selection
    a = task_aligned_assign(scores, pred_boxes, pts, tiny, torch.tensor([0]), small_side_floor=16)
    assert a["fg"].sum() > 0
    # topk2=1 leaves exactly one positive per gt (YOLO26 one-to-one branch)
    a = task_aligned_assign(scores, pred_boxes, pts, gt, torch.tensor([1]), topk=7, topk2=1)
    assert a["fg"].sum() == 1


def test_simota_dynamic_k():
    pts, st = make_grid_points([(32, 32)], [8])
    gt = torch.tensor([[40.0, 40, 200, 200], [220, 220, 236, 236]])
    pred = torch.cat((pts - 40, pts + 40), 1)
    idx, ious = simota_assign(torch.full((len(pts), 2), 0.5), pred, pts, st, gt, torch.tensor([0, 1]))
    assert (idx == 0).sum() > (idx == 1).sum() >= 1  # big, well-covered object gets more positives


def test_hungarian_is_one_to_one():
    q = 20
    logits = torch.randn(q, 4)
    boxes = torch.rand(q, 2) * 0.5
    boxes = torch.cat((boxes, boxes + 0.3), 1)
    gt = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9], [0.2, 0.6, 0.3, 0.8]])
    idx = hungarian_assign(detr_match_cost(logits, boxes, torch.tensor([0, 1, 2]), gt))
    assert (idx >= 0).sum() == 3 and len(set(idx[idx >= 0].tolist())) == 3


def test_focal_reduces_to_bce_and_dfl_expectation():
    x = torch.randn(100)
    y = (torch.rand(100) > 0.5).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(x, y, reduction="none")
    assert torch.allclose(sigmoid_focal_loss(x, y, alpha=-1, gamma=0), bce, atol=1e-6)
    # DFL optimum: two-hot logits around the target -> expectation equals target
    target = torch.tensor([[3.3, 0.0, 7.75, 14.2]])
    logits = torch.full((1, 4, 16), -30.0)
    for i, t in enumerate(target[0]):
        lo = int(t)
        logits[0, i, lo] = torch.log(lo + 1 - t)
        if lo + 1 < 16:
            logits[0, i, lo + 1] = torch.log(t - lo + 1e-12)
    assert torch.allclose(dfl_decode(logits.view(1, -1)), target, atol=1e-3)
    assert dfl_loss(logits.view(1, -1), target).item() < dfl_loss(torch.zeros(1, 64), target).item()


def test_letterbox_roundtrip():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100.0, 50, 300, 400], [0, 0, 640, 480]])
    out, r, pad = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == 1.0 and pad == (0, 80)
    lb = letterbox_boxes(boxes, r, pad)
    assert np.allclose(unletterbox_boxes(lb, r, pad, img.shape), boxes)
    out, r, pad = letterbox(img, 320, auto=True)
    assert out.shape[0] % 32 == 0 and out.shape[1] == 320


def test_mosaic_boxes_inside_canvas():
    rng = np.random.default_rng(0)
    samples = [make_sample(rng, 128) for _ in range(4)]
    img, boxes, labels = mosaic4([s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples], size=128, rng=np.random.RandomState(0))
    assert img.shape == (128, 128, 3) and len(boxes) == len(labels)
    assert (boxes >= 0).all() and (boxes <= 128).all()


def test_yolo_txt_roundtrip():
    rng = np.random.default_rng(1)
    img, boxes, labels = make_sample(rng, 200)
    b2, l2 = from_yolo_lines(to_yolo_lines(boxes, labels, 200, 200), 200, 200)
    assert np.allclose(b2, boxes, atol=1e-3) and (l2 == labels).all()


def test_tiny_yolo_shapes_and_overfit():
    torch.manual_seed(0)
    ds = SyntheticShapes(8, 128, seed=0)
    x, boxes, labels = collate([ds[i] for i in range(8)])
    for reg_max, e2e in [(16, False), (1, True)]:
        m = TinyYOLO(nc=3, reg_max=reg_max, end2end=e2e, img_size=128)
        out = m(x)
        assert out["boxes"].shape == (8, 16 * 16 + 8 * 8 + 4 * 4, 4 * reg_max)
        assert out["scores"].shape == (8, 336, 3)
        loss_fn = DetectionLoss(m)
        opt = torch.optim.AdamW(m.parameters(), 3e-3)
        first = None
        for _ in range(40):
            loss, _ = loss_fn(m(x), boxes, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < 0.6 * first, (reg_max, e2e, first, loss.item())
        m.eval()
        res = m.predict(x, conf=0.01)
        assert len(res) == 8 and all(r["boxes"].shape[-1] == 4 for r in res)
