import math

import torch
import torchvision.ops as tvo

from odlab.anchors import best_possible_recall, kmeans_anchors, make_anchor_boxes, make_grid_points
from odlab.boxes import (
    cxcywh_to_xyxy,
    elementwise_iou,
    nwd,
    pairwise_iou,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_xywh,
)
from odlab.nms import batched_nms, diou_nms, matrix_nms, nms, soft_nms, topk_select, weighted_boxes_fusion


def rand_boxes(n, seed=0, size=200.0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * size
    wh = torch.rand(n, 2, generator=g) * size / 3 + 1
    return torch.cat((xy, xy + wh), 1)


def test_format_roundtrip():
    b = rand_boxes(50)
    assert torch.allclose(xywh_to_xyxy(xyxy_to_xywh(b)), b, atol=1e-4)
    assert torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b, atol=1e-4)


def test_iou_family_matches_torchvision():
    a, b = rand_boxes(40, 1), rand_boxes(30, 2)
    assert torch.allclose(pairwise_iou(a, b, "iou"), tvo.box_iou(a, b), atol=1e-5)
    assert torch.allclose(pairwise_iou(a, b, "giou"), tvo.generalized_box_iou(a, b), atol=1e-5)
    assert torch.allclose(pairwise_iou(a, b, "diou"), tvo.distance_box_iou(a, b), atol=1e-5)
    assert torch.allclose(pairwise_iou(a, b, "ciou"), tvo.complete_box_iou(a, b), atol=1e-5)


def test_worked_example_from_chapter_03():
    # A = [0,0,4,4], B = [2,2,6,6]: inter 4, union 28, IoU = 1/7; enclosing 6x6 = 36
    a = torch.tensor([[0.0, 0, 4, 4]])
    b = torch.tensor([[2.0, 2, 6, 6]])
    assert math.isclose(elementwise_iou(a, b, "iou").item(), 1 / 7, rel_tol=1e-5)
    assert math.isclose(elementwise_iou(a, b, "giou").item(), 1 / 7 - 8 / 36, rel_tol=1e-5)
    # centre distance^2 = 8, diagonal^2 = 72
    assert math.isclose(elementwise_iou(a, b, "diou").item(), 1 / 7 - 8 / 72, rel_tol=1e-5)
    # same aspect ratio -> v = 0 -> CIoU == DIoU
    assert math.isclose(elementwise_iou(a, b, "ciou").item(), elementwise_iou(a, b, "diou").item(), rel_tol=1e-5)


def test_disjoint_boxes_have_gradient_under_giou_not_iou():
    a = torch.tensor([[0.0, 0, 2, 2]], requires_grad=True)
    b = torch.tensor([[5.0, 5, 7, 7]])
    (1 - elementwise_iou(a, b, "iou")).sum().backward()
    assert a.grad.abs().sum() == 0
    a.grad = None
    (1 - elementwise_iou(a, b, "giou")).sum().backward()
    assert a.grad.abs().sum() > 0


def test_siou_and_eiou_are_one_for_identical_boxes():
    b = rand_boxes(10)
    assert torch.allclose(elementwise_iou(b, b, "siou"), torch.ones(10), atol=1e-4)
    assert torch.allclose(elementwise_iou(b, b, "eiou"), torch.ones(10), atol=1e-4)


def test_nwd_scale_behaviour():
    small = torch.tensor([[0.0, 0, 6, 6]])
    small_shift = small + 2
    big = torch.tensor([[0.0, 0, 60, 60]])
    big_shift = big + 2
    # IoU punishes the same 2-px shift far more on the small box; NWD does not
    iou_gap = elementwise_iou(big, big_shift).item() - elementwise_iou(small, small_shift).item()
    nwd_gap = nwd(big, big_shift).item() - nwd(small, small_shift).item()
    assert iou_gap > 0.3 and abs(nwd_gap) < 1e-6


def test_nms_matches_torchvision():
    for seed in range(5):
        b = rand_boxes(80, seed)
        s = torch.rand(80, generator=torch.Generator().manual_seed(seed + 10))
        assert torch.equal(nms(b, s, 0.5), tvo.nms(b, s, 0.5))
        lab = torch.randint(0, 4, (80,), generator=torch.Generator().manual_seed(seed + 20))
        assert set(batched_nms(b, s, lab, 0.5).tolist()) == set(tvo.batched_nms(b, s, lab, 0.5).tolist())


def test_soft_nms_keeps_more_than_hard_nms():
    b = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
    s = torch.tensor([0.9, 0.8, 0.7])
    assert len(nms(b, s, 0.5)) == 2
    keep, scores = soft_nms(b, s, method="gaussian")
    assert len(keep) == 3 and scores[1] < 0.8  # overlapped box survives with a decayed score


def test_diou_nms_spares_far_centres():
    # IoU = 1/3, but the centres are 10 px apart inside a 10x30 enclosing box: DIoU = 1/3 - 0.1
    b = torch.tensor([[0.0, 0, 10, 20], [0, 10, 10, 30]])
    s = torch.tensor([0.9, 0.8])
    assert len(nms(b, s, 0.3)) == 1
    assert len(diou_nms(b, s, 0.3)) == 2


def test_matrix_nms_decays_duplicates():
    b = torch.tensor([[0.0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5], [40, 40, 50, 50]])
    s = torch.tensor([0.9, 0.85, 0.8])
    d = matrix_nms(b, s)
    assert d[0] == s[0] and d[2] == s[2] and d[1] < 0.5


def test_wbf_averages():
    b1 = torch.tensor([[0.0, 0, 10, 10]])
    b2 = torch.tensor([[2.0, 2, 12, 12]])
    fb, fs, fl = weighted_boxes_fusion([b1, b2], [torch.tensor([0.9]), torch.tensor([0.9])], [torch.tensor([0]), torch.tensor([0])], iou_thr=0.4)
    assert len(fb) == 1 and torch.allclose(fb[0], torch.tensor([1.0, 1, 11, 11]))


def test_topk_select_shapes():
    s = torch.rand(100, 5)
    idx, lab, val = topk_select(s, 10)
    assert idx.shape == lab.shape == val.shape == (10,)
    assert torch.allclose(s[idx, lab], val)


def test_grid_points_count_8400():
    pts, st = make_grid_points([(80, 80), (40, 40), (20, 20)], [8, 16, 32])
    assert pts.shape == (8400, 2) and st.shape == (8400, 1)
    assert pts[0].tolist() == [4.0, 4.0] and pts[-1].tolist() == [624.0, 624.0]


def test_anchor_boxes_and_kmeans():
    a = make_anchor_boxes((2, 3), 16, sizes=(32,), ratios=(0.5, 1, 2))
    assert a.shape == (2 * 3 * 3, 4)
    g = torch.Generator().manual_seed(0)
    wh = torch.cat((torch.rand(300, 2, generator=g) * 20 + 10, torch.rand(300, 2, generator=g) * 100 + 100))
    anchors, fit = kmeans_anchors(wh, k=6)
    assert anchors.shape == (6, 2) and fit > 0.6
    assert best_possible_recall(wh, anchors) > 0.95
