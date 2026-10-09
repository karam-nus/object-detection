"""Host-side pre/post-processing for exported YOLO26n ONNX: timing and head agreement on real images."""
import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import time, json, numpy as np, cv2, onnxruntime as ort, torch, torchvision
W = WORK + "/weights/"
so = ort.SessionOptions(); so.intra_op_num_threads = 4

def letterbox(im, new=640, color=114):
    h, w = im.shape[:2]; r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (new - nh) // 2, (new - nw) // 2
    out = np.full((new, new, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)

def preprocess(bgr):
    lb, r, pad = letterbox(bgr)
    x = lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW, [0,1]
    return np.ascontiguousarray(x), r, pad

def post_raw(y, r, pad, conf=0.25, iou=0.7):
    p = y[0].T                                   # (8400, 4 + nc): cx, cy, w, h, class scores (already sigmoid)
    scores = p[:, 4:].max(1); cls = p[:, 4:].argmax(1)
    keep = scores > conf
    b, s, c = p[keep, :4], scores[keep], cls[keep]
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
    k = torchvision.ops.batched_nms(torch.from_numpy(xyxy), torch.from_numpy(s), torch.from_numpy(c), iou).numpy()[:300]
    xyxy, s, c = xyxy[k], s[k], c[k]
    xyxy[:, [0, 2]] -= pad[0]; xyxy[:, [1, 3]] -= pad[1]; xyxy /= r
    return xyxy, s, c, int(keep.sum())

def post_e2e(y, r, pad, conf=0.25):
    d = y[0]; d = d[d[:, 4] > conf]               # (300, 6): x1, y1, x2, y2, score, class
    xyxy = d[:, :4].copy(); xyxy[:, [0, 2]] -= pad[0]; xyxy[:, [1, 3]] -= pad[1]; xyxy /= r
    return xyxy, d[:, 4], d[:, 5].astype(int)

def t(f, n=30):
    for _ in range(3): f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) * 1e3)
    return round(float(np.median(ts)), 2)

out = {}
raw = ort.InferenceSession(W + "yolo26n_nmsNone.onnx", so, providers=["CPUExecutionProvider"])
e2e = ort.InferenceSession(W + "yolo26n_nmsFalse.onnx", so, providers=["CPUExecutionProvider"])
y11 = ort.InferenceSession(W + "yolo11n_nmsNone.onnx", so, providers=["CPUExecutionProvider"])
for img in ("bus.jpg", "zidane.jpg"):
    bgr = cv2.imread(W + img)
    x, r, pad = preprocess(bgr)
    yr = raw.run(None, {"images": x})[0]; ye = e2e.run(None, {"images": x})[0]
    rec = {"input": list(bgr.shape[:2]), "preprocess_ms": t(lambda: preprocess(bgr)),
           "yolo26n_raw_infer_ms": t(lambda: raw.run(None, {"images": x})),
           "yolo26n_e2e_infer_ms": t(lambda: e2e.run(None, {"images": x})),
           "yolo11n_raw_infer_ms": t(lambda: y11.run(None, {"images": x}))}
    for conf in (0.001, 0.25):
        rec[f"nms_post_ms@{conf}"] = t(lambda: post_raw(yr, r, pad, conf))
        rec[f"candidates@{conf}"] = post_raw(yr, r, pad, conf)[3]
    rec["e2e_post_ms"] = t(lambda: post_e2e(ye, r, pad))
    a = post_raw(yr, r, pad, 0.25); b = post_e2e(ye, r, pad, 0.25)
    rec["dets_o2m_nms@0.25"] = len(a[0]); rec["dets_e2e@0.25"] = len(b[0])
    if len(a[0]) and len(b[0]):
        iou = torchvision.ops.box_iou(torch.from_numpy(a[0]).float(), torch.from_numpy(b[0]).float()).numpy()
        rec["mean_best_iou_o2m_vs_e2e"] = round(float(iou.max(1).mean()), 3)
    rec["classes_o2m"] = sorted(a[2].tolist()); rec["classes_e2e"] = sorted(b[2].tolist())
    out[img] = rec
    print(img, json.dumps(rec), flush=True)
json.dump(out, open(W + "postproc_study.json", "w"), indent=1)
