"""PTQ study: ONNX Runtime static INT8 (QDQ) of YOLO26n / YOLO11n, evaluated on a COCO128 split.
Calibrate on images 0-63, validate on images 64-127 (disjoint)."""
import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import os, glob, json, time, numpy as np, cv2, onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from ultralytics import YOLO
W = WORK + "/weights/"
D = WORK + "/datasets/coco128"
imgs = sorted(glob.glob(f"{D}/images/train2017/*.jpg"))
calib, val = imgs[:64], imgs[64:]
os.makedirs(f"{D}/splits", exist_ok=True)
open(f"{D}/splits/val64.txt", "w").write("\n".join(val))
names = YOLO(W + "yolo26n.pt").names
import yaml
yaml.safe_dump({"path": D, "train": "splits/val64.txt", "val": "splits/val64.txt", "names": names}, open(f"{D}/val64.yaml", "w"))

def letterbox(im, new=640):
    h, w = im.shape[:2]; r = min(new / h, new / w); nh, nw = round(h * r), round(w * r)
    out = np.full((new, new, 3), 114, np.uint8); t, l = (new - nh) // 2, (new - nw) // 2
    out[t:t + nh, l:l + nw] = cv2.resize(im, (nw, nh)); return out

class Reader(CalibrationDataReader):
    def __init__(self, files): self.it = iter(files)
    def get_next(self):
        f = next(self.it, None)
        if f is None: return None
        x = letterbox(cv2.imread(f))[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        return {"images": np.ascontiguousarray(x)}

def head_output_nodes(path):
    g = onnx.load(path).graph
    # final 1x1 convs of every box/class tower (".../cv2.k/cv2.k.2/Conv", ".../cv3.k/cv3.k.2/Conv") and everything after the head convs
    convs = [n.name for n in g.node if n.op_type == "Conv" and (n.name.endswith(".2/Conv") and ("cv2" in n.name or "cv3" in n.name))]
    post = []
    head_started = False
    for n in g.node:
        if n.name in convs: head_started = True
        if head_started and (n.op_type != "Conv" or "dfl" in n.name): post.append(n.name)
    return convs, post

results = {}
for base in ("yolo26n_nmsNone", "yolo11n_nmsNone"):
    src = W + base + ".onnx"
    convs, post = head_output_nodes(src)
    variants = {"int8_all": [], "int8_keep_head_out_float": convs + post}
    for v, excl in variants.items():
        dst = W + f"{base}_{v}.onnx"
        quantize_static(src, dst, Reader(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=excl,
                        extra_options={"ActivationSymmetric": False})
    for tag in ("", "_int8_all", "_int8_keep_head_out_float"):
        p = W + base + tag + ".onnx"
        m = YOLO(p, task="detect")
        r = m.val(data=f"{D}/val64.yaml", imgsz=640, batch=1, rect=False, verbose=False, plots=False, device="cpu")
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, 640, 640).astype(np.float32)
        for _ in range(5): s.run(None, {"images": x})
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); s.run(None, {"images": x}); ts.append((time.perf_counter() - t0) * 1e3)
        results[base + tag] = {"map50_95": round(float(r.box.map), 4), "map50": round(float(r.box.map50), 4),
                               "cpu_ms": round(float(np.median(ts)), 1), "size_mb": round(os.path.getsize(p) / 1e6, 2),
                               "excluded_nodes": len(variants.get(tag[1:], []))}
        print(base + tag, json.dumps(results[base + tag]), flush=True)
json.dump(results, open(W + "int8_study.json", "w"), indent=1)
