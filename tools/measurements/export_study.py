"""Export YOLO26n/YOLO11n to ONNX with each head option; record shapes, ops, size, CPU latency."""
import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import os, time, json, shutil, collections
import numpy as np, onnx, onnxruntime as ort
from ultralytics import YOLO
W = WORK + "/weights"
os.chdir(W)
res = {}
for name, kw in [("yolo26n", dict(nms=None)), ("yolo26n", dict(nms=False)), ("yolo26n", dict(nms=True)), ("yolo11n", dict(nms=None)), ("yolo11n", dict(nms=True))]:
    tag = f"{name}_nms{kw['nms']}"
    m = YOLO(f"{name}.pt")
    f = m.export(format="onnx", imgsz=640, **kw, verbose=False)
    dst = f"{tag}.onnx"; shutil.move(f, dst)
    g = onnx.load(dst)
    ops = collections.Counter(n.op_type for n in g.graph.node)
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    for _ in range(5): sess.run(None, {sess.get_inputs()[0].name: x})
    t = []
    for _ in range(30):
        t0 = time.perf_counter(); sess.run(None, {sess.get_inputs()[0].name: x}); t.append((time.perf_counter() - t0) * 1e3)
    res[tag] = dict(shapes=[list(o.shape) for o in outs], nodes=len(g.graph.node), size_mb=round(os.path.getsize(dst) / 1e6, 2),
                    special={k: v for k, v in ops.items() if k in ("TopK", "NonMaxSuppression", "Softmax", "GatherElements", "Gather", "ReduceMax", "ArgMax", "Mod", "Div", "MatMul", "Einsum")},
                    median_ms=round(float(np.median(t)), 1), p90_ms=round(float(np.percentile(t, 90)), 1))
    print(tag, json.dumps(res[tag]), flush=True)
json.dump(res, open("export_study.json", "w"), indent=1)
