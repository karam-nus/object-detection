import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import sys, torch
from ultralytics.nn.tasks import DetectionModel
cfg = sys.argv[1]
m = DetectionModel(cfg, verbose=False).eval()
shapes = {}
hooks = []
for i, layer in enumerate(m.model):
    def h(mod, inp, out, i=i):
        if isinstance(out, torch.Tensor): shapes[i] = tuple(out.shape)
        elif isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor): shapes[i] = tuple(out[0].shape)
        else: shapes[i] = type(out).__name__
    hooks.append(layer.register_forward_hook(h))
with torch.no_grad():
    m(torch.zeros(1, 3, 640, 640))
tot = 0
for i, layer in enumerate(m.model):
    p = sum(x.numel() for x in layer.parameters()); tot += p
    print(f"{i:>2} {str(layer.f):>9} {layer.type.split('.')[-1]:<10} params={p:>9,} out={shapes.get(i)}")
print("total", f"{tot:,}")
d = m.model[-1]
print("head cv2[0]", d.cv2[0]); print("head cv3[0]", d.cv3[0])
try:
    from thop import profile
except Exception: pass
