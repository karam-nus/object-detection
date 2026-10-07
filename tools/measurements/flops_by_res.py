import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import sys, torch, collections
from ultralytics.nn.tasks import DetectionModel
m = DetectionModel(sys.argv[1], verbose=False).eval()
d = m.model[-1]
# drop the unused one2one branch to mirror deployment
macs = collections.Counter(); params = collections.Counter()
def hook(mod, inp, out):
    k = mod.kernel_size[0] * mod.kernel_size[1]
    mac = mod.in_channels // mod.groups * k * mod.out_channels * out.shape[2] * out.shape[3]
    macs[out.shape[2]] += mac
    params[out.shape[2]] += sum(p.numel() for p in mod.parameters())
for name, mod in m.named_modules():
    if isinstance(mod, torch.nn.Conv2d) and "one2one" not in name:
        mod.register_forward_hook(hook)
with torch.no_grad(): m(torch.zeros(1, 3, 640, 640))
tm, tp = sum(macs.values()), sum(params.values())
for r in sorted(macs, reverse=True):
    print(f"{r:>4}x{r:<4} stride {640//r:>2}: GMACs {macs[r]/1e9:6.3f} ({100*macs[r]/tm:4.1f}%)  conv params {params[r]:>9,} ({100*params[r]/tp:4.1f}%)")
print(f"total conv GMACs {tm/1e9:.3f} (x2 = {2*tm/1e9:.2f} GFLOPs), conv params {tp:,}")
