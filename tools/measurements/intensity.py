import os as _os, pathlib as _pl
REPO = str(_pl.Path(__file__).resolve().parents[2])
WORK = _os.environ.get("ODBOOK_WORK", _os.path.join(REPO, "work"))  # weights/, datasets/ live here

import sys, copy, torch
from ultralytics.nn.tasks import DetectionModel
for cfg in sys.argv[1:]:
    m = copy.deepcopy(DetectionModel(cfg, verbose=False)).fuse(verbose=False).eval()
    stats = dict(macs=0, act=0, w=0)
    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        stats["macs"] += mod.in_channels // mod.groups * k * mod.out_channels * out.shape[2] * out.shape[3]
        stats["act"] += inp[0].numel() + out.numel()          # read input + write output, per conv
        stats["w"] += mod.weight.numel()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Conv2d): mod.register_forward_hook(hook)
    with torch.no_grad(): m(torch.zeros(1, 3, 640, 640))
    flops = 2 * stats["macs"]; bytes_fp16 = 2 * (stats["act"] + stats["w"])
    print(f"{cfg}: conv GFLOPs {flops/1e9:.2f}, conv activation traffic {2*stats['act']/1e6:.1f} MB, weights {2*stats['w']/1e6:.1f} MB (FP16), intensity {flops/bytes_fp16:.0f} FLOP/byte")
