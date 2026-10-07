"""Regenerate assets/images/yolo_pareto_t4.svg from assets/data/detectors.json (run from the repo root)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("assets/data/detectors.json"))["models"]
e2e26 = {"YOLO26n": 40.1, "YOLO26s": 47.8, "YOLO26m": 52.5, "YOLO26l": 54.4, "YOLO26x": 56.9}
fams = [("YOLOv8", "#8a8a8a"), ("YOLOv10", "#60a5fa"), ("YOLO11", "#a78bfa"), ("YOLOv12", "#f472b6"),
        ("YOLO26", "#f59e0b"), ("D-FINE", "#4ade80"), ("DEIMv2", "#22d3ee"), ("RF-DETR", "#ef4444"),
        ("LW-DETR", "#facc15"), ("RT-DETR", "#93c5fd")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": "#c8c8c8",
                     "axes.labelcolor": "#c8c8c8", "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a",
                     "axes.edgecolor": "#3a3a3a", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(8.2, 5.2))
fig.patch.set_alpha(0); ax.set_facecolor("none")
for fam, col in fams:
    rows = [r for r in d if r["family"] == fam and r["latency_hw"] == "T4 TRT FP16" and r["latency_ms"] and r["coco_ap"]]
    if fam == "RT-DETR": rows = [r for r in rows if r["name"].startswith("RT-DETRv4")]
    if fam == "D-FINE": rows = [r for r in rows if "O365" not in r["name"]]
    if fam == "YOLO26":
        for r in rows: r["coco_ap"] = e2e26.get(r["name"], r["coco_ap"])
    rows = [r for r in rows if r["coco_ap"] >= 37 and r["latency_ms"] <= 17.5]
    rows.sort(key=lambda r: r["latency_ms"])
    xs = [r["latency_ms"] for r in rows]; ys = [r["coco_ap"] for r in rows]
    needs_nms = rows[0]["nms"] is True
    lw = 2.4 if fam == "YOLO26" else 1.4
    name = {"RT-DETR": "RT-DETRv4", "YOLO26": "YOLO26 (nms=False AP)"}.get(fam, fam)
    ax.plot(xs, ys, "-", color=col, lw=lw, alpha=0.9, zorder=3 if fam == "YOLO26" else 2, label=name, marker="o", markersize=5, markerfacecolor="none" if needs_nms else col, markeredgecolor=col)
    ax.scatter(xs, ys, s=34 if fam == "YOLO26" else 24, facecolors="none" if needs_nms else col,
               edgecolors=col, linewidths=1.4, zorder=4)
    label = fam + (" (e2e AP)" if fam == "YOLO26" else "") + (" ◦ NMS not timed" if needs_nms else "")
ax.set_xlim(0, 18); ax.set_ylim(36, 61.5)
ax.set_xlabel("Latency, ms  (NVIDIA T4, TensorRT FP16, batch 1, as reported by each source)")
ax.set_ylabel("COCO val2017 AP50:95")
ax.grid(color="#2a2a2a", lw=0.7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.17), ncol=2, frameon=False, fontsize=8, labelcolor="#c8c8c8")
ax.text(17.8, 36.6, "hollow = needs NMS, NMS time excluded from latency\nfilled = NMS-free, latency is end-to-end\nYOLO26 plotted with its one-to-one (nms=False) AP to match its latency",
        ha="right", va="bottom", fontsize=7.5, color="#9a9a9a")
fig.tight_layout()
fig.savefig("assets/images/yolo_pareto_t4.svg", transparent=True)
print("ok")
