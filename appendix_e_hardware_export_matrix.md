---
title: "Appendix E — Hardware & Export Matrix"
---

[← Back to Table of Contents](./README.md)

# Appendix E — Hardware & Export Matrix

A lookup table for deployment: for each export target, the hardware it serves, the precisions it
supports, how YOLO26's head options behave, and the traps seen most often. The `format=` names and head
behaviour follow the Ultralytics exporter (8.4.x, 2026). Toolchains change quickly: verify on your
versions. Chapters 36, 44, 45 and 46 explain the background.

---

## Export targets

| `format=` | Runtime | Hardware | Precisions (`quantize=`) | YOLO26 `nms=False` (one-to-one) | `nms=True` (embedded NMS) | Known traps |
|---|---|---|---|---|---|---|
| `onnx` | ONNX Runtime and others | Anywhere | FP32, FP16, INT8 (via the consumer) | ✅ | ✅ (ONNX `NonMaxSuppression`) | Opset differences; check the consumer supports TopK/NMS |
| `engine` | TensorRT | NVIDIA GPU, Jetson (+ DLA) | FP16, INT8 (Q/DQ via ModelOpt on TensorRT 11), FP8 on new GPUs | ✅ | ✅ | Engine tied to GPU and TensorRT version; old TRT (< 8.5) and TRT 10.3 INT8 on JetPack 6 fall back to one-to-many |
| `openvino` | OpenVINO | Intel CPU, iGPU, NPU | FP32, FP16, INT8 (NNCF) | ✅ | ✅ | NPU plugin op coverage differs from CPU |
| `coreml` | Core ML | Apple CPU, GPU, Neural Engine | FP32, FP16, INT8 weights | ✅ | ✅ (static shapes; enables Xcode preview) | Neural Engine placement depends on ops and shapes |
| `coreai` | Apple Core AI | Apple devices | FP16, quantised | check | — | New format; check model support |
| `saved_model`, `pb` | TensorFlow | CPU, GPU | FP32, FP16, INT8 | ✅ (`saved_model`) | ✅ (`saved_model`) | `pb` is legacy |
| `litert` | LiteRT (TFLite) | Android CPU/GPU/NPU, embedded Linux | FP32, FP16, INT8, `w8a16` | ✅ FP; ❌ INT8/`w8a16` (fall back to one-to-many) | — | Full-integer models for NPUs need representative data |
| `edgetpu` | Edge TPU | Coral | INT8 only | ❌ (one-to-many) | — | Every op must map to the TPU or the model splits |
| `ncnn` | NCNN | ARM/x86 CPU, Vulkan GPU | FP32, FP16, INT8 | ❌ (one-to-many) | — | Fastest CPU option on Raspberry Pi 5 (Chapter 16) |
| `mnn` | MNN | Mobile CPU/GPU | FP32, FP16, INT8 | ✅ | ✅ (static) | — |
| `executorch` | ExecuTorch | Mobile and embedded | FP32, quantised via delegates | ❌ (one-to-many) | — | Delegate coverage varies |
| `paddle` | Paddle Inference | CPU, GPU | FP32, FP16 | ❌ (one-to-many) | — | — |
| `rknn` | RKNN | Rockchip NPU (RK3588 …) | INT8 | ❌ (one-to-many) | — | Host NMS; calibration images required |
| `qnn` | Qualcomm QNN | Snapdragon NPU/DSP | INT8, FP16 | ❌ (one-to-many) | — | Host NMS |
| `hailo` | Hailo Dataflow Compiler | Hailo-8/8L/10 | INT8 (some INT4) | ✅ selectable; default raw tensors + host NMS | — | Compilation needs calibration data |
| `imx` | Sony IMX500 toolchain | IMX500 smart camera | INT8 | — | ✅ required (selected automatically) | The export table lists support for YOLOv8n and YOLO11n only |
| `axelera` | Axelera Voyager | Axelera Metis | INT8 | check | — | Vendor toolchain |
| `deepx` | DEEPX DX-COM | DEEPX NPUs | INT8 | check | — | `optimize` flag trades compile time for speed |
| `ascend` | Huawei CANN | Ascend NPUs | FP16, INT8 | check | — | Vendor toolchain |
| `xilinx` | AMD Vitis AI | AMD/Xilinx FPGAs, NPUs | INT8 | check | — | Vendor toolchain |
| `torchscript` | PyTorch (LibTorch) | CPU, GPU | FP32, FP16 | ✅ | ✅ | Mostly for C++ PyTorch serving |

"✅/❌" for `nms=False` reflect the Ultralytics end-to-end guide's list of formats that fall back to the
one-to-many path. "check" means the documentation does not state it for that format.

---

## Detector operators by hardware class

| Operator | Datacentre GPU | Jetson GPU / DLA | x86 / ARM CPU | Mobile NPU | Edge NPU | Micro-NPU / MCU |
|---|---|---|---|---|---|---|
| Conv / DW-conv / concat / add | ✅ | ✅ / ✅ | ✅ | ✅ | ✅ | ✅ |
| SiLU | ✅ | ✅ / partial | ✅ | usually | usually (LUT) | LUT or ReLU |
| Nearest upsample | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MaxPool 5×5 | ✅ | ✅ | ✅ | ✅ | usually | check kernel limits |
| Softmax (DFL, attention) | ✅ | ✅ / ❌ | ✅ | usually | varies | CPU |
| MatMul, LayerNorm (attention) | ✅ | ✅ / ❌ | ✅ | newer NPUs | varies | Ethos-U85 yes; older no |
| TopK, Gather | ✅ | ✅ / ❌ | ✅ | often CPU | often unsupported | CPU |
| NMS | ✅ (plugin / ONNX op) | ✅ / ❌ | ✅ | CPU | host | CPU |
| GridSample (deformable attention) | ✅ (plugin) | ✅ / ❌ | ✅ | rare | rare | ❌ |
| Dynamic shapes | ✅ (profiles) | ✅ | ✅ | limited | ❌ | ❌ |

---

## Model families and deployment friendliness

| Family | GPU | CPU | Mobile / edge NPU | MCU | Notes |
|---|---|---|---|---|---|
| YOLO26 (n/s) | ✅ | ✅ (fastest at n/s) | ✅ (INT8; one-to-many on some NPUs) | ❌ (too large for most) | No DFL; optional NMS |
| YOLO11 / YOLOv8 | ✅ | ✅ | ✅ (DFL in head: keep head float in INT8) | ❌ | Mature toolchain support |
| YOLOv12 / YOLOv13 | ✅ (attention kernels) | slower | limited | ❌ | Attention and custom ops |
| YOLOX / RTMDet / PP-YOLOE / PicoDet | ✅ | ✅ | ✅ | tiny variants | Permissive licences |
| RT-DETR / D-FINE / DEIM / RF-DETR / LW-DETR | ✅ (best AP per ms) | slow | limited (attention, deformable ops) | ❌ | Permissive (except RF-DETR XL/2XL) |
| Open-vocabulary (YOLOE) | ✅ | ✅ (after re-parameterisation) | ✅ (as closed-set export) | ❌ | Fix the prompt list before export |
| FOMO, Yolo-Fastest, TinyissimoYOLO | ✅ | ✅ | ✅ | ✅ | Chapter 15 |

---

## Pre-flight checklist for any target

1. Export with static shapes and the head you intend to ship; read the exporter's warnings (fallbacks).
2. Inspect the compiler's per-layer placement report: nothing unexpected on the CPU.
3. Run the golden-image test on the device (Chapter 36).
4. Validate the artifact's accuracy at the deployment precision (Chapters 33, 45).
5. Measure sustained, end-to-end latency with the real pipeline (Chapter 46).
6. Record versions: model, exporter, runtime/SDK, firmware (Chapter 47).
