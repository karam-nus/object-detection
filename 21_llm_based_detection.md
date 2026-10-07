---
title: "Chapter 21 — LLM-Based Detection"
---

[← Back to Table of Contents](./README.md)

# Chapter 21 — LLM-Based Detection

> *"A detector emits 300 boxes in one forward pass. A language model writes them one digit at a time."*

## Overview

Multimodal large language models (MLLMs) can now answer "where?" by **writing bounding boxes as
text**. That turns detection into instruction following: "find the screws that are missing a washer",
"box every person not wearing a hard hat", "which button submits this form?". No closed-set detector
can express these queries. It also inherits the costs of autoregressive generation: sequential
decoding, missed objects in crowded scenes, no native confidence scores, and coordinate drift. This
chapter covers how coordinates are tokenised, the model lineage (Kosmos-2 to Qwen3-VL, Florence-2,
PaliGemma, Gemini, Rex-Omni, LocateAnything), how they are trained (SFT, then reinforcement learning with
IoU rewards), how they are evaluated, their speed and failure modes, and the **hybrid pipelines**
that are, in 2026, the practical way to use them.

<div class="diagram">
<div class="diagram-title">Five ways to get a box out of a language model</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card blue"><div class="card-title">1. Plain-text numbers</div><div class="card-desc">"[123, 45, 310, 290]" — Shikra, Qwen-VL family, Gemini (JSON box_2d)</div></div>
  <div class="diagram-card green"><div class="card-title">2. Location tokens</div><div class="card-desc">dedicated vocabulary &lt;loc_0&gt;…&lt;loc_999&gt; — Kosmos-2, Florence-2, PaliGemma</div></div>
  <div class="diagram-card accent"><div class="card-title">3. Quantised coordinate tokens</div><div class="card-desc">1,000 tokens = coordinates 0–999, 4 per box — Rex-Omni</div></div>
  <div class="diagram-card purple"><div class="card-title">4. Select from proposals</div><div class="card-desc">a detector proposes; the LLM outputs indices — ChatRex, Groma-style region tokens</div></div>
  <div class="diagram-card pink"><div class="card-title">5. Parallel box decoding</div><div class="card-desc">a whole box in one step instead of token by token — LocateAnything</div></div>
</div>
</div>

---

## Why put detection inside a language model?

| Capability | Closed-set detector | OV detector (Ch 20) | MLLM |
|---|:---:|:---:|:---:|
| Fixed category list | ✓ | ✓ | ✓ |
| Arbitrary noun phrases | ✗ | ✓ | ✓ |
| Referring expressions ("the cup left of the laptop") | ✗ | partial | ✓ |
| Reasoning ("the item that doesn't belong") | ✗ | ✗ | ✓ |
| Multi-step instructions, dialogue, explanation | ✗ | ✗ | ✓ |
| GUI / document / OCR grounding | ✗ | weak | ✓ |
| 100+ objects per image, real-time | ✓ | ✓ | ✗ |
| Calibrated scores for ranking | ✓ | ✓ | usually ✗ |

The last two rows are why MLLMs complement detectors rather than replace them.

---

## How coordinates become tokens

| Model | Format | Coordinate space | Order |
|---|---|---|---|
| **Shikra** (2023) | plain numbers, 3 decimals | normalised [0, 1] | x1, y1, x2, y2 |
| **Kosmos-2** (2023) | `<patch_index_….>` tokens: the image is a 32×32 grid → 1,024 location tokens; a box = top-left and bottom-right cells | grid | corners |
| **Qwen-VL** (2023) | `<box>(x1,y1),(x2,y2)</box>` as text | integers in [0, 1000) | x, y |
| **Qwen2.5-VL** (2025) | JSON with `bbox_2d` | **absolute pixels on the resized input** | x1, y1, x2, y2 |
| **Qwen3-VL** (2025) | JSON with `bbox_2d` | **relative 0–1000** again | x1, y1, x2, y2 |
| **Florence-2** (2024) | `<loc_0>`…`<loc_999>` tokens | 1,000 bins per axis | x1, y1, x2, y2 |
| **PaliGemma / PaliGemma 2** | `<loc0000>`…`<loc1023>` tokens | 1,024 bins | **y**min, **x**min, **y**max, **x**max |
| **Gemini (API)** | JSON `box_2d` | normalised 0–1000 | **y**min, **x**min, **y**max, **x**max |
| **Rex-Omni** (2025) | 1,000 coordinate tokens (0–999) mapped into the vocabulary | quantised | 4 tokens per box |

<div class="callout warn"><span class="callout-title">Trap</span>The table has three coordinate
spaces (normalised, 0–1000 relative, absolute pixels of the <em>resized</em> image) and two orders
(x-first, y-first). Qwen switched space twice between versions. Most "the VLM's boxes are shifted"
bugs are a convention mismatch. Read the model card, then verify on one image with a known box.</div>

**Quantisation error.** With 1,000 bins over a 1,920-pixel-wide image, each bin is 1.92 px. The maximum
rounding error is ±0.96 px per coordinate. Harmless for large objects, measurable for 10-pixel ones
(Chapter 3: one pixel costs an 8×8 box 22% of its IoU). Plain-text numbers on the resized image have the
same issue at the resize ratio.

---

## The model lineage

| Model | Org, year | Size | Detection-relevant idea |
|---|---|---|---|
| **Kosmos-2** | Microsoft, 2023 | 1.6 B | grounding via location tokens; GRIT dataset of grounded image–text pairs |
| **Shikra** | 2023 | 7–13 B | referential dialogue with plain-number coordinates |
| **Ferret / Ferret-v2** | Apple, 2023–24 | 7–13 B | hybrid region representation (points, boxes, free-form shapes) as *inputs* and outputs |
| **Qwen-VL → Qwen2.5-VL → Qwen3-VL** | Alibaba, 2023–25 | 2 B–235 B | grounding as a first-class skill; Qwen3-VL-235B-A22B reports **48.6 mAP on ODinW-13** |
| **Florence-2** | Microsoft, 2024 | 0.23 B / 0.77 B | one seq2seq model for captioning, detection, grounding, OCR; trained on FLD-5B (5.4 B annotations on 126 M images) |
| **PaliGemma / 2** | Google, 2024 | 3 B–28 B | `<loc>` tokens; strong fine-tuning base for detection-as-text |
| **Gemini 2.x** | Google | — | JSON `box_2d` output through the API; scores 14.4 cgF1 on SAM 3's SA-Co/Gold box benchmark |
| **ChatRex** | IDEA, 2024 | — | decoupled perception: a proposal network supplies boxes, the LLM answers with *indices* |
| **VLM-R1** | Om AI Lab, 2025 | 3 B | GRPO reinforcement learning for referring-expression comprehension and OV detection |
| **Rex-Omni** | IDEA, 2025 (CVPR 2026) | 3 B (Qwen2.5-VL-3B) | quantised coordinate tokens; SFT on ~22 M samples then **GRPO with geometry-aware rewards** |
| **LocateAnything-3B** | NVIDIA, 2026 | 3 B (MoonViT + Qwen2.5-3B) | **parallel box decoding**; 138 M training samples; non-commercial license |
| **Moondream** | vikhyat, 2024– | ~2 B | small VLM with built-in `detect` and `point` skills |

### Rex-Omni in detail

| | |
|---|---|
| **Introduced** | Jiang et al. (IDEA), 2025 (arXiv:2510.12798, CVPR 2026) |
| **Lineage** | Qwen2.5-VL-3B |
| **Paradigm** | generative: next-token prediction of quantised coordinates |
| **Assignment** | n/a (sequence loss), then RL rewards based on box matching |
| **Post-processing** | parse tokens; no NMS (duplicates are a known failure that GRPO reduces) |
| **COCO AP (sizes)** | reported as F1 at IoU thresholds in a zero-shot setting; competitive with Grounding DINO-SwinT and closed-set DINO-R50 at IoU 0.5 |
| **Latency** | ~5 boxes/s decoding (NVIDIA's comparison); AWQ-quantised version halves storage |
| **Pre-training** | Qwen2.5-VL + ~22 M grounding samples |
| **License** | see repository (IDEA) |
| **Known failure modes** | dense scenes; duplicate predictions and size misalignment after SFT (reduced by GRPO) |

Rex-Omni repurposes 1,000 vocabulary tokens as the integers 0–999, so a box costs **four tokens**
rather than 15–20 characters of text. Its two-stage training is the template for 2025–2026 MLLM
detectors:

1. **SFT** on large-scale grounding data teaches the format and the basics.
2. **GRPO** (group relative policy optimisation, as in reasoning LLMs) samples several outputs per
   image, scores each with a *geometry-aware reward* (IoU-based matching to the ground truth), and
   reinforces the better ones. The paper reports that RL fixes the characteristic SFT errors:
   **duplicate boxes** (the model "says" the same object twice) and **size misalignment**.

RL works here because detection has a cheap, exact reward. Matching predicted boxes to ground truth
*is* the metric. VLM-R1 and Perception-R1 use the same recipe.

### LocateAnything: attacking the decoding cost

NVIDIA's LocateAnything-3B (May 2026) predicts **all four coordinates of a box in a single parallel
step** (Parallel Box Decoding) instead of token by token. Its reported throughput:

| Model | Decoding style | Boxes per second |
|---|---|:---:|
| Qwen3-VL | textual coordinates | 1.1 |
| Rex-Omni | quantised tokens | 5.0 |
| **LocateAnything-3B** | parallel box decoding | **12.7** |

These figures are NVIDIA's own comparison. Even 12.7 boxes/s is about four orders of magnitude below a
dense detector that emits 300 candidates in under 2 ms (≈150,000 boxes/s), and is why the hybrid
designs below exist.

---

## A cost model for generative detection

Let an image contain $M$ objects. A generative detector produces about $c$ tokens per object
(coordinates + label + separators) and decodes at $r$ tokens per second:

$$t_{\text{detect}} \approx t_{\text{prefill}} + \frac{c \cdot M}{r}$$

**Numerical check.** JSON output like `{"bbox_2d":[412,233,618,507],"label":"cup"},` is about 20 tokens
per object. At 50 tokens/s (a few-billion-parameter model on one GPU, batch 1), an image with
$M = 20$ objects needs 8 seconds of decoding. With Rex-Omni-style 4-token boxes plus a short label
(about 6 tokens), the same image needs about 2.4 s. A YOLO26n takes 1.7 ms on a T4 regardless of $M$.

Two consequences:

1. **Latency grows with the number of objects.** Dense scenes (shelves, crowds, aerial images) are
   exactly where generative detection is slowest.
2. **Models learn to stop early.** Long outputs are rare in training data, and generation has an
   end-of-sequence token. Recall in dense scenes suffers. This is the "missed objects" failure that RL
   rewards for recall partly address.

---

## Failure modes

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Early stopping / low recall in dense scenes** | sequence-length bias; EOS token | RL with recall-aware rewards; tile the image; hybrid proposal + LLM |
| **Duplicates** | no one-to-one constraint over the generated set | GRPO penalties (Rex-Omni); post-hoc NMS on parsed boxes |
| **Hallucinated boxes for absent objects** | language prior: "a kitchen has a microwave" | presence checks; negative prompts; verification by a detector |
| **Coordinate convention errors** | normalised vs absolute vs resized; x-first vs y-first | Read the card; unit-test with a known image |
| **Resolution limits** | images resized to the model's pixel budget; small objects vanish | tiling (SAHI-style, Chapter 41); higher-resolution settings |
| **No calibrated confidence** | text has no score | use token log-probabilities as a proxy; or verify with a detector; report F1 not AP |

---

## Evaluating MLLM detectors

AP requires a ranking score for every prediction. Most MLLMs do not emit one. Common practice:

- **F1 (or precision/recall) at IoU 0.5 and at 0.5:0.95**, treating every generated box as a positive
  prediction (Rex-Omni's protocol).
- **Referring-expression accuracy**: Acc@0.5 on RefCOCO/+/g (the single box for a description).
- **ODinW mAP** when the model exposes scores or log-probabilities (Qwen3-VL's 48.6 on ODinW-13).
- **cgF1 on SA-Co**, which also penalises predictions for absent concepts (Chapter 20).

Compare an MLLM with a detector **at the detector's deployment threshold**: F1 vs F1. Comparing an MLLM's
F1 with a detector's AP is meaningless.

---

## Hybrid pipelines: how MLLM detection is actually used

<div class="diagram">
<div class="diagram-title">Four hybrid patterns</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent"><div class="card-title">A. Detector proposes, LLM decides</div><div class="card-desc">YOLO / Grounding DINO / SAM 3 give candidate boxes with indices; the MLLM answers "which ones match the instruction?" (ChatRex-style). Fast, complete, reasoning-capable.</div></div>
  <div class="diagram-card green"><div class="card-title">B. LLM plans, detector executes</div><div class="card-desc">An agent decomposes a request and calls detection tools; the detector supplies dense, scored boxes. Works with any LLM.</div></div>
  <div class="diagram-card purple"><div class="card-title">C. Detector gates, LLM describes</div><div class="card-desc">Run a cheap detector every frame; call the MLLM only when something relevant appears (the pattern in Ultralytics' LLM docs: call the LLM only when YOLO detects a person).</div></div>
  <div class="diagram-card blue"><div class="card-title">D. LLM labels, detector learns</div><div class="card-desc">Use an MLLM or OV model to auto-label, humans correct, train a small closed-set model (Chapter 49). The MLLM's cost is paid once, offline.</div></div>
</div>
</div>

Pattern C in code, with Ultralytics' `LLM` interface (an OpenAI-compatible client in the `ultralytics`
package):

```python
from ultralytics import LLM, YOLO

yolo = YOLO("yolo26n.pt")
llm = LLM("<your-multimodal-model>")       # any OpenAI-compatible endpoint
result = yolo("frame.jpg")[0]
if any(result.names[int(c)] == "person" for c in result.boxes.cls):   # cheap gate, every frame
    reply = llm("Is anyone not wearing a hard hat? Answer yes/no.", image="frame.jpg")   # rare, expensive
```

Pattern A with Qwen-style output, sketched:

```python
boxes = detector(image)                         # dense, scored, complete
prompt = "Candidates:\n" + "\n".join(f"[{i}] {b.label} at {b.xyxy}" for i, b in enumerate(boxes))
prompt += "\nReturn the indices of objects that are 'damaged'."
indices = parse_ints(mllm(prompt, image=image))  # LLM outputs a few tokens, not coordinates
```

The LLM writes a handful of index tokens instead of coordinates. Speed and recall come from the
detector, judgement from the language model.

---

## When to use what

| Situation | Use |
|---|---|
| Fixed classes, real-time, many objects | a closed-set detector (Parts III–IV) |
| New classes by name, real-time | an OV detector (YOLOE-26, YOLO-World) |
| Descriptive or relational queries, few objects, latency of seconds acceptable | an MLLM (Qwen3-VL, Rex-Omni, LocateAnything) |
| Descriptive queries over dense scenes | Pattern A: detector proposals + MLLM selection |
| No labels yet | Pattern D: MLLM/OV auto-labelling → small detector |
| GUI elements, document fields, OCR boxes | an MLLM with grounding (Qwen3-VL, LocateAnything, Florence-2) |

---

## Key Takeaways

- MLLMs write boxes as text, as location tokens, as quantised coordinate tokens, as indices into
  proposals, or (LocateAnything) as parallel box predictions. Coordinate conventions differ by model
  and even by version.
- They add referring, reasoning and instruction following that detectors cannot express. They pay with
  latency that grows with object count: 1–13 boxes/s versus about 150,000 for a dense detector.
- The 2025–2026 recipe is SFT then reinforcement learning (GRPO) with IoU-based rewards. Detection
  supplies an exact, cheap reward, and RL fixes duplicates and size errors.
- Without native scores, MLLM detectors are evaluated by F1 at IoU thresholds, RefCOCO accuracy or
  cgF1. Don't compare their F1 with a detector's AP.
- In practice, MLLMs work best in hybrids: detector proposes and LLM decides; LLM plans and detector
  executes; detector gates and LLM describes; LLM labels and detector learns.

## Check Yourself

<details class="check"><summary>A Qwen2.5-VL pipeline gives correct boxes on 1024×1024 images but shifted boxes on 1920×1080 images. What is the likely cause?</summary>
Qwen2.5-VL returns absolute coordinates on the *resized* input (its pixel budget and patch rounding
change the size), not on the original image. Map from the model's processed size back to the original
size. Qwen3-VL uses relative 0–1000 coordinates instead, so code written for one version breaks with
the other.</details>

<details class="check"><summary>Estimate decoding time for 40 objects with a model that emits 6 tokens per box at 60 tokens/s, and explain why tiling would not help speed.</summary>
40 × 6 / 60 = 4 seconds plus prefill. Tiling splits the image into more model calls. Each tile pays its
own prefill and the total number of output tokens is unchanged, so tiling helps recall for small
objects but not latency. Batching tiles helps throughput, not per-image latency.</details>

<details class="check"><summary>Why does reinforcement learning work unusually well for MLLM detection, compared with open-ended text tasks?</summary>
The reward is exact and cheap: match the generated boxes to the ground truth with IoU and compute
F1-like scores. Duplicates, misses and loose boxes all lower the reward directly. Open-ended text
needs learned or human reward models. Detection does not.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Kosmos-2 | Peng et al., 2023 | arXiv:2306.14824 | Location tokens on a 32×32 grid, GRIT |
| Shikra | Chen et al., 2023 | arXiv:2306.15195 | Plain-number coordinates |
| Ferret | You et al., 2023 | arXiv:2310.07704 | Hybrid region representation |
| Qwen-VL; Qwen2.5-VL; Qwen3-VL Technical Report | Bai et al., 2023; 2025; Qwen team, 2025 | arXiv:2308.12966; 2502.13923; 2511.21631 | Coordinate conventions; ODinW-13 48.6 |
| Florence-2 | Xiao et al., 2024 | arXiv:2311.06242 | Unified seq2seq, FLD-5B |
| PaliGemma | Beyer et al., 2024 | arXiv:2407.07726 | `<loc>` tokens, y-first |
| Gemini API documentation | Google | ai.google.dev | `box_2d`, ymin-first, 0–1000 |
| ChatRex | Jiang et al., 2024 | arXiv:2411.18363 | Decoupled proposal + LLM selection |
| VLM-R1 | Shen et al., 2025 | arXiv:2504.07615 + github.com/om-ai-lab/VLM-R1 | GRPO for REC / OVD |
| Detect Anything via Next Point Prediction (Rex-Omni) | Jiang et al., 2025 | arXiv:2510.12798 + README | Quantised tokens, SFT + GRPO, F1 evaluation |
| LocateAnything | NVIDIA, 2026 | research.nvidia.com/labs/lpr/locate-anything | Parallel box decoding; 12.7 vs 5.0 vs 1.1 boxes/s |
| SAM 3 README | Meta, 2025 | github.com/facebookresearch/sam3 | Gemini 2.5 cgF1 on SA-Co/Gold |
| Ultralytics LLM interface docs | Ultralytics, 2026 | docs/en/models/llm.md | Detector-gated LLM pattern |

---

**Next:** [Chapter 22 — The YOLO Matrix](./22_yolo_matrix.md) — Part IV, the heart of the book: YOLO
taken apart along eleven orthogonal axes.
