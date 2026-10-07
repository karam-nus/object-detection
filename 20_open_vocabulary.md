---
title: "Chapter 20 — Open-Vocabulary & Grounded Detection"
---

[← Back to Table of Contents](./README.md)

# Chapter 20 — Open-Vocabulary & Grounded Detection

> *"A closed-set detector answers 'which of these 80 things is here?'. An open-vocabulary detector answers 'where is the thing I just described?'"*

## Overview

Every detector so far ends in a $K$-way classifier fixed at training time. **Open-vocabulary (OV)
detection** replaces it with a similarity between region features and *text embeddings*. Any phrase
then becomes a class, at inference time, with no retraining. **Grounding** extends this from category
names to referring phrases ("the red mug on the left"). This chapter covers the mechanism, the model
lineage (OWL-ViT/v2, GLIP, Grounding DINO, YOLO-World, YOLOE, DINO-X, T-Rex2, LLMDet, SAM 3), how OV
models are evaluated and how those evaluations leak, and how to use them in practice: as detectors, as
auto-labellers, and as the zero-shot first stage of a pipeline that ends in a small closed-set model
(Chapter 49).

<div class="diagram">
<div class="diagram-title">Closed-set vs open-vocabulary heads</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Closed set</div>
    <ul>
      <li>region feature f ∈ ℝᵈ</li>
      <li>logits = W f, W ∈ ℝ^(K×d) learned</li>
      <li>K fixed at training</li>
      <li>new class ⇒ new data + retrain</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Open vocabulary</div>
    <ul>
      <li>region feature f ∈ ℝᵈ</li>
      <li>logits = τ · cos(f, T(prompt_k)) for each prompt</li>
      <li>prompts chosen at inference</li>
      <li>new class ⇒ new text (or an example image)</li>
    </ul>
  </div>
</div>
</div>

---

## The mechanism

A text encoder $T$ (CLIP's, or BERT in GLIP/Grounding DINO) maps each prompt to an embedding $t_k$.
A region feature $f_i$ (from an anchor point, a query or a box) is projected into the same space. The
class logit is a scaled similarity:

$$\text{logit}_{ik} = \frac{\langle f_i, t_k \rangle}{\tau \,\|f_i\| \|t_k\|} + b$$

Training needs **region–text pairs**: detection datasets with class names (Objects365, COCO, LVIS),
grounding datasets with phrases (GoldG, RefCOCO, Flickr30k Entities), and, at scale, **pseudo-labelled
image–caption pairs** from the web. The last source is what gives OV detectors their breadth, and
also their evaluation problems (below).

Two architectural styles exist:

| Style | Text interacts with image features… | Examples | Speed | Strength |
|---|---|---|---|---|
| **Late fusion** (dual encoder) | only at the final dot product | OWL-ViT/v2, YOLO-World (offline mode), YOLOE | fast; text embeddings can be cached or folded into weights | Real-time, many classes |
| **Early / deep fusion** | throughout the encoder and decoder (cross-attention) | GLIP, Grounding DINO, DINO-X, MDETR | slower; recomputed per prompt set | Referring expressions, compositional phrases |

---

## The lineage

### OWL-ViT and OWLv2 (Google)

**OWL-ViT** (2022) removes CLIP's final pooling and attaches box and class heads to each ViT output
token. It is a minimal late-fusion design that also supports **one-shot image-conditioned detection**:
use an example crop's embedding instead of text. **OWLv2** (2023) scales training with **OWL-ST**
self-training: an OWL-ViT teacher pseudo-labels about 10⁹ web images with n-gram queries derived from
their captions, and a student trains on those labels. In SAM 3's comparison table, OWLv2 scores 45.5 box
AP on LVIS and 46.1 on COCO.

### GLIP — detection as phrase grounding

GLIP (2022) unified detection and grounding. It treats every detection dataset as grounding data
("prompt = all class names joined") and uses **deep fusion** (cross-attention between image and text
features in the encoder). It scaled to millions of grounding pairs with self-training.

### Grounding DINO (IDEA)

| | |
|---|---|
| **Introduced** | Liu et al., 2023 (arXiv:2303.05499, ECCV 2024) |
| **Lineage** | DINO + GLIP-style fusion |
| **Paradigm** | query-based, deep fusion, NMS-free |
| **Assignment** | Hungarian with contrastive region–token alignment loss |
| **Post-processing** | box and text thresholds per token |
| **COCO AP (sizes)** | **52.5 zero-shot** (L, without COCO data); 48.4 zero-shot (T); 63.0 / 57.2 fine-tuned |
| **Latency** | not real-time (Swin backbones, BERT, cross-modal layers) |
| **Pre-training** | Objects365, GoldG, Cap4M, OpenImages (L) |
| **License** | Apache-2.0 |
| **Known failure modes** | prompt sensitivity; long prompts truncated (BERT token limit); counting errors in dense scenes |

Its three fusion points are a **feature enhancer** (bi-directional image↔text cross-attention),
**language-guided query selection** (queries initialised from image tokens most similar to the text),
and a **cross-modality decoder** (queries attend to both modalities). Grounding DINO became the default
zero-shot detector of 2023–2024 and the core of the Grounded-SAM auto-labelling pipelines.
**Grounding DINO 1.5/1.6** (Pro and Edge) continued the line as closed-weight API models. **DINO-X**
(2024) is its successor.

### DINO-X (IDEA)

DINO-X Pro reports **56.0 AP on COCO zero-shot, 59.8 on LVIS-minival and 52.4 on LVIS-val**, with large
gains on LVIS rare classes over Grounding DINO 1.6 Pro. It was trained on a "Grounding-100M" dataset and
supports text prompts, visual prompts and a **prompt-free** "detect everything" mode. It is available
through an API, not as open weights.

### T-Rex2 — visual prompts

T-Rex2 (2024) adds **visual prompts**: draw boxes or points on a few examples and detect all similar
objects. It handles things that are easy to show and hard to name: a specific screw type, a cell
morphology, a product package. Text and visual prompts complement each other in its design.

### YOLO-World (Tencent AI Lab)

| | |
|---|---|
| **Introduced** | Cheng et al., 2024 (arXiv:2401.17270, CVPR 2024) |
| **Lineage** | YOLOv8 + CLIP text encoder |
| **Paradigm** | dense, late-ish fusion (RepVL-PAN) |
| **Assignment** | TAL with region–text contrastive loss |
| **Post-processing** | NMS |
| **COCO AP (sizes)** | v2 zero-shot COCO (Ultralytics): S 37.7 · M 43.0 · L 45.8 · X 47.1 |
| **Latency** | original paper: 35.4 AP on LVIS at 52 FPS on a V100 (L) |
| **Pre-training** | Objects365 + GoldG + CC3M pseudo-labels |
| **License** | GPL-3.0 (original repo); AGPL-3.0 in Ultralytics |
| **Known failure modes** | text–image fusion weaker than deep-fusion models on complex phrases |

Its key deployment idea is **prompt-then-detect**. Encode your vocabulary once, then *re-parameterise*
the text embeddings into the network's weights (its RepVL-PAN and classification layers). The deployed
model is a plain YOLO with no text encoder at runtime. In Ultralytics: `model.set_classes([...])`,
then `model.save()` or `export()`.

### YOLOE (Tsinghua → Ultralytics YOLOE-26)

YOLOE (2025) generalises YOLO-World to three prompt types with **zero inference overhead** after
re-parameterisation:

- **RepRTA** (re-parameterisable region–text alignment): a lightweight auxiliary network refines text
  embeddings during training and folds into the classification head afterwards.
- **SAVPE** (semantic-activated visual prompt encoder): encodes example boxes into prompt embeddings.
- **LRPC** (lazy region–prompt contrast): **prompt-free** mode that matches regions against a large
  built-in vocabulary only when needed.

The authors report YOLOE-v8-S at +3.5 AP over YOLO-Worldv2-S on LVIS with 3× less training cost.
**YOLOE-26** (Ultralytics, 2026) reaches **40.6 AP** on LVIS minival with text prompts, 38.5 with visual
prompts and 31.1 prompt-free (YOLOE-26x, the non-e2e head; the NMS-free head trails by 1.1–2.3 AP).

### LLMDet — supervision from a language model

LLMDet (CVPR 2025) co-trains an open-vocabulary detector with an LLM that generates detailed
**image-level captions** and **region-level short captions** (the GroundingCap-1M dataset). The
caption-generation loss improves the detector's vision–language alignment, and the improved detector in
turn helps build a better multimodal LLM. It is the clearest example of LLMs used as *supervisors* for
detectors rather than as detectors (Chapter 21 covers the latter).

### SAM 3 — concepts, not classes

| | |
|---|---|
| **Introduced** | Meta, 2025 (arXiv:2511.16719; ICLR 2026) |
| **Lineage** | SAM 2 tracker + a DETR-style concept detector sharing one vision encoder |
| **Paradigm** | promptable concept segmentation: text noun phrase and/or image exemplars → all instances |
| **Assignment** | DETR-style matching, with a **presence head** |
| **Post-processing** | thresholds; tracker for video |
| **COCO AP (sizes)** | 56.4 box AP (COCO), 53.6 box AP (LVIS), per the SAM 3 README |
| **Latency** | 848 M parameters; GPU-class |
| **Pre-training** | data engine with ~4 M unique concepts including hard negatives |
| **License** | SAM license (see repo) |
| **Known failure modes** | still well below humans on SA-Co/Gold (55.7 vs 74.0 cgF1 for boxes) |

SAM 3's **presence head** decouples *recognition* ("is this concept in the image at all?") from
*localisation* ("which instances?"). OV detectors are notoriously prone to false positives for absent
concepts, and the presence head targets exactly that. On its SA-Co/Gold benchmark, SAM 3 scores 55.7
cgF1 for box detection vs 24.5 for OWLv2, 22.5 for DINO-X and 14.4 for Gemini 2.5. Humans score 74.0.
Fine-tuned on RF100-VL, it reaches 61.6 AP (reported in its paper and quoted by RF-DETR's README).

---

## How OV detectors are evaluated, and how evaluations leak

| Benchmark | What it measures | Note |
|---|---|---|
| **LVIS minival / val, AP$_r$** | 1,203 classes, long tail | The main OV metric; AP$_r$ = rare classes |
| **COCO zero-shot** | 80 classes, model trained without COCO *images* | COCO's classes are almost all in Objects365, so the *classes* are not novel |
| **ODinW-13 / 35** | transfer to 13 / 35 diverse small datasets | Closer to real deployment; Qwen3-VL reports 48.6 mAP on ODinW-13 (Ch 21) |
| **RF100-VL** | 100 domains, zero/few-shot and fine-tuned | Includes medical, aerial, industrial |
| **SA-Co** | 4 M-concept benchmark with hard negatives | SAM 3's; measures "is it there at all" as well |

<div class="callout warn"><span class="callout-title">Trap</span>"Zero-shot" usually means
"no images from this benchmark in training", not "never saw these categories". Objects365 contains
nearly every COCO class. Some grounding corpora contain COCO <em>images</em> (GroundingCap-1M does,
as one 2026 paper explicitly notes). For your own task, the only meaningful zero-shot test is your own
data, with your own class names.</div>

---

## Using OV detectors in practice

**Prompting.** Class names are not always the best prompts. Try synonyms and descriptive phrases
("forklift" vs "industrial forklift truck"), and keep a small validation set to choose prompts
empirically. Several prompts for one class can be ensembled. Deep-fusion models are sensitive to prompt
*order* and length. BERT's 256-token limit truncates long vocabularies in Grounding DINO, so batch
large vocabularies.

**Thresholds.** OV scores are similarity scores calibrated differently per prompt. A single global
threshold is rarely right. Calibrate per class on validation data (Chapter 47).

**Speed.** If your vocabulary is fixed at deployment, use a re-parameterisable model (YOLO-World,
YOLOE, YOLOE-26) and fold the text in. You then pay closed-set YOLO latency.

**The auto-label → distil pattern.** The most common production use of OV detectors in 2026 is not
deployment but **labelling**:

<div class="diagram">
<div class="flow-h">
  <div class="flow-node purple">unlabelled images</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">OV detector / SAM 3 / VLM<small>text prompts</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node">human review<small>fix, reject</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">train YOLO26n / D-FINE-N<small>closed set, fast</small></div>
</div>
</div>

Chapter 49 runs this end to end and measures what is lost against human labels.

---

## Choosing an OV model

| Need | Choose | Why |
|---|---|---|
| Real-time, fixed vocabulary, edge | YOLOE-26 / YOLO-World (re-parameterised) | Closed-set YOLO latency after folding |
| Best open-weight zero-shot boxes | Grounding DINO-L, LLMDet, OWLv2-L | Strong deep or late fusion; open weights |
| Best quality, API acceptable | DINO-X Pro | 56.0 COCO / 59.8 LVIS-minival zero-shot |
| Concepts with masks + video tracking | SAM 3 | Presence head; detector + tracker |
| Things that are hard to name | T-Rex2, YOLOE visual prompts, OWL one-shot | Prompt by example |
| Referring expressions and reasoning | MLLMs (Chapter 21) | Language understanding beyond nouns |

---

## Key Takeaways

- OV detection replaces a fixed $K$-way classifier with region–text similarity. Classes become
  inference-time inputs.
- Late fusion (OWL, YOLO-World, YOLOE) is fast and can fold text into the weights. Deep fusion
  (GLIP, Grounding DINO, DINO-X) handles complex phrases at higher cost.
- Grounding DINO (52.5 zero-shot COCO) set the open-weight standard. DINO-X Pro (56.0) and SAM 3 (56.4
  box AP COCO, presence head) lead in 2025–2026.
- YOLO-World and YOLOE make OV real-time. YOLOE-26x reaches 40.6 LVIS AP with text prompts.
- "Zero-shot" benchmarks leak through overlapping categories (Objects365 ⊃ COCO classes) and sometimes
  overlapping images. Test on your own data.
- In production, OV models earn their keep as auto-labellers for small closed-set detectors.

## Check Yourself

<details class="check"><summary>How does YOLO-World run with no text encoder at deployment time?</summary>
With a fixed vocabulary, the text embeddings are constants. They are computed once and re-parameterised
into the network's weights (the vision–language PAN and the classification layer's weights). The
exported model is a standard YOLO whose class logits are those precomputed similarities.</details>

<details class="check"><summary>Why does SAM 3 add a presence head, and which OV failure does it target?</summary>
OV detectors produce false positives for concepts that are absent: something always scores highest
against the prompt. The presence head predicts once per image whether the concept is present at all,
which decouples recognition from per-instance localisation. Low presence suppresses all instances. This
targets the hard-negative case that SA-Co explicitly tests.</details>

<details class="check"><summary>A paper claims 50 AP "zero-shot on COCO". What should you ask before trusting that for your warehouse-robot task?</summary>
Which datasets were used for training. If Objects365, COCO's classes were seen. If a grounding corpus,
COCO images may have been included. Then ask how the model performs on *your* classes and imagery, with
your prompts. Measure it on a small labelled sample of your data, ideally with per-class thresholds.</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| CLIP | Radford et al., 2021 | arXiv:2103.00020 | Image–text embeddings |
| Simple Open-Vocabulary Object Detection (OWL-ViT) | Minderer et al., 2022 | arXiv:2205.06230 | Late-fusion OV, one-shot |
| Scaling Open-Vocabulary Object Detection (OWLv2) | Minderer et al., 2023 | arXiv:2306.09683 | OWL-ST self-training |
| Grounded Language-Image Pre-training (GLIP) | Li et al., 2022 | arXiv:2112.03857 | Detection as grounding |
| Grounding DINO | Liu et al., 2023 | arXiv:2303.05499 + README | Fusion design; 52.5 zero-shot, 63.0 fine-tuned |
| DINO-X | Ren et al., 2024 | arXiv:2411.14347 + DINO-X-API README | 56.0 COCO / 59.8 LVIS-minival / 52.4 LVIS-val |
| T-Rex2 | Jiang et al., 2024 | arXiv:2403.14610 | Visual prompts |
| YOLO-World | Cheng et al., 2024 | arXiv:2401.17270 + README; Ultralytics docs | Prompt-then-detect; 35.4 LVIS @ 52 FPS; zero-shot COCO table |
| YOLOE | Wang et al., 2025 | arXiv:2503.07465 + README | RepRTA, SAVPE, LRPC |
| Ultralytics YOLO26 docs (YOLOE-26) | Ultralytics, 2026 | docs/en/models/yolo26.md | 40.6 / 38.5 / 31.1 LVIS AP |
| LLMDet | Fu et al., 2025 | CVPR 2025 / github.com/iSEE-Laboratory/LLMDet | LLM-supervised OV detection, GroundingCap-1M |
| SAM 3: Segment Anything with Concepts | Carion et al. (Meta), 2025 | arXiv:2511.16719 + README | Presence head, 848 M params, SA-Co results table |
| VL-SAM-v3 | 2026 | arXiv:2605.03456 | Note that GroundingCap-1M includes COCO images |
| RF-DETR README | Roboflow, 2026 | github.com/roboflow/rf-detr | SAM 3 RF100-VL 61.6 |

---

**Next:** [Chapter 21 — LLM-Based Detection](./21_llm_based_detection.md) — from text *prompts* to text
*outputs*: multimodal language models that write boxes as tokens.
