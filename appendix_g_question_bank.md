---
title: "Appendix G — Question Bank"
---

[← Back to Table of Contents](./README.md)

# Appendix G — Question Bank

120 self-test and interview questions, grouped by part, each with a short answer and the chapter that
develops it. Try to answer before opening each one.

---

## Part I — Foundations (Chapters 0–10)

<details class="check"><summary>1. What does a detector output for each object, and in what coordinate conventions?</summary>
A box, a class label and a confidence score. Boxes appear as corners (x1, y1, x2, y2), centre–size
(cx, cy, w, h; YOLO labels, normalised) or top-left–size (x, y, w, h; COCO JSON). Mixing them is one of
the most common bugs. <em>(Ch. 2)</em></details>

<details class="check"><summary>2. Why is plain IoU a poor loss for boxes that do not overlap?</summary>
With zero intersection, small moves of the prediction do not change IoU, so the gradient is zero and the
box is not pulled toward the target. <em>(Ch. 3, App. B)</em></details>

<details class="check"><summary>3. What do GIoU, DIoU and CIoU add to IoU?</summary>
GIoU penalises the empty part of the smallest enclosing box; DIoU penalises centre distance normalised
by the enclosing diagonal; CIoU adds an aspect-ratio consistency term to DIoU. <em>(Ch. 3)</em></details>

<details class="check"><summary>4. Contrast anchors, points and queries as prediction references.</summary>
Anchors are predefined boxes per cell (predict offsets); points are grid-cell centres (predict distances
to edges); queries are learned embeddings decoded into boxes by attention (one per object, DETR).
<em>(Ch. 4)</em></details>

<details class="check"><summary>5. Why must YOLO input sizes be multiples of 32?</summary>
The network downsamples five times by 2 (stride 32 at P5). Each stage must halve an even size exactly so
that the upsample-and-concatenate paths of the neck line up. <em>(Ch. 4, 29)</em></details>

<details class="check"><summary>6. What is label assignment and why does it matter so much?</summary>
It decides which predictions are trained as positives for which objects, and which as background. It
sets what each output learns; most YOLO accuracy gains from v3 to v8 came from better assignment.
<em>(Ch. 5, 30)</em></details>

<details class="check"><summary>7. How does ATSS choose positives without a fixed IoU threshold?</summary>
For each object it takes the top-k closest anchors per level, computes their IoUs, and uses mean + std
of those IoUs as the object's own threshold. <em>(Ch. 5)</em></details>

<details class="check"><summary>8. What is SimOTA's dynamic k?</summary>
Each object gets k = max(1, floor(sum of its top-10 IoUs)) positives, the cheapest under a classification
+ 3 × IoU cost. Well-predicted objects get more positives. <em>(Ch. 5, App. B)</em></details>

<details class="check"><summary>9. What is the TAL metric, and why is β = 6?</summary>
t = score^α · IoU^β with α = 0.5, β = 6 in Ultralytics. The large β makes localisation quality dominate
the ranking of candidates, so well-localised boxes become positives and scores learn to track IoU.
<em>(Ch. 5)</em></details>

<details class="check"><summary>10. Why does one-to-one (Hungarian) matching make NMS unnecessary?</summary>
Each object has exactly one positive; a second prediction on the same object is trained as background,
so the model learns to suppress its own duplicates. <em>(Ch. 5, 18)</em></details>

<details class="check"><summary>11. What problem does focal loss solve?</summary>
Dense detectors evaluate tens of thousands of mostly easy background locations. Focal loss down-weights
well-classified examples by (1 − p)^γ so the hard ones dominate the gradient. <em>(Ch. 6)</em></details>

<details class="check"><summary>12. Why can DFL never reach zero?</summary>
Its minimum is the entropy of the two-hot target, about 0.5 nats per edge on average. With the 1.5 gain,
logged DFL plateaus near 0.75 or above. <em>(Ch. 6, App. B)</em></details>

<details class="check"><summary>13. What is a soft classification target and why use one?</summary>
A target below 1 that encodes localisation quality (IoU or the normalised TAL score). It makes the class
score rank well-localised boxes first, which NMS and AP reward. <em>(Ch. 5, 6)</em></details>

<details class="check"><summary>14. How does greedy NMS fail in crowds?</summary>
True neighbouring objects overlap above the IoU threshold, so the lower-scoring one is deleted.
Soft-NMS, adaptive thresholds or one-to-one heads help. <em>(Ch. 7, 43)</em></details>

<details class="check"><summary>15. What does Soft-NMS change?</summary>
Instead of deleting overlapping boxes it decays their scores (linearly or by a Gaussian of IoU), so a
true neighbour can survive with a lower score. <em>(Ch. 7)</em></details>

<details class="check"><summary>16. How is COCO AP computed?</summary>
Per class, the interpolated precision envelope is averaged at 101 recall points; the result is averaged
over 10 IoU thresholds (0.50:0.05:0.95) and over classes, with at most 100 detections per image.
<em>(Ch. 8)</em></details>

<details class="check"><summary>17. What does a large gap between AP50 and AP75 tell you?</summary>
Objects are found but boxes are loose: localisation quality, label policy, quantisation of the box path
or small objects. <em>(Ch. 8, 45)</em></details>

<details class="check"><summary>18. In COCO, what counts as a small object, and measured where?</summary>
Area below 32² pixels in the original image, not at the network input. <em>(Ch. 8, 41)</em></details>

<details class="check"><summary>19. TIDE shows many background errors. What is a common non-model cause?</summary>
Missing labels: the "false positives" are real, unlabelled objects. <em>(Ch. 8)</em></details>

<details class="check"><summary>20. Why does label consistency matter more than label volume beyond a point?</summary>
Inconsistent rules (box extent, occlusion, class boundaries) put contradictory targets on similar
images, capping AP and especially AP75. <em>(Ch. 9)</em></details>

<details class="check"><summary>21. What does mosaic buy, and what does it cost?</summary>
More objects and contexts per image, scale variety and implicit batch diversity; it costs realism (odd
contexts, shrunken objects), which is why it is switched off for the final epochs. <em>(Ch. 10, 28)</em></details>

<details class="check"><summary>22. Why letterbox rather than resize?</summary>
It preserves aspect ratio, so objects keep their shapes; the padding (114 grey) is undone when mapping
boxes back. <em>(Ch. 10, 28, 36)</em></details>

<details class="check"><summary>23. Why should validation use real, unaugmented images at the deployment resolution?</summary>
Validation estimates deployment performance; augmentations and a different input size change object
sizes and statistics. <em>(Ch. 10, 33)</em></details>

<details class="check"><summary>24. What does a "background image" do in training?</summary>
It provides pure negatives and reduces false positives on scenes without objects; 0–10% of the set is a
common guideline. <em>(Ch. 27)</em></details>

<details class="check"><summary>25. Name two ways to split a dataset that prevent leakage.</summary>
Split by video or capture session, and by location or scene; then deduplicate near-identical frames
across splits. <em>(Ch. 9, 40)</em></details>

---

## Part II — Building blocks and the landscape (Chapters 11–21)

<details class="check"><summary>26. What is the CSP principle?</summary>
Split the channels, transform only part through the heavy blocks, then concatenate and fuse; it adds
gradient paths and cuts compute. <em>(Ch. 11, 29)</em></details>

<details class="check"><summary>27. FPN vs PAN?</summary>
FPN fuses top-down (semantics into high-resolution maps); PAN adds a bottom-up path (localisation detail
back into low-resolution maps). <em>(Ch. 12)</em></details>

<details class="check"><summary>28. What does BiFPN add?</summary>
Repeated bidirectional fusion with learned, normalised weights per input. <em>(Ch. 12)</em></details>

<details class="check"><summary>29. Why decouple the classification and box heads?</summary>
The two tasks prefer different features; separate towers converge faster and improve AP. <em>(Ch. 13)</em></details>

<details class="check"><summary>30. Two-stage vs one-stage detectors: the essential trade-off?</summary>
Two-stage detectors refine a sparse set of proposals (accurate, slower); one-stage detectors predict
densely in one pass (faster, needed focal loss and better assignment to catch up). <em>(Ch. 14)</em></details>

<details class="check"><summary>31. On a microcontroller, what usually decides feasibility first?</summary>
Peak SRAM: the largest activation working set of any layer must fit, before compute or flash.
<em>(Ch. 15)</em></details>

<details class="check"><summary>32. What does FOMO output instead of boxes?</summary>
Object centroids on a coarse grid (per-cell class heat-map), which suits counting and locating on MCUs.
<em>(Ch. 15)</em></details>

<details class="check"><summary>33. List three properties of an edge-friendly detector.</summary>
Static shapes, no NMS or DFL softmax in the graph, NPU-supported activations, plain or re-parameterised
convs, attention only at low resolution, INT8 robustness. <em>(Ch. 16)</em></details>

<details class="check"><summary>34. Why is T4 TensorRT latency only a proxy for NPU latency?</summary>
GPUs run every operator; NPUs have restricted operator sets, INT8-only execution and different memory
hierarchies, so rankings can change. <em>(Ch. 16, 44)</em></details>

<details class="check"><summary>35. What makes the most accurate detectors accurate?</summary>
Large foundation backbones, large-scale pre-training (Objects365 and beyond), strong DETR heads with
auxiliary one-to-many training, high resolution and sometimes test-time augmentation. <em>(Ch. 17)</em></details>

<details class="check"><summary>36. What is DETR's set-prediction formulation?</summary>
A fixed set of queries each predicts one box or "no object"; Hungarian matching assigns one query per
object and the loss is computed on the matched pairs. <em>(Ch. 18)</em></details>

<details class="check"><summary>37. What does deformable attention fix?</summary>
Dense attention over all pixels is slow to train and expensive; deformable attention samples a few
learned points around a reference, making multi-scale attention tractable and convergence much faster.
<em>(Ch. 18)</em></details>

<details class="check"><summary>38. Why do denoising queries help DETR training?</summary>
Noised ground-truth boxes fed as queries give a stable, direct box-refinement signal that does not
depend on unstable Hungarian matches early in training. <em>(Ch. 18)</em></details>

<details class="check"><summary>39. How did RT-DETR make a DETR real-time?</summary>
An efficient hybrid encoder (attention only at the lowest resolution, AIFI, plus CNN cross-scale fusion)
and IoU-aware query selection, with a light decoder. <em>(Ch. 19)</em></details>

<details class="check"><summary>40. What is D-FINE's fine-grained distribution refinement?</summary>
Each decoder layer predicts residual distributions over edge offsets relative to the previous layer's
box, with non-uniform bins, plus self-distillation from the last layer to earlier ones. <em>(Ch. 6, 19)</em></details>

<details class="check"><summary>41. Why do DINOv2-based detectors transfer well to unusual datasets?</summary>
Their backbones learned general visual features from web-scale self-supervision rather than from COCO's
categories, so they adapt better to new domains with little data. <em>(Ch. 19, 34)</em></details>

<details class="check"><summary>42. What replaces the class layer in an open-vocabulary detector?</summary>
Similarity between region features and text (or visual-prompt) embeddings, so classes can be specified
at inference. <em>(Ch. 20)</em></details>

<details class="check"><summary>43. How can YOLO-World/YOLOE run without a text encoder at deployment?</summary>
For a fixed prompt list, the text embeddings are computed once and folded into conv weights
(re-parameterisation), giving an ordinary closed-set detector. <em>(Ch. 20, 38)</em></details>

<details class="check"><summary>44. When would you choose Grounding DINO over YOLOE?</summary>
For offline labelling or complex phrases where accuracy matters more than speed; YOLOE for real-time or
edge use. <em>(Ch. 20, 49)</em></details>

<details class="check"><summary>45. How do multimodal LLMs output boxes?</summary>
As text tokens: quantised coordinates (often normalised to a fixed range) generated autoregressively,
sometimes with special location tokens. <em>(Ch. 21)</em></details>

<details class="check"><summary>46. Why are MLLM detectors slow on images with many objects?</summary>
Each box costs several generated tokens, produced sequentially, so latency grows with the number of
objects; parallel decoding methods attack this. <em>(Ch. 21)</em></details>

<details class="check"><summary>47. What is the most practical use of an MLLM in a detection system today?</summary>
As an annotator or a verifier on candidate crops from a fast detector, handling complex or open-ended
descriptions. <em>(Ch. 21, 47)</em></details>

<details class="check"><summary>48. Name two failure modes of MLLM detection.</summary>
Hallucinated boxes for absent objects, and missed or imprecise boxes for small or numerous objects.
<em>(Ch. 21)</em></details>

---

## Part IV — YOLO in depth (Chapters 22–39)

<details class="check"><summary>49. Why does the book analyse YOLO along orthogonal axes?</summary>
Version numbers bundle changes in data, augmentation, architecture, assignment, recipe and pre-training;
separating the axes shows which change produced which gain and what transfers. <em>(Ch. 22)</em></details>

<details class="check"><summary>50. Two YOLO versions differ by 1.4 AP. What must match before attributing it to architecture?</summary>
Pre-training data, training schedule and recipe, head used for the reported number (o2m+NMS vs o2o),
evaluator, input size and latency conditions. <em>(Ch. 22, 34)</em></details>

<details class="check"><summary>51. How did YOLOv1 decide which prediction learns an object?</summary>
The cell containing the object centre is responsible; of its two predictors, the one with the higher
current IoU becomes the positive. <em>(Ch. 24, 30)</em></details>

<details class="check"><summary>52. Why was YOLOv3's supervision sparse?</summary>
Each object had exactly one positive anchor across all scales; other overlapping predictions were
ignored. <em>(Ch. 24, 30)</em></details>

<details class="check"><summary>53. What are YOLOv4's "bag of freebies" and "bag of specials"?</summary>
Training-only improvements that cost nothing at inference (augmentation, loss, label smoothing) and
small inference-time additions with large gains (SPP, PAN, Mish/CSP blocks). <em>(Ch. 25)</em></details>

<details class="check"><summary>54. Why does YOLOv5 decode centres as 2σ(t) − 0.5?</summary>
Its assignment also uses the two neighbouring cells, which must be able to predict a centre outside
their own cell; the range −0.5…1.5 allows it. <em>(Ch. 30)</em></details>

<details class="check"><summary>55. Name YOLOX's main contributions.</summary>
Anchor-free prediction, a decoupled head, SimOTA assignment and strong augmentation with a no-augmentation
finish. <em>(Ch. 25)</em></details>

<details class="check"><summary>56. What is YOLOv7's lead-guided assignment?</summary>
The auxiliary head is trained on coarse targets and the lead head on fine targets, both derived from the
lead head's predictions. <em>(Ch. 25, 30)</em></details>

<details class="check"><summary>57. Summarise YOLOv8's changes from YOLOv5.</summary>
C2f blocks, anchor-free decoupled head with DFL, no objectness, TAL assignment, and the unified
`ultralytics` package. <em>(Ch. 26)</em></details>

<details class="check"><summary>58. What is YOLOv10's consistent dual assignment?</summary>
Train a one-to-many head (top-10) and a one-to-one head (top-1) with the same matching metric; deploy only
the one-to-one head, so no NMS is needed. <em>(Ch. 5, 26)</em></details>

<details class="check"><summary>59. What did YOLO11 change architecturally?</summary>
C3k2 blocks (C2f with C3k inner blocks at larger scales), C2PSA attention at P5, and a depthwise-separable
class head. <em>(Ch. 26, 29)</em></details>

<details class="check"><summary>60. What is YOLOv12's area attention and its deployment caveat?</summary>
Self-attention within a few strips of the feature map, cutting cost about 4×; it relies on FlashAttention
for GPU speed and is harder to deploy on CPUs and NPUs. <em>(Ch. 26, 29)</em></details>

<details class="check"><summary>61. Name four deployment-driven changes in YOLO26.</summary>
DFL removed (L1 on distances), optional NMS-free one-to-one head, ProgLoss and STAL in training, MuSGD,
plus Objects365 pre-training. <em>(Ch. 26)</em></details>

<details class="check"><summary>62. Which head does YOLO26 use by default for predict, val and export?</summary>
The one-to-many head with NMS (`nms=None`); `nms=False` selects the one-to-one head with `(N, 300, 6)`
output. <em>(Ch. 26, 36)</em></details>

<details class="check"><summary>63. Describe the YOLO detection label format.</summary>
One text file per image (same stem), one line per object: zero-based class, then cx, cy, w, h
normalised to [0, 1]. <em>(Ch. 27)</em></details>

<details class="check"><summary>64. Training reports "0 images, 1,200 backgrounds". What happened?</summary>
No label files were found: wrong `labels/` path, a directory name breaking the images → labels
substitution, or mismatched stems. <em>(Ch. 27)</em></details>

<details class="check"><summary>65. What does Ultralytics' `cls_pw` do?</summary>
Sets the power of inverse-frequency class weights in the BCE classification loss (0 off, 1 full); in
YOLOv5 the same name meant BCE positive weight. <em>(Ch. 27, 30)</em></details>

<details class="check"><summary>66. What were the YOLO26 checkpoints pre-trained on?</summary>
Objects365v1 for 150 epochs, then COCO for 40–245 epochs depending on size; no COCO-from-scratch run.
<em>(Ch. 27, 31)</em></details>

<details class="check"><summary>67. What is `close_mosaic` and why does it help?</summary>
Turning mosaic off for the last epochs (10 by default) so the model finishes on realistic, full images
matching validation. <em>(Ch. 28)</em></details>

<details class="check"><summary>68. Why can an exported model score lower than `model.val()` without any bug?</summary>
`val` uses rectangular batches with minimal padding; a fixed square export pads more and shrinks objects
relative to the input; FP16/INT8 and NMS settings also differ. <em>(Ch. 28, 36)</em></details>

<details class="check"><summary>69. Why are three sequential 5×5 max pools equivalent to SPP(5, 9, 13)?</summary>
Stride-1 max pools compose: two 5×5 equal a 9×9 and three equal a 13×13, so the intermediate outputs are
exactly SPP's branches. <em>(Ch. 29)</em></details>

<details class="check"><summary>70. In YOLO26n, where do parameters and FLOPs live?</summary>
About 72% of parameters are at stride 32, but most compute is at strides 8–16 (30% and 32% of MACs).
<em>(Ch. 29)</em></details>

<details class="check"><summary>71. Why do YOLO26m and YOLO26l have identical widths?</summary>
Both use width 1.0 with `max_channels` 512; they differ only in depth (one vs two repeats per C3k2).
<em>(Ch. 29)</em></details>

<details class="check"><summary>72. Why is YOLO26n's head a third the size of YOLO11n's?</summary>
The box tower width is max(16, ch/4, 4·reg_max); with reg_max = 1 it drops from 64 to 16 channels.
<em>(Ch. 29)</em></details>

<details class="check"><summary>73. Does TAL assign objects to pyramid levels by size?</summary>
No. Candidates come from all levels at once and the top-10 by alignment metric win; large objects end up
on coarse levels because those predict them better. <em>(Ch. 30)</em></details>

<details class="check"><summary>74. What does STAL do?</summary>
For candidate selection only, boxes with sides under 16 px are enlarged to 16 px around their centre, so
tiny objects always get grid points; regression still targets the true box. <em>(Ch. 30, 41)</em></details>

<details class="check"><summary>75. What is ProgLoss?</summary>
YOLO26's schedule for combining head losses: the one-to-many weight decays linearly from 0.8 to 0.1 over
training while the one-to-one weight rises. <em>(Ch. 30)</em></details>

<details class="check"><summary>76. Why does the one-to-one head train on detached features?</summary>
So its sparse, harder loss updates only its own head and does not disturb the backbone, which is shaped
by the dense one-to-many loss. <em>(Ch. 29, 30)</em></details>

<details class="check"><summary>77. Your YOLO26 log shows higher box loss than YOLO11's. Is localisation worse?</summary>
Not necessarily: YOLO26 logs the one-to-one branch's losses (one positive per object, harder to fit), and
gains differ. Compare mAP and AP75 instead. <em>(Ch. 30)</em></details>

<details class="check"><summary>78. Why might your `lr0` be ignored?</summary>
With `optimizer=auto`, the trainer chooses MuSGD (lr 0.01) for over 10,000 steps or AdamW with
lr = 0.002·5/(4+nc), overriding `lr0`. <em>(Ch. 31)</em></details>

<details class="check"><summary>79. What is `nbs` and what does it imply for batch 128?</summary>
The nominal batch of 64: smaller batches accumulate gradients to 64 images per step; at 128 the step and
weight decay are about twice as large. <em>(Ch. 31)</em></details>

<details class="check"><summary>80. Why does warm-up start the bias learning rate high?</summary>
Early steps mostly learn output biases (class priors, box offsets) while freshly initialised weights are
protected by near-zero learning rates. <em>(Ch. 31)</em></details>

<details class="check"><summary>81. Why does validation mAP lag early in training?</summary>
Validation uses the EMA weights, which still average over older, worse weights while the decay ramps up.
<em>(Ch. 31)</em></details>

<details class="check"><summary>82. What does `cls_remap` do when fine-tuning?</summary>
Copies class-head rows from the checkpoint by class name, so classes shared with the pre-training
dataset (exact names) start from trained weights. <em>(Ch. 31)</em></details>

<details class="check"><summary>83. Which metric selects `best.pt` in current Ultralytics?</summary>
mAP50-95 alone (older releases used 0.1·mAP50 + 0.9·mAP50-95). <em>(Ch. 31)</em></details>

<details class="check"><summary>84. Describe the Ultralytics tuner's mutation step.</summary>
Pick a parent from the top 9 weighted by fitness, mutate each normalised gene with probability 0.5 by
Gaussian noise (σ = 0.2 × gain), and after 30 results sometimes sample from a covariance fitted to the
elite. Each iteration is a full training run. <em>(Ch. 32)</em></details>

<details class="check"><summary>85. What bias do short tuning runs introduce?</summary>
They favour fast-converging settings (high LR, weak augmentation) that lose in full-length training.
<em>(Ch. 32)</em></details>

<details class="check"><summary>86. How do you know a hyperparameter change is real?</summary>
Compare means over several seeds against the seed spread, ideally with paired bootstrap intervals on a
held-out test set. <em>(Ch. 32, 33)</em></details>

<details class="check"><summary>87. Why does `val` use conf = 0.001?</summary>
To keep nearly all candidates so the precision–recall curve reaches high recall; a higher threshold
truncates the curve and lowers mAP. <em>(Ch. 33)</em></details>

<details class="check"><summary>88. At what confidence are the printed P and R measured?</summary>
At the threshold maximising the smoothed mean F1 over classes for that run, not at a fixed or deployment
threshold. <em>(Ch. 33)</em></details>

<details class="check"><summary>89. How does Ultralytics' matching differ from COCO's?</summary>
Per IoU threshold it matches greedily by IoU, while COCO walks predictions in descending confidence; AP
integration also uses a trapezoid rule over 101 points. Measured differences were within 0.4 AP.
<em>(Ch. 33)</em></details>

<details class="check"><summary>90. How wide is a 95% bootstrap interval on mAP with 200 validation images?</summary>
In the book's measurement, about 3.8 points (±1.9); with 50 images 6.8, with 1,000 images 1.5.
<em>(Ch. 33)</em></details>

<details class="check"><summary>91. On an accuracy–latency plot, why are hollow (NMS-needing) points optimistic?</summary>
Their latency excludes NMS, while NMS-free points report end-to-end time. <em>(Ch. 34)</em></details>

<details class="check"><summary>92. Is YOLO26 faster than YOLO11 on CPUs at every size?</summary>
No. Ultralytics' CPU ONNX table shows YOLO26 faster at n (−31%) and s (−3%) but slower at m, l and x.
<em>(Ch. 34)</em></details>

<details class="check"><summary>93. What does RF100-VL show that COCO does not?</summary>
Fine-tuning transfer across 100 diverse datasets; rankings there differ from COCO (in RF-DETR's harness,
YOLO26-N scored below YOLO11-N). <em>(Ch. 34)</em></details>

<details class="check"><summary>94. What extra obligation does AGPL-3.0 add over GPL-3.0?</summary>
Users interacting with the software over a network must be offered the source; GPL triggers only on
distribution. <em>(Ch. 35)</em></details>

<details class="check"><summary>95. Can you ship YOLO-NAS pre-trained weights commercially?</summary>
No; the code is Apache-2.0 but the pre-trained weights carry a non-commercial licence. <em>(Ch. 35)</em></details>

<details class="check"><summary>96. What are the output shapes of a YOLO26 ONNX export with nms=None and nms=False at 640?</summary>
(N, 4 + nc, 8400) raw one-to-many (cx, cy, w, h + sigmoid scores) and (N, 300, 6) one-to-one
(x1, y1, x2, y2, score, class). <em>(Ch. 36)</em></details>

<details class="check"><summary>97. All scores of an exported model lie between 0.5 and 0.73. Why?</summary>
A second sigmoid was applied to scores already sigmoided in the graph. <em>(Ch. 36)</em></details>

<details class="check"><summary>98. Why might an NMS-free export still need host NMS on an NPU?</summary>
The toolchain cannot run TopK/Gather on the accelerator, so the exporter falls back to the one-to-many
head. <em>(Ch. 36, 44)</em></details>

<details class="check"><summary>99. In the book's TinyYOLO, what did removing DFL and NMS cost?</summary>
About 2.3 AP each (0.833 → 0.810 → 0.787), with the one-to-one head losing mostly on large objects while
having the best recall. <em>(Ch. 37, 39)</em></details>

<details class="check"><summary>100. What distinguishes evidence grade A from B in Chapter 37?</summary>
A: measured in the book or reproduced across independent codebases; B: an ablation in the authors' own
paper and codebase only. <em>(Ch. 37)</em></details>

<details class="check"><summary>101. How does YOLO instance segmentation produce masks?</summary>
A Proto module outputs 32 prototype masks at stride 4; each box predicts 32 coefficients; the mask is the
sigmoid of their linear combination, cropped to the box. <em>(Ch. 38)</em></details>

<details class="check"><summary>102. Why do OBB models use ProbIoU?</summary>
Rotated boxes as Gaussians give a smooth overlap measure free of the angle periodicity and boundary
discontinuities that break direct angle regression. <em>(Ch. 38, 43)</em></details>

<details class="check"><summary>103. Why must low-confidence detections reach ByteTrack?</summary>
Its second association pass uses them to keep tracks alive through occlusion; filtering at 0.5 first
breaks tracks. <em>(Ch. 38)</em></details>

<details class="check"><summary>104. Why did TinyYOLO's end-to-end model have more parameters than the DFL model?</summary>
It carries two full heads during training; the narrower box towers do not offset the second head.
Deployment keeps one. <em>(Ch. 39)</em></details>

<details class="check"><summary>105. What is the first sanity check when building a detector from scratch?</summary>
Overfit a single small batch with augmentation off; failure means a pipeline bug, not a hard problem.
<em>(Ch. 39, 40)</em></details>

---

## Parts V–VII — Training strategy, deployment and practice (Chapters 40–52)

<details class="check"><summary>106. You merge your dataset with COCO to gain a "person" class. Why might person AP be poor on your images?</summary>
Unlabelled people in your images were trained as background. Pseudo-label them or use a partial-label
loss. <em>(Ch. 40)</em></details>

<details class="check"><summary>107. What is repeat-factor sampling?</summary>
Images are repeated by r = max over their classes of max(1, √(t / f_c)), oversampling images with rare
classes (t = 0.001 on LVIS). <em>(Ch. 40)</em></details>

<details class="check"><summary>108. Why does gradient accumulation not fix small-batch BatchNorm?</summary>
BN statistics come from each forward batch; accumulation only enlarges the optimiser's batch. Use frozen
BN, SyncBN or GroupNorm. <em>(Ch. 40)</em></details>

<details class="check"><summary>109. A 30-px object in a 1920 × 1080 frame: how big is it at imgsz 640?</summary>
Scale 640/1920 = 0.33, so about 10 px: in the tiny regime. <em>(Ch. 41)</em></details>

<details class="check"><summary>110. What does a P2 head cost YOLO26n at 640?</summary>
About +36% FLOPs (5.5 → 7.5 GFLOPs) and 34,000 instead of 8,400 candidate points, with almost no
parameter change. <em>(Ch. 41)</em></details>

<details class="check"><summary>111. Why does SAHI merge tile detections with intersection-over-smaller?</summary>
An object cut by a tile edge yields a partial box inside a fuller one; their IoU is low but IoS high, so
IoS merges these duplicates. <em>(Ch. 41)</em></details>

<details class="check"><summary>112. What is confirmation bias in semi-supervised detection?</summary>
The student learns the teacher's pseudo-label mistakes, which then reinforce themselves; thresholds,
EMA teachers and robust assignment mitigate it. <em>(Ch. 42)</em></details>

<details class="check"><summary>113. When does detector distillation pay off?</summary>
When the teacher is clearly better on your data, the student is capacity-limited, and you compare
against the same student trained without distillation, with seeds. <em>(Ch. 42)</em></details>

<details class="check"><summary>114. What does streaming AP measure?</summary>
Accuracy of the latest available output at each moment, so slow detectors are penalised for describing
an older scene. <em>(Ch. 43)</em></details>

<details class="check"><summary>115. Why is YOLO26n memory- or overhead-bound on a T4?</summary>
Its arithmetic intensity is about 62 FLOP/byte against a ridge of about 200, and small kernels leave the
GPU underused; n → s costs 1.5× latency for 3.8× FLOPs. <em>(Ch. 44)</em></details>

<details class="check"><summary>116. Why did whole-graph INT8 give 0.0 mAP in the book's measurement?</summary>
Coordinates (up to ~660 px) and scores (0–1) share one quantised output tensor; every score rounds to
the same value. Keeping the head output and decoding in float recovered all but 0.8–1.2 points.
<em>(Ch. 45)</em></details>

<details class="check"><summary>117. What does Ultralytics' QAT (`train(quantize=8)`) keep in float?</summary>
The head's output layers and the DFL conv; activation and weight ranges are calibrated once and then
fixed while the weights adapt. <em>(Ch. 45)</em></details>

<details class="check"><summary>118. A GPU timing loop reports 0.3 ms for YOLO26n. What is likely missing?</summary>
Device synchronisation (timing kernel launches instead of execution) and warm-up. <em>(Ch. 46)</em></details>

<details class="check"><summary>119. After retraining, alarms double although mAP improved. Why?</summary>
Thresholds were not re-tuned for the new model's score distribution. <em>(Ch. 47)</em></details>

<details class="check"><summary>120. In what order does the decision guide ask its questions, and why?</summary>
Hardware, licence, labels, object properties, then the accuracy–latency point: from the constraints that
cannot move to those that can be traded. <em>(Ch. 51)</em></details>
