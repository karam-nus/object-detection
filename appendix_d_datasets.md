---
title: "Appendix D — Dataset Index"
---

[← Back to Table of Contents](./README.md)

# Appendix D — Dataset Index

Detection datasets by purpose. Sizes are approximate and refer to the commonly used releases; check each
dataset's page for the exact split you use. **Licence terms matter for commercial use** (Chapter 35):
many datasets allow research use only, and image licences can differ from annotation licences. The
licence column summarises the common understanding; read the dataset's own terms before relying on it.

---

## General-purpose benchmarks

| Dataset | Images | Classes | Boxes | Notes | Licence (summary) | Good for |
|---|---|---|---|---|---|---|
| **PASCAL VOC 2007/2012** | ≈ 16.5k trainval (07+12) | 20 | ≈ 40k | AP50 metric (11-point for 2007) | Flickr images; research use | History, quick baselines |
| **COCO 2017** | 118k train, 5k val, ≈ 41k test | 80 | ≈ 860k (train) | AP50:95, crowd regions, area ranges | Annotations CC BY 4.0; images under Flickr terms | The standard benchmark |
| **Objects365** | ≈ 1.7–2M | 365 | ≈ 30M | Pre-training for YOLO26, DETRs | Research terms; check | Pre-training |
| **Open Images V7** | ≈ 1.9M (boxed subset) | 600 boxable | ≈ 16M | Hierarchy, group-of boxes, verified negatives | Annotations CC BY 4.0; images CC BY 2.0 (per image) | Large-scale, permissive-ish data |
| **LVIS v1** | ≈ 164k (COCO images) | 1,203 | ≈ 2M masks | Long tail, federated labels | CC BY 4.0 annotations | Long-tail evaluation |
| **V3Det** | ≈ 245k | ≈ 13,000 | ≈ 1.7M | Vast vocabulary | Research; check | Large-vocabulary detection |
| **COCO8 / COCO128** | 8 / 128 | 80 | — | Ultralytics smoke-test subsets of COCO train | As COCO | Pipeline tests only |

## Transfer and multi-domain

| Dataset | Content | Good for |
|---|---|---|
| **ODinW** (Object Detection in the Wild) | 35 public datasets | Zero/few-shot transfer of open-vocabulary detectors |
| **RF100 / RF100-VL** | 100 datasets across aerial, medical, industrial, documents and more | Fine-tuning transfer; the most informative benchmark for "will it work on my data" |
| **Roboflow Universe** | Hundreds of thousands of community datasets | Finding a starting dataset (check each licence and label quality) |

## Crowds, people and faces

| Dataset | Size | Notes |
|---|---|---|
| **CrowdHuman** | 15k train, 4.4k val, 5k test; ≈ 470k people | Full, visible and head boxes; about 23 people per image |
| **CityPersons** | ≈ 5k (Cityscapes images) | Pedestrians in driving scenes; MR⁻² metric |
| **WIDER FACE** | ≈ 32k images, ≈ 394k faces | Scale, occlusion and pose variation |
| **TinyPerson** | ≈ 1.6k images, ≈ 72k people | People under 20 px |

## Driving and 3D

| Dataset | Notes |
|---|---|
| **KITTI** | 7.5k training images with 2D/3D boxes; LiDAR + stereo |
| **Cityscapes** | 5k finely annotated frames (segmentation; boxes derived); used for domain shift (Foggy Cityscapes) |
| **BDD100K** | 100k images with boxes, 10 classes; weather and time-of-day attributes |
| **nuScenes** | 1,000 scenes, 6 cameras + LiDAR + radar; NDS metric |
| **Waymo Open Dataset** | Large-scale LiDAR + camera; APH metric |
| **ACDC** | Adverse conditions (fog, night, rain, snow) |
| **SODA-D** | Small objects in driving scenes |

## Aerial, satellite and rotated boxes

| Dataset | Size | Notes |
|---|---|---|
| **DOTA v1.0 / v1.5 / v2.0** | 2.8k / 2.8k / ≈ 11k large images; up to ≈ 1.8M instances | OBB, 15–18 classes; tiling required |
| **DIOR / DIOR-R** | ≈ 23k images, 20 classes | Horizontal and rotated versions |
| **HRSC2016** | ≈ 1k images | Ships, OBB |
| **xView** | ≈ 1M objects, 60 classes | 0.3 m satellite imagery; very large images |
| **VisDrone** | ≈ 10k images (DET) + video | Drone views, small objects |
| **AI-TOD** | ≈ 28k images, ≈ 700k objects, 8 classes | Mean object size ≈ 12.8 px |

## Video and tracking

| Dataset | Notes |
|---|---|
| **ImageNet VID** | 30 classes; video object detection |
| **MOT17 / MOT20** | Pedestrian tracking; MOT20 very crowded |
| **DanceTrack** | Similar appearance, complex motion: association-focused |
| **TAO** | Large-vocabulary tracking |

## Domain-specific

| Dataset | Domain | Notes |
|---|---|---|
| **SKU-110K** | Retail shelves | ≈ 11.8k images, ≈ 1.7M densely packed boxes |
| **DocLayNet / PubLayNet** | Document layout | 11 / 5 classes; extreme aspect ratios |
| **FLIR ADAS (thermal)** | Thermal driving | Paired RGB–thermal |
| **DeepLesion** | Medical CT | ≈ 32k lesions; FROC metric |
| **Global Wheat Head** | Agriculture | Dense small objects; domain shift across sites |

---

## Choosing a dataset for a purpose

| Purpose | Use |
|---|---|
| Comparing with published numbers | COCO val2017, with pycocotools (Chapter 33) |
| Predicting fine-tuning success on your data | RF100-VL or ODinW subsets close to your domain, plus your own data |
| Pre-training | Objects365, Open Images (check licences) |
| Long tail | LVIS |
| Small objects | AI-TOD, VisDrone, TinyPerson, SODA-D |
| Crowds | CrowdHuman, MOT20, SKU-110K |
| Rotated boxes | DOTA, DIOR-R, HRSC2016 |
| Robustness | ACDC, BDD100K splits, Foggy Cityscapes |
| Pipeline smoke tests | COCO8, COCO128, `odlab.data.SyntheticShapes` |
