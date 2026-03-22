# NorgesGruppen Product Detection Solution

**NM i AI 2026 Competition Entry**  
**Team:** The AlchemYsts  
**Team Captain:** Yassine Elhallaoui

---

## Overview

This repository contains our complete solution for the NorgesGruppen product detection challenge. The task involves detecting and classifying grocery products on store shelves from images — a classic computer vision problem with real-world retail applications.

The competition scoring combines two objectives:
- **70% Detection mAP@0.5** - How well we find product bounding boxes
- **30% Classification mAP@0.5** - How accurately we identify the product type

Our approach uses a two-stage pipeline: first detecting all products with an object detector, then classifying each detection using a dedicated product classifier enhanced with reference image embeddings.

---

## Solution Architecture

### Stage 1: Object Detection (YOLO11)

We use **YOLO11m** as our detection backbone. The detector was trained with a two-stage recipe:

1. **Initial Training** - Train on a specific data fold with aggressive augmentation
2. **Fine-tuning** - Continue training on the full dataset with milder augmentation

Key training decisions:
- Single-class detection (we detect "product" first, classify later)
- Input resolution of 960x960 for good speed/accuracy balance
- ONNX export for efficient inference

### Stage 2: Product Classification (ConvNeXt-Small + ArcFace)

After detection, each crop is classified using:

- **Backbone:** ConvNeXt-Small (timm variant)
- **Loss:** ArcFace with scale=30, margin=0.25
- **Input size:** 384x384 pixels
- **Training:** 15 epochs with MixUp/CutMix augmentation

The classifier was trained on:
- Crops from training images (ground truth boxes)
- Reference product images (repeated 10x for balance)

### Reference Embedding Enhancement

We built reference embeddings from the official product photos. At inference time, the classifier logits are blended with similarity scores to the reference images:

```
final_score = 0.6 * classifier_probs + 0.4 * reference_similarity
```

This helps when products appear in unusual orientations or lighting conditions.

### SAHI Sliced Inference (Optional)

For images with many small products, SAHI (Slicing Aided Hyper Inference) can be enabled to:
- Slice the image into overlapping patches
- Run detection on each patch + full image
- Merge results using Weighted Boxes Fusion

This helps catch small products that might be missed in full-image inference.

---

## Repository Structure

```
final submission/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── config.json                  # Default inference configuration
│
├── submission/                  # Competition submission package
│   └── YassY_yolo11_yolo11m_convnext_small_i384_arcface_sahi.zip
│
├── YassY_yolo11_yolo11m_convnext_small_i384_arcface_sahi/  # Unpacked solution
│   ├── run.py                   # Main entry point for inference
│   ├── solution_utils_onnx_sahi.py  # Core inference pipeline
│   ├── crop_classifier_utils.py     # Classifier utilities
│   ├── settings.json            # Inference hyperparameters
│   ├── classifier.pt            # Trained classifier weights
│   ├── reference_embeddings.npy # Product reference embeddings
│   ├── reference_metadata.json  # Reference metadata
│   └── weights/
│       └── best.onnx            # YOLO11 detection model
│
├── training/                    # Training scripts and utilities
│   ├── train_yolo11_detector.py     # Train the detection model
│   ├── train_improved_classifier.py # Train the classifier
│   ├── build_reference_embeddings.py # Generate reference embeddings
│   └── crop_classifier_utils.py     # Shared classifier code
│
└── validation/                  # Validation and evaluation
    ├── evaluate_competition.py  # Evaluate on validation set
    ├── solution_utils.py        # Original inference utilities
    └── crop_classifier_utils.py # Shared classifier code
```

---

## How to Use

### Inference (Running the Solution)

The solution expects a directory of shelf images and outputs a JSON file with predictions.

```bash
cd "YassY_yolo11_yolo11m_convnext_small_i384_arcface_sahi"

python run.py \
    --input /path/to/test/images \
    --output /path/to/output/predictions.json
```

Optional: use a custom settings file for experimentation:

```bash
python run.py \
    --input /path/to/test/images \
    --output /path/to/output/predictions.json \
    --settings /path/to/custom_settings.json
```

#### Key Settings (settings.json)

```json
{
  "weights": ["weights/best.onnx"],
  "imgsz": 960,
  "conf": 0.001,
  "iou": 0.6,
  "max_det": 300,
  "half": true,
  "use_sahi": false,
  "use_wbf": true,
  "wbf_iou": 0.5,
  "classifier_weights": "classifier.pt",
  "classifier_threshold": 0.05,
  "classifier_batch": 16,
  "classifier_imgsz": 384,
  "classifier_score_blend": 1.0
}
```

| Parameter | Description |
|-----------|-------------|
| `conf` | Detection confidence threshold (lower = more detections) |
| `iou` | NMS IoU threshold |
| `use_sahi` | Enable sliced inference for small objects |
| `sahi_slice_size` | Patch size for SAHI (if enabled) |
| `classifier_threshold` | Minimum confidence to accept classifier prediction |
| `classifier_score_blend` | Blend factor for detection/classifier scores |

### Training the Detector

```bash
python training/train_yolo11_detector.py \
    --fold 0 \
    --model yolo11m.pt \
    --imgsz 960 \
    --batch 4 \
    --stage1-epochs 10 \
    --stage2-epochs 6 \
    --export-onnx
```

This will:
1. Train on fold 0 for 10 epochs
2. Fine-tune on full data for 6 epochs
3. Export to ONNX format

### Training the Classifier

```bash
python training/train_improved_classifier.py \
    --fold 0 \
    --architecture convnext_small \
    --imgsz 384 \
    --loss-type arcface \
    --epochs 15 \
    --batch 32 \
    --reference-repeat 10
```

For final model (train on all data):

```bash
python training/train_improved_classifier.py \
    --full-data \
    --architecture convnext_small \
    --imgsz 384 \
    --loss-type arcface \
    --epochs 12 \
    --batch 32
```

### Building Reference Embeddings

```bash
python training/build_reference_embeddings.py \
    --classifier-weights /path/to/classifier.pt \
    --imgsz 384 \
    --aggregation views \
    --output-embeddings reference_embeddings.npy \
    --output-metadata reference_metadata.json
```

### Validation

```bash
python validation/evaluate_competition.py \
    --fold 0 \
    --weights /path/to/detector.onnx \
    --classifier-weights /path/to/classifier.pt \
    --imgsz 960 \
    --conf 0.001
```

---

## Experiments and Development

### Detection Experiments

| Model | Input Size | Augment | mAP@0.5 |
|-------|-----------|---------|---------|
| YOLOv8m | 960 | Standard | ~0.78 |
| YOLOv8l | 1280 | Heavy | ~0.81 |
| **YOLO11m** | **960** | **Two-stage** | **~0.83** |

Key findings:
- Two-stage training (fold → full) consistently outperformed single-stage
- Single-class detection worked better than multi-class for this task
- Resolution of 960 was the sweet spot (1280 was slower with marginal gains)

### Classification Experiments

| Backbone | Input | Loss | Top-1 Acc | Top-5 Acc |
|----------|-------|------|-----------|-----------|
| ConvNeXt-Tiny | 224 | CE | ~78% | ~94% |
| ConvNeXt-Small | 224 | CE | ~80% | ~95% |
| **ConvNeXt-Small** | **384** | **ArcFace** | **~84%** | **~97%** |
| EfficientNet-V2-S | 384 | ArcFace | ~82% | ~96% |
| Swin-T | 224 | ArcFace | ~81% | ~95% |

Key findings:
- ArcFace loss significantly improved embedding quality
- 384px input captured details 224px missed
- Reference embeddings added ~2% final score improvement

### Final Configuration

Our best single-model configuration:
- **Detector:** YOLO11m @ 960px, ONNX format
- **Classifier:** ConvNeXt-Small @ 384px with ArcFace
- **Reference:** Views aggregation, weight=0.4
- **Inference:** SAHI disabled (faster), WBF enabled

---

## Hardware and Training Time

All experiments were conducted on:
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: AMD Ryzen 9 7950X
- RAM: 64GB DDR5

Typical training times:
- Detector training: ~2 hours (16 epochs total)
- Classifier training: ~3 hours (15 epochs)
- Full pipeline training: ~5 hours

---

## Dependencies

Core requirements:
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
onnxruntime-gpu>=1.15.0
timm>=0.9.0
ensemble-boxes>=1.0.9
numpy>=1.24.0
pillow>=9.0.0
```

See `requirements-train.txt` for full training dependencies.

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

This solution was developed for the **NM i AI 2026** competition hosted by **NorgesGruppen**. Special thanks to the competition organizers for providing the dataset and evaluation framework.

**Team:** The AlchemYsts  
**Captain:** Yassine Elhallaoui
