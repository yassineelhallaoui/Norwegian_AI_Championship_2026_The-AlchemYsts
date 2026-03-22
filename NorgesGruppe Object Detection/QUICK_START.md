# Quick Start Guide

## Running Inference

### 1. Navigate to the solution folder

```bash
cd "YassY_yolo11_yolo11m_convnext_small_i384_arcface_sahi"
```

### 2. Run on test images

```bash
python run.py \
    --input /path/to/test/images \
    --output predictions.json
```

### 3. Output format

The output `predictions.json` contains a list of detections:

```json
[
  {
    "image_id": 42,
    "category_id": 123,
    "bbox": [100.5, 200.0, 50.0, 80.0],
    "score": 0.95
  }
]
```

Each entry represents one detected product with:
- `image_id`: ID of the source image
- `category_id`: Predicted product class (1-356)
- `bbox`: Bounding box in COCO format `[x, y, width, height]`
- `score`: Confidence score

## Competition Submission

The file in `submission/YassY_yolo11_yolo11m_convnext_small_i384_arcface_sahi.zip` is ready for upload.

## Training from Scratch

See the `training/` folder for scripts to reproduce the models.
