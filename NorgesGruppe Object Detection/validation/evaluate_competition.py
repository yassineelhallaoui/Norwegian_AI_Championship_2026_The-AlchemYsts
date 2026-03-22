"""
Evaluate predictions on validation set.

This script runs the full pipeline (detection + classification) on validation images
and computes metrics that approximate the competition scoring:

  - Detection AP@0.5: How well we find product boxes
  - Classification mAP@0.5: How accurately we classify found products
  - Competition proxy score: 0.7 * detection_AP + 0.3 * classification_mAP

Usage:
    python evaluate_competition.py \
        --fold 0 \
        --weights /path/to/detector.onnx \
        --classifier-weights /path/to/classifier.pt \
        --imgsz 960 \
        --conf 0.001 \
        --save-predictions val_preds.json \
        --save-summary val_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from solution_utils import (
    DEFAULT_SETTINGS,
    detections_to_submission_rows,
    infer_device,
    load_classifier_bundle,
    load_models,
    load_reference_bundle,
    parse_image_id,
    predict_image,
    rerank_detections_with_classifier,
)


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate detection and classification on validation set")
    parser.add_argument("--fold", type=int, default=0, help="Which fold to validate")
    parser.add_argument(
        "--val-txt",
        type=Path,
        default=None,
        help="Optional override for validation image list",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(code_root.parent / "DataSets" / "NM_NGD_coco_dataset" / "train" / "annotations.json").resolve(),
        help="Path to COCO annotations",
    )
    parser.add_argument("--weights", nargs="+", required=True, help="Model weights file(s)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.01, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--augment", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--disable-wbf", action="store_true", help="Disable Weighted Boxes Fusion")
    parser.add_argument("--save-predictions", type=Path, default=None, help="Save predictions to file")
    parser.add_argument("--save-summary", type=Path, default=None, help="Save metrics summary to file")
    parser.add_argument("--classifier-weights", type=Path, default=None, help="Classifier checkpoint")
    parser.add_argument("--classifier-threshold", type=float, default=0.35, help="Classifier confidence threshold")
    parser.add_argument("--classifier-batch", type=int, default=64, help="Classifier batch size")
    parser.add_argument("--classifier-min-det-score", type=float, default=0.0, help="Min detection score to classify")
    parser.add_argument("--classifier-topk-per-image", type=int, default=0, help="Max boxes to classify per image (0=all)")
    parser.add_argument("--classifier-imgsz", type=int, default=0, help="Classifier input size (0=from checkpoint)")
    parser.add_argument("--classifier-score-blend", type=float, default=0.0, help="Blend detection/classifier scores")
    parser.add_argument("--reference-embeddings", type=Path, default=None, help="Reference embeddings .npy file")
    parser.add_argument("--reference-metadata", type=Path, default=None, help="Reference metadata .json file")
    parser.add_argument("--reference-weight", type=float, default=0.0, help="Weight for reference similarity (0-1)")
    parser.add_argument("--reference-temperature", type=float, default=0.10, help="Temperature for reference softmax")
    parser.add_argument("--reference-topk", type=int, default=5, help="Top-K reference classes to consider")
    return parser.parse_args()


def load_image_list(val_txt: Path) -> list[Path]:
    """Load list of image paths from text file."""
    return [Path(line.strip()) for line in val_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def coco_xywh_to_xyxy(box: list[float]) -> list[float]:
    """Convert COCO format [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculate Intersection over Union between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Compute intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    if inter_area <= 0.0:
        return 0.0

    # Compute union
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    
    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_average_precision(
    gt_by_class: dict[int, dict[int, list[list[float]]]],
    preds_by_class: dict[int, list[dict]],
    iou_thr: float = 0.5
) -> tuple[float, dict[int, float]]:
    """
    Compute Average Precision per class using VOC-style metric.
    
    Args:
        gt_by_class: Ground truth boxes grouped by class then image
        preds_by_class: Predictions grouped by class
        iou_thr: IoU threshold for matching predictions to ground truth
    
    Returns:
        mean AP across classes, and per-class AP
    """
    ap_by_class: dict[int, float] = {}

    for class_id, gt_images in gt_by_class.items():
        total_gt = sum(len(boxes) for boxes in gt_images.values())
        if total_gt == 0:
            continue

        # Track which ground truth boxes have been matched
        matched = {
            image_id: [False] * len(boxes)
            for image_id, boxes in gt_images.items()
        }
        
        # Sort predictions by confidence (high to low)
        predictions = sorted(preds_by_class.get(class_id, []), key=lambda item: item["score"], reverse=True)

        # Accumulate true positives and false positives
        tp = np.zeros(len(predictions), dtype=float)
        fp = np.zeros(len(predictions), dtype=float)

        for idx, prediction in enumerate(predictions):
            image_id = int(prediction["image_id"])
            predicted_box = prediction["bbox_xyxy"]
            gt_boxes = gt_images.get(image_id, [])

            # Find best matching ground truth box
            best_iou = 0.0
            best_match = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(predicted_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = gt_idx

            # Check if this is a true positive
            if best_iou >= iou_thr and best_match >= 0 and not matched[image_id][best_match]:
                tp[idx] = 1.0
                matched[image_id][best_match] = True
            else:
                fp[idx] = 1.0

        if len(predictions) == 0:
            ap_by_class[class_id] = 0.0
            continue

        # Compute precision-recall curve
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(total_gt, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

        # VOC-style AP: interpolate precision-recall curve
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for idx in range(len(mpre) - 2, -1, -1):
            mpre[idx] = max(mpre[idx], mpre[idx + 1])
        changing_points = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))
        ap_by_class[class_id] = ap

    if not ap_by_class:
        return 0.0, {}
    return float(np.mean(list(ap_by_class.values()))), ap_by_class


def build_ground_truth(
    coco: dict,
    image_ids: set[int]
) -> tuple[dict[int, dict[int, list[list[float]]]], dict[int, dict[int, list[list[float]]]]]:
    """
    Build ground truth lookup tables.
    
    Returns:
        det_gt: All boxes grouped for detection evaluation (single class)
        cls_gt: Boxes grouped by product class for classification evaluation
    """
    cls_gt: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    det_gt: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_ids:
            continue
        class_id = int(ann["category_id"])
        box_xyxy = coco_xywh_to_xyxy(ann["bbox"])
        cls_gt[class_id][image_id].append(box_xyxy)
        det_gt[0][image_id].append(box_xyxy)  # Class 0 for detection (all products)
    
    return det_gt, cls_gt


def build_prediction_tables(predictions: list[dict]) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """
    Build prediction lookup tables grouped by class.
    
    Returns:
        det_preds: All predictions for detection evaluation
        cls_preds: Predictions grouped by predicted class
    """
    cls_preds: dict[int, list[dict]] = defaultdict(list)
    det_preds: dict[int, list[dict]] = defaultdict(list)
    
    for prediction in predictions:
        pred_entry = {
            "image_id": int(prediction["image_id"]),
            "score": float(prediction["score"]),
            "bbox_xyxy": coco_xywh_to_xyxy(prediction["bbox"]),
        }
        cls_preds[int(prediction["category_id"])].append(pred_entry)
        det_preds[0].append(pred_entry)  # Single class for detection
    
    return det_preds, cls_preds


def summarize_predictions(predictions: list[dict], coco: dict, image_ids: set[int]) -> dict[str, Any]:
    """
    Compute all evaluation metrics.
    
    Competition score = 0.7 * detection_AP + 0.3 * classification_mAP
    """
    det_gt, cls_gt = build_ground_truth(coco, image_ids=image_ids)
    det_preds, cls_preds = build_prediction_tables(predictions)

    detection_ap, detection_per_class = compute_average_precision(det_gt, det_preds, iou_thr=0.5)
    classification_map, classification_per_class = compute_average_precision(cls_gt, cls_preds, iou_thr=0.5)
    final_score = 0.7 * detection_ap + 0.3 * classification_map

    return {
        "validation_images": len(image_ids),
        "num_predictions": len(predictions),
        "detection_ap50": detection_ap,
        "classification_map50": classification_map,
        "competition_proxy_score": final_score,
        "num_detection_classes_evaluated": len(detection_per_class),
        "num_classification_classes_evaluated": len(classification_per_class),
    }


def evaluate_weight_paths(
    image_paths: list[Path],
    coco: dict,
    weight_paths: list[Path],
    settings: dict,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Run inference on validation images and compute metrics.
    """
    image_ids = {parse_image_id(path) for path in image_paths}
    device, half = infer_device(prefer_half=bool(settings["half"]))
    models = load_models(weight_paths)
    classifier_bundle = load_classifier_bundle(settings=settings, device=device)
    reference_bundle = load_reference_bundle(settings=settings)

    predictions: list[dict] = []
    for image_path in image_paths:
        image_id = parse_image_id(image_path)
        
        # Detect products
        detections = predict_image(models, image_path, settings, device=device, half=half)
        
        # Classify detections
        detections = rerank_detections_with_classifier(
            image_path=image_path,
            detections=detections,
            classifier_bundle=classifier_bundle,
            reference_bundle=reference_bundle,
        )
        
        predictions.extend(detections_to_submission_rows(image_id, detections))

    summary = summarize_predictions(predictions=predictions, coco=coco, image_ids=image_ids)
    return predictions, summary


def main() -> None:
    args = parse_args()
    code_root = Path(__file__).resolve().parent.parent
    
    # Determine validation split
    val_txt = args.val_txt or (code_root / "working" / "yolo_dataset" / "splits" / f"fold_{args.fold}_val.txt")

    # Load annotations and image list
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_paths = load_image_list(val_txt)

    # Build settings from arguments
    settings = dict(DEFAULT_SETTINGS)
    settings.update({
        "weights": [str(Path(weight).resolve()) for weight in args.weights],
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "augment": args.augment,
        "use_wbf": not args.disable_wbf,
        "classifier_weights": str(args.classifier_weights.resolve()) if args.classifier_weights else None,
        "classifier_threshold": args.classifier_threshold,
        "classifier_batch": args.classifier_batch,
        "classifier_min_det_score": args.classifier_min_det_score,
        "classifier_topk_per_image": args.classifier_topk_per_image,
        "classifier_imgsz": args.classifier_imgsz,
        "classifier_score_blend": args.classifier_score_blend,
        "reference_embeddings": str(args.reference_embeddings.resolve()) if args.reference_embeddings else None,
        "reference_metadata": str(args.reference_metadata.resolve()) if args.reference_metadata else None,
        "reference_weight": args.reference_weight,
        "reference_temperature": args.reference_temperature,
        "reference_topk": args.reference_topk,
    })

    # Run evaluation
    predictions, summary = evaluate_weight_paths(
        image_paths=image_paths,
        coco=coco,
        weight_paths=[Path(weight).resolve() for weight in settings["weights"]],
        settings=settings,
    )

    # Save results if requested
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.save_predictions.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    if args.save_summary:
        args.save_summary.parent.mkdir(parents=True, exist_ok=True)
        args.save_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print summary
    print(f"Validation images: {summary['validation_images']}")
    print(f"Predictions: {summary['num_predictions']}")
    print(f"Detection AP@0.5: {summary['detection_ap50']:.6f}")
    print(f"Classification mAP@0.5: {summary['classification_map50']:.6f}")
    print(f"Competition proxy score: {summary['competition_proxy_score']:.6f}")


if __name__ == "__main__":
    main()
