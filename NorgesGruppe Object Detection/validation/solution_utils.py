"""
Core inference utilities for product detection.

This module contains the main pipeline for running inference:
- Load YOLO detection models
- Run detection on images
- Load and apply the classifier
- Optionally blend with reference embeddings
- Format outputs for submission

This is the ultralytics-based version (uses .pt files directly).
For ONNX-based inference, see solution_utils_onnx_sahi.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from crop_classifier_utils import load_classifier_checkpoint, predict_crops_with_classifier

def enable_torch_load_compatibility() -> None:
    # ultralytics 8.1.0 expects the pre-2.6 torch.load behavior for trusted checkpoints.
    if getattr(torch.load, "_ngd_patched", False):
        return

    original_torch_load = torch.load

    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    compatible_torch_load._ngd_patched = True  # type: ignore[attr-defined]
    torch.load = compatible_torch_load


enable_torch_load_compatibility()

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - handled at runtime
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:  # pragma: no cover - import success path
    ULTRALYTICS_IMPORT_ERROR = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_SETTINGS = {
    "weights": [],
    "imgsz": 1280,
    "conf": 0.01,
    "iou": 0.60,
    "max_det": 300,
    "half": True,
    "augment": False,
    "agnostic_nms": False,
    "use_wbf": True,
    "wbf_iou": 0.55,
    "wbf_skip_box_thr": 0.0001,
    "classifier_weights": None,
    "classifier_threshold": 0.35,
    "classifier_batch": 64,
    "classifier_min_det_score": 0.0,
    "classifier_topk_per_image": 0,
    "classifier_imgsz": 0,
    "classifier_score_blend": 0.0,
    "reference_embeddings": None,
    "reference_metadata": None,
    "reference_weight": 0.0,
    "reference_temperature": 0.10,
    "reference_topk": 5,
}


def _require_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "ultralytics is required. Install the training requirements first."
        ) from ULTRALYTICS_IMPORT_ERROR


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_settings(code_root: Path, settings_path: Path | None = None) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    path = settings_path or code_root / "settings.json"
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        settings.update(loaded)
    settings["weights"] = discover_weight_files(code_root, settings.get("weights", []))
    classifier_path = settings.get("classifier_weights")
    if classifier_path:
        settings["classifier_weights"] = str(resolve_path(classifier_path, code_root))
    reference_embeddings = settings.get("reference_embeddings")
    if reference_embeddings:
        settings["reference_embeddings"] = str(resolve_path(reference_embeddings, code_root))
    reference_metadata = settings.get("reference_metadata")
    if reference_metadata:
        settings["reference_metadata"] = str(resolve_path(reference_metadata, code_root))
    if not settings["weights"]:
        raise FileNotFoundError(
            "No model weights found. Add a weights file or update Code/settings.json."
        )
    return settings


def discover_weight_files(code_root: Path, configured_weights: Iterable[str]) -> list[Path]:
    weight_paths: list[Path] = []
    for item in configured_weights:
        resolved = resolve_path(item, code_root)
        if resolved.exists():
            weight_paths.append(resolved)
    if weight_paths:
        return weight_paths

    for pattern in ("weights/*.onnx", "weights/*.pt", "*.onnx", "*.pt"):
        for candidate in sorted(code_root.glob(pattern)):
            if candidate.is_file():
                weight_paths.append(candidate.resolve())
    return weight_paths


def infer_device(prefer_half: bool) -> tuple[int | str, bool]:
    if torch.cuda.is_available():
        return 0, bool(prefer_half)
    return "cpu", False


def iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_image_id(image_path: Path) -> int:
    try:
        return int(image_path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse image id from {image_path.name}") from exc


def load_models(weight_paths: list[Path]) -> list["YOLO"]:
    _require_ultralytics()
    return [YOLO(str(weight_path)) for weight_path in weight_paths]


def load_classifier_bundle(settings: dict, device: int | str) -> dict | None:
    classifier_path = settings.get("classifier_weights")
    if not classifier_path:
        return None
    checkpoint_path = Path(classifier_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {checkpoint_path}")
    torch_device = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    classifier_imgsz = int(settings.get("classifier_imgsz", 0))
    model, transform, checkpoint = load_classifier_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch_device,
        imgsz_override=classifier_imgsz if classifier_imgsz > 0 else None,
    )
    return {
        "model": model,
        "transform": transform,
        "device": torch_device,
        "threshold": float(settings.get("classifier_threshold", 0.35)),
        "batch_size": int(settings.get("classifier_batch", 64)),
        "min_det_score": float(settings.get("classifier_min_det_score", 0.0)),
        "topk_per_image": int(settings.get("classifier_topk_per_image", 0)),
        "imgsz": classifier_imgsz,
        "score_blend": float(settings.get("classifier_score_blend", 0.0)),
        "checkpoint": checkpoint,
    }


def load_reference_bundle(settings: dict) -> dict | None:
    embeddings_path = settings.get("reference_embeddings")
    metadata_path = settings.get("reference_metadata")
    if not embeddings_path or not metadata_path:
        return None

    embeddings = np.load(Path(embeddings_path))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    category_ids = [int(value) for value in metadata["category_ids"]]
    if embeddings.ndim != 2:
        raise ValueError(f"Reference embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(category_ids):
        raise ValueError("Reference embedding rows do not match category_ids length")

    reference_embeddings = torch.tensor(embeddings, dtype=torch.float32)
    reference_embeddings = torch.nn.functional.normalize(reference_embeddings, dim=1)
    aggregation_mode = str(metadata.get("aggregation_mode", "prototype"))
    return {
        "embeddings": reference_embeddings,
        "category_ids": category_ids,
        "category_ids_tensor": torch.tensor(category_ids, dtype=torch.long),
        "aggregation_mode": aggregation_mode,
        "weight": float(settings.get("reference_weight", 0.0)),
        "temperature": float(settings.get("reference_temperature", 0.10)),
        "topk": int(settings.get("reference_topk", 5)),
    }


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _predict_single_model(model: "YOLO", image_path: Path, settings: dict, device, half: bool) -> tuple[list[dict], tuple[int, int]]:
    results = model.predict(
        source=str(image_path),
        imgsz=int(settings["imgsz"]),
        conf=float(settings["conf"]),
        iou=float(settings["iou"]),
        max_det=int(settings["max_det"]),
        half=half,
        device=device,
        augment=bool(settings["augment"]),
        agnostic_nms=bool(settings["agnostic_nms"]),
        verbose=False,
    )
    result = results[0]
    image_h, image_w = result.orig_shape
    detections: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections, (image_w, image_h)

    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()

    for box, score, class_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = _to_float_list(box)
        detections.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "score": float(score),
                "category_id": int(class_id),
            }
        )
    return detections, (image_w, image_h)


def _clip_box(box: list[float], image_w: int, image_h: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_w), x1))
    y1 = max(0.0, min(float(image_h), y1))
    x2 = max(0.0, min(float(image_w), x2))
    y2 = max(0.0, min(float(image_h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _merge_with_wbf(per_model_predictions: list[list[dict]], image_size: tuple[int, int], settings: dict) -> list[dict]:
    if len(per_model_predictions) == 1 or not settings.get("use_wbf", True):
        return per_model_predictions[0]

    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        return per_model_predictions[0]

    image_w, image_h = image_size
    boxes_list = []
    scores_list = []
    labels_list = []

    for predictions in per_model_predictions:
        boxes = []
        scores = []
        labels = []
        for pred in predictions:
            x1, y1, x2, y2 = pred["xyxy"]
            boxes.append(
                [
                    max(0.0, min(1.0, x1 / image_w)),
                    max(0.0, min(1.0, y1 / image_h)),
                    max(0.0, min(1.0, x2 / image_w)),
                    max(0.0, min(1.0, y2 / image_h)),
                ]
            )
            scores.append(float(pred["score"]))
            labels.append(int(pred["category_id"]))
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if not any(boxes_list):
        return []

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=float(settings["wbf_iou"]),
        skip_box_thr=float(settings["wbf_skip_box_thr"]),
    )

    merged: list[dict] = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        merged.append(
            {
                "xyxy": _clip_box(
                    [x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h],
                    image_w,
                    image_h,
                ),
                "score": float(score),
                "category_id": int(label),
            }
        )
    return merged


def predict_image(models: list["YOLO"], image_path: Path, settings: dict, device, half: bool) -> list[dict]:
    per_model_predictions: list[list[dict]] = []
    image_size: tuple[int, int] | None = None
    for model in models:
        predictions, image_size = _predict_single_model(model, image_path, settings, device, half)
        per_model_predictions.append(predictions)
    if image_size is None:
        raise RuntimeError(f"Could not determine image size for {image_path}")
    return _merge_with_wbf(per_model_predictions, image_size, settings)


def rerank_detections_with_classifier(
    image_path: Path,
    detections: list[dict],
    classifier_bundle: dict | None,
    reference_bundle: dict | None = None,
) -> list[dict]:
    if classifier_bundle is None or not detections:
        return detections

    min_det_score = float(classifier_bundle["min_det_score"])
    selected_indices = [
        index for index, det in enumerate(detections) if float(det["score"]) >= min_det_score
    ]
    topk_per_image = int(classifier_bundle.get("topk_per_image", 0))
    if topk_per_image > 0 and len(selected_indices) > topk_per_image:
        selected_indices = sorted(
            selected_indices,
            key=lambda index: float(detections[index]["score"]),
            reverse=True,
        )[:topk_per_image]
    if not selected_indices:
        return detections

    logits, crop_embeddings = predict_crops_with_classifier(
        classifier=classifier_bundle["model"],
        transform=classifier_bundle["transform"],
        image_path=image_path,
        boxes_xyxy=[detections[index]["xyxy"] for index in selected_indices],
        device=classifier_bundle["device"],
        batch_size=int(classifier_bundle["batch_size"]),
    )
    if logits.numel() == 0:
        return detections

    classifier_probs = torch.softmax(logits, dim=1)
    combined_probs = classifier_probs.clone()
    reference_weight = 0.0
    if reference_bundle is not None:
        reference_weight = float(reference_bundle.get("weight", 0.0))
    if reference_bundle is not None and reference_weight > 0.0 and crop_embeddings.numel() > 0:
        ref_embeddings = reference_bundle["embeddings"]
        sims = crop_embeddings @ ref_embeddings.T
        temperature = max(float(reference_bundle.get("temperature", 0.10)), 1e-6)
        topk = int(reference_bundle.get("topk", 5))
        allowed_mask = None
        if topk > 0:
            classifier_topk = classifier_probs.topk(
                k=min(topk, classifier_probs.shape[1]),
                dim=1,
            ).indices
            allowed_mask = torch.zeros_like(classifier_probs, dtype=torch.bool)
            allowed_mask.scatter_(1, classifier_topk, True)

        aggregation_mode = str(reference_bundle.get("aggregation_mode", "prototype"))
        full_ref_probs = torch.zeros_like(classifier_probs)
        if aggregation_mode == "views":
            ref_logits = sims / temperature
            if allowed_mask is not None:
                ref_allowed = allowed_mask[:, reference_bundle["category_ids_tensor"]]
                ref_logits = ref_logits.masked_fill(~ref_allowed, float("-inf"))
            aggregated_logits = torch.full_like(classifier_probs, float("-inf"))
            index = reference_bundle["category_ids_tensor"].unsqueeze(0).expand(ref_logits.shape[0], -1)
            aggregated_logits.scatter_reduce_(1, index, ref_logits, reduce="amax", include_self=True)
            valid_rows = torch.isfinite(aggregated_logits).any(dim=1)
            if valid_rows.any():
                full_ref_probs[valid_rows] = torch.softmax(aggregated_logits[valid_rows], dim=1)
        else:
            ref_probs_small = torch.softmax(sims / temperature, dim=1)
            if allowed_mask is not None:
                ref_allowed = allowed_mask[:, reference_bundle["category_ids_tensor"]]
                ref_probs_small = ref_probs_small * ref_allowed.to(dtype=ref_probs_small.dtype)
                row_sums = ref_probs_small.sum(dim=1, keepdim=True)
                valid_rows = row_sums.squeeze(1) > 0
                if valid_rows.any():
                    ref_probs_small[valid_rows] = ref_probs_small[valid_rows] / row_sums[valid_rows]
            full_ref_probs[:, reference_bundle["category_ids_tensor"]] = ref_probs_small
        combined_probs = (1.0 - reference_weight) * classifier_probs + reference_weight * full_ref_probs

    combined_conf, combined_cls = combined_probs.max(dim=1)
    threshold = float(classifier_bundle["threshold"])
    score_blend = float(classifier_bundle.get("score_blend", 0.0))
    reranked: list[dict] = []
    replacement_map = {
        index: (int(cls_id), float(cls_conf))
        for index, cls_id, cls_conf in zip(
            selected_indices,
            combined_cls.tolist(),
            combined_conf.tolist(),
        )
    }
    for index, det in enumerate(detections):
        updated = dict(det)
        replacement = replacement_map.get(index)
        if replacement is not None:
            cls_id, cls_conf = replacement
            if cls_conf >= threshold:
                updated["category_id"] = int(cls_id)
            if score_blend > 0.0:
                updated["score"] = float(
                    float(det["score"]) * ((1.0 - score_blend) + score_blend * cls_conf)
                )
        reranked.append(updated)
    return reranked


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def detections_to_submission_rows(image_id: int, detections: list[dict]) -> list[dict]:
    rows = []
    for det in detections:
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(det["category_id"]),
                "bbox": [round(float(v), 2) for v in xyxy_to_xywh(det["xyxy"])],
                "score": round(float(det["score"]), 6),
            }
        )
    return rows


def run_inference(input_dir: Path, output_path: Path, code_root: Path, settings_path: Path | None = None) -> list[dict]:
    settings = load_settings(code_root=code_root, settings_path=settings_path)
    device, half = infer_device(prefer_half=bool(settings["half"]))
    models = load_models(settings["weights"])
    classifier_bundle = load_classifier_bundle(settings=settings, device=device)
    reference_bundle = load_reference_bundle(settings=settings)
    image_paths = iter_image_paths(input_dir)

    predictions: list[dict] = []
    for image_path in image_paths:
        image_id = parse_image_id(image_path)
        detections = predict_image(models, image_path, settings, device=device, half=half)
        detections = rerank_detections_with_classifier(
            image_path=image_path,
            detections=detections,
            classifier_bundle=classifier_bundle,
            reference_bundle=reference_bundle,
        )
        predictions.extend(detections_to_submission_rows(image_id, detections))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return predictions
