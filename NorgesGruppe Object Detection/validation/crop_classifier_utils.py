"""
Classifier utilities for product identification.

This module provides everything needed to train and use product classifiers:
- ArcFace loss implementation for better embeddings
- Support for multiple backbones (ConvNeXt, ResNet, EfficientNet, Swin, timm models)
- Dataset handling for crop classification
- Training and inference utilities
- ROI-based classification from full images

Key components:
  - ArcMarginProduct: ArcFace loss layer for metric learning
  - TimmEmbeddingClassifier: Wrapper for timm models with ArcFace
  - ProductCropDataset: Dataset for training on product crops
  - build_classifier_model: Factory for creating classifier models
  - predict_crops_with_classifier: Efficient inference using ROI align
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    convnext_small,
    convnext_tiny,
    efficientnet_v2_s,
    resnet18,
    resnet50,
    swin_t,
)
from torchvision.ops import roi_align
from torchvision.transforms.functional import pil_to_tensor

try:
    import timm
    from timm.data import resolve_model_data_config
except ImportError:  # pragma: no cover - timm is available in the competition sandbox
    timm = None
    resolve_model_data_config = None

try:
    import torch._dynamo as torch_dynamo
except ImportError:  # pragma: no cover
    torch_dynamo = None

try:
    from torchvision.io import ImageReadMode, read_image
except ImportError:  # pragma: no cover - torchvision.io should exist in runtime
    ImageReadMode = None
    read_image = None


ROI_ALIGN = torch_dynamo.disable(roi_align) if torch_dynamo is not None else roi_align


UNKNOWN_PRODUCT_ID = 355
SUPPORTED_CLASSIFIER_ARCHITECTURES = {
    "convnext_small",
    "resnet18",
    "resnet50",
    "convnext_tiny",
    "efficientnet_v2_s",
    "swin_t",
}


def is_timm_architecture(architecture: str) -> bool:
    return architecture.lower().startswith("timm/")


def supported_architectures_message() -> str:
    return f"{sorted(SUPPORTED_CLASSIFIER_ARCHITECTURES)} plus timm/<model_name>"


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.20) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        if labels is None:
            return cosine * self.scale

        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = cosine * (1.0 - one_hot) + target_logits * one_hot
        return logits * self.scale


class TimmEmbeddingClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        loss_type: str = "ce",
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.20,
    ) -> None:
        super().__init__()
        if timm is None or resolve_model_data_config is None:
            raise ImportError("timm is required for timm/* classifier architectures.")

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.num_features = int(getattr(self.backbone, "num_features"))
        self.ngd_architecture = f"timm/{model_name}"
        self.ngd_loss_type = loss_type
        self.ngd_accepts_labels = True
        self.ngd_preprocess_spec = resolve_model_data_config(self.backbone)
        if loss_type == "arcface":
            self.classifier = ArcMarginProduct(
                in_features=self.num_features,
                out_features=num_classes,
                scale=arcface_scale,
                margin=arcface_margin,
            )
        else:
            self.classifier = nn.Linear(self.num_features, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.extract_features(x)
        if self.ngd_loss_type == "arcface":
            return self.classifier(embeddings, labels=labels)
        return self.classifier(embeddings)


def normalize_name(name: str) -> str:
    return " ".join(name.upper().split())


def load_annotations(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


@dataclass
class CropSample:
    image_path: str
    category_id: int
    bbox_xyxy: tuple[float, float, float, float] | None
    source: str


def coco_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x, y, x + w, y + h


def expand_box(box: tuple[float, float, float, float], image_size: tuple[int, int], expand_ratio: float = 0.04) -> tuple[float, float, float, float]:
    image_w, image_h = image_size
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_pad = w * expand_ratio
    y_pad = h * expand_ratio
    return (
        max(0.0, x1 - x_pad),
        max(0.0, y1 - y_pad),
        min(float(image_w), x2 + x_pad),
        min(float(image_h), y2 + y_pad),
    )


def build_fold_samples(
    annotations: dict[str, Any],
    image_paths: list[Path],
    include_unknown: bool = True,
) -> list[CropSample]:
    path_by_id = {image_id_from_path(path): path for path in image_paths}
    valid_image_ids = set(path_by_id)
    samples: list[CropSample] = []
    for ann in annotations["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in valid_image_ids:
            continue
        category_id = int(ann["category_id"])
        if not include_unknown and category_id == UNKNOWN_PRODUCT_ID:
            continue
        image_path = path_by_id[image_id]
        samples.append(
            CropSample(
                image_path=str(image_path),
                category_id=category_id,
                bbox_xyxy=coco_to_xyxy(ann["bbox"]),
                source="shelf",
            )
        )
    return samples


def build_reference_samples(
    annotations: dict[str, Any],
    metadata: dict[str, Any],
    metadata_root: Path,
    repeat: int = 1,
) -> list[CropSample]:
    category_lookup = {
        normalize_name(category["name"]): int(category["id"])
        for category in annotations["categories"]
    }
    preferred_order = ["front", "main", "left", "right", "top", "bottom", "back"]
    samples: list[CropSample] = []
    for product in metadata["products"]:
        category_id = category_lookup.get(normalize_name(product["product_name"]))
        if category_id is None:
            continue
        image_types = [image_type for image_type in preferred_order if image_type in product.get("image_types", [])]
        for image_type in image_types:
            image_path = metadata_root / product["product_code"] / f"{image_type}.jpg"
            if not image_path.exists():
                continue
            sample = CropSample(
                image_path=str(image_path.resolve()),
                category_id=category_id,
                bbox_xyxy=None,
                source=f"reference:{image_type}",
            )
            for _ in range(max(1, repeat)):
                samples.append(sample)
    return samples


class ProductCropDataset(Dataset):
    def __init__(self, samples: list[CropSample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if sample.bbox_xyxy is not None:
            box = expand_box(sample.bbox_xyxy, image.size)
            image = image.crop(box)
        return self.transform(image), sample.category_id


def get_default_classifier_weights(architecture: str) -> Any:
    architecture = architecture.lower()
    if is_timm_architecture(architecture):
        if timm is None or resolve_model_data_config is None:
            raise ImportError("timm is required for timm/* classifier architectures.")
        model_name = architecture.split("/", 1)[1]
        model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        return resolve_model_data_config(model)
    if architecture == "resnet18":
        return ResNet18_Weights.DEFAULT
    if architecture == "resnet50":
        return ResNet50_Weights.DEFAULT
    if architecture == "convnext_tiny":
        return ConvNeXt_Tiny_Weights.DEFAULT
    if architecture == "convnext_small":
        return ConvNeXt_Small_Weights.DEFAULT
    if architecture == "efficientnet_v2_s":
        return EfficientNet_V2_S_Weights.DEFAULT
    if architecture == "swin_t":
        return Swin_T_Weights.DEFAULT
    raise ValueError(
        f"Unsupported classifier architecture: {architecture}. "
        f"Supported: {supported_architectures_message()}"
    )


def build_classifier_model(
    num_classes: int,
    architecture: str = "resnet18",
    pretrained: bool = True,
    loss_type: str = "ce",
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.20,
) -> tuple[nn.Module, Any]:
    architecture = architecture.lower()
    default_weights = get_default_classifier_weights(architecture)
    weights = default_weights if pretrained else None

    if is_timm_architecture(architecture):
        model_name = architecture.split("/", 1)[1]
        model = TimmEmbeddingClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            loss_type=loss_type,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
        )
        return model, (weights or default_weights)

    if architecture == "resnet18":
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == "resnet50":
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == "convnext_tiny":
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif architecture == "convnext_small":
        model = convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif architecture == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif architecture == "swin_t":
        model = swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported classifier architecture: {architecture}. "
            f"Supported: {supported_architectures_message()}"
        )

    setattr(model, "ngd_architecture", architecture)
    setattr(model, "ngd_loss_type", loss_type)
    setattr(model, "ngd_accepts_labels", False)
    return model, (weights or default_weights)


def resolve_preprocess_spec(weights: Any) -> dict[str, Any]:
    if isinstance(weights, dict):
        input_size = weights.get("input_size", (3, 224, 224))
        return {
            "mean": tuple(float(value) for value in weights.get("mean", (0.485, 0.456, 0.406))),
            "std": tuple(float(value) for value in weights.get("std", (0.229, 0.224, 0.225))),
            "crop_pct": float(weights.get("crop_pct", 0.875)),
            "input_size": tuple(int(value) for value in input_size),
        }

    normalize = weights.transforms()
    crop_size = getattr(normalize, "crop_size", (224, 224))
    if isinstance(crop_size, int):
        crop_height = crop_width = int(crop_size)
    elif len(crop_size) == 1:
        crop_height = crop_width = int(crop_size[0])
    else:
        crop_height = int(crop_size[0])
        crop_width = int(crop_size[1])
    return {
        "mean": tuple(float(value) for value in normalize.mean),
        "std": tuple(float(value) for value in normalize.std),
        "crop_pct": float(getattr(normalize, "crop_pct", 0.875)),
        "input_size": (3, crop_height, crop_width),
    }


def create_transforms(weights: Any, train: bool, imgsz: int) -> Any:
    spec = resolve_preprocess_spec(weights)
    mean = spec["mean"]
    std = spec["std"]
    if train:
        from torchvision.transforms import (
            ColorJitter,
            Compose,
            Normalize,
            RandomAffine,
            RandomErasing,
            RandomResizedCrop,
            ToTensor,
        )

        return Compose(
            [
                RandomResizedCrop(imgsz, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
                RandomAffine(degrees=4, translate=(0.03, 0.03), scale=(0.9, 1.08)),
                ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
            ]
        )

    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

    crop_pct = float(spec["crop_pct"])
    resize_size = int(round(imgsz / crop_pct))
    return Compose(
        [
            Resize(resize_size),
            CenterCrop(imgsz),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def create_inference_spec(weights: Any, imgsz: int) -> dict[str, Any]:
    spec = resolve_preprocess_spec(weights)
    return {
        "imgsz": int(imgsz),
        "mean": [float(value) for value in spec["mean"]],
        "std": [float(value) for value in spec["std"]],
        "crop_pct": float(spec["crop_pct"]),
    }


def save_classifier_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    num_classes: int,
    imgsz: int,
    architecture: str,
    loss_type: str,
    summary: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "imgsz": imgsz,
            "architecture": architecture,
            "loss_type": loss_type,
            "summary": summary,
        },
        checkpoint_path,
    )


def load_classifier_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    imgsz_override: int | None = None,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "resnet18")
    loss_type = checkpoint.get("loss_type", "ce")
    model, weights = build_classifier_model(
        num_classes=int(checkpoint["num_classes"]),
        architecture=architecture,
        pretrained=False,
        loss_type=loss_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    imgsz = int(imgsz_override or checkpoint.get("imgsz", 224))
    inference_spec = create_inference_spec(weights=weights, imgsz=imgsz)
    return model, inference_spec, checkpoint


def load_image_tensor(image_path: Path) -> torch.Tensor:
    if read_image is not None and ImageReadMode is not None:
        return read_image(str(image_path), mode=ImageReadMode.RGB)
    image = Image.open(image_path).convert("RGB")
    return pil_to_tensor(image)


def forward_classifier(classifier: nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    architecture = getattr(classifier, "ngd_architecture", None)
    loss_type = getattr(classifier, "ngd_loss_type", "ce")

    if is_timm_architecture(str(architecture)):
        embeddings = classifier.extract_features(batch)
        if loss_type == "arcface":
            logits = classifier.classifier(embeddings, labels=None)
        else:
            logits = classifier.classifier(embeddings)
        return logits, embeddings

    if architecture in {"resnet18", "resnet50"}:
        x = classifier.conv1(batch)
        x = classifier.bn1(x)
        x = classifier.relu(x)
        x = classifier.maxpool(x)

        x = classifier.layer1(x)
        x = classifier.layer2(x)
        x = classifier.layer3(x)
        x = classifier.layer4(x)

        x = classifier.avgpool(x)
        embeddings = torch.flatten(x, 1)
        logits = classifier.fc(embeddings)
        return logits, embeddings

    if architecture in {"convnext_tiny", "convnext_small"}:
        x = classifier.features(batch)
        x = classifier.avgpool(x)
        x = classifier.classifier[0](x)
        embeddings = classifier.classifier[1](x)
        logits = classifier.classifier[2](embeddings)
        return logits, embeddings

    if architecture == "efficientnet_v2_s":
        x = classifier.features(batch)
        x = classifier.avgpool(x)
        embeddings = torch.flatten(x, 1)
        logits = classifier.classifier(embeddings)
        return logits, embeddings

    if architecture == "swin_t":
        x = classifier.features(batch)
        x = classifier.norm(x)
        x = classifier.permute(x)
        x = classifier.avgpool(x)
        embeddings = classifier.flatten(x)
        logits = classifier.head(embeddings)
        return logits, embeddings

    raise ValueError(f"Unsupported classifier architecture for embedding extraction: {architecture}")


def get_classifier_output_dims(classifier: nn.Module) -> tuple[int, int]:
    architecture = getattr(classifier, "ngd_architecture", None)
    if is_timm_architecture(str(architecture)):
        if getattr(classifier, "ngd_loss_type", "ce") == "arcface":
            return int(classifier.classifier.out_features), int(classifier.classifier.in_features)
        return int(classifier.classifier.out_features), int(classifier.classifier.in_features)
    if architecture in {"resnet18", "resnet50"}:
        return int(classifier.fc.out_features), int(classifier.fc.in_features)
    if architecture in {"convnext_tiny", "convnext_small"}:
        return int(classifier.classifier[2].out_features), int(classifier.classifier[2].in_features)
    if architecture == "efficientnet_v2_s":
        return int(classifier.classifier[1].out_features), int(classifier.classifier[1].in_features)
    if architecture == "swin_t":
        return int(classifier.head.out_features), int(classifier.head.in_features)
    raise ValueError(f"Unsupported classifier architecture for output dims: {architecture}")


def predict_crops_with_classifier(
    classifier: nn.Module,
    transform,
    image_path: Path,
    boxes_xyxy: list[list[float]],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not boxes_xyxy:
        num_classes, embed_dim = get_classifier_output_dims(classifier)
        return (
            torch.empty((0, num_classes), dtype=torch.float32),
            torch.empty((0, embed_dim), dtype=torch.float32),
        )

    image_tensor = load_image_tensor(image_path).to(device=device, dtype=torch.float32) / 255.0
    _, image_h, image_w = image_tensor.shape
    rois = []
    for box in boxes_xyxy:
        expanded = expand_box((float(box[0]), float(box[1]), float(box[2]), float(box[3])), (image_w, image_h))
        rois.append([0.0, expanded[0], expanded[1], expanded[2], expanded[3]])

    imgsz = int(transform["imgsz"])
    mean = torch.tensor(transform["mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(transform["std"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    image_input = image_tensor.unsqueeze(0)

    logits_batches: list[torch.Tensor] = []
    embedding_batches: list[torch.Tensor] = []
    autocast_enabled = device.type == "cuda"

    # Batch both ROI align and classifier to avoid OOM
    with torch.inference_mode():
        for start in range(0, len(rois), batch_size):
            batch_rois = torch.tensor(rois[start : start + batch_size], device=device, dtype=torch.float32)
            crops = ROI_ALIGN(
                input=image_input,
                boxes=batch_rois,
                output_size=(imgsz, imgsz),
                spatial_scale=1.0,
                aligned=True,
            )
            crops = (crops - mean) / std
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                logits, embeddings = forward_classifier(classifier, crops)
            logits_batches.append(logits.detach().float().cpu())
            embedding_batches.append(F.normalize(embeddings.detach().float(), dim=1).cpu())
            del crops, batch_rois
    del image_input
    return torch.cat(logits_batches, dim=0), torch.cat(embedding_batches, dim=0)


def classify_crops(
    classifier: nn.Module,
    transform,
    image_path: Path,
    boxes_xyxy: list[list[float]],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[list[int], list[float]]:
    logits, _ = predict_crops_with_classifier(
        classifier=classifier,
        transform=transform,
        image_path=image_path,
        boxes_xyxy=boxes_xyxy,
        device=device,
        batch_size=batch_size,
    )
    if logits.numel() == 0:
        return [], []
    probs = torch.softmax(logits, dim=1)
    conf, cls = probs.max(dim=1)
    predicted_classes = [int(item) for item in cls.tolist()]
    confidences = [float(item) for item in conf.tolist()]
    return predicted_classes, confidences
