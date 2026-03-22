"""
Build reference embeddings from product images.

The competition provides official product photos for each of the 356 products.
We use these to create embeddings that help classify products at inference time.

The idea is: if a detected crop looks similar to a product's official photo,
it's probably that product. This is especially helpful when products appear
in unusual orientations or lighting conditions.

We support two aggregation modes:
  - prototype: Average all reference embeddings per class into one vector
  - views: Keep individual embeddings (useful if products have multiple views)

Usage:
    python build_reference_embeddings.py \
        --classifier-weights /path/to/classifier.pt \
        --imgsz 384 \
        --aggregation views \
        --output-embeddings reference_embeddings.npy \
        --output-metadata reference_metadata.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crop_classifier_utils import (
    ProductCropDataset,
    build_classifier_model,
    build_reference_samples,
    create_transforms,
    forward_classifier,
    load_annotations,
    load_classifier_checkpoint,
    load_metadata,
)


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build reference embeddings from product images")
    parser.add_argument("--classifier-weights", type=Path, required=True,
                        help="Path to trained classifier checkpoint")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(code_root.parent / "DataSets" / "NM_NGD_coco_dataset" / "train" / "annotations.json").resolve(),
        help="Path to COCO annotations",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(code_root.parent / "DataSets" / "NM_NGD_product_images" / "metadata.json").resolve(),
        help="Path to product metadata",
    )
    parser.add_argument("--imgsz", type=int, default=384,
                        help="Input image size (should match classifier training)")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--aggregation", choices=["prototype", "views"], default="views",
                        help="How to aggregate embeddings: prototype=mean per class, views=keep all")
    parser.add_argument(
        "--output-embeddings",
        type=Path,
        default=(code_root / "working" / "reference_embeddings.npy").resolve(),
        help="Where to save the embedding matrix",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=(code_root / "working" / "reference_embeddings_meta.json").resolve(),
        help="Where to save the metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load annotations and metadata
    annotations = load_annotations(args.annotations.resolve())
    metadata = load_metadata(args.metadata.resolve())
    metadata_root = args.metadata.resolve().parent

    # Load the trained classifier
    model, _, checkpoint = load_classifier_checkpoint(
        checkpoint_path=args.classifier_weights.resolve(),
        device=device,
        imgsz_override=args.imgsz,
    )
    
    # Rebuild the model architecture to get the preprocessing transforms
    _, weights = build_classifier_model(
        num_classes=int(checkpoint["num_classes"]),
        architecture=str(checkpoint.get("architecture", "resnet18")),
        pretrained=False,
    )
    transform = create_transforms(weights=weights, train=False, imgsz=args.imgsz)

    # Build dataset of reference product images
    samples = build_reference_samples(
        annotations=annotations,
        metadata=metadata,
        metadata_root=metadata_root,
        repeat=1,  # No repeats needed for embedding generation
    )
    dataset = ProductCropDataset(samples=samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,  # Keep order consistent
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    # Storage for embeddings
    embedding_rows: list[np.ndarray] = []
    embedding_labels: list[int] = []
    
    # For prototype aggregation: accumulate sums per class
    sums: dict[int, torch.Tensor] = {}
    counts: defaultdict[int, int] = defaultdict(int)

    # Generate embeddings
    autocast_enabled = device.type == "cuda"
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                _, embeddings = forward_classifier(model, images)
            
            # Normalize embeddings (important for cosine similarity)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
            
            # Store individual embeddings and accumulate for prototype
            for embedding, label in zip(embeddings, labels):
                class_id = int(label.item())
                embedding_cpu = embedding.detach().cpu()
                embedding_rows.append(embedding_cpu.numpy())
                embedding_labels.append(class_id)
                
                # Accumulate for prototype calculation
                if class_id not in sums:
                    sums[class_id] = embedding_cpu.clone()
                else:
                    sums[class_id] += embedding_cpu
                counts[class_id] += 1

    # Aggregate embeddings based on chosen mode
    if args.aggregation == "views":
        # Keep all individual embeddings
        # This preserves multiple views per product but uses more memory
        category_ids = embedding_labels
        embeddings = np.stack(embedding_rows, axis=0).astype(np.float32)
    else:
        # Create prototypes: one embedding per class
        # This is more memory-efficient but loses view diversity
        category_ids = sorted(sums)
        prototypes = []
        for category_id in category_ids:
            prototype = sums[category_id] / max(counts[category_id], 1)
            prototype = torch.nn.functional.normalize(prototype, dim=0)
            prototypes.append(prototype.numpy())
        embeddings = np.stack(prototypes, axis=0).astype(np.float32)

    # Build metadata
    unique_category_ids = sorted(counts)
    metadata_out = {
        "category_ids": category_ids,
        "counts": {str(category_id): counts[category_id] for category_id in unique_category_ids},
        "num_reference_images": len(samples),
        "num_categories": len(set(embedding_labels)),
        "num_rows": len(category_ids),
        "imgsz": args.imgsz,
        "aggregation_mode": args.aggregation,
        "classifier_weights": str(args.classifier_weights.resolve()),
    }

    # Save outputs
    args.output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_embeddings.resolve(), embeddings)
    args.output_metadata.resolve().write_text(
        json.dumps(metadata_out, indent=2),
        encoding="utf-8",
    )

    print(f"Reference categories: {len(category_ids)}")
    print(f"Reference images: {len(samples)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings written to: {args.output_embeddings.resolve()}")
    print(f"Metadata written to: {args.output_metadata.resolve()}")


if __name__ == "__main__":
    main()
