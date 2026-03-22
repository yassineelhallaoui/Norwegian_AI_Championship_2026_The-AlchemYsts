"""
Train an improved crop classifier for product identification.

This classifier takes detected product crops and identifies which of the
356 product types it is. The key improvements over a basic ResNet:

  - ConvNeXt-Small backbone (better than ResNet, still fast)
  - 384px input resolution (more detail than standard 224px)
  - ArcFace loss (creates better-separated embeddings)
  - MixUp/CutMix augmentation (improves generalization)
  - Reference image repeats (balances class distribution)
  - Warmup + cosine LR schedule (stable training)

The classifier is trained on:
  1. Crops from shelf images (ground truth boxes)
  2. Reference product images (official product photos)

Usage:
  # Train on fold 0 for validation
  python train_improved_classifier.py --fold 0 --epochs 15 --imgsz 384 --architecture convnext_small
  
  # Train on full data for final submission
  python train_improved_classifier.py --full-data --epochs 12 --imgsz 384 --loss-type arcface
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crop_classifier_utils import (
    ProductCropDataset,
    build_classifier_model,
    build_fold_samples,
    build_reference_samples,
    create_transforms,
    load_annotations,
    load_metadata,
    save_classifier_checkpoint,
)


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Improved crop classifier training")
    parser.add_argument("--fold", type=int, default=0, help="Which fold to validate on")
    parser.add_argument("--architecture", default="convnext_small",
                        help="Backbone: convnext_small, resnet50, timm/<model>, etc.")
    parser.add_argument("--loss-type", choices=["ce", "arcface"], default="arcface",
                        help="Loss function: cross-entropy or ArcFace")
    parser.add_argument("--arcface-scale", type=float, default=30.0, help="ArcFace scale parameter")
    parser.add_argument("--arcface-margin", type=float, default=0.25, help="ArcFace margin parameter")
    parser.add_argument("--full-data", action="store_true", help="Train on full dataset (no validation)")
    parser.add_argument("--train-txt", type=Path, default=None, help="Custom train image list")
    parser.add_argument("--val-txt", type=Path, default=None, help="Custom val image list")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Initialize from checkpoint")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=384, help="Input resolution (384 recommended)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (32 fits in 12GB at 384px)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--reference-repeat", type=int, default=10,
                        help="How many times to repeat reference images (balances classes)")
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp alpha (0=disabled)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0, help="CutMix alpha (0=disabled)")
    parser.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying MixUp/CutMix")
    parser.add_argument("--name", default=None, help="Run name (auto-generated if not provided)")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--project", type=Path,
                        default=(code_root / "working" / "classifier_runs").resolve())
    return parser.parse_args()


def forward_model(model: nn.Module, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    """Forward pass that handles both regular and ArcFace models."""
    if getattr(model, "ngd_accepts_labels", False):
        return model(images, labels=labels)
    return model(images)


class MixUpCutMix:
    """
    Applies MixUp or CutMix augmentation to a batch.
    
    MixUp blends two images and their labels proportionally.
    CutMix cuts a patch from one image and pastes it into another.
    Both improve generalization by creating more diverse training examples.
    """
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0,
                 prob: float = 0.5, num_classes: int = 357):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.size(0)
        
        # Convert labels to one-hot for soft blending
        targets_onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        targets_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Randomly decide whether to apply augmentation
        if torch.rand(1).item() > self.prob:
            return images, targets_onehot

        # Randomly shuffle indices for mixing
        indices = torch.randperm(batch_size, device=images.device)
        
        # Choose between CutMix and MixUp
        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or torch.rand(1).item() > 0.5)

        if use_cutmix:
            # CutMix: cut and paste a patch
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            _, _, h, w = images.shape
            cut_ratio = math.sqrt(1.0 - lam)
            cut_h = int(h * cut_ratio)
            cut_w = int(w * cut_ratio)
            
            # Random center point
            cy = torch.randint(0, h, (1,)).item()
            cx = torch.randint(0, w, (1,)).item()
            
            # Calculate patch bounds
            y1 = max(0, cy - cut_h // 2)
            y2 = min(h, cy + cut_h // 2)
            x1 = max(0, cx - cut_w // 2)
            x2 = min(w, cx + cut_w // 2)
            
            # Apply the cut
            images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            # Adjust lambda based on actual patch area
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
        else:
            # MixUp: blend images
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            images = lam * images + (1.0 - lam) * images[indices]

        # Blend labels with same lambda
        targets_onehot = lam * targets_onehot + (1.0 - lam) * targets_onehot[indices]
        return images, targets_onehot


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross entropy loss that works with soft targets (from MixUp/CutMix)."""
    if label_smoothing > 0:
        num_classes = targets.size(1)
        targets = targets * (1.0 - label_smoothing) + label_smoothing / num_classes
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = forward_model(model=model, images=images, labels=None)
            loss = criterion(logits, labels)
            
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            
            # Top-1 accuracy
            top1 = logits.argmax(dim=1)
            correct1 += int((top1 == labels).sum().item())
            
            # Top-5 accuracy
            top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            correct5 += int((top5 == labels.unsqueeze(1)).any(dim=1).sum().item())
            
    return {
        "loss": total_loss / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top5": correct5 / max(total, 1),
    }


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    Warmup helps stabilize training in early epochs when gradients are large.
    Cosine decay smoothly reduces LR for better convergence.
    """
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lrs[0], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    code_root = Path(__file__).resolve().parent.parent

    # Generate run name if not provided
    split_tag = "full" if args.full_data else f"fold{args.fold}"
    loss_tag = args.loss_type
    run_name = args.name or f"cls_{split_tag}_{args.architecture}_{args.imgsz}_{loss_tag}_e{args.epochs}"
    save_dir = (args.project / run_name).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations and metadata
    annotations = load_annotations(
        (code_root.parent / "DataSets" / "NM_NGD_coco_dataset" / "train" / "annotations.json").resolve()
    )
    metadata = load_metadata(
        (code_root.parent / "DataSets" / "NM_NGD_product_images" / "metadata.json").resolve()
    )
    metadata_root = (code_root.parent / "DataSets" / "NM_NGD_product_images").resolve()

    # Determine train/val splits
    if args.train_txt is not None:
        train_txt = args.train_txt.resolve()
    elif args.full_data:
        train_txt = (code_root / "working" / "yolo_dataset" / "splits" / "full_train.txt").resolve()
    else:
        train_txt = (code_root / "working" / "yolo_dataset" / "splits" / f"fold_{args.fold}_train.txt").resolve()

    val_txt = None
    if args.val_txt is not None:
        val_txt = args.val_txt.resolve()
    elif not args.full_data:
        val_txt = (code_root / "working" / "yolo_dataset" / "splits" / f"fold_{args.fold}_val.txt").resolve()

    # Load image lists
    def load_image_list(txt_path: Path) -> list[Path]:
        return [Path(line.strip()) for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_images = load_image_list(train_txt)
    val_images = load_image_list(val_txt) if val_txt is not None else []

    # Build training samples from shelf images
    train_samples = build_fold_samples(annotations=annotations, image_paths=train_images, include_unknown=True)
    val_samples = build_fold_samples(annotations=annotations, image_paths=val_images, include_unknown=True)

    # Add reference product images (with repeats for class balance)
    # Some products have few shelf examples, so we boost them with reference images
    train_samples.extend(
        build_reference_samples(
            annotations=annotations,
            metadata=metadata,
            metadata_root=metadata_root,
            repeat=args.reference_repeat,
        )
    )

    num_classes = len(annotations["categories"])
    print(f"Training samples: {len(train_samples)}, Validation: {len(val_samples)}, Classes: {num_classes}")

    # Build the model
    model, weights = build_classifier_model(
        num_classes=num_classes,
        architecture=args.architecture,
        pretrained=True,
        loss_type=args.loss_type,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
    )

    # Optionally load from checkpoint
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint.resolve(), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"Loaded init checkpoint: {args.init_checkpoint}")

    # Create transforms
    train_transform = create_transforms(weights=weights, train=True, imgsz=args.imgsz)
    val_transform = create_transforms(weights=weights, train=False, imgsz=args.imgsz)

    train_dataset = ProductCropDataset(train_samples, train_transform)
    val_dataset = ProductCropDataset(val_samples, val_transform) if val_samples else None

    # Create dataloaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        )

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, args.min_lr)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # Setup MixUp/CutMix if enabled
    mix_augment = None
    use_mix = args.mixup_alpha > 0 or args.cutmix_alpha > 0
    if use_mix:
        mix_augment = MixUpCutMix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            num_classes=num_classes,
        )

    best_metric = float("-inf")
    history: list[dict] = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                if mix_augment is not None:
                    # Apply MixUp/CutMix augmentation
                    mixed_images, soft_targets = mix_augment(images, labels)
                    logits = forward_model(model=model, images=mixed_images, labels=labels)
                    loss = soft_cross_entropy(logits, soft_targets, label_smoothing=args.label_smoothing)
                else:
                    logits = forward_model(model=model, images=images, labels=labels)
                    loss = nn.functional.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_summary = {"epoch": epoch, "train_loss": round(train_loss, 5), "lr": round(current_lr, 8)}

        # Validation
        selection_metric = -train_loss  # Fallback if no validation
        if val_loader is not None:
            val_metrics = evaluate(model=model, loader=val_loader, device=device)
            epoch_summary["val_loss"] = round(val_metrics["loss"], 5)
            epoch_summary["val_top1"] = round(val_metrics["top1"], 5)
            epoch_summary["val_top5"] = round(val_metrics["top5"], 5)
            selection_metric = float(val_metrics["top1"])

        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

        # Save best model
        if selection_metric >= best_metric:
            best_metric = selection_metric
            summary = {
                "run_name": run_name,
                "fold": None if args.full_data else args.fold,
                "full_data": bool(args.full_data),
                "architecture": args.architecture,
                "loss_type": args.loss_type,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "reference_repeat": args.reference_repeat,
                "num_train_samples": len(train_samples),
                "num_val_samples": len(val_samples),
                "mixup_alpha": args.mixup_alpha,
                "cutmix_alpha": args.cutmix_alpha,
                "label_smoothing": args.label_smoothing,
                "history": history,
            }
            if val_loader is not None:
                summary["best_val_top1"] = val_metrics["top1"]
                summary["best_val_top5"] = val_metrics["top5"]
            save_classifier_checkpoint(
                checkpoint_path=save_dir / "best.pt",
                model=model,
                num_classes=num_classes,
                imgsz=args.imgsz,
                architecture=args.architecture,
                loss_type=args.loss_type,
                summary=summary,
            )
            (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBest checkpoint: {save_dir / 'best.pt'}")
    if val_loader is not None:
        print(f"Best val top1: {best_metric:.4f}")


if __name__ == "__main__":
    main()
