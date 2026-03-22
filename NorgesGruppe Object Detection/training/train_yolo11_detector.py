"""
Train YOLO11 detector for product detection.

This uses a two-stage training recipe that worked well in our experiments:
  Stage 1: Train on a single fold with heavy augmentation
  Stage 2: Fine-tune on full dataset with milder augmentation

The idea is to first learn good feature representations from limited data,
then refine them on the complete dataset without overfitting.

We train as single-class detector (just find "products") because the
classification is handled separately by a dedicated classifier.

Usage:
  # Train fold 0 with default settings
  python train_yolo11_detector.py --fold 0 --stage1-epochs 10 --stage2-epochs 6
  
  # Train with different model or resolution
  python train_yolo11_detector.py --fold 0 --model yolo11l.pt --imgsz 1280 --batch 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

# Fix compatibility between PyTorch 2.6+ and older ultralytics versions
# PyTorch 2.6 changed torch.load to require weights_only=True by default,
# but ultralytics models were saved with weights_only=False
if not getattr(torch.load, "_ngd_patched", False):
    _original_load = torch.load
    def _compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)
    _compatible_load._ngd_patched = True
    torch.load = _compatible_load


def create_singlecls_yaml(src_yaml: Path, dst_yaml: Path) -> Path:
    """Create a single-class version of a YOLO dataset yaml file."""
    config = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    config["names"] = {0: "product"}  # Only one class: product
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    dst_yaml.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    return dst_yaml


def create_singlecls_labels(labels_dir: Path, singlecls_dir: Path) -> None:
    """Convert multi-class labels to single-class by changing all class IDs to 0."""
    singlecls_dir.mkdir(parents=True, exist_ok=True)
    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = "0"  # Set class to 0
                new_lines.append(" ".join(parts))
        (singlecls_dir / label_file.name).write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="YOLO11 two-stage single-class detector training")
    parser.add_argument("--fold", type=int, default=0, help="Which fold to train on (0-4)")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="YOLO11 model variant")
    parser.add_argument("--imgsz", type=int, default=960, help="Input image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--stage1-epochs", type=int, default=10, help="Stage 1 epochs (fold training)")
    parser.add_argument("--stage2-epochs", type=int, default=6, help="Stage 2 epochs (full fine-tune)")
    parser.add_argument("--device", default="0", help="GPU device ID")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export to ONNX")
    parser.add_argument("--export-imgsz", type=int, default=960, help="ONNX export image size")
    parser.add_argument("--project", type=Path,
                        default=(code_root / "working" / "runs").resolve())
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    code_root = Path(__file__).resolve().parent.parent
    dataset_root = code_root / "working" / "yolo_dataset"

    # Prepare single-class labels if they don't exist
    singlecls_labels = dataset_root / "labels_singlecls"
    if not singlecls_labels.exists():
        print("Creating single-class labels...")
        create_singlecls_labels(dataset_root / "labels", singlecls_labels)

    # Create single-class YAML configs for this fold
    model_stem = Path(args.model).stem
    fold_yaml = dataset_root / "folds" / f"fold_{args.fold}.yaml"
    singlecls_fold_yaml = dataset_root / "folds" / f"fold_{args.fold}_singlecls_{model_stem}.yaml"
    full_yaml = dataset_root / "folds" / "full.yaml"
    singlecls_full_yaml = dataset_root / "folds" / f"full_singlecls_{model_stem}.yaml"

    # Modify yamls to use single-class labels
    for src, dst in [(fold_yaml, singlecls_fold_yaml), (full_yaml, singlecls_full_yaml)]:
        config = yaml.safe_load(src.read_text(encoding="utf-8"))
        config["path"] = str(dataset_root)
        config["names"] = {0: "product"}
        dst.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

    # We need to swap the labels directory during training
    labels_orig = dataset_root / "labels"
    labels_backup = dataset_root / "labels_multiclass"

    def swap_to_singlecls():
        """Temporarily use single-class labels for training."""
        if labels_orig.exists() and not labels_backup.exists():
            labels_orig.rename(labels_backup)
        if not labels_orig.exists():
            singlecls_labels_src = dataset_root / "labels_singlecls"
            labels_orig.symlink_to(singlecls_labels_src)

    def swap_to_multiclass():
        """Restore original multi-class labels."""
        if labels_orig.is_symlink():
            labels_orig.unlink()
        if labels_backup.exists() and not labels_orig.exists():
            labels_backup.rename(labels_orig)

    # ========== STAGE 1: Train on fold ==========
    stage1_name = f"yolo11_{model_stem}_singlecls_fold{args.fold}_i{args.imgsz}_e{args.stage1_epochs}_s1"
    print(f"\n{'='*80}")
    print(f"STAGE 1: Training {args.model} single-class on fold {args.fold}")
    print(f"  Epochs: {args.stage1_epochs}, ImgSz: {args.imgsz}, Batch: {args.batch}")
    print(f"{'='*80}\n")

    swap_to_singlecls()
    try:
        model = YOLO(args.model)
        model.train(
            data=str(singlecls_fold_yaml.resolve()),
            epochs=args.stage1_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(args.project),
            name=stage1_name,
            device=args.device,
            workers=args.workers,
            seed=42,
            deterministic=True,
            cache="disk",
            patience=args.patience,
            close_mosaic=max(3, args.stage1_epochs // 3),
            amp=True,
            single_cls=True,
            # Stage 1: Heavy augmentation for robust feature learning
            mosaic=0.6,
            scale=0.25,
            translate=0.05,
            degrees=2.0,
            hsv_h=0.015,
            hsv_s=0.40,
            hsv_v=0.30,
            flipud=0.0,  # No vertical flip (products are always upright)
            fliplr=0.0,  # No horizontal flip (might confuse classifiers)
            mixup=0.05,
            copy_paste=0.0,
            erasing=0.1,
        )
        stage1_weights = args.project / stage1_name / "weights" / "best.pt"
    finally:
        swap_to_multiclass()

    # ========== STAGE 2: Fine-tune on full dataset ==========
    stage2_name = f"yolo11_{model_stem}_singlecls_fold{args.fold}_i{args.imgsz}_e{args.stage1_epochs}_ft{args.stage2_epochs}"
    print(f"\n{'='*80}")
    print(f"STAGE 2: Fine-tuning on full dataset for {args.stage2_epochs} epochs")
    print(f"  Starting from: {stage1_weights}")
    print(f"{'='*80}\n")

    swap_to_singlecls()
    try:
        model = YOLO(str(stage1_weights))
        model.train(
            data=str(singlecls_full_yaml.resolve()),
            epochs=args.stage2_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(args.project),
            name=stage2_name,
            device=args.device,
            workers=args.workers,
            seed=42,
            deterministic=True,
            cache="disk",
            patience=args.patience,
            close_mosaic=max(2, args.stage2_epochs // 3),
            amp=True,
            single_cls=True,
            # Stage 2: Milder augmentation, focus on refinement
            mosaic=0.5,
            scale=0.2,
            translate=0.05,
            degrees=2.0,
            hsv_h=0.015,
            hsv_s=0.40,
            hsv_v=0.30,
            flipud=0.0,
            fliplr=0.0,
            mixup=0.0,  # Disable mixup in fine-tuning
            copy_paste=0.0,
            auto_augment="randaugment",
            erasing=0.4,
        )
        stage2_weights = args.project / stage2_name / "weights" / "best.pt"
    finally:
        swap_to_multiclass()

    # ========== EXPORT TO ONNX ==========
    if args.export_onnx:
        print(f"\n{'='*80}")
        print(f"Exporting to ONNX (imgsz={args.export_imgsz})")
        print(f"{'='*80}\n")

        export_model = YOLO(str(stage2_weights))
        export_path = export_model.export(
            format="onnx",
            imgsz=args.export_imgsz,
            opset=17,
            simplify=True,
            half=False,  # FP32 for compatibility
            dynamic=True,  # Dynamic batch size support
        )
        print(f"ONNX exported to: {export_path}")

        # Save experiment summary for reference
        summary = {
            "model": args.model,
            "fold": args.fold,
            "imgsz": args.imgsz,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage1_weights": str(stage1_weights),
            "stage2_weights": str(stage2_weights),
            "onnx_path": str(export_path),
            "export_imgsz": args.export_imgsz,
        }
        summary_path = args.project / stage2_name / "experiment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"  Stage 1 weights: {stage1_weights}")
    print(f"  Stage 2 weights: {stage2_weights}")
    if args.export_onnx:
        print(f"  ONNX model: {export_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
