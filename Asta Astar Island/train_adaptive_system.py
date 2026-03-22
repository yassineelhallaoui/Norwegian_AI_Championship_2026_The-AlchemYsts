"""Train the complete adaptive ensemble system.

This script trains all components of the adaptive system:
1. Ensemble of diverse base models
2. Regime detector for round-specific adaptation

Training uses leave-one-round-out validation to ensure realistic performance estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_ensemble import AdaptiveConfig, AdaptiveEnsembleSystem
from data_io import load_historical_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the adaptive ensemble system.")
    parser.add_argument(
        "--historical-dir",
        default=str(Path(__file__).resolve().parent / "historical_data"),
        help="Directory with historical rounds and ground truth.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "models"),
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble (use single model).",
    )
    parser.add_argument(
        "--no-regime-detection",
        action="store_true",
        help="Disable regime detection.",
    )
    parser.add_argument(
        "--regime-likelihood-scale",
        type=float,
        default=40.0,
        help="Likelihood scale for regime weighting.",
    )

    args = parser.parse_args()

    # Load historical data
    print("Loading historical data...")
    historical_rounds = load_historical_rounds(args.historical_dir)
    print(f"Loaded {len(historical_rounds)} historical rounds")

    # Configure the adaptive system
    config = AdaptiveConfig(
        use_ensemble=not args.no_ensemble,
        use_regime_detection=not args.no_regime_detection,
        regime_likelihood_scale=args.regime_likelihood_scale,
        enable_fallback=True,
    )

    print("\nTraining adaptive ensemble system...")
    print(f"  Ensemble: {config.use_ensemble}")
    print(f"  Regime detection: {config.use_regime_detection}")

    # Train the full system
    system = AdaptiveEnsembleSystem(config=config).fit(historical_rounds)

    # Save trained models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving models to {output_dir}...")
    system.save(output_dir)

    # Save training metadata for reference
    metadata = {
        "num_historical_rounds": len(historical_rounds),
        "config": {
            "use_ensemble": config.use_ensemble,
            "use_regime_detection": config.use_regime_detection,
            "regime_likelihood_scale": config.regime_likelihood_scale,
            "enable_fallback": config.enable_fallback,
        },
    }

    if system.ensemble_model is not None:
        metadata["ensemble"] = {
            "num_base_models": len(system.ensemble_model.base_models),
            "model_weights": system.ensemble_model.model_weights,
            "model_configs": [
                {
                    "name": model.config.name,
                    "type": model.config.model_type,
                    "temperature": model.config.temperature_alpha,
                    "validation_score": model.validation_score,
                }
                for model in system.ensemble_model.base_models
            ],
        }

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved training metadata to {metadata_path}")
    print("\nTraining complete!")

    # Display ensemble weights for visibility
    if system.ensemble_model is not None:
        print("\nEnsemble model weights (from validation performance):")
        for name, weight in system.ensemble_model.model_weights.items():
            print(f"  {name}: {weight:.4f}")


if __name__ == "__main__":
    main()
