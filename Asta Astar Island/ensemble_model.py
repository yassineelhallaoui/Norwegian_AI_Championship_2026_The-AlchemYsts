"""Ensemble of diverse base models for robust predictions across different round regimes.

This module implements the first layer of the adaptive system: multiple models with
different biases that together provide more reliable predictions than any single model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

from features import build_feature_matrix
from scoring import floor_and_normalize


def apply_temperature_scaling(prediction: np.ndarray, alpha: float, floor: float = 1e-4) -> np.ndarray:
    """Flatten or sharpen a probability grid without changing its support.
    
    Temperature scaling adjusts how "peaked" the probability distribution is:
    - alpha > 1.0 flattens (more uniform, less confident)
    - alpha < 1.0 sharpens (more peaked, more confident)
    - alpha = 1.0 leaves unchanged
    """
    if alpha <= 0:
        raise ValueError("Temperature alpha must be positive.")

    adjusted = np.clip(np.asarray(prediction, dtype=float), floor, None)
    if abs(alpha - 1.0) > 1e-12:
        adjusted = np.power(adjusted, 1.0 / alpha)
    return adjusted / adjusted.sum(axis=-1, keepdims=True)


@dataclass
class BaseModelConfig:
    """Configuration for a single base model variant."""

    model_type: Literal["ridge", "extra_trees", "random_forest", "blend", "gradient_boosting"]
    name: str

    # Ridge parameters
    ridge_alpha: float = 1.0

    # Tree parameters
    n_estimators: int = 160
    min_samples_leaf: int = 2
    max_depth: int | None = None
    train_sample_limit: int = 48000

    # Blend parameters (for ridge+trees hybrid)
    tree_weight: float = 0.7

    # Temperature
    temperature_alpha: float = 1.0


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble of models."""

    base_model_configs: list[BaseModelConfig]
    ensemble_method: Literal["average", "weighted", "stacking"] = "average"
    validation_holdout: float = 0.15


class BaseModel:
    """Wrapper for a single base model."""

    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config
        self.model = None
        self.ridge_model = None
        self.tree_model = None
        self.validation_score: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "BaseModel":
        """Train the base model according to its configuration."""
        if self.config.model_type == "ridge":
            self.model = Ridge(alpha=self.config.ridge_alpha)
            self.model.fit(features, targets)

        elif self.config.model_type == "extra_trees":
            self.model = ExtraTreesRegressor(
                n_estimators=self.config.n_estimators,
                min_samples_leaf=self.config.min_samples_leaf,
                max_depth=self.config.max_depth,
                random_state=0,
                n_jobs=-1,
            )
            # Subsample if too much data (speed/performance tradeoff)
            if len(features) > self.config.train_sample_limit:
                rng = np.random.default_rng(0)
                indices = rng.choice(len(features), size=self.config.train_sample_limit, replace=False)
                self.model.fit(features[indices], targets[indices])
            else:
                self.model.fit(features, targets)

        elif self.config.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                min_samples_leaf=self.config.min_samples_leaf,
                max_depth=self.config.max_depth,
                random_state=0,
                n_jobs=-1,
            )
            if len(features) > self.config.train_sample_limit:
                rng = np.random.default_rng(0)
                indices = rng.choice(len(features), size=self.config.train_sample_limit, replace=False)
                self.model.fit(features[indices], targets[indices])
            else:
                self.model.fit(features, targets)

        elif self.config.model_type == "blend":
            # Hybrid: combine linear (Ridge) and non-linear (ExtraTrees) predictions
            self.ridge_model = Ridge(alpha=self.config.ridge_alpha)
            self.ridge_model.fit(features, targets)

            self.tree_model = ExtraTreesRegressor(
                n_estimators=self.config.n_estimators,
                min_samples_leaf=self.config.min_samples_leaf,
                max_depth=self.config.max_depth,
                random_state=0,
                n_jobs=-1,
            )
            if len(features) > self.config.train_sample_limit:
                rng = np.random.default_rng(0)
                indices = rng.choice(len(features), size=self.config.train_sample_limit, replace=False)
                self.tree_model.fit(features[indices], targets[indices])
            else:
                self.tree_model.fit(features, targets)

        elif self.config.model_type == "gradient_boosting":
            base_estimator = HistGradientBoostingRegressor(
                max_iter=self.config.n_estimators,
                min_samples_leaf=self.config.min_samples_leaf,
                max_depth=self.config.max_depth,
                random_state=0,
            )
            self.model = MultiOutputRegressor(base_estimator)
            if len(features) > self.config.train_sample_limit:
                rng = np.random.default_rng(0)
                indices = rng.choice(len(features), size=self.config.train_sample_limit, replace=False)
                self.model.fit(features[indices], targets[indices])
            else:
                self.model.fit(features, targets)

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.config.model_type == "blend":
            # Weighted combination of ridge and tree predictions
            ridge_pred = self.ridge_model.predict(features)
            tree_pred = self.tree_model.predict(features)
            blend = self.config.tree_weight
            return (1.0 - blend) * ridge_pred + blend * tree_pred
        else:
            return self.model.predict(features)


class EnsembleModel:
    """Ensemble of base models with automatic validation and weighting.
    
    The ensemble combines multiple models trained with different configurations.
    Each model gets a weight based on its leave-one-round-out validation performance.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or self._default_config()
        self.base_models: list[BaseModel] = []
        self.model_weights: dict[str, float] = {}

    @staticmethod
    def _default_config() -> EnsembleConfig:
        """Create default ensemble configuration with diverse models.
        
        Each model has different characteristics to cover various scenarios:
        - Conservative blend: Balanced approach, usually the safest
        - Flattened blend: Less confident predictions, better for uncertain rounds
        - Tree-heavy blend: More aggressive tree-based predictions
        - Deep ExtraTrees: Pure tree approach with depth limits
        - Random Forest: Different tree algorithm for diversity
        - Gradient Boosting: Sequential error correction
        """
        return EnsembleConfig(
            base_model_configs=[
                # Conservative blend (current best)
                BaseModelConfig(
                    model_type="blend",
                    name="conservative_blend",
                    ridge_alpha=1.0,
                    n_estimators=160,
                    tree_weight=0.7,
                    temperature_alpha=1.0,
                ),
                # Slightly flattened blend
                BaseModelConfig(
                    model_type="blend",
                    name="flattened_blend",
                    ridge_alpha=1.0,
                    n_estimators=160,
                    tree_weight=0.7,
                    temperature_alpha=1.10,
                ),
                # More trees, less regularization
                BaseModelConfig(
                    model_type="blend",
                    name="tree_heavy_blend",
                    ridge_alpha=0.5,
                    n_estimators=200,
                    tree_weight=0.85,
                    temperature_alpha=1.0,
                ),
                # Pure ExtraTrees with depth limit
                BaseModelConfig(
                    model_type="extra_trees",
                    name="deep_trees",
                    n_estimators=200,
                    max_depth=25,
                    min_samples_leaf=1,
                    temperature_alpha=1.0,
                ),
                # Random Forest for diversity
                BaseModelConfig(
                    model_type="random_forest",
                    name="random_forest",
                    n_estimators=150,
                    max_depth=20,
                    temperature_alpha=1.05,
                ),
                # Gradient Boosting (NEW for V2)
                BaseModelConfig(
                    model_type="gradient_boosting",
                    name="gradient_boosting",
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=20,
                    temperature_alpha=1.0,
                ),
            ],
            ensemble_method="weighted",
        )

    def fit(self, historical_rounds: list[dict]) -> "EnsembleModel":
        """Train all base models with leave-one-round-out validation.
        
        This validation strategy is crucial: for each round, we train on all other
        rounds and test on the held-out round. This simulates how the model will
        perform on truly unseen rounds.
        """
        # Organize data by round for leave-one-out validation
        round_data: list[dict] = []

        for round_idx, round_bundle in enumerate(historical_rounds):
            round_features = []
            round_targets = []
            for initial_state, ground_truth in zip(round_bundle["initial_states"], round_bundle["ground_truths"]):
                round_features.append(build_feature_matrix(initial_state))
                round_targets.append(np.asarray(ground_truth, dtype=float).reshape(-1, 6))

            round_data.append(
                {
                    "round_idx": round_idx,
                    "features": np.concatenate(round_features, axis=0),
                    "targets": np.concatenate(round_targets, axis=0),
                    "round_bundle": round_bundle,
                }
            )

        # Concatenate all data for final training
        all_features = np.concatenate([rd["features"] for rd in round_data], axis=0)
        all_targets = np.concatenate([rd["targets"] for rd in round_data], axis=0)

        # Leave-one-round-out validation for model weighting
        round_scores: dict[str, list[float]] = {cfg.name: [] for cfg in self.config.base_model_configs}

        for holdout_idx in range(len(historical_rounds)):
            # Build train and val sets by excluding/including specific rounds
            train_features_list = [rd["features"] for rd in round_data if rd["round_idx"] != holdout_idx]
            train_targets_list = [rd["targets"] for rd in round_data if rd["round_idx"] != holdout_idx]
            val_features = round_data[holdout_idx]["features"]
            val_targets = round_data[holdout_idx]["targets"]

            if not train_features_list:
                continue

            train_features = np.concatenate(train_features_list, axis=0)
            train_targets = np.concatenate(train_targets_list, axis=0)

            for config in self.config.base_model_configs:
                model = BaseModel(config).fit(train_features, train_targets)
                val_pred = model.predict(val_features)

                # Apply temperature if configured
                if config.temperature_alpha != 1.0:
                    grid_shape = historical_rounds[holdout_idx]["initial_states"][0]["grid"]
                    h, w = np.array(grid_shape).shape
                    num_seeds = len(historical_rounds[holdout_idx]["initial_states"])

                    reshaped_pred = val_pred.reshape(num_seeds, h, w, 6)
                    for i in range(len(reshaped_pred)):
                        reshaped_pred[i] = apply_temperature_scaling(reshaped_pred[i], config.temperature_alpha)
                    val_pred = reshaped_pred.reshape(-1, 6)

                # Compute validation score (negative KL divergence approximation)
                val_pred = floor_and_normalize(val_pred.reshape(val_targets.shape), floor=1e-4)
                score = -np.mean(np.sum(val_targets * np.log(np.clip(val_pred / (val_targets + 1e-6), 1e-6, 1e6)), axis=-1))
                round_scores[config.name].append(score)

        # Compute average validation scores across all held-out rounds
        avg_scores = {name: np.mean(scores) for name, scores in round_scores.items()}

        # Train final models on all data (no holdout)
        for config in self.config.base_model_configs:
            model = BaseModel(config).fit(all_features, all_targets)
            model.validation_score = avg_scores[config.name]
            self.base_models.append(model)

        # Compute ensemble weights based on validation performance
        if self.config.ensemble_method == "weighted":
            # Softmax weighting: better validation performance = higher weight
            scores = np.array([model.validation_score for model in self.base_models])
            weights = np.exp(-scores)  # Negative because lower KL is better
            weights = weights / weights.sum()
            self.model_weights = {model.config.name: float(w) for model, w in zip(self.base_models, weights)}
        else:
            # Equal weighting
            self.model_weights = {model.config.name: 1.0 / len(self.base_models) for model in self.base_models}

        return self

    def predict(self, initial_state: dict, return_individual: bool = False) -> np.ndarray | dict[str, np.ndarray]:
        """Generate ensemble prediction for a single initial state."""
        features = build_feature_matrix(initial_state)
        grid = np.asarray(initial_state["grid"], dtype=int)
        h, w = grid.shape

        individual_predictions = {}
        ensemble_prediction = np.zeros((h, w, 6), dtype=float)

        for model in self.base_models:
            pred = model.predict(features)
            pred = pred.reshape(h, w, 6)

            # Apply temperature if configured
            if model.config.temperature_alpha != 1.0:
                pred = apply_temperature_scaling(pred, model.config.temperature_alpha)

            pred = floor_and_normalize(pred, floor=1e-4)
            individual_predictions[model.config.name] = pred

            # Add weighted contribution to ensemble
            weight = self.model_weights[model.config.name]
            ensemble_prediction += weight * pred

        # Normalize ensemble
        ensemble_prediction = floor_and_normalize(ensemble_prediction, floor=1e-4)

        if return_individual:
            return {
                "ensemble": ensemble_prediction,
                "individual": individual_predictions,
                "weights": self.model_weights,
            }

        return ensemble_prediction

    def predict_with_uncertainty(self, initial_state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Generate prediction with epistemic uncertainty estimate.
        
        Uncertainty is measured as the variance across ensemble member predictions.
        High variance means the models disagree = we should be less confident.
        """
        result = self.predict(initial_state, return_individual=True)
        ensemble = result["ensemble"]
        individual = result["individual"]

        # Compute variance across models as uncertainty measure
        predictions_stack = np.stack([individual[name] for name in individual.keys()], axis=0)
        uncertainty = np.var(predictions_stack, axis=0).sum(axis=-1)  # Sum over classes

        return ensemble, uncertainty

    def save(self, path: str | Path) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "base_models": self.base_models,
                "model_weights": self.model_weights,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleModel":
        """Load ensemble from disk."""
        payload = joblib.load(Path(path))
        ensemble = cls(config=payload["config"])
        ensemble.base_models = payload["base_models"]
        ensemble.model_weights = payload["model_weights"]
        return ensemble
