"""Unified adaptive system combining ensemble models with query-pattern regime detection.

This is the main orchestration layer that ties everything together:
- Uses the ensemble for base predictions and uncertainty estimation
- Decides whether to use regime detection or robust calibration based on confidence
- Applies fallback protection if queries cause problematic shifts
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import from Code_querypattern for regime detection
CODE_QUERYPATTERN_DIR = Path(__file__).resolve().parent.parent / "Code_querypattern"
if str(CODE_QUERYPATTERN_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_QUERYPATTERN_DIR))

from querypattern_model import QueryPatternConfig, QueryPatternMixture  # noqa: E402

from adaptive_querying import update_predictions_with_robust_calibration
from ensemble_model import EnsembleModel
from scoring import floor_and_normalize


@dataclass
class AdaptiveConfig:
    """Configuration for the full adaptive system."""

    # Ensemble model settings
    use_ensemble: bool = True

    # Query pattern regime detection
    use_regime_detection: bool = True
    regime_likelihood_scale: float = 40.0
    regime_use_local_posterior: bool = True

    # Fallback strategy
    enable_fallback: bool = True
    fallback_confidence_threshold: float = 0.15  # If model disagreement > this, use conservative approach


class AdaptiveEnsembleSystem:
    """Complete adaptive system with ensemble, regime detection, and robust fallback.
    
    This is the main interface for predictions. It orchestrates:
    1. Base predictions from the ensemble
    2. Uncertainty estimation to decide strategy
    3. Either regime detection (when confident) or robust calibration (when uncertain)
    4. Fallback blending if queries cause large divergence
    """

    def __init__(self, config: AdaptiveConfig | None = None) -> None:
        self.config = config or AdaptiveConfig()
        self.ensemble_model: EnsembleModel | None = None
        self.regime_detector: QueryPatternMixture | None = None

    def fit(self, historical_rounds: list[dict]) -> "AdaptiveEnsembleSystem":
        """Train both ensemble and regime detector on historical data."""
        if self.config.use_ensemble:
            self.ensemble_model = EnsembleModel().fit(historical_rounds)

        if self.config.use_regime_detection:
            regime_config = QueryPatternConfig(
                temperature_alpha=1.0,  # Ensemble already handles temperature
                context_shrink=300.0,
                likelihood_scale=self.config.regime_likelihood_scale,
                use_local_posterior=self.config.regime_use_local_posterior,
                include_round_experts=True,
                include_pooled_expert=True,
                include_base_expert=True,
            )
            self.regime_detector = QueryPatternMixture(config=regime_config).fit(historical_rounds)

        return self

    def predict_with_queries(
        self,
        initial_states: list[dict],
        query_observations: dict[int, list[dict]],
        return_diagnostics: bool = False,
    ) -> tuple[list[np.ndarray], dict | None]:
        """Generate predictions using the full adaptive pipeline.
        
        The flow is:
        1. Get ensemble predictions + uncertainty for each seed
        2. If no queries, return ensemble predictions
        3. If queries exist, decide strategy based on average uncertainty:
           - Low uncertainty (< 0.15): Use regime detection (more aggressive)
           - High uncertainty (>= 0.15): Use robust calibration (more conservative)
        4. Check if query-updated predictions diverge too much from ensemble
        5. If divergence > 0.5, blend back toward ensemble
        """
        diagnostics = {}

        # Step 1: Generate base ensemble predictions with uncertainty
        if self.ensemble_model is not None:
            ensemble_priors = []
            uncertainties = []

            for state in initial_states:
                prior, uncertainty = self.ensemble_model.predict_with_uncertainty(state)
                ensemble_priors.append(prior)
                uncertainties.append(uncertainty)

            diagnostics["ensemble_uncertainty"] = [float(u.mean()) for u in uncertainties]
        else:
            raise RuntimeError("Ensemble model must be fitted before prediction.")

        # Step 2: Check if we have any queries
        has_queries = any(query_observations.values())

        if not has_queries:
            # No queries - just return ensemble predictions
            diagnostics["strategy"] = "ensemble_only"
            return (ensemble_priors, diagnostics) if return_diagnostics else (ensemble_priors, None)

        # Step 3: Decide strategy based on model confidence
        # Low uncertainty = models agree = we can be more aggressive with regime detection
        # High uncertainty = models disagree = use conservative robust calibration
        avg_uncertainty = np.mean([u.mean() for u in uncertainties])
        use_regime_detection = self.config.use_regime_detection and avg_uncertainty < self.config.fallback_confidence_threshold

        if use_regime_detection and self.regime_detector is not None:
            # Use regime detection for query integration
            # This builds round-specific experts and weights by query likelihood
            prepared_round = {
                "initial_states": initial_states,
                "base_predictions": ensemble_priors,
                "expert_predictions": self.regime_detector._expert_predictions(initial_states, ensemble_priors),
            }

            predictions, regime_diagnostics = self.regime_detector.combine_prepared_round(
                prepared_round,
                query_observations,
                likelihood_scale=self.config.regime_likelihood_scale,
                use_local_posterior=self.config.regime_use_local_posterior,
            )

            diagnostics["strategy"] = "regime_detection"
            diagnostics["regime"] = regime_diagnostics

        else:
            # Use robust calibration approach
            # More conservative: doesn't assume we know the round regime
            predictions, class_weights = update_predictions_with_robust_calibration(
                ensemble_priors,
                query_observations,
                prior_strength=15.0,
                calibration_blend=0.8,
                max_weight_ratio=3.0,
            )

            diagnostics["strategy"] = "robust_calibration"
            diagnostics["class_weights"] = class_weights.tolist()

        # Step 4: Fallback check - compare with ensemble-only predictions
        # This is the safety net: if queries caused a huge shift, something might be wrong
        if self.config.enable_fallback and has_queries:
            divergence = self._compute_kl_divergence(predictions, ensemble_priors)
            avg_divergence = np.mean(divergence)

            diagnostics["query_update_divergence"] = float(avg_divergence)

            # If query update caused large divergence, blend with original ensemble
            if avg_divergence > 0.5:  # Threshold for "something went wrong"
                blend_weight = 0.7  # Trust ensemble more when queries seem noisy
                blended_predictions = []
                for pred, prior in zip(predictions, ensemble_priors):
                    blended = blend_weight * prior + (1.0 - blend_weight) * pred
                    blended = floor_and_normalize(blended, floor=1e-4)
                    blended_predictions.append(blended)

                predictions = blended_predictions
                diagnostics["fallback_activated"] = True
                diagnostics["fallback_blend_weight"] = blend_weight
            else:
                diagnostics["fallback_activated"] = False

        return (predictions, diagnostics) if return_diagnostics else (predictions, None)

    @staticmethod
    def _compute_kl_divergence(predictions1: list[np.ndarray], predictions2: list[np.ndarray]) -> list[float]:
        """Compute KL divergence between two sets of predictions.
        
        KL divergence measures how different two probability distributions are.
        High divergence means the queries changed our predictions significantly.
        """
        divergences = []
        for p1, p2 in zip(predictions1, predictions2):
            # KL(p1 || p2) per cell, averaged
            kl = np.sum(p1 * np.log(np.clip(p1 / (p2 + 1e-8), 1e-8, 1e8)), axis=-1)
            divergences.append(float(kl.mean()))
        return divergences

    def save(self, directory: str | Path) -> None:
        """Save all components."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.ensemble_model is not None:
            self.ensemble_model.save(directory / "ensemble_model.joblib")

        if self.regime_detector is not None:
            import joblib
            joblib.dump(self.regime_detector, directory / "regime_detector.joblib")

        import joblib
        joblib.dump(self.config, directory / "adaptive_config.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "AdaptiveEnsembleSystem":
        """Load all components."""
        directory = Path(directory)
        import joblib

        config = joblib.load(directory / "adaptive_config.joblib")
        system = cls(config=config)

        if (directory / "ensemble_model.joblib").exists():
            system.ensemble_model = EnsembleModel.load(directory / "ensemble_model.joblib")

        if (directory / "regime_detector.joblib").exists():
            system.regime_detector = joblib.load(directory / "regime_detector.joblib")

        return system
