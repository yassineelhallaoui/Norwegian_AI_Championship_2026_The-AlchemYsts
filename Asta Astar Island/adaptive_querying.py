"""Uncertainty-aware query planning and adaptive posterior updates.

This module handles two critical tasks:
1. Deciding WHERE to query (query selection strategy)
2. Updating predictions based on query results (robust calibration)
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
from scipy.ndimage import convolve

from constants import EMPTY_CLASS, MOUNTAIN_CLASS, VIEWPORT_SIZE
from scoring import entropy_map, floor_and_normalize


# Kernel for detecting coastal cells (cells adjacent to ocean)
CARDINAL_KERNEL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def allocate_query_budget_by_uncertainty(
    priors: list[np.ndarray],
    uncertainties: list[np.ndarray],
    total_queries: int,
    min_queries_per_seed: int = 1,
) -> list[int]:
    """Allocate query budget across seeds based on uncertainty.
    
    Instead of equal allocation, we give more queries to seeds where:
    - The prediction entropy is high (data uncertainty)
    - The ensemble models disagree (epistemic uncertainty)
    
    This ensures we spend our query budget where it provides the most information.
    """
    if total_queries <= 0:
        return [0 for _ in priors]

    num_seeds = len(priors)

    # Compute combined uncertainty metric for each seed
    combined_scores = []
    for prior, uncertainty in zip(priors, uncertainties):
        entropy_sum = float(entropy_map(prior).sum())
        uncertainty_sum = float(uncertainty.sum())
        # Blend entropy (data uncertainty) with model disagreement (epistemic uncertainty)
        combined_scores.append(entropy_sum + 0.5 * uncertainty_sum)

    combined_scores = np.array(combined_scores)

    # Reserve minimum queries per seed so no seed is completely ignored
    reserved = min_queries_per_seed * num_seeds
    if total_queries <= reserved:
        return [total_queries // num_seeds + (1 if i < total_queries % num_seeds else 0) for i in range(num_seeds)]

    # Allocate remaining budget proportionally to uncertainty
    allocation = np.full(num_seeds, min_queries_per_seed, dtype=int)
    remaining = total_queries - reserved

    raw = remaining * combined_scores / max(float(combined_scores.sum()), 1.0)
    extra_allocation = np.floor(raw).astype(int)
    remainder = remaining - int(extra_allocation.sum())

    allocation += extra_allocation

    # Distribute remainder to highest-uncertainty seeds
    if remainder > 0:
        order = np.argsort(-(raw - extra_allocation))
        for idx in order[:remainder]:
            allocation[idx] += 1

    return allocation.tolist()


def select_queries_with_thompson_sampling(
    initial_state: dict,
    prior: np.ndarray,
    uncertainty: np.ndarray,
    num_queries: int,
    existing_observations: Iterable[dict] | None = None,
    exploration_weight: float = 0.3,
) -> list[dict]:
    """Select query positions using uncertainty-guided Thompson sampling.
    
    Thompson sampling balances exploitation (querying high-information areas)
    with exploration (trying new areas). We prioritize:
    1. Around settlements (highest variance)
    2. Coastal areas (port formation potential)
    3. High uncertainty regions
    4. Coverage gaps (avoid redundant queries)
    """
    if num_queries <= 0:
        return []

    positions = _candidate_positions(initial_state)
    grid = np.asarray(initial_state["grid"], dtype=int)

    # Compute scoring components for each cell
    entropy = entropy_map(prior)
    dynamic_mass = prior[:, :, 1:5].sum(axis=-1)  # Settlement, Port, Ruin, Forest probabilities
    coastal = convolve((grid == 10).astype(float), CARDINAL_KERNEL, mode="constant") > 0
    forest_mask = grid == 4
    food_potential = convolve(forest_mask.astype(float), np.ones((7, 7)), mode="constant")

    # Track coverage to avoid redundant queries
    coverage = np.zeros(prior.shape[:2], dtype=float)
    used_positions: set[tuple[int, int]] = set()

    for observation in existing_observations or []:
        viewport_x = int(observation["viewport_x"])
        viewport_y = int(observation["viewport_y"])
        viewport_w = int(observation["viewport_w"])
        viewport_h = int(observation["viewport_h"])
        coverage[viewport_y : viewport_y + viewport_h, viewport_x : viewport_x + viewport_w] += 1.0
        used_positions.add((viewport_x, viewport_y))

    positions = [pos for pos in positions if pos not in used_positions]
    selected: list[dict] = []

    # Iteratively select the best position, updating coverage each time
    for iteration in range(num_queries):
        best_score = None
        best_position = None

        # Score all candidate positions
        for viewport_x, viewport_y in positions:
            y_slice = slice(viewport_y, viewport_y + VIEWPORT_SIZE)
            x_slice = slice(viewport_x, viewport_x + VIEWPORT_SIZE)

            # Information gain components
            entropy_gain = entropy[y_slice, x_slice].sum()
            uncertainty_gain = uncertainty[y_slice, x_slice].sum()
            dynamic_gain = dynamic_mass[y_slice, x_slice].sum()
            coastal_bonus = 0.4 * coastal[y_slice, x_slice].sum()
            food_bonus = 0.1 * food_potential[y_slice, x_slice].sum()

            # Coverage penalty (avoid redundant queries)
            coverage_penalty = 0.5 * coverage[y_slice, x_slice].sum()

            # Thompson sampling: add exploration noise
            exploration_noise = 0.0
            if exploration_weight > 0:
                rng = np.random.default_rng(seed=42 + iteration)
                exploration_noise = exploration_weight * rng.normal(0, entropy_gain * 0.1)

            # Combined score
            score = (
                entropy_gain + 0.7 * uncertainty_gain + 0.3 * dynamic_gain + coastal_bonus + food_bonus - coverage_penalty + exploration_noise
            )

            if best_score is None or score > best_score:
                best_score = score
                best_position = (viewport_x, viewport_y)

        if best_position is None:
            break

        viewport_x, viewport_y = best_position
        selected.append(
            {
                "viewport_x": viewport_x,
                "viewport_y": viewport_y,
                "viewport_w": VIEWPORT_SIZE,
                "viewport_h": VIEWPORT_SIZE,
            }
        )

        # Update coverage so next query avoids this area
        coverage[viewport_y : viewport_y + VIEWPORT_SIZE, viewport_x : viewport_x + VIEWPORT_SIZE] += 1.0
        positions.remove(best_position)

    return selected


def _candidate_positions(initial_state: dict) -> list[tuple[int, int]]:
    """Generate candidate viewport positions with strategic priorities.
    
    Returns a list of (x, y) positions for potential queries, prioritized by:
    1. Around settlements (these areas have highest variance)
    2. Grid coverage (systematic exploration)
    3. Edge coverage (fjords and ocean interactions)
    """
    grid = np.asarray(initial_state["grid"], dtype=int)
    height, width = grid.shape
    max_x = max(0, width - VIEWPORT_SIZE)
    max_y = max(0, height - VIEWPORT_SIZE)

    seen: set[tuple[int, int]] = set()

    # Priority 1: Around settlements (highest variance in predictions)
    for settlement in initial_state["settlements"]:
        for dx, dy in [(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (-3, 3), (3, -3), (3, 3)]:
            x = min(max(settlement["x"] - VIEWPORT_SIZE // 2 + dx, 0), max_x)
            y = min(max(settlement["y"] - VIEWPORT_SIZE // 2 + dy, 0), max_y)
            seen.add((x, y))

    # Priority 2: Grid coverage (systematic exploration)
    step = 6
    for x in range(0, max_x + 1, step):
        for y in range(0, max_y + 1, step):
            seen.add((x, y))

    # Priority 3: Dense coverage near edges (fjords and ocean interactions)
    edge_step = 4
    for x in range(0, max_x + 1, edge_step):
        seen.add((x, 0))
        seen.add((x, max_y))
    for y in range(0, max_y + 1, edge_step):
        seen.add((0, y))
        seen.add((max_x, y))

    return sorted(seen)


def execute_adaptive_queries(
    client,
    round_id: str,
    initial_states: list[dict],
    priors: list[np.ndarray],
    uncertainties: list[np.ndarray],
    total_queries: int,
    existing_observations: dict[int, list[dict]] | None = None,
    delay_s: float = 0.25,
) -> dict[int, list[dict]]:
    """Execute query allocation with adaptive uncertainty-aware strategy.
    
    This is the main entry point for live querying:
    1. Allocates query budget across seeds based on uncertainty
    2. Selects optimal positions for each seed
    3. Executes queries via API
    4. Returns observations for prediction update
    """
    allocation = allocate_query_budget_by_uncertainty(priors, uncertainties, total_queries)
    observations: dict[int, list[dict]] = {}
    existing_observations = existing_observations or {}

    for seed_index, (state, prior, uncertainty, num_queries) in enumerate(
        zip(initial_states, priors, uncertainties, allocation)
    ):
        planned_queries = select_queries_with_thompson_sampling(
            state,
            prior,
            uncertainty,
            num_queries,
            existing_observations=existing_observations.get(seed_index, []),
        )

        seed_entries: list[dict] = []
        for query in planned_queries:
            result = client.simulate(round_id, seed_index, **query)
            viewport = result["viewport"]

            # Map terrain codes to class indices
            class_grid = np.vectorize(lambda code: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}[code])(
                np.asarray(result["grid"], dtype=int)
            )

            seed_entries.append(
                {
                    "seed_index": seed_index,
                    "viewport_x": viewport["x"],
                    "viewport_y": viewport["y"],
                    "viewport_w": viewport["w"],
                    "viewport_h": viewport["h"],
                    "class_grid": class_grid.astype(int),
                    "raw_grid": result["grid"],
                    "settlements": result.get("settlements", []),
                }
            )
            time.sleep(delay_s)

        observations[seed_index] = seed_entries

    return observations


def update_predictions_with_robust_calibration(
    priors: list[np.ndarray],
    query_observations: dict[int, Iterable[dict]],
    prior_strength: float = 15.0,
    calibration_blend: float = 0.8,
    max_weight_ratio: float = 3.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Update predictions with robust query calibration that avoids over-fitting.
    
    Key improvements over naive approaches:
    - Higher prior_strength (15 vs 12): Queries need stronger evidence to override priors
    - Calibration_blend < 1.0: Only partially trust class weight adjustments
    - Max_weight_ratio: Caps extreme adjustments to prevent over-correction
    
    The logic: observe what classes appear in queries vs what we expected,
    compute adjustment weights, but constrain them to avoid wild swings.
    """
    local_counts = [np.zeros_like(prior) for prior in priors]
    observed_counts = np.zeros(priors[0].shape[-1], dtype=float)
    expected_counts = np.zeros(priors[0].shape[-1], dtype=float)

    # Count what we observed in queries vs what we expected
    for seed_index, observations in query_observations.items():
        prior = priors[seed_index]
        for observation in observations:
            class_grid = np.asarray(observation["class_grid"], dtype=int)
            viewport_x = observation["viewport_x"]
            viewport_y = observation["viewport_y"]
            viewport_h, viewport_w = class_grid.shape

            for y_offset in range(viewport_h):
                for x_offset in range(viewport_w):
                    y = viewport_y + y_offset
                    x = viewport_x + x_offset
                    observed_class = int(class_grid[y_offset, x_offset])
                    local_counts[seed_index][y, x, observed_class] += 1.0
                    observed_counts[observed_class] += 1.0
                    expected_counts += prior[y, x]

    # Compute class weights: observed / expected
    # If we see more of a class than expected, weight it up; less, weight it down
    class_weights = (observed_counts + 1.0) / (expected_counts + 1.0)

    # Reduce weight adjustments for static classes (empty, mountain)
    # These shouldn't change much, so be conservative
    class_weights[EMPTY_CLASS] = 1.0 + 0.2 * (class_weights[EMPTY_CLASS] - 1.0)
    class_weights[MOUNTAIN_CLASS] = 1.0 + 0.2 * (class_weights[MOUNTAIN_CLASS] - 1.0)

    # Cap extreme weights to prevent over-correction
    class_weights = np.clip(class_weights, 1.0 / max_weight_ratio, max_weight_ratio)

    # Geometric mean normalization (keeps relative adjustments)
    class_weights /= np.exp(np.mean(np.log(class_weights)))

    updated_predictions: list[np.ndarray] = []
    for prior, counts in zip(priors, local_counts):
        # Apply calibration with blending (don't fully trust class weights)
        calibrated_prior = prior * np.power(class_weights, calibration_blend)
        calibrated_prior = calibrated_prior / calibrated_prior.sum(axis=-1, keepdims=True)

        # Bayesian update with strong prior
        posterior = prior_strength * calibrated_prior + counts
        updated_predictions.append(floor_and_normalize(posterior, floor=1e-4))

    return updated_predictions, class_weights
