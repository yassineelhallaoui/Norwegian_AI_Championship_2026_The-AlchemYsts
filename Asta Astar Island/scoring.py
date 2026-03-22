"""Offline scoring helpers."""

from __future__ import annotations

import math

import numpy as np

from constants import PROBABILITY_FLOOR


def floor_and_normalize(probabilities: np.ndarray, floor: float = PROBABILITY_FLOOR) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), floor, None)
    return clipped / clipped.sum(axis=-1, keepdims=True)


def entropy_map(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    return -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)


def entropy_weighted_kl_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    p = np.asarray(ground_truth, dtype=float)
    q = floor_and_normalize(np.asarray(prediction, dtype=float))
    ent = entropy_map(p)
    kl = (p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum(axis=-1)
    mask = ent > 1e-12
    if not np.any(mask):
        return 100.0
    weighted_kl = float((ent[mask] * kl[mask]).sum() / ent[mask].sum())
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * weighted_kl)))
