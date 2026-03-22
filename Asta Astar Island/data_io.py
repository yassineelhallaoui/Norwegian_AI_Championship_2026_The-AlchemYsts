"""Local data loading and snapshot helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from constants import INTERNAL_TO_PREDICTION_CLASS


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_historical_rounds(historical_dir: str | Path) -> list[dict]:
    historical_dir = Path(historical_dir)
    round_dirs = sorted(path for path in historical_dir.iterdir() if path.is_dir())

    bundles: list[dict] = []
    for round_dir in round_dirs:
        initial_states = json.loads((round_dir / "initial_states.json").read_text(encoding="utf-8"))
        ground_truths = []
        for seed_index in range(len(initial_states)):
            ground_truth_path = round_dir / f"seed_{seed_index}_ground_truth.json"
            ground_truths.append(json.loads(ground_truth_path.read_text(encoding="utf-8"))["ground_truth"])

        summary = json.loads((round_dir / "summary.json").read_text(encoding="utf-8"))
        bundles.append(
            {
                "path": str(round_dir),
                "name": round_dir.name,
                "round_id": summary.get("round_id", summary.get("id")),
                "round_number": summary["round_number"],
                "initial_states": initial_states,
                "ground_truths": ground_truths,
            }
        )
    return bundles


def round_snapshot_dir(base_dir: str | Path, round_details: dict) -> Path:
    return Path(base_dir) / f"round_{round_details['round_number']}_{round_details['id']}"


def save_live_snapshot(
    output_dir: str | Path,
    round_details: dict,
    budget: dict,
    my_rounds: list[dict],
    my_predictions: list[dict],
    leaderboard: list[dict],
) -> Path:
    snapshot_dir = round_snapshot_dir(output_dir, round_details)
    save_json(snapshot_dir / "round_details.json", round_details)
    save_json(snapshot_dir / "budget.json", budget)
    save_json(snapshot_dir / "my_rounds.json", my_rounds)
    save_json(snapshot_dir / "my_predictions_summary.json", my_predictions)
    save_json(snapshot_dir / "leaderboard_top20.json", leaderboard[:20])
    return snapshot_dir


def find_latest_saved_query_dir(query_root: str | Path, round_id: str) -> Path | None:
    query_root = Path(query_root)
    if not query_root.exists():
        return None
    matches = [path for path in query_root.iterdir() if path.is_dir() and round_id in path.name]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _raw_grid_to_class_grid(grid: list[list[int]]) -> np.ndarray:
    array = np.asarray(grid, dtype=int)
    mapper = np.vectorize(INTERNAL_TO_PREDICTION_CLASS.get)
    return mapper(array).astype(int)


def load_saved_query_observations(query_dir: str | Path) -> dict[int, list[dict]]:
    """Load saved live query logs into a seed-indexed structure."""

    query_dir = Path(query_dir)
    observations: dict[int, list[dict]] = {}
    for seed_dir in sorted(path for path in query_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        seed_index = int(seed_dir.name.split("_")[-1])
        entries: list[dict] = []
        for query_file in sorted(seed_dir.glob("query_*.json")):
            payload = json.loads(query_file.read_text(encoding="utf-8"))
            result = payload["result"]
            viewport = result["viewport"]
            entries.append(
                {
                    "source": str(query_file),
                    "seed_index": seed_index,
                    "timestamp": payload.get("timestamp"),
                    "viewport_x": viewport["x"],
                    "viewport_y": viewport["y"],
                    "viewport_w": viewport["w"],
                    "viewport_h": viewport["h"],
                    "class_grid": _raw_grid_to_class_grid(result["grid"]),
                    "raw_grid": result["grid"],
                    "settlements": result.get("settlements", []),
                }
            )
        observations[seed_index] = entries
    return observations


def prediction_summary(predictions: list[np.ndarray]) -> list[dict]:
    summary: list[dict] = []
    for seed_index, prediction in enumerate(predictions):
        argmax_grid = prediction.argmax(axis=-1)
        confidence_grid = prediction.max(axis=-1)
        summary.append(
            {
                "seed_index": seed_index,
                "argmax_grid": argmax_grid.tolist(),
                "confidence_grid": np.round(confidence_grid, 4).tolist(),
                "confidence_mean": float(confidence_grid.mean()),
                "confidence_min": float(confidence_grid.min()),
                "confidence_max": float(confidence_grid.max()),
            }
        )
    return summary
