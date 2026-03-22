"""Production runner for adaptive ensemble system on live rounds.

This is the main script for submitting predictions to active rounds.
It handles:
- Loading trained models
- Connecting to the API
- Managing saved queries from previous runs
- Executing additional live queries
- Generating predictions with the adaptive system
- Submitting results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adaptive_ensemble import AdaptiveEnsembleSystem
from adaptive_querying import execute_adaptive_queries
from api_client import AstarIslandClient
from data_io import (
    find_latest_saved_query_dir,
    load_saved_query_observations,
    prediction_summary,
    save_json,
    save_live_snapshot,
)


def _json_ready_query_observations(query_observations: dict[int, list[dict]]) -> dict[str, list[dict]]:
    """Convert query observations to JSON-serializable format."""
    payload: dict[str, list[dict]] = {}
    for seed_index, entries in query_observations.items():
        payload[str(seed_index)] = []
        for entry in entries:
            serializable = dict(entry)
            if "class_grid" in serializable:
                serializable["class_grid"] = np.asarray(serializable["class_grid"], dtype=int).tolist()
            payload[str(seed_index)].append(serializable)
    return payload


def _load_query_observations_file(path: Path) -> dict[int, list[dict]]:
    """Load query observations from a JSON file."""
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    observations: dict[int, list[dict]] = {}
    for seed_index_text, entries in payload.items():
        seed_index = int(seed_index_text)
        restored_entries: list[dict] = []
        for entry in entries:
            restored = dict(entry)
            if "class_grid" in restored:
                restored["class_grid"] = np.asarray(restored["class_grid"], dtype=int)
            restored_entries.append(restored)
        observations[seed_index] = restored_entries
    return observations


def _merge_query_observations(
    query_observations: dict[int, list[dict]],
    additional_observations: dict[int, list[dict]],
) -> None:
    """Merge additional observations into the main collection, avoiding duplicates.
    
    Deduplication is based on viewport position and size.
    """
    for seed_index, entries in additional_observations.items():
        target_entries = query_observations.setdefault(seed_index, [])
        seen = {
            (
                int(entry["viewport_x"]),
                int(entry["viewport_y"]),
                int(entry["viewport_w"]),
                int(entry["viewport_h"]),
            )
            for entry in target_entries
        }
        for entry in entries:
            key = (
                int(entry["viewport_x"]),
                int(entry["viewport_y"]),
                int(entry["viewport_w"]),
                int(entry["viewport_h"]),
            )
            if key in seen:
                continue
            target_entries.append(entry)
            seen.add(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive ensemble system on live round.")
    parser.add_argument(
        "--token-file",
        default=str(Path(__file__).resolve().parent.parent / "Code" / ".token"),
        help="Path to JWT token file.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parent / "models"),
        help="Directory with trained models.",
    )
    parser.add_argument(
        "--query-data-root",
        default=str(Path(__file__).resolve().parent.parent / "Code_kimi" / "query_data"),
        help="Root directory for saved live query logs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "artifacts"),
        help="Directory for snapshots and predictions.",
    )
    parser.add_argument(
        "--additional-live-queries",
        type=int,
        default=0,
        help="Additional live queries beyond saved queries.",
    )
    parser.add_argument(
        "--skip-saved-queries",
        action="store_true",
        help="Ignore saved query logs.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit predictions (omit for dry-run).",
    )

    args = parser.parse_args()

    # Load trained system
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading adaptive system from {model_dir}...")
    system = AdaptiveEnsembleSystem.load(model_dir)

    # Connect to API and find active round
    client = AstarIslandClient.from_token_file(args.token_file)
    rounds = client.get_rounds()
    active_round = next((r for r in rounds if r["status"] == "active"), None)

    if active_round is None:
        raise SystemExit("No active round found.")

    # Fetch round data
    round_details = client.get_round_details(active_round["id"])
    budget = client.get_budget()
    my_rounds = client.get_my_rounds()
    my_predictions = client.get_my_predictions(active_round["id"])
    leaderboard = client.get_leaderboard()

    # Save a snapshot of the current state for analysis
    snapshot_dir = save_live_snapshot(
        args.output_dir,
        round_details,
        budget,
        my_rounds,
        my_predictions,
        leaderboard,
    )

    # Get base predictions with uncertainty from the ensemble
    print("\nGenerating ensemble predictions with uncertainty...")
    base_priors = []
    uncertainties = []

    for state in round_details["initial_states"]:
        prior, uncertainty = system.ensemble_model.predict_with_uncertainty(state)
        base_priors.append(prior)
        uncertainties.append(uncertainty)

    avg_uncertainty = np.mean([u.mean() for u in uncertainties])
    print(f"Average epistemic uncertainty: {avg_uncertainty:.4f}")

    # Collect query observations from multiple sources
    query_observations: dict[int, list[dict]] = {i: [] for i in range(round_details["seeds_count"])}
    used_saved_query_dir = None

    # 1. Load from Code_kimi query data (previous runs)
    if not args.skip_saved_queries:
        saved_query_dir = find_latest_saved_query_dir(args.query_data_root, round_details["id"])
        if saved_query_dir is not None:
            _merge_query_observations(query_observations, load_saved_query_observations(saved_query_dir))
            used_saved_query_dir = str(saved_query_dir)
            print(f"Loaded saved queries from: {used_saved_query_dir}")

    # 2. Load from local artifact directory (this script's previous runs)
    local_query_file = snapshot_dir / "query_observations.json"
    _merge_query_observations(query_observations, _load_query_observations_file(local_query_file))

    # 3. Execute additional live queries if requested
    if args.additional_live_queries > 0:
        print(f"\nExecuting {args.additional_live_queries} additional queries...")
        live_observations = execute_adaptive_queries(
            client,
            round_details["id"],
            round_details["initial_states"],
            base_priors,
            uncertainties,
            args.additional_live_queries,
            existing_observations=query_observations,
        )
        _merge_query_observations(query_observations, live_observations)
        print(f"Executed queries: {sum(len(obs) for obs in live_observations.values())}")

    # Generate final predictions with the adaptive system
    print("\nGenerating final predictions with adaptive system...")
    predictions, diagnostics = system.predict_with_queries(
        round_details["initial_states"],
        query_observations,
        return_diagnostics=True,
    )

    print(f"Strategy used: {diagnostics.get('strategy', 'unknown')}")
    if diagnostics.get("fallback_activated"):
        print(f"Fallback activated (blend weight: {diagnostics.get('fallback_blend_weight', 'N/A')})")

    # Save all results
    predictions_path = snapshot_dir / "adaptive_predictions.json"
    save_json(predictions_path, [pred.tolist() for pred in predictions])
    save_json(snapshot_dir / "prediction_summary.json", prediction_summary(predictions))
    save_json(snapshot_dir / "query_observations.json", _json_ready_query_observations(query_observations))
    save_json(
        snapshot_dir / "run_metadata.json",
        {
            "round_id": round_details["id"],
            "round_number": round_details["round_number"],
            "model_source": str(model_dir),
            "used_saved_query_dir": used_saved_query_dir,
            "saved_query_counts": {str(k): len(v) for k, v in query_observations.items()},
            "additional_live_queries": args.additional_live_queries,
            "diagnostics": diagnostics,
            "submitted": args.submit,
        },
    )

    print(f"\nActive round: {round_details['round_number']} ({round_details['id']})")
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']}")
    print(f"Saved snapshot: {snapshot_dir}")
    print(f"Saved predictions: {predictions_path}")

    # Submit if requested
    if args.submit:
        print("\nSubmitting predictions...")
        for seed_index, prediction in enumerate(predictions):
            response = client.submit_prediction(round_details["id"], seed_index, prediction.tolist())
            print(f"  Seed {seed_index}: {response['status']}")
            time.sleep(0.6)
        print("Submission complete!")
    else:
        print("\nDry-run mode: predictions not submitted (use --submit to submit)")


if __name__ == "__main__":
    main()
