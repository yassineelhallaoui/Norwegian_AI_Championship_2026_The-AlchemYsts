"""Sync completed-round historical data and ground truth for Code_YassY.

This module fetches historical rounds from the Astar Island API and stores them
locally for training. It preserves ground truth data so we can learn from past
rounds and improve future predictions.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from api_client import AstarIslandClient
from data_io import save_json


def _copy_local_seed_archive(seed_dir: Path, output_dir: Path) -> None:
    """Copy any existing local seed data to preserve older rounds."""
    if not seed_dir.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for round_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir() and path.name.startswith("round_")):
        destination = output_dir / round_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for source_file in round_dir.iterdir():
            if source_file.is_file():
                target = destination / source_file.name
                if not target.exists():
                    shutil.copy2(source_file, target)


def sync_historical_data(
    token_file: str | Path,
    output_dir: str | Path,
    local_seed_dir: str | Path | None = None,
) -> dict:
    """Fetch completed rounds from API and save locally.
    
    For each completed round, we save:
    - summary.json: Round metadata
    - initial_states.json: Starting map configurations
    - seed_X_ground_truth.json: Final terrain for each seed (for training)
    - my_predictions.json: Our past submissions (for analysis)
    
    Returns a manifest tracking what was synced.
    """
    client = AstarIslandClient.from_token_file(token_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, copy any local data to preserve older rounds not in API
    if local_seed_dir is not None:
        _copy_local_seed_archive(Path(local_seed_dir), output_dir)

    rounds = sorted(client.get_rounds(), key=lambda item: item["round_number"])
    manifest = {
        "rounds_seen": [],
        "rounds_synced": [],
        "rounds_pending_ground_truth": [],
    }

    for round_info in rounds:
        manifest["rounds_seen"].append(
            {
                "round_number": round_info["round_number"],
                "round_id": round_info["id"],
                "status": round_info["status"],
            }
        )

        # Only sync completed/scoring rounds (have ground truth)
        if round_info["status"] not in {"completed", "scoring"}:
            manifest["rounds_pending_ground_truth"].append(
                {
                    "round_number": round_info["round_number"],
                    "round_id": round_info["id"],
                    "status": round_info["status"],
                }
            )
            continue

        round_details = client.get_round_details(round_info["id"])
        round_dir = output_dir / f"round_{round_info['round_number']}_{round_info['id']}"
        round_dir.mkdir(parents=True, exist_ok=True)

        save_json(round_dir / "summary.json", round_details)
        save_json(round_dir / "initial_states.json", round_details["initial_states"])

        # Save our past predictions (if available) for analysis
        try:
            my_predictions = client.get_my_predictions(round_info["id"])
            save_json(round_dir / "my_predictions.json", my_predictions)
        except Exception as exc:
            save_json(round_dir / "my_predictions_error.json", {"error": str(exc)})

        # Save ground truth for each seed (the "labels" for training)
        seed_results = []
        for seed_index in range(round_details["seeds_count"]):
            seed_record = {
                "seed_index": seed_index,
                "analysis_saved": False,
            }
            try:
                analysis = client.get_analysis(round_info["id"], seed_index)
                payload = {
                    "score": analysis.get("score"),
                    "ground_truth": analysis.get("ground_truth"),
                    "prediction": analysis.get("prediction"),
                    "initial_grid": analysis.get("initial_grid"),
                    "width": analysis.get("width"),
                    "height": analysis.get("height"),
                }
                save_json(round_dir / f"seed_{seed_index}_ground_truth.json", payload)
                seed_record["analysis_saved"] = True
                seed_record["score"] = analysis.get("score")
            except Exception as exc:
                seed_record["error"] = str(exc)
            seed_results.append(seed_record)

        manifest["rounds_synced"].append(
            {
                "round_number": round_info["round_number"],
                "round_id": round_info["id"],
                "status": round_info["status"],
                "output_dir": str(round_dir),
                "seed_results": seed_results,
            }
        )

    save_json(output_dir / "historical_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync completed Astar Island rounds into Code_YassY/historical_data.")
    parser.add_argument(
        "--token-file",
        default=str(Path(__file__).resolve().parent.parent / "Code" / ".token"),
        help="Path to the JWT token file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "historical_data"),
        help="Where to store synchronized historical rounds.",
    )
    parser.add_argument(
        "--local-seed-dir",
        default=str(Path(__file__).resolve().parent.parent / "Code_kimi" / "historical_data"),
        help="Optional local archive to copy first so older data is preserved.",
    )
    args = parser.parse_args()

    manifest = sync_historical_data(args.token_file, args.output_dir, args.local_seed_dir)
    print("Saved manifest to", Path(args.output_dir) / "historical_manifest.json")
    print("Completed/scoring rounds synced:", len(manifest["rounds_synced"]))
    print("Pending rounds:", len(manifest["rounds_pending_ground_truth"]))
    for round_item in manifest["rounds_synced"]:
        saved = sum(1 for seed_item in round_item["seed_results"] if seed_item["analysis_saved"])
        print(
            "round",
            round_item["round_number"],
            "status",
            round_item["status"],
            "analysis_saved",
            f"{saved}/{len(round_item['seed_results'])}",
        )
    for round_item in manifest["rounds_pending_ground_truth"]:
        print("pending", round_item["round_number"], round_item["status"])


if __name__ == "__main__":
    main()
