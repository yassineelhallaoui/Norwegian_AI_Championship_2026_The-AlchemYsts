"""Comprehensive offline evaluation with leave-one-round-out validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from adaptive_ensemble import AdaptiveConfig, AdaptiveEnsembleSystem
from adaptive_querying import (
    allocate_query_budget_by_uncertainty,
    select_queries_with_thompson_sampling,
)
from data_io import load_historical_rounds


def compute_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute score based on entropy-weighted KL divergence (simplified version)."""
    prediction = np.asarray(prediction, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    # Ensure valid probabilities
    prediction = np.clip(prediction, 1e-6, 1.0)
    ground_truth = np.clip(ground_truth, 1e-6, 1.0)

    # KL divergence per cell
    kl = np.sum(ground_truth * np.log(ground_truth / prediction), axis=-1)

    # Entropy per cell
    entropy = -np.sum(ground_truth * np.log(ground_truth), axis=-1)

    # Weighted KL
    weighted_kl = (entropy * kl).sum() / max(entropy.sum(), 1e-8)

    # Score formula
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return float(score)


def simulate_queries_for_round(
    initial_states: list[dict],
    ground_truths: list[np.ndarray],
    priors: list[np.ndarray],
    uncertainties: list[np.ndarray],
    num_queries: int,
) -> dict[int, list[dict]]:
    """Simulate query observations from ground truth."""
    allocation = allocate_query_budget_by_uncertainty(priors, uncertainties, num_queries)
    observations: dict[int, list[dict]] = {}

    for seed_index, (state, gt, prior, uncertainty, n_queries) in enumerate(
        zip(initial_states, ground_truths, priors, uncertainties, allocation)
    ):
        planned_queries = select_queries_with_thompson_sampling(
            state,
            prior,
            uncertainty,
            n_queries,
        )

        seed_entries: list[dict] = []
        for query in planned_queries:
            x = query["viewport_x"]
            y = query["viewport_y"]
            w = query["viewport_w"]
            h = query["viewport_h"]

            # Sample from ground truth distribution
            gt_region = gt[y : y + h, x : x + w]
            class_grid = np.zeros((h, w), dtype=int)

            for i in range(h):
                for j in range(w):
                    # Sample class according to ground truth probabilities
                    class_grid[i, j] = np.random.choice(6, p=gt_region[i, j])

            seed_entries.append(
                {
                    "seed_index": seed_index,
                    "viewport_x": x,
                    "viewport_y": y,
                    "viewport_w": w,
                    "viewport_h": h,
                    "class_grid": class_grid,
                }
            )

        observations[seed_index] = seed_entries

    return observations


def evaluate_leave_one_out(
    historical_rounds: list[dict],
    query_budgets: list[int],
    num_repeats: int = 10,
    config: AdaptiveConfig | None = None,
) -> dict:
    """Perform leave-one-round-out evaluation."""
    results = []

    for holdout_idx in range(len(historical_rounds)):
        print(f"\nEvaluating holdout round {holdout_idx + 1}/{len(historical_rounds)}")

        # Split data
        train_rounds = [r for i, r in enumerate(historical_rounds) if i != holdout_idx]
        test_round = historical_rounds[holdout_idx]

        # Train system
        system = AdaptiveEnsembleSystem(config=config).fit(train_rounds)

        # Evaluate on test round
        initial_states = test_round["initial_states"]
        ground_truths = [np.asarray(gt, dtype=float) for gt in test_round["ground_truths"]]

        # Get base predictions with uncertainty
        base_priors = []
        uncertainties = []
        for state in initial_states:
            prior, uncertainty = system.ensemble_model.predict_with_uncertainty(state)
            base_priors.append(prior)
            uncertainties.append(uncertainty)

        # Compute zero-query scores
        zero_query_scores = [compute_score(pred, gt) for pred, gt in zip(base_priors, ground_truths)]

        round_result = {
            "round_name": test_round.get("round_name", f"round_{holdout_idx}"),
            "round_number": test_round.get("round_number", holdout_idx),
            "zero_query_average": float(np.mean(zero_query_scores)),
            "zero_query_scores": [float(s) for s in zero_query_scores],
            "query_budgets": {},
        }

        # Evaluate each query budget
        for budget in query_budgets:
            if budget == 0:
                continue

            budget_scores = []

            for repeat in tqdm(range(num_repeats), desc=f"Budget {budget}", leave=False):
                # Simulate queries
                np.random.seed(42 + repeat)
                observations = simulate_queries_for_round(
                    initial_states,
                    ground_truths,
                    base_priors,
                    uncertainties,
                    budget,
                )

                # Get predictions with queries
                predictions, diagnostics = system.predict_with_queries(
                    initial_states,
                    observations,
                    return_diagnostics=True,
                )

                # Compute scores
                scores = [compute_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
                budget_scores.append(
                    {
                        "average_score": float(np.mean(scores)),
                        "seed_scores": [float(s) for s in scores],
                        "strategy": diagnostics.get("strategy", "unknown"),
                    }
                )

            round_result["query_budgets"][str(budget)] = {
                "average_score": float(np.mean([r["average_score"] for r in budget_scores])),
                "std_score": float(np.std([r["average_score"] for r in budget_scores])),
                "best_score": float(np.max([r["average_score"] for r in budget_scores])),
                "runs": budget_scores,
            }

        results.append(round_result)

    return {
        "evaluation_type": "leave_one_round_out",
        "query_budgets": query_budgets,
        "num_repeats_per_budget": num_repeats,
        "rounds": results,
        "overall_zero_query_average": float(np.mean([r["zero_query_average"] for r in results])),
        "overall_query_averages": {
            str(budget): float(np.mean([r["query_budgets"][str(budget)]["average_score"] for r in results if str(budget) in r["query_budgets"]]))
            for budget in query_budgets
            if budget > 0
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adaptive ensemble system.")
    parser.add_argument(
        "--historical-dir",
        default=str(Path(__file__).resolve().parent / "historical_data"),
        help="Directory with historical rounds and ground truth.",
    )
    parser.add_argument(
        "--output-file",
        default=str(Path(__file__).resolve().parent / "evaluation_results.json"),
        help="Output file for results.",
    )
    parser.add_argument(
        "--query-budgets",
        nargs="+",
        type=int,
        default=[0, 10, 20, 30, 50],
        help="Query budgets to evaluate.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=20,
        help="Number of repeats per query budget.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer repeats and budgets.",
    )

    args = parser.parse_args()

    if args.fast:
        query_budgets = [0, 20]
        num_repeats = 5
    else:
        query_budgets = args.query_budgets
        num_repeats = args.num_repeats

    print("Loading historical data...")
    historical_rounds = load_historical_rounds(args.historical_dir)
    print(f"Loaded {len(historical_rounds)} historical rounds")

    config = AdaptiveConfig(
        use_ensemble=True,
        use_regime_detection=True,
        enable_fallback=True,
    )

    print(f"\nRunning evaluation with {num_repeats} repeats per budget...")
    print(f"Query budgets: {query_budgets}")

    results = evaluate_leave_one_out(historical_rounds, query_budgets, num_repeats, config)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\nSummary:")
    print(f"  Zero-query average: {results['overall_zero_query_average']:.2f}")
    for budget, score in results["overall_query_averages"].items():
        print(f"  Budget {budget}: {score:.2f}")


if __name__ == "__main__":
    main()
