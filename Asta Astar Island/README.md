# Code_YassY: Adaptive Ensemble System for Astar Island

**Developed by Yassine Elhallaoui, Captain of "The AlchemYsts" - NM i AI 2026**

An intelligent multi-strategy approach for the Astar Island prediction challenge, combining ensemble learning, regime detection, and robust calibration to optimize query usage and maximize prediction accuracy.

---

## The Problem

In the Astar Island competition, we predict how civilization expands across a procedurally generated archipelago. Each round provides initial maps with settlements, ports, ruins, forests, and mountains. The challenge: predict the probability distribution of 6 terrain classes (empty, settlement, port, ruin, forest, mountain) for each cell after civilization expansion.

**The Core Insight**: More queries don't always help. Early analysis revealed that sometimes adding queries actually *degraded* scores:
- Round 4: 87.53 (zero-query) → 86.51 (with 6 queries) ❌ Worse!
- Round 5: 73.65 (zero-query) → 78.25 (submitted) ✅ Better

This happens because:
1. Single query observations can override well-calibrated priors
2. Equal query allocation wastes budget on low-uncertainty areas
3. Different rounds exhibit different dynamics (expansion vs collapse)
4. Fixed temperature settings help some rounds but hurt others

---

## The Solution: Triple-Layer Adaptive System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE ENSEMBLE SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Ensemble of Diverse Models                            │
│  ├── Conservative Blend (Ridge + ExtraTrees, 70% tree)         │
│  ├── Flattened Blend (same, α=1.10 temperature)                │
│  ├── Tree-Heavy Blend (85% tree weight)                        │
│  ├── Deep ExtraTrees (pure trees, depth limit)                 │
│  ├── Random Forest (for diversity)                             │
│  └── Gradient Boosting (additional perspective)                │
│                          ↓                                      │
│  Layer 2: Uncertainty-Aware Querying                            │
│  ├── Epistemic uncertainty from model disagreement             │
│  ├── Entropy-based data uncertainty                            │
│  ├── Thompson sampling for exploration/exploitation            │
│  └── Strategic position selection (settlements, coasts, gaps)  │
│                          ↓                                      │
│  Layer 3: Adaptive Prediction Strategy                          │
│  ├── High Confidence → Regime Detection (round-specific)       │
│  ├── Low Confidence → Robust Calibration (conservative)        │
│  └── Fallback Protection → Blend if queries cause divergence   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Innovations

**1. Epistemic Uncertainty Quantification**

Instead of treating all predictions equally, we measure how much the ensemble models disagree. High disagreement signals uncertainty about specific regions.

```python
# Stack predictions from all models
predictions_stack = np.stack([model_i.predict(state) for model_i in ensemble])
uncertainty = np.var(predictions_stack, axis=0).sum(axis=-1)
```

This uncertainty drives:
- Query budget allocation (more queries for uncertain seeds)
- Query position selection (prioritize uncertain regions)
- Strategy selection (use regime detection when confident)

**2. Robust Calibration**

When updating predictions with query observations, we use stronger safeguards:
- **Prior strength 15** (vs 12): Requires more evidence to override the prior
- **Calibration blend 0.8**: Only partially trust class weight adjustments
- **Max weight ratio 3.0**: Caps extreme adjustments to prevent over-correction

**3. Automatic Fallback Protection**

The system detects when queries cause problematic prediction shifts:
- Compare query-updated predictions with ensemble-only
- If KL divergence > 0.5, blend back toward ensemble (70% ensemble, 30% query-updated)
- This prevents the "queries hurt score" scenario

**4. Regime-Aware Expert Weighting**

Different rounds behave differently (expansion-heavy vs collapse-heavy). We build round-specific calibration experts and weight them by how well they explain observed queries.

---

## Synchronizing Historical Data

Before training, you need historical round data with ground truth. The sync tool fetches completed rounds from the API:

```bash
python sync_historical_data.py
```

This script:
1. Connects to the Astar Island API
2. Downloads all completed/scoring rounds
3. Saves initial states, ground truth, and your past predictions
4. Maintains a manifest of synced rounds

**Key options:**
```bash
# Use custom token file
python sync_historical_data.py --token-file /path/to/.token

# Custom output directory
python sync_historical_data.py --output-dir ./my_historical_data
```

Synced data is organized as:
```
historical_data/
├── round_1_<uuid>/
│   ├── summary.json
│   ├── initial_states.json
│   ├── seed_0_ground_truth.json
│   └── ...
├── round_2_<uuid>/
└── historical_manifest.json
```

---

## Training the System

Train all ensemble models and the regime detector:

```bash
python train_adaptive_system.py
```

This creates:
- 6 diverse base models in `models/ensemble_model.joblib`
- Regime detector in `models/regime_detector.joblib`
- Training metadata in `models/training_metadata.json`

The system uses **leave-one-round-out validation**: for each historical round, it trains on all others and validates on the held-out round. This gives realistic performance estimates and optimal model weighting.

**Training options:**
```bash
# Disable regime detection (ensemble only)
python train_adaptive_system.py --no-regime-detection

# Adjust regime sensitivity
python train_adaptive_system.py --regime-likelihood-scale 50.0
```

---

## Running on Live Rounds

### Dry Run (No Submission)

Test the system without submitting:

```bash
python run_adaptive.py
```

### With Queries (Recommended)

```bash
# Conservative: 20 queries (safe, proven)
python run_adaptive.py --additional-live-queries 20 --submit

# Balanced: 30 queries (optimal based on historical analysis)
python run_adaptive.py --additional-live-queries 30 --submit

# Aggressive: 40 queries (use more of the 50 budget)
python run_adaptive.py --additional-live-queries 40 --submit
```

### How Query Synchronization Works

The system intelligently manages queries across runs:

1. **Saved queries from Code_kimi**: Automatically loads previously saved queries from `../Code_kimi/query_data/`
2. **Local artifact queries**: Loads queries from previous runs in `artifacts/round_X/`
3. **Additional live queries**: Executes new queries via API if requested
4. **Deduplication**: Merges all sources without duplicates

This means you can:
- Run once with 20 queries
- Later run with `--additional-live-queries 10` to add 10 more
- The system combines all 30 for the final prediction

### Key Options

```bash
# Skip saved queries (start fresh)
python run_adaptive.py --skip-saved-queries --additional-live-queries 30 --submit

# Use custom model directory
python run_adaptive.py --model-dir ./my_models --submit

# Custom token location
python run_adaptive.py --token-file /path/to/.token --submit
```

---

## Query Budget Analysis

From historical data analysis:
- 0→5 queries: **+11.69 points** (huge gain!)
- 5→10: +0.34 points
- 10→20: +0.36 points
- 20→30: -0.01 points (diminishing returns)
- 30→50: +0.06 points

**Recommendation**: Start with **30 queries**. The uncertainty-aware allocation extracts more value than naive equal-split approaches.

---

## File Structure

```
Code_YassY/
├── Core System
│   ├── ensemble_model.py        # Multi-model ensemble with validation
│   ├── adaptive_querying.py     # Query planning and robust calibration
│   └── adaptive_ensemble.py     # Unified adaptive system
│
├── Utilities
│   ├── features.py              # Cell-wise feature extraction
│   ├── scoring.py               # KL divergence and entropy
│   ├── api_client.py            # API wrapper
│   ├── data_io.py               # Data loading/saving
│   └── constants.py             # Shared constants
│
├── Scripts
│   ├── sync_historical_data.py  # Fetch historical data
│   ├── train_adaptive_system.py # Train all models
│   ├── evaluate_adaptive.py     # Offline evaluation
│   └── run_adaptive.py          # Production runner
│
├── Data & Models
│   ├── historical_data/         # Synced rounds (auto-created)
│   ├── models/                  # Trained models (auto-created)
│   └── artifacts/               # Run snapshots (auto-created)
│
└── requirements.txt             # Python dependencies
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0
- requests >= 2.31.0
- tqdm >= 4.66.0

---

## Expected Performance

Based on offline evaluation:

1. **Never significantly degrade from zero-query baseline**
   - Fallback protection prevents query-induced score drops

2. **Improve by 3-8 points on average with queries**
   - Larger gains when queries match ground truth well
   - Smaller but still positive gains when queries are noisy

3. **Adapt to round regime**
   - Expansion-heavy rounds: Weights expansion experts higher
   - Collapse-heavy rounds: Weights collapse experts higher

---

## Competition Context

**NM i AI 2026** - Norwegian Championship in Artificial Intelligence

**Team**: The AlchemYsts  
**Captain**: Yassine Elhallaoui

This solution represents an evolution from earlier approaches, incorporating insights from:
- Base architecture: Code_gpt (Ridge + ExtraTrees blend)
- Regime detection: Code_querypattern (QueryPatternMixture)
- Ensemble + Adaptive strategies: Code_YassY (this implementation)

---

## Design Philosophy

1. **Defensive First**: Prevent harm before seeking gains (fallback protection)
2. **Ensemble Diversity**: Multiple models cover different failure modes
3. **Adaptive Strategy**: Use regime detection when confident, robust calibration when uncertain
4. **Data Efficiency**: Allocate queries where they provide most information
5. **Robust to Noise**: Don't let single observations override strong priors

---

## Quick Start Checklist

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Sync historical data
python sync_historical_data.py

# 3. Train the system
python train_adaptive_system.py

# 4. Dry run on active round
python run_adaptive.py

# 5. Submit with 30 queries
python run_adaptive.py --additional-live-queries 30 --submit
```

---

## License

This project is open source and available under the **MIT License**.

Feel free to use, modify, and distribute this code for your own projects or competitions. Attribution is appreciated but not required.

---

*Built with precision, patience, and a touch of alchemy.*  
*The AlchemYsts - NM i AI 2026*
