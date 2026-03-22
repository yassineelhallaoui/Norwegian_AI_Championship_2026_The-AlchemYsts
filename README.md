# Norwegian AI Championship 2026 - The AlchemYsts

**Team:** The AlchemYsts  
**Captain:** Yassine Elhallaoui  
**Event:** NM i AI 2026 (Norwegian Championship in Artificial Intelligence)

---

This repository contains our complete solutions for the Norwegian AI Championship 2026 (NM i AI 2026), featuring three distinct AI challenges covering prediction systems, computer vision, and autonomous accounting agents.

## Projects Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE ALCHEMYSTS - NM i AI 2026                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🏝️ Asta Astar Island                                               │
│     Adaptive Ensemble Prediction System                             │
│     └── Multi-strategy ensemble with regime detection               │
│                                                                     │
│  🔍 NorgesGruppe Object Detection                                   │
│     Product Detection on Store Shelves                              │
│     └── YOLO11 + ConvNeXt with ArcFace classification               │
│                                                                     │
│  🤖 Tripletex AI Accounting Agent                                   │
│     Autonomous Accounting AI Agent                                  │
│     └── Natural language to API with self-correction                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏝️ Project 1: Asta Astar Island - Adaptive Ensemble System

**Location:** [`Asta Astar Island/`](./Asta%20Astar%20Island/)

### The Challenge

Predict civilization expansion across procedurally generated archipelagos. Given initial maps with settlements, ports, ruins, forests, and mountains, predict the probability distribution of 6 terrain classes after civilization expansion.

### Key Insight

More queries don't always help. Early analysis revealed that sometimes adding queries actually *degraded* scores. This happens because:
1. Single query observations can override well-calibrated priors
2. Equal query allocation wastes budget on low-uncertainty areas
3. Different rounds exhibit different dynamics (expansion vs collapse)

### The Solution: Triple-Layer Adaptive System

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

- **Epistemic Uncertainty Quantification**: Measure ensemble disagreement to guide query allocation
- **Robust Calibration**: Prior strength 15, calibration blend 0.8, max weight ratio 3.0
- **Automatic Fallback Protection**: Detect when queries cause problematic shifts (KL > 0.5)
- **Regime-Aware Expert Weighting**: Different weights for expansion vs collapse rounds

### Quick Start

```bash
cd "Asta Astar Island"

# 1. Install dependencies
pip install -r requirements.txt

# 2. Sync historical data
python sync_historical_data.py

# 3. Train the system
python train_adaptive_system.py

# 4. Dry run on active round
python run_adaptive.py

# 5. Submit with 30 queries (recommended)
python run_adaptive.py --additional-live-queries 30 --submit
```

---

## 🔍 Project 2: NorgesGruppe Object Detection

**Location:** [`NorgesGruppe Object Detection/`](./NorgesGruppe%20Object%20Detection/)

### The Challenge

Detect and classify grocery products on store shelves from images. The competition scoring combines:
- **70% Detection mAP@0.5** - Finding product bounding boxes
- **30% Classification mAP@0.5** - Identifying the product type

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              TWO-STAGE DETECTION PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: Object Detection (YOLO11m)                       │
│  ├── Single-class detection (detect "product" first)       │
│  ├── Input resolution: 960x960                             │
│  ├── Two-stage training (fold → full)                      │
│  └── ONNX export for efficient inference                   │
│                          ↓                                  │
│  Stage 2: Product Classification                           │
│  ├── ConvNeXt-Small backbone (timm)                        │
│  ├── ArcFace loss (scale=30, margin=0.25)                  │
│  ├── Input size: 384x384                                   │
│  └── Reference embedding enhancement (weight=0.4)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **YOLO11m Detector**: Two-stage training (fold-specific → full dataset)
- **ConvNeXt-Small Classifier**: ArcFace loss with 384px input
- **Reference Embeddings**: Similarity scoring with official product photos
- **SAHI Support**: Optional sliced inference for small objects

### Quick Start

```bash
cd "NorgesGruppe Object Detection"

# Inference
python validation/solution_utils.py \
    --input /path/to/test/images \
    --output /path/to/output/predictions.json

# Train detector
python training/train_yolo11_detector.py \
    --fold 0 --model yolo11m.pt --imgsz 960 --export-onnx

# Train classifier
python training/train_improved_classifier.py \
    --fold 0 --architecture convnext_small --imgsz 384 --loss-type arcface
```

### Performance

| Component | Configuration | Performance |
|-----------|--------------|-------------|
| Detector | YOLO11m @ 960px | ~0.83 mAP@0.5 |
| Classifier | ConvNeXt-S @ 384px + ArcFace | ~84% Top-1, ~97% Top-5 |
| Combined | + Reference embeddings | +2% final score |

---

## 🤖 Project 3: Tripletex AI Accounting Agent

**Location:** [`Tripletex AI Accounting Agent/`](./Tripletex%20AI%20Accounting%20Agent/)

### The Challenge

Build an autonomous AI agent that integrates with the Tripletex Accounting API to perform accounting tasks through natural language instructions.

**Supported Languages:** Norwegian, English, Spanish, Portuguese, German, French, Nynorsk

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   main.py       │────▶│   V3Agent        │────▶│  Tripletex API  │
│   (FastAPI)     │     │   (core/agent.py)│     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ LLM Engine   │ │ Autonomous   │ │ Knowledge    │
        │ (Gemini 2.5) │ │ Corrector    │ │ Graph        │
        └──────────────┘ └──────────────┘ └──────────────┘
```

### Core Components

| Module | Purpose |
|--------|---------|
| `core/agent.py` | Main orchestrator with ReAct loop |
| `core/autonomous_corrector.py` | Self-healing error correction |
| `core/error_analyzer.py` | Parses Norwegian API errors |
| `core/schema_intelligence.py` | OpenAPI schema navigation |
| `core/llm_engine.py` | Gemini API integration |
| `core/knowledge_graph.py` | Persistent rule storage |

### Example Tasks

The agent handles tasks like:
- "Create an invoice for Acme AS with 3 line items"
- "Register a supplier invoice from Luna SL for 45,000 NOK"
- "Create employee Hans Müller with email hans@example.com"
- "Record a journal entry for office expenses"

### Autonomous Correction System

When the Tripletex API returns an error:
1. **Analyzes** the error message (Norwegian or English)
2. **Checks schema** for valid field names
3. **Suggests fixes** using pattern matching + LLM
4. **Verifies** the correction with Gemini
5. **Retries** with corrected payload

### Quick Start

```bash
cd "Tripletex AI Accounting Agent"

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_offline.py           # Offline scenario tests
python test_core_scenarios.py    # Core scenario tests
python test_real_world_corrections.py  # Real-world correction tests

# Deploy to Cloud Run
./deploy.sh
```

### API Usage

```bash
curl -X POST https://tripletex-agent-yassy-auto-l3gtp4syqq-ez.a.run.app/solve \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create invoice for customer Acme AS",
    "files": [],
    "tripletex_credentials": {
      "base_url": "https://api.tripletex.io/v2",
      "session_token": "your-session-token"
    }
  }'
```

---

## Repository Structure

```
Norwegian_AI_Championship_2026_The-AlchemYsts/
│
├── 📁 Asta Astar Island/              # Prediction challenge
│   ├── ensemble_model.py              # Multi-model ensemble
│   ├── adaptive_querying.py           # Query planning
│   ├── train_adaptive_system.py       # Training script
│   ├── run_adaptive.py                # Production runner
│   └── README.md                      # Detailed docs
│
├── 📁 NorgesGruppe Object Detection/  # Computer vision challenge
│   ├── training/                      # Training scripts
│   ├── validation/                    # Evaluation code
│   ├── config.json                    # Inference config
│   └── README.md                      # Detailed docs
│
├── 📁 Tripletex AI Accounting Agent/  # Accounting agent challenge
│   ├── core/                          # Core modules
│   ├── main.py                        # FastAPI server
│   ├── deploy.sh                      # Cloud Run deploy
│   └── README.md                      # Detailed docs
│
├── .gitignore                         # Excludes models/weights
└── README.md                          # This file
```

---

## Common Dependencies

Each project has its own `requirements.txt`. Common packages used across projects:

```bash
# Core ML/Data
numpy >= 1.24.0
scipy >= 1.10.0
scikit-learn >= 1.3.0
joblib >= 1.3.0

# Deep Learning (Object Detection)
torch >= 2.0.0
torchvision >= 0.15.0
ultralytics >= 8.0.0
timm >= 0.9.0

# API/HTTP
requests >= 2.31.0
fastapi >= 0.100.0

# Utilities
tqdm >= 4.66.0
pillow >= 9.0.0
```

---

## Design Philosophy

Across all three projects, we follow these principles:

1. **Defensive First**: Prevent harm before seeking gains
2. **Ensemble Diversity**: Multiple models cover different failure modes
3. **Adaptive Strategy**: Adjust to context (regime, input size, error type)
4. **Data Efficiency**: Allocate resources where they provide most value
5. **Robust to Noise**: Don't let single observations override strong priors

---

## Competition Context

**NM i AI 2026** is Norway's premier AI competition, bringing together top talent to solve real-world problems from Norwegian companies.

- **Team:** The AlchemYsts
- **Captain:** Yassine Elhallaoui
- **Events:** 3 parallel code competitions
- **Sponsors:** Asta (Schibsted), NorgesGruppen, Tripletex

---

## License

All projects are open source and available under the **MIT License**.

Feel free to use, modify, and distribute this code for your own projects or competitions. Attribution is appreciated but not required.

---

*Built with precision, patience, and a touch of alchemy.*  
**The AlchemYsts - NM i AI 2026**
