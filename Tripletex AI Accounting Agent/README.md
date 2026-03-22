# Tripletex AI Accounting Agent V3

**Norwegian AI Championship 2026 - Code Competition Entry**  
**Team: The AlchemYsts**  
**Captain: Yassine Elhallaoui**

---

## Overview

This is an autonomous AI agent that integrates with the Tripletex Accounting API to perform accounting tasks through natural language instructions. The agent handles multiple languages (Norwegian, English, Spanish, Portuguese, German, French, Nynorsk) and self-corrects API errors using schema intelligence and LLM verification.

Built for the NM i AI 2026 (Norwegian AI Championship) code competition.

---

## What It Does

The agent receives natural language accounting tasks like:

- "Create an invoice for Acme AS with 3 line items"
- "Register a supplier invoice from Luna SL for 45,000 NOK"
- "Create employee Hans Müller with email hans@example.com"
- "Record a journal entry for office expenses"

The agent then:
1. Interprets the task using Gemini 2.5 Pro
2. Maps it to the correct Tripletex API endpoint
3. Builds the proper payload structure
4. Handles API errors autonomously
5. Retries with corrected parameters until success or max attempts

---

## Architecture

### Core Components

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

### Key Modules

**core/agent.py** - Main orchestrator with ReAct loop  
**core/autonomous_corrector.py** - Self-healing error correction  
**core/error_analyzer.py** - Parses Norwegian API errors  
**core/schema_intelligence.py** - OpenAPI schema navigation  
**core/llm_engine.py** - Gemini API integration  
**core/knowledge_graph.py** - Persistent rule storage

---

## Deployment

### Google Cloud Run

The service is deployed on Google Cloud Run (europe-west4):

```bash
# Deploy to Cloud Run
./deploy.sh
```

**Service URL:** `https://tripletex-agent-yassy-auto-l3gtp4syqq-ez.a.run.app`

**Environment Variables:**
- `GEMINI_API_KEY` - Google AI API key
- `TRIPLETEX_BASE_URL` - Tripletex API endpoint
- `AGENT_API_KEY` - Optional API key for service access

---

## Testing

### Run All Tests

```bash
# Offline scenario tests (real log replay)
python test_offline.py

# Core scenario tests
python test_core_scenarios.py

# Real-world correction tests
python test_real_world_corrections.py
```

### Test Coverage

The test suite includes 350+ test cases covering:
- 19 real failure scenarios from production logs
- 16 real-world error correction patterns
- Multi-language task interpretation
- API error recovery flows

---

## Usage

### API Endpoint

**POST** `/solve`

**Request Body:**
```json
{
  "prompt": "Create invoice for customer Acme AS",
  "files": [],
  "tripletex_credentials": {
    "base_url": "https://api.tripletex.io/v2",
    "session_token": "your-session-token"
  }
}
```

**Response:**
```json
{
  "status": "completed",
  "result_status": "success",
  "attempts": 2,
  "elapsed_seconds": 45.3
}
```

### Example Tasks

The agent handles tasks like:

- Creating customers, employees, suppliers
- Generating invoices and orders
- Recording payments
- Managing projects and timesheets
- Processing supplier invoices
- Creating journal entries and vouchers
- Handling travel expenses

---

## Autonomous Correction System

When the Tripletex API returns an error, the agent:

1. **Analyzes** the error message (Norwegian or English)
2. **Checks schema** for valid field names
3. **Suggests fixes** using pattern matching + LLM
4. **Verifies** the correction with Gemini
5. **Retries** with corrected payload

Example error recovery:
```
Error: "department.id | Ugyldig avdeling" (Invalid department)
Fix: Auto-fetch first available department ID
```

---

## Competition Context

**Event:** NM i AI 2026 (Norwegian AI Championship)  
**Track:** Code Competition - Accounting Agent  
**Team:** The AlchemYsts  
**Captain:** Yassine Elhallaoui

This agent was built to compete in the Tripletex Accounting Agent challenge, where participants create AI systems that can perform real accounting tasks through natural language.

---

## License

This project is open source and available under the **MIT License**.
