"""
Tripletex AI Accounting Agent - Main Entry Point

FastAPI application that exposes the accounting agent as a REST API.
Deployed on Google Cloud Run for the NM i AI 2026 competition.

Team: The AlchemYsts
Captain: Yassine Elhallaoui
Competition: NM i AI 2026 (Norwegian AI Championship)

Endpoints:
- GET  /         - Health check
- GET  /health   - Detailed health with knowledge graph stats  
- POST /solve    - Main endpoint for accounting tasks
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from core.agent import V3Agent
from schemas import SolveRequest
from tripletex_client import TripletexClient

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Also set level for our modules
logging.getLogger("core.agent").setLevel(logging.INFO)
logging.getLogger("core.llm_engine").setLevel(logging.INFO)
logging.getLogger("core.knowledge_graph").setLevel(logging.INFO)
logging.getLogger("core.openapi_context").setLevel(logging.INFO)

app = FastAPI(title="Tripletex AI Accounting Agent V3", version="1.1.0")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")

@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "tripletex-code-yassy-v3", "version": "1.1.0"}

@app.get("/health")
async def detailed_health() -> dict[str, Any]:
    """Detailed health check including knowledge graph stats."""
    from core.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    stats = kg.get_stats()
    return {
        "status": "ok",
        "version": "1.1.0",
        "knowledge_graph": stats
    }

@app.post("/solve")
async def solve(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
    started_at = time.time()
    request_id = f"{int(started_at * 1000)}"
    
    logger.info(f"[{request_id}] ====== NEW REQUEST ======")

    if AGENT_API_KEY:
        if not authorization or authorization.replace("Bearer ", "", 1) != AGENT_API_KEY:
            logger.warning(f"[{request_id}] Invalid or missing API key")
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        payload = SolveRequest.model_validate(await request.json())
        logger.info(f"[{request_id}] Task: {payload.prompt[:100]}...")
    except Exception as exc:
        logger.error(f"[{request_id}] Invalid request body: {exc}")
        raise HTTPException(status_code=400, detail="Invalid request body")

    client = TripletexClient(
        base_url=payload.tripletex_credentials.base_url,
        session_token=payload.tripletex_credentials.session_token,
    )
    agent = V3Agent(client)

    try:
        result = agent.process_task(payload.prompt)
        elapsed = time.time() - started_at
        
        status = result.get("status", "unknown")
        attempts = result.get("attempts", 0)
        history_count = len(result.get("history", []))
        
        logger.info(
            f"[{request_id}] Completed in {elapsed:.1f}s status={status} attempts={attempts} api_calls={history_count}"
        )
        
        if status == "failed":
            error_info = result.get("error", {})
            if isinstance(error_info, dict):
                logger.error(f"[{request_id}] Task failed: {error_info.get('message', 'Unknown error')}")
            else:
                logger.error(f"[{request_id}] Task failed: {error_info}")
        
        # Return success status based on actual result
        # Note: The competition platform might check the status field
        return JSONResponse({
            "status": "completed" if status == "success" else "failed",
            "result_status": status,
            "attempts": attempts,
            "elapsed_seconds": elapsed
        }, status_code=200)

    except Exception as exc:
        elapsed = time.time() - started_at
        logger.error(
            f"[{request_id}] Task crashed after {elapsed:.1f}s: {exc}\nTraceback:\n{traceback.format_exc()}"
        )
        return JSONResponse({
            "status": "failed",
            "error": str(exc),
            "elapsed_seconds": elapsed
        }, status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    
    logger.info(f"Starting Tripletex AI Accounting Agent V3 on port {port}")
    logger.info(f"Gemini Model: {os.environ.get('GEMINI_MODEL', 'default')}")
    logger.info(f"API Key configured: {bool(os.environ.get('GEMINI_API_KEY'))}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
