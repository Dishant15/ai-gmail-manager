"""
app/api.py - FastAPI application with all REST endpoints.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.agent import (
    get_reply_logs,
    load_agent_config,
    run_polling_cycle,
    update_agent_config,
)
from app.config import (
    AgentConfig,
    AgentConfigUpdate,
    DocumentInfo,
    PollingStatus,
    ReplyLog,
    StatusResponse,
    settings,
)
from app.rag_engine import delete_document, get_document_count, ingest_pdf, list_documents

# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Email Agent",
    description="Automatically replies to Gmail using a local Mistral LLM with RAG capabilities.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Background scheduler
# ──────────────────────────────────────────────────────────────

scheduler = BackgroundScheduler()
_scheduler_started = False


def _start_scheduler():
    global _scheduler_started
    if not _scheduler_started:
        scheduler.add_job(
            run_polling_cycle,
            "interval",
            seconds=settings.email_poll_interval,
            id="email_poller",
            replace_existing=True,
            max_instances=1,
        )
        scheduler.start()
        _scheduler_started = True
        logger.info(
            "Email poller scheduled every {} seconds.", settings.email_poll_interval
        )


@app.on_event("startup")
async def startup_event():
    _start_scheduler()
    logger.info("RAG Email Agent API started.")


@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped.")


# ──────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {"message": "RAG Email Agent is running", "docs": "/docs"}


@app.get("/health", response_model=StatusResponse, tags=["health"])
async def health():
    return StatusResponse(status="ok", message="Service is healthy")


# ──────────────────────────────────────────────────────────────
# Polling control
# ──────────────────────────────────────────────────────────────

@app.get("/polling/status", response_model=PollingStatus, tags=["polling"])
async def polling_status():
    config = load_agent_config()
    # Check the job's own state, not the scheduler's — the scheduler stays
    # alive even when the job is paused, so scheduler.running is always True.
    job = scheduler.get_job("email_poller")
    job_running = job is not None and job.next_run_time is not None
    return PollingStatus(
        running=job_running,
        interval_seconds=settings.email_poll_interval,
        gmail_address=settings.gmail_address,
        auto_reply_enabled=config.auto_reply_enabled,
        documents_indexed=get_document_count(),
    )


@app.post("/polling/trigger", response_model=List[ReplyLog], tags=["polling"])
async def trigger_poll():
    """Manually trigger an email polling cycle."""
    results = run_polling_cycle()
    return results


@app.post("/polling/start", response_model=StatusResponse, tags=["polling"])
async def start_polling():
    _start_scheduler()
    job = scheduler.get_job("email_poller")
    if job:
        # Job exists but was paused — resume it
        job.resume()
    else:
        # Job was never created (shouldn't normally happen) — add it fresh
        scheduler.add_job(
            run_polling_cycle,
            "interval",
            seconds=settings.email_poll_interval,
            id="email_poller",
            replace_existing=True,
            max_instances=1,
        )
    return StatusResponse(status="ok", message="Polling started")


@app.post("/polling/stop", response_model=StatusResponse, tags=["polling"])
async def stop_polling():
    job = scheduler.get_job("email_poller")
    if job:
        job.pause()
        return StatusResponse(status="ok", message="Polling paused")
    return StatusResponse(status="warning", message="No active polling job found")


# ──────────────────────────────────────────────────────────────
# Agent configuration
# ──────────────────────────────────────────────────────────────

@app.get("/agent/config", response_model=AgentConfig, tags=["agent"])
async def get_agent_config():
    return load_agent_config()


@app.put("/agent/config", response_model=AgentConfig, tags=["agent"])
async def set_agent_config(update: AgentConfigUpdate):
    """Update one or more agent configuration fields."""
    config = update_agent_config(**update.model_dump(exclude_none=True))
    return config


@app.post("/agent/config/reset", response_model=AgentConfig, tags=["agent"])
async def reset_agent_config():
    """Reset agent configuration to defaults."""
    from app.agent import save_agent_config
    default = AgentConfig()
    save_agent_config(default)
    return default


# ──────────────────────────────────────────────────────────────
# Knowledge base (PDF management)
# ──────────────────────────────────────────────────────────────

@app.get("/knowledge/documents", response_model=List[DocumentInfo], tags=["knowledge"])
async def get_documents():
    """List all indexed PDF documents."""
    return list_documents()


@app.post(
    "/knowledge/upload",
    response_model=DocumentInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["knowledge"],
)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF file into the knowledge base."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    upload_path = Path(settings.upload_dir) / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        doc_info = ingest_pdf(upload_path)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index PDF: {exc}",
        )

    return doc_info


@app.delete(
    "/knowledge/documents/{filename}",
    response_model=StatusResponse,
    tags=["knowledge"],
)
async def remove_document(filename: str):
    """Remove a document from the knowledge base."""
    success = delete_document(filename)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found.",
        )

    # Also remove the file from disk
    file_path = Path(settings.upload_dir) / filename
    file_path.unlink(missing_ok=True)

    return StatusResponse(status="ok", message=f"Document '{filename}' deleted.")


# ──────────────────────────────────────────────────────────────
# Reply logs
# ──────────────────────────────────────────────────────────────

@app.get("/logs", response_model=List[ReplyLog], tags=["logs"])
async def get_logs(limit: int = 50):
    """Retrieve recent reply logs."""
    return get_reply_logs(limit=min(limit, 200))
