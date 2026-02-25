"""
api
---
FastAPI web application for the Multi-Agent Orchestrator.

Re-exports ``app`` so that ``uvicorn api:app`` continues to work.
"""

from api.app import app

__all__ = ["app"]
