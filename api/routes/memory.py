"""HTTP routes for ChromaDB-backed memory."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.orchestrator import Orchestrator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    n_results: int = Field(default=10, ge=1, le=100)
    filter: dict[str, Any] | None = None


def build_router(orchestrator: Orchestrator) -> APIRouter:
    router = APIRouter()

    def _memory():
        # ``_init_bundle`` is internal; used here intentionally to share state.
        return orchestrator._init_bundle().memory  # noqa: SLF001

    @router.get("/stats")
    def stats() -> dict[str, Any]:
        return _memory().get_collection_stats()

    @router.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            hits = _memory().search(request.query, n_results=request.n_results, filter=request.filter)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"query": request.query, "count": len(hits), "results": hits}

    @router.delete("/{doc_id}")
    def delete(doc_id: str) -> dict[str, Any]:
        try:
            _memory().delete(doc_id)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"deleted": doc_id}

    return router
