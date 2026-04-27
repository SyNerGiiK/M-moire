"""FastAPI server exposing the orchestrator over HTTP.

Endpoints:
    GET  /health
    GET  /status
    POST /agents/run                  body: {"agent": "...", "subcommand": "..."}
    POST /agents/cycle                run a full orchestration cycle
    POST /memory/search               body: {"query": "...", "n_results": 10}
    GET  /memory/stats
"""
from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI

from agents.orchestrator import Orchestrator
from api.routes.agents import build_router as agents_router
from api.routes.memory import build_router as memory_router
from config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    orchestrator = Orchestrator(settings=settings)

    app = FastAPI(
        title="Second Brain",
        version="0.1.0",
        description="Local-first AI Second Brain — agents over an Obsidian vault.",
    )
    app.state.orchestrator = orchestrator
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "second-brain"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        return orchestrator.get_status()

    app.include_router(agents_router(orchestrator), prefix="/agents", tags=["agents"])
    app.include_router(memory_router(orchestrator), prefix="/memory", tags=["memory"])
    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="second-brain-api")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    uvicorn.run("api.main:create_app", host=host, port=port, factory=True, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
