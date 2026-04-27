"""HTTP routes that trigger agents."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.orchestrator import Orchestrator


class AgentRequest(BaseModel):
    agent: str = Field(..., description="Agent name: researcher | arxiv | curator | tagger")
    subcommand: str | None = Field(default=None, description="Optional subcommand for curator/tagger")
    topics: list[str] | None = Field(default=None, description="Optional topic override")
    days_back: int | None = Field(default=None, description="arxiv: lookback window")


def build_router(orchestrator: Orchestrator) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def list_agents() -> dict[str, Any]:
        return {
            "agents": ["researcher", "arxiv", "curator", "tagger"],
            "curator_subcommands": [
                "process_inbox",
                "deduplicate_vault",
                "generate_connections",
                "update_mocs",
            ],
            "tagger_subcommands": ["tag_all_untagged", "normalize_tags"],
        }

    @router.post("/run")
    def run(request: AgentRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if request.subcommand:
            kwargs["subcommand"] = request.subcommand
        if request.topics is not None:
            kwargs["topics"] = request.topics
        if request.days_back is not None:
            kwargs["days_back"] = request.days_back
        try:
            result = orchestrator.run_agent(request.agent, **kwargs)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result.as_dict()

    @router.post("/cycle")
    def cycle() -> dict[str, Any]:
        results = orchestrator.run_full_cycle()
        return {key: r.as_dict() for key, r in results.items()}

    return router
