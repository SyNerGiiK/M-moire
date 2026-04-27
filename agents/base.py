"""Shared agent infrastructure: result type, base class, log writer."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from skills.note_writer import NoteWriter


@dataclass
class AgentResult:
    agent: str
    success: bool = True
    notes_created: int = 0
    notes_updated: int = 0
    topics_processed: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    finished_at: str | None = None

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.success = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "success": self.success,
            "notes_created": self.notes_created,
            "notes_updated": self.notes_updated,
            "topics_processed": list(self.topics_processed),
            "sources_used": list(self.sources_used),
            "errors": list(self.errors),
            "extra": dict(self.extra),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class BaseAgent:
    """Base for every agent — provides logging into ``vault/07_Agents_Log/``."""

    name: str = "base"

    def __init__(self, note_writer: NoteWriter, log_folder: str = "07_Agents_Log") -> None:
        self.note_writer = note_writer
        self.log_folder = log_folder

    def _new_result(self) -> AgentResult:
        return AgentResult(agent=self.name)

    # ------------------------------------------------------------ logging API

    def write_log(
        self,
        result: AgentResult,
        summary: str = "",
        created_notes: list[dict[str, Any]] | None = None,
    ) -> str:
        """Append an Agent Log entry to the vault and return its path."""
        result.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        title = f"{result.agent}_{datetime.now().strftime('%Y-%m-%d_%H%M')}"
        body = _render_log_body(result, summary=summary, created_notes=created_notes or [])
        metadata = {
            "agent": result.agent,
            "run_date": result.finished_at,
            "topics_processed": result.topics_processed,
            "notes_created": result.notes_created,
            "notes_updated": result.notes_updated,
            "sources_used": result.sources_used,
            "type": "log",
            "agent_generated": True,
            "tags": ["agent-log", result.agent.replace("_", "-")],
        }
        path = self.note_writer.create_note(
            folder=self.log_folder,
            title=title,
            content=body,
            metadata=metadata,
        )
        logger.info("[{}] log -> {}", result.agent, Path(path).name)
        return path


def _render_log_body(
    result: AgentResult,
    summary: str,
    created_notes: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Agent Log: {result.agent} — {result.finished_at}")
    lines.append("")
    lines.append("## Summary")
    lines.append(summary or "_no summary provided_")
    lines.append("")
    lines.append("## Stats")
    lines.append(f"- Notes created: **{result.notes_created}**")
    lines.append(f"- Notes updated: **{result.notes_updated}**")
    lines.append(f"- Started: {result.started_at}")
    lines.append(f"- Finished: {result.finished_at}")
    lines.append(f"- Success: {'yes' if result.success else 'no'}")
    if result.topics_processed:
        lines.append("")
        lines.append("## Topics Processed")
        for topic in result.topics_processed:
            lines.append(f"- {topic}")
    if result.sources_used:
        lines.append("")
        lines.append("## Sources Used")
        for src in result.sources_used[:50]:
            lines.append(f"- {src}")
    if created_notes:
        lines.append("")
        lines.append("## Notes Created")
        for note in created_notes[:100]:
            title = note.get("title") or "untitled"
            path = note.get("path") or ""
            lines.append(f"- [[{Path(path).stem}]] — {title}")
    if result.errors:
        lines.append("")
        lines.append("## Errors")
        for err in result.errors:
            lines.append(f"- `{err}`")
    if result.extra:
        lines.append("")
        lines.append("## Extra")
        lines.append("```yaml")
        lines.append(yaml.safe_dump(result.extra, allow_unicode=True, sort_keys=False).strip())
        lines.append("```")
    return "\n".join(lines) + "\n"
