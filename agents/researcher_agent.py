"""Autonomous web research agent.

Pipeline per topic:
    1. Web-search for sources
    2. Fetch & extract clean text
    3. Summarize each source into atomic-note format
    4. Skip near-duplicates against ChromaDB
    5. Generate tags + connections to existing vault notes
    6. Write to ``vault/00_Inbox/``
    7. Store in ChromaDB with metadata
    8. Append a structured log entry to ``vault/07_Agents_Log/``
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from agents.base import AgentResult, BaseAgent
from skills.note_writer import NoteWriter
from skills.summarizer import Summarizer
from skills.vector_memory import VectorMemory
from skills.web_search import WebSearch


_DEPTH_TO_SOURCES = {"quick": 3, "medium": 10, "deep": 30}


class ResearcherAgent(BaseAgent):
    name = "researcher"

    def __init__(
        self,
        note_writer: NoteWriter,
        memory: VectorMemory,
        summarizer: Summarizer,
        web_search: WebSearch | None = None,
        config: dict[str, Any] | None = None,
        topics_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(note_writer)
        self.memory = memory
        self.summarizer = summarizer
        self.web_search = web_search or WebSearch()
        self.config = config or {}
        self.topics_config = topics_config or {}
        self.dedup_threshold = float(
            (self.topics_config.get("settings") or {}).get("dedup_threshold", 0.90)
        )
        self.max_notes_per_run = int(
            (self.topics_config.get("settings") or {}).get("max_notes_per_run", 50)
        )

    # --------------------------------------------------------------- API

    def research_topic(self, topic: str, depth: str = "medium") -> list[dict[str, Any]]:
        max_sources = _DEPTH_TO_SOURCES.get(depth, 10)
        max_sources = int(self.config.get("max_sources_per_topic", max_sources))
        logger.info("[researcher] topic={} depth={} max_sources={}", topic, depth, max_sources)
        hits = self.web_search.search(topic, max_results=max_sources)
        out: list[dict[str, Any]] = []
        for hit in hits:
            url = hit.get("url") or ""
            if not url:
                continue
            page = self.web_search.fetch_page(url)
            if not page.get("content"):
                continue
            out.append(
                {
                    "topic": topic,
                    "title": page.get("title") or hit.get("title") or url,
                    "url": url,
                    "date": page.get("date"),
                    "content": page["content"],
                    "snippet": hit.get("snippet", ""),
                }
            )
        return out

    def run(self, topics: list[str] | None = None) -> AgentResult:
        result = self._new_result()
        created_notes: list[dict[str, Any]] = []

        try:
            topic_list = topics if topics is not None else _topics_from_config(self.topics_config)
            if not topic_list:
                result.add_error("no topics configured")
                self.write_log(result, summary="No topics to process.", created_notes=created_notes)
                return result

            depth = self.config.get("search_depth", "medium")
            for topic in topic_list:
                if result.notes_created >= self.max_notes_per_run:
                    logger.info("Reached max_notes_per_run={}", self.max_notes_per_run)
                    break
                result.topics_processed.append(topic)
                try:
                    sources = self.research_topic(topic, depth=depth)
                except Exception as exc:
                    result.add_error(f"research_topic({topic!r}): {exc}")
                    continue
                for source in sources:
                    if result.notes_created >= self.max_notes_per_run:
                        break
                    note = self._process_source(topic, source, result)
                    if note:
                        created_notes.append(note)

            summary = (
                f"Researched {len(result.topics_processed)} topic(s); "
                f"created {result.notes_created} new note(s)."
            )
            self.write_log(result, summary=summary, created_notes=created_notes)
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("ResearcherAgent crashed")
            result.add_error(f"fatal: {exc}")
            self.write_log(result, summary="Crashed.", created_notes=created_notes)
        return result

    # ------------------------------------------------------------- internal

    def _process_source(
        self,
        topic: str,
        source: dict[str, Any],
        result: AgentResult,
    ) -> dict[str, Any] | None:
        url = source.get("url") or ""
        title = source.get("title") or url
        content = source.get("content") or ""
        if len(content.split()) < 50:
            return None  # too short to be useful

        # Dedup BEFORE we spend an LLM call.
        existing = self.memory.is_duplicate(content, threshold=self.dedup_threshold)
        if existing is not None:
            logger.debug("Skip duplicate of {} (sim={:.2f})", existing.get("id"), existing.get("similarity"))
            return None

        try:
            summary_md = self.summarizer.summarize(content, style="atomic")
        except Exception as exc:
            result.add_error(f"summarize({url}): {exc}")
            return None

        try:
            tags = self.summarizer.generate_tags(content) or []
        except Exception:
            tags = []

        body_parts = [
            f"# {title}",
            "",
            "## Core Idea",
            summary_md or "_summary unavailable_",
            "",
            "## Evidence / Sources",
            f"- {url}",
        ]
        if source.get("snippet"):
            body_parts.extend(["", f"> {source['snippet']}"])
        body_parts.extend(["", "## Connections", "- Related: [[]]", "", "## Questions Raised", ""])
        body = "\n".join(body_parts)

        metadata = {
            "title": title,
            "source": url,
            "tags": tags + [_slug(topic)],
            "type": "atomic",
            "agent_generated": True,
            "agent": self.name,
            "topic": topic,
            "confidence": 0.7,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        try:
            path = self.note_writer.create_note(
                folder="00_Inbox",
                title=title,
                content=body,
                metadata=metadata,
            )
        except Exception as exc:
            result.add_error(f"create_note({title!r}): {exc}")
            return None

        try:
            self.memory.add(
                text=summary_md or content[:2000],
                metadata={
                    "source": url,
                    "url": url,
                    "type": "atomic",
                    "tags": metadata["tags"],
                    "agent": self.name,
                    "confidence": 0.7,
                    "note_path": str(Path(path).relative_to(self.note_writer.vault_path)),
                    "title": title,
                },
            )
        except Exception as exc:
            result.add_error(f"memory.add({title!r}): {exc}")

        result.notes_created += 1
        result.sources_used.append(url)
        return {"title": title, "path": path, "url": url}


def _topics_from_config(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for domain in cfg.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        sources = domain.get("sources") or {}
        if not sources.get("web", True):
            continue
        for sub in domain.get("subtopics") or []:
            out.append(str(sub))
        if not domain.get("subtopics") and domain.get("name"):
            out.append(str(domain["name"]))
    return out


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9\-]+", "-", value.lower()).strip("-")[:40]
