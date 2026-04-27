"""Scientific paper monitoring agent.

For each topic in ``config/topics.yaml`` (where ``sources.arxiv: true``):
    1. Search arXiv for the last N days
    2. Skip papers already indexed in ChromaDB
    3. Generate structured note: abstract, contributions, methodology, implications
    4. Write to ``vault/04_Resources/Papers/``
    5. Add embedding to ChromaDB
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from agents.base import AgentResult, BaseAgent
from skills.arxiv_fetcher import ArxivFetcher
from skills.note_writer import NoteWriter
from skills.summarizer import Summarizer
from skills.vector_memory import VectorMemory


class ArxivAgent(BaseAgent):
    name = "arxiv"

    def __init__(
        self,
        note_writer: NoteWriter,
        memory: VectorMemory,
        summarizer: Summarizer,
        arxiv_fetcher: ArxivFetcher | None = None,
        config: dict[str, Any] | None = None,
        topics_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(note_writer)
        self.memory = memory
        self.summarizer = summarizer
        self.arxiv = arxiv_fetcher or ArxivFetcher()
        self.config = config or {}
        self.topics_config = topics_config or {}

    # --------------------------------------------------------------- public

    def monitor_topics(
        self,
        topics: list[str] | None = None,
        days_back: int | None = None,
    ) -> AgentResult:
        result = self._new_result()
        created_notes: list[dict[str, Any]] = []

        days_back = int(days_back if days_back is not None else self.config.get("days_back", 7))
        max_per_topic = int(self.config.get("max_papers_per_topic", 5))

        topic_list = topics if topics is not None else _arxiv_topics(self.topics_config)
        if not topic_list:
            result.add_error("no arxiv topics configured")
            self.write_log(result, summary="No topics to monitor.", created_notes=created_notes)
            return result

        for topic in topic_list:
            result.topics_processed.append(topic)
            try:
                papers = self.arxiv.search(topic, max_results=max_per_topic, days_back=days_back)
            except Exception as exc:
                result.add_error(f"arxiv.search({topic!r}): {exc}")
                continue
            for paper in papers:
                note = self.process_paper(paper, topic, result)
                if note:
                    created_notes.append(note)

        summary = (
            f"Monitored {len(result.topics_processed)} topic(s); "
            f"added {result.notes_created} paper note(s)."
        )
        self.write_log(result, summary=summary, created_notes=created_notes)
        return result

    def process_paper(
        self,
        paper: dict[str, Any],
        topic: str = "",
        result: AgentResult | None = None,
    ) -> dict[str, Any] | None:
        arxiv_id = paper.get("id") or ""
        title = (paper.get("title") or "").strip()
        if not arxiv_id or not title:
            return None
        # Dedup against memory: arxiv_id is the canonical key.
        existing = self.memory.search(
            query=f"arxiv:{arxiv_id}",
            n_results=1,
            filter={"arxiv_id": arxiv_id},
        )
        if existing:
            return None

        abstract = (paper.get("abstract") or "").strip()
        try:
            summary_md = self.summarizer.summarize(abstract, style="detailed") if abstract else ""
        except Exception as exc:
            if result:
                result.add_error(f"summarize({arxiv_id}): {exc}")
            summary_md = abstract

        try:
            tags = self.summarizer.generate_tags(abstract) if abstract else []
        except Exception:
            tags = []
        tags = list(dict.fromkeys(tags + paper.get("categories", []) + (["arxiv"])))

        body = _render_paper_body(paper, summary_md)
        metadata = {
            "title": title,
            "arxiv_id": arxiv_id,
            "authors": paper.get("authors", []),
            "published": paper.get("date", ""),
            "source": paper.get("url", ""),
            "tags": tags,
            "type": "paper",
            "agent_generated": True,
            "agent": self.name,
            "topic": topic,
            "confidence": 0.85,
        }
        try:
            path = self.note_writer.create_note(
                folder="04_Resources/Papers",
                title=f"{arxiv_id} - {title}",
                content=body,
                metadata=metadata,
            )
        except Exception as exc:
            if result:
                result.add_error(f"create_note(paper {arxiv_id}): {exc}")
            return None

        # Store in memory.
        try:
            self.memory.add(
                text=f"{title}\n\n{abstract}",
                metadata={
                    "source": paper.get("url", ""),
                    "url": paper.get("url", ""),
                    "type": "paper",
                    "arxiv_id": arxiv_id,
                    "tags": tags,
                    "agent": self.name,
                    "confidence": 0.85,
                    "note_path": str(Path(path).relative_to(self.note_writer.vault_path)),
                    "title": title,
                },
            )
        except Exception as exc:
            if result:
                result.add_error(f"memory.add(paper {arxiv_id}): {exc}")

        if result is not None:
            result.notes_created += 1
            result.sources_used.append(paper.get("url", ""))
        return {"title": title, "path": path, "url": paper.get("url", "")}


# ====================================================================
# Helpers
# ====================================================================


def _render_paper_body(paper: dict[str, Any], summary_md: str) -> str:
    authors = ", ".join(paper.get("authors") or []) or "_unknown_"
    cats = ", ".join(paper.get("categories") or []) or "_uncategorized_"
    pdf_url = paper.get("pdf_url") or ""
    abstract = paper.get("abstract") or ""

    parts = [
        f"# {paper.get('title') or 'Untitled paper'}",
        "",
        f"- **arXiv:** [{paper.get('id', '')}]({paper.get('url', '')})",
        f"- **Authors:** {authors}",
        f"- **Published:** {paper.get('date', '')}",
        f"- **Categories:** {cats}",
        f"- **PDF:** {pdf_url}",
        "",
        "## Abstract",
        abstract or "_no abstract_",
        "",
        "## Summary",
        summary_md or "_summary unavailable_",
        "",
        "## Key Contributions",
        "- ",
        "",
        "## Methodology",
        "- ",
        "",
        "## Implications",
        "- ",
        "",
        "## Connections",
        "- Related: [[]]",
        "",
    ]
    return "\n".join(parts)


def _arxiv_topics(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for domain in cfg.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        sources = domain.get("sources") or {}
        if not sources.get("arxiv", False):
            continue
        for sub in domain.get("subtopics") or []:
            out.append(str(sub))
        if not domain.get("subtopics") and domain.get("name"):
            out.append(str(domain["name"]))
    return out
