"""Documentation crawl agent.

Given a root URL (e.g. ``https://nextjs.org/docs``), walk every internal
link of the same domain, extract clean text, chunk it, and ingest each
chunk into :class:`VectorMemory` for later RAG retrieval.

The agent also drops a small index note in
``vault/04_Resources/Crawls/`` so the run is visible inside Obsidian.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from agents.base import AgentResult, BaseAgent
from skills.note_writer import NoteWriter
from skills.vector_memory import VectorMemory
from skills.web_crawler import WebCrawler


class CrawlerAgent(BaseAgent):
    """Aspirate an entire documentation site into the vector store."""

    name = "crawler"

    def __init__(
        self,
        note_writer: NoteWriter,
        memory: VectorMemory,
        crawler: WebCrawler | None = None,
        config: dict[str, Any] | None = None,
        topics_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(note_writer)
        self.memory = memory
        self.crawler = crawler or WebCrawler()
        self.config = config or {}
        self.topics_config = topics_config or {}
        self.dedup_threshold = float(
            (self.topics_config.get("settings") or {}).get("dedup_threshold", 0.92)
        )

    # ------------------------------------------------------------------- run

    def run(
        self,
        url: str | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        path_prefix: str | None = None,
        allow_subdomains: bool | None = None,
        target_chars: int | None = None,
        overlap_chars: int | None = None,
        tag: str | None = None,
    ) -> AgentResult:
        result = self._new_result()
        url = (url or self.config.get("url") or "").strip()
        if not url:
            result.add_error("crawler.run: missing 'url'")
            self.write_log(result, summary="No URL provided.")
            return result

        max_pages = int(max_pages if max_pages is not None else self.config.get("max_pages", 200))
        max_depth = int(max_depth if max_depth is not None else self.config.get("max_depth", 3))
        target_chars = int(target_chars or self.config.get("target_chars", 1200))
        overlap_chars = int(overlap_chars or self.config.get("overlap_chars", 150))
        allow_subdomains = (
            self.config.get("allow_subdomains", True)
            if allow_subdomains is None else bool(allow_subdomains)
        )
        path_prefix = path_prefix if path_prefix is not None else self.config.get("path_prefix")

        host = urlparse(url).netloc or url
        crawl_tag = tag or _slug(host)
        result.topics_processed.append(host)

        logger.info("[crawler] starting crawl of {}", url)
        try:
            pages = self.crawler.crawl(
                root_url=url,
                max_pages=max_pages,
                max_depth=max_depth,
                allow_subdomains=allow_subdomains,
                path_prefix=path_prefix,
            )
        except Exception as exc:
            logger.exception("CrawlerAgent crawl failed")
            result.add_error(f"crawl({url}): {exc}")
            self.write_log(result, summary="Crawl crashed.")
            return result

        chunks_added = 0
        chunks_skipped = 0
        pages_indexed = 0
        from skills.web_crawler import chunk_text

        for page in pages:
            pieces = chunk_text(page.content, target_chars=target_chars, overlap_chars=overlap_chars)
            if not pieces:
                continue
            pages_indexed += 1
            result.sources_used.append(page.url)
            total = len(pieces)
            for idx, piece in enumerate(pieces):
                try:
                    dup = self.memory.is_duplicate(piece, threshold=self.dedup_threshold)
                except Exception:
                    dup = None
                if dup is not None:
                    chunks_skipped += 1
                    continue
                metadata = {
                    "source": page.url,
                    "url": page.url,
                    "title": page.title,
                    "type": "crawl",
                    "agent": self.name,
                    "host": host,
                    "depth": page.depth,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "tags": [crawl_tag, "crawl", "rag"],
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                try:
                    self.memory.add(text=piece, metadata=metadata)
                    chunks_added += 1
                except Exception as exc:
                    result.add_error(f"memory.add({page.url}#{idx}): {exc}")

        result.notes_created = pages_indexed
        result.extra.update(
            {
                "url": url,
                "host": host,
                "pages_fetched": len(pages),
                "pages_indexed": pages_indexed,
                "chunks_added": chunks_added,
                "chunks_skipped_duplicates": chunks_skipped,
                "max_pages": max_pages,
                "max_depth": max_depth,
                "path_prefix": path_prefix,
                "crawl_tag": crawl_tag,
            }
        )

        self._write_index_note(url, host, pages, chunks_added, chunks_skipped, crawl_tag)
        summary = (
            f"Crawled {host}: {len(pages)} page(s), {chunks_added} chunk(s) ingested, "
            f"{chunks_skipped} duplicate(s) skipped."
        )
        try:
            self.write_log(result, summary=summary)
        except Exception as exc:
            logger.warning("Could not write agent log: {}", exc)
        return result

    # ----------------------------------------------------------------- helpers

    def _write_index_note(
        self,
        url: str,
        host: str,
        pages: list,
        chunks_added: int,
        chunks_skipped: int,
        crawl_tag: str,
    ) -> None:
        title = f"Crawl — {host}"
        body_lines = [
            f"# {title}",
            "",
            f"- Root: {url}",
            f"- Pages fetched: {len(pages)}",
            f"- Chunks ingested: {chunks_added}",
            f"- Duplicates skipped: {chunks_skipped}",
            f"- Tag: `{crawl_tag}`",
            "",
            "## Pages",
        ]
        for page in pages[:500]:
            page_title = getattr(page, "title", "") or page.url
            body_lines.append(f"- [{page_title}]({page.url})")
        metadata = {
            "title": title,
            "type": "crawl-index",
            "source": url,
            "host": host,
            "tags": [crawl_tag, "crawl", "rag"],
            "agent": self.name,
            "agent_generated": True,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        try:
            self.note_writer.create_note(
                folder="04_Resources/Crawls",
                title=title,
                content="\n".join(body_lines),
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Could not write crawl index note: {}", exc)


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9\-]+", "-", value.lower()).strip("-")[:40] or "crawl"
