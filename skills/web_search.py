"""Web search and clean-text extraction.

Uses the keyless DuckDuckGo Search package for queries and trafilatura
for high-quality body extraction. Network errors degrade to empty
results — agents are expected to handle that gracefully.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from loguru import logger


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass
class FetchedPage:
    url: str
    title: str
    content: str
    date: str | None
    word_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "date": self.date,
            "word_count": self.word_count,
        }


_HREF_RE = re.compile(r"href=[\"\']([^\"\']+)[\"\']")


class WebSearch:
    def __init__(self, user_agent: str = "SecondBrain/0.1 (+https://localhost)") -> None:
        self.user_agent = user_agent

    # ----------------------------------------------------------------- search

    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        try:
            from duckduckgo_search import DDGS
        except ImportError:  # pragma: no cover
            logger.error("duckduckgo-search is not installed.")
            return []

        results: list[dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    hit = SearchHit(
                        title=str(item.get("title") or "").strip(),
                        url=str(item.get("href") or item.get("url") or "").strip(),
                        snippet=str(item.get("body") or "").strip(),
                    )
                    if hit.url:
                        results.append(hit.as_dict())
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for {!r}: {}", query, exc)
        return results[:max_results]

    # ------------------------------------------------------------------ fetch

    def fetch_page(self, url: str) -> dict[str, Any]:
        if not url:
            return {"url": url, "title": "", "content": "", "date": None, "word_count": 0}
        try:
            import trafilatura
        except ImportError:  # pragma: no cover
            logger.error("trafilatura is not installed.")
            return {"url": url, "title": "", "content": "", "date": None, "word_count": 0}

        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            logger.debug("Failed to download {}", url)
            return {"url": url, "title": "", "content": "", "date": None, "word_count": 0}
        try:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                with_metadata=True,
                output_format="json",
                favor_precision=True,
            )
        except Exception as exc:
            logger.warning("trafilatura failed on {}: {}", url, exc)
            extracted = None

        if not extracted:
            return {"url": url, "title": "", "content": "", "date": None, "word_count": 0}

        import json

        try:
            data = json.loads(extracted)
        except json.JSONDecodeError:
            data = {"text": extracted, "title": "", "date": None}

        text = (data.get("text") or "").strip()
        page = FetchedPage(
            url=url,
            title=(data.get("title") or "").strip(),
            content=text,
            date=(data.get("date") or None),
            word_count=len(text.split()) if text else 0,
        )
        return page.as_dict()

    # ------------------------------------------------------------ deep search

    def deep_search(
        self,
        query: str,
        depth: int = 2,
        max_per_level: int = 5,
    ) -> list[dict[str, Any]]:
        """Search, fetch top results, and follow outbound links one level deep.

        ``depth`` controls how many levels are walked (1 = search only,
        2 = also follow outbound links from fetched pages).
        """
        seen: set[str] = set()
        out: list[dict[str, Any]] = []

        seeds = self.search(query, max_results=max_per_level)
        frontier: list[str] = []
        for hit in seeds:
            url = hit["url"]
            if url and url not in seen:
                seen.add(url)
                page = self.fetch_page(url)
                if page["content"]:
                    page["depth"] = 1
                    page["seed_query"] = query
                    out.append(page)
                    if depth > 1:
                        frontier.extend(_extract_links(page["content"], url)[:max_per_level])

        if depth > 1:
            for url in frontier:
                if url in seen:
                    continue
                seen.add(url)
                page = self.fetch_page(url)
                if page["content"]:
                    page["depth"] = 2
                    page["seed_query"] = query
                    out.append(page)

        return out


def _extract_links(text: str, base_url: str) -> list[str]:
    """Extract outbound HTTP(S) links found verbatim in extracted text."""
    candidates = re.findall(r"https?://[^\s)\]>]+", text)
    base_host = urlparse(base_url).netloc
    out: list[str] = []
    for candidate in candidates:
        host = urlparse(candidate).netloc
        if host and host != base_host:
            out.append(candidate.rstrip(".,;:"))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    result: list[str] = []
    for link in out:
        if link not in seen:
            seen.add(link)
            result.append(link)
    return result
