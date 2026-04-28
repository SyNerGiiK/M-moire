"""Recursive same-domain web crawler with clean text extraction + chunking.

Walks every internal link from a root URL using a breadth-first frontier,
extracts the readable body with ``trafilatura`` (already a project dep),
and yields chunked documents ready for embedding in :class:`VectorMemory`.

Network failures (DNS, timeouts, 4xx/5xx, malformed HTML) degrade silently
to empty results — callers iterate what was successfully fetched.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urldefrag, urljoin, urlparse

from loguru import logger


_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#\s>]+)""", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z",
    ".mp3", ".mp4", ".mov", ".avi", ".webm",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".xml", ".rss", ".atom",
}


@dataclass
class CrawledPage:
    url: str
    title: str
    content: str
    word_count: int
    depth: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "word_count": self.word_count,
            "depth": self.depth,
        }


@dataclass
class Chunk:
    text: str
    url: str
    title: str
    chunk_index: int
    total_chunks: int
    depth: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _normalize_url(url: str) -> str:
    """Drop fragments and trailing slash so we don't crawl the same page twice."""
    if not url:
        return ""
    url, _ = urldefrag(url.strip())
    if url.endswith("/") and len(url) > 1:
        url = url[:-1]
    return url


def _looks_like_asset(url: str) -> bool:
    path = urlparse(url).path.lower()
    for ext in _SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


def _same_domain(url: str, root_host: str, allow_subdomains: bool = True) -> bool:
    host = urlparse(url).netloc.lower()
    if not host:
        return False
    if host == root_host:
        return True
    if allow_subdomains and host.endswith("." + root_host):
        return True
    return False


def _path_under(url: str, root_path: str) -> bool:
    """Constrain crawl to a path prefix (e.g. /docs)."""
    if not root_path or root_path == "/":
        return True
    p = urlparse(url).path or "/"
    return p.startswith(root_path)


def _extract_links(html: str, base_url: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in _HREF_RE.findall(html or ""):
        if raw.startswith(("mailto:", "javascript:", "tel:")):
            continue
        absolute = urljoin(base_url, raw)
        if absolute.startswith(("http://", "https://")):
            normalized = _normalize_url(absolute)
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
    return out


def chunk_text(
    text: str,
    target_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[str]:
    """Paragraph-aware chunker; falls back to hard slicing for huge blocks.

    Targets ``target_chars`` per chunk with ``overlap_chars`` carry-over so
    sentences spanning a boundary remain searchable in both chunks.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        # If a single paragraph is bigger than target, hard-slice it.
        if len(para) > target_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            start = 0
            while start < len(para):
                end = min(start + target_chars, len(para))
                chunks.append(para[start:end].strip())
                if end == len(para):
                    break
                start = end - overlap_chars if end - overlap_chars > start else end
            continue

        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) <= target_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
            # carry over the tail of the previous chunk for context
            tail = buffer[-overlap_chars:] if buffer and overlap_chars > 0 else ""
            buffer = (tail + "\n\n" + para).strip() if tail else para

    if buffer:
        chunks.append(buffer.strip())
    return [c for c in chunks if c]


class WebCrawler:
    """Same-domain breadth-first crawler with chunked output.

    Public surface:
      - :meth:`crawl` — returns a list of :class:`CrawledPage`.
      - :meth:`crawl_and_chunk` — yields :class:`Chunk` ready for embedding.

    Network errors are swallowed and logged at DEBUG. The crawler never
    raises on per-page failures; only programming errors propagate.
    """

    def __init__(
        self,
        user_agent: str = "SecondBrain-Crawler/0.1 (+https://localhost)",
        request_delay: float = 0.2,
        timeout: int = 20,
    ) -> None:
        self.user_agent = user_agent
        self.request_delay = max(0.0, float(request_delay))
        self.timeout = int(timeout)

    # ------------------------------------------------------------------ fetch

    def _download(self, url: str) -> str | None:
        """Download raw HTML. Returns None on any failure."""
        try:
            import trafilatura
        except ImportError:  # pragma: no cover
            logger.error("trafilatura is not installed; crawl disabled.")
            return None
        try:
            html = trafilatura.fetch_url(url, no_ssl=True)
        except Exception as exc:
            logger.debug("fetch_url failed for {}: {}", url, exc)
            return None
        return html or None

    def _extract(self, html: str, url: str) -> tuple[str, str]:
        """Return (title, clean_text). Empty strings on failure."""
        try:
            import trafilatura
        except ImportError:  # pragma: no cover
            return "", ""
        title = ""
        match = _TITLE_RE.search(html or "")
        if match:
            title = re.sub(r"\s+", " ", match.group(1)).strip()
        try:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_precision=False,
                no_fallback=False,
                url=url,
            ) or ""
        except Exception as exc:
            logger.debug("trafilatura.extract failed on {}: {}", url, exc)
            text = ""
        return title, text.strip()

    # ------------------------------------------------------------------- crawl

    def crawl(
        self,
        root_url: str,
        max_pages: int = 200,
        max_depth: int = 3,
        allow_subdomains: bool = True,
        path_prefix: str | None = None,
        min_words: int = 30,
    ) -> list[CrawledPage]:
        """Breadth-first crawl from ``root_url`` constrained to its domain.

        Parameters
        ----------
        root_url:
            Seed URL. Its host (and path prefix, if provided) defines the scope.
        max_pages:
            Hard cap on successfully fetched pages.
        max_depth:
            How many link-hops away from the root to walk (0 = root only).
        allow_subdomains:
            Treat ``foo.example.com`` as same-domain when seeded from ``example.com``.
        path_prefix:
            Optional path constraint, e.g. ``/docs`` to limit a crawl to a docs section.
            Defaults to the seed URL's path prefix when it isn't ``/``.
        min_words:
            Skip pages whose extracted text is shorter than this (boilerplate, 404s).
        """
        root_url = _normalize_url(root_url)
        if not root_url:
            return []
        parsed_root = urlparse(root_url)
        if parsed_root.scheme not in ("http", "https") or not parsed_root.netloc:
            logger.warning("Invalid root URL: {!r}", root_url)
            return []

        root_host = parsed_root.netloc.lower()
        if path_prefix is None:
            # Default: if seed is /docs/foo, restrict to /docs ; else allow all.
            seed_path = parsed_root.path or "/"
            if seed_path not in ("", "/"):
                # take the first segment as prefix
                segments = [s for s in seed_path.split("/") if s]
                path_prefix = "/" + segments[0] if segments else "/"
            else:
                path_prefix = "/"

        seen: set[str] = {root_url}
        frontier: deque[tuple[str, int]] = deque([(root_url, 0)])
        pages: list[CrawledPage] = []
        attempts = 0
        attempt_cap = max_pages * 5  # bounded safety net for dead links

        logger.info(
            "[crawler] start url={} max_pages={} max_depth={} prefix={}",
            root_url, max_pages, max_depth, path_prefix,
        )
        while frontier and len(pages) < max_pages and attempts < attempt_cap:
            url, depth = frontier.popleft()
            attempts += 1
            html = self._download(url)
            if self.request_delay:
                time.sleep(self.request_delay)
            if not html:
                continue
            title, text = self._extract(html, url)
            words = len(text.split())
            if words >= min_words:
                pages.append(
                    CrawledPage(
                        url=url,
                        title=title or url,
                        content=text,
                        word_count=words,
                        depth=depth,
                    )
                )
                logger.debug("[crawler] +page depth={} words={} url={}", depth, words, url)
            else:
                logger.debug("[crawler] -skip (short) url={}", url)

            if depth >= max_depth:
                continue
            for link in _extract_links(html, url):
                if link in seen:
                    continue
                if _looks_like_asset(link):
                    continue
                if not _same_domain(link, root_host, allow_subdomains=allow_subdomains):
                    continue
                if not _path_under(link, path_prefix):
                    continue
                seen.add(link)
                frontier.append((link, depth + 1))

        logger.info(
            "[crawler] done pages={} attempted={} frontier_left={}",
            len(pages), attempts, len(frontier),
        )
        return pages

    # ------------------------------------------------------------------ chunks

    def crawl_and_chunk(
        self,
        root_url: str,
        max_pages: int = 200,
        max_depth: int = 3,
        allow_subdomains: bool = True,
        path_prefix: str | None = None,
        target_chars: int = 1200,
        overlap_chars: int = 150,
        min_words: int = 30,
    ) -> Iterator[Chunk]:
        """Crawl + chunk in one call. Yields chunks lazily."""
        pages = self.crawl(
            root_url=root_url,
            max_pages=max_pages,
            max_depth=max_depth,
            allow_subdomains=allow_subdomains,
            path_prefix=path_prefix,
            min_words=min_words,
        )
        for page in pages:
            pieces = chunk_text(page.content, target_chars=target_chars, overlap_chars=overlap_chars)
            total = len(pieces)
            for idx, piece in enumerate(pieces):
                yield Chunk(
                    text=piece,
                    url=page.url,
                    title=page.title,
                    chunk_index=idx,
                    total_chunks=total,
                    depth=page.depth,
                )
