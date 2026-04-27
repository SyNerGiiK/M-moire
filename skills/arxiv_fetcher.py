"""arXiv search and paper download.

Wraps the ``arxiv`` package and adds PDF parsing via :mod:`pymupdf`.
Network/parse errors return empty results — callers should always
check the output.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ArxivPaper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    url: str
    pdf_url: str
    date: str
    categories: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArxivFetcher:
    """Search arXiv and pull paper text on demand."""

    def __init__(self, download_dir: str | Path | None = None) -> None:
        self.download_dir = Path(download_dir) if download_dir else None
        if self.download_dir:
            self.download_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        max_results: int = 20,
        days_back: int | None = 7,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        try:
            import arxiv
        except ImportError:  # pragma: no cover
            logger.error("arxiv package is not installed.")
            return []

        client = arxiv.Client(page_size=min(50, max_results), delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        cutoff = None
        if days_back is not None:
            cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)

        out: list[dict[str, Any]] = []
        try:
            for result in client.results(search):
                published = getattr(result, "published", None)
                if cutoff and published and published.replace(tzinfo=dt.timezone.utc) < cutoff:
                    continue
                paper = ArxivPaper(
                    id=_short_id(result.entry_id),
                    title=(result.title or "").strip(),
                    abstract=(result.summary or "").strip(),
                    authors=[str(a) for a in (result.authors or [])],
                    url=str(result.entry_id),
                    pdf_url=str(result.pdf_url),
                    date=published.strftime("%Y-%m-%d") if published else "",
                    categories=list(result.categories or []),
                )
                out.append(paper.as_dict())
                if len(out) >= max_results:
                    break
        except Exception as exc:
            logger.warning("arXiv search failed for {!r}: {}", query, exc)
        return out

    # --------------------------------------------------------------- single fetch

    def fetch_paper(self, arxiv_id: str) -> dict[str, Any]:
        if not arxiv_id:
            return {}
        try:
            import arxiv
        except ImportError:  # pragma: no cover
            return {}
        client = arxiv.Client()
        try:
            search = arxiv.Search(id_list=[_short_id(arxiv_id)])
            result = next(client.results(search), None)
        except Exception as exc:
            logger.warning("arXiv fetch failed for {}: {}", arxiv_id, exc)
            return {}
        if result is None:
            return {}
        paper = ArxivPaper(
            id=_short_id(result.entry_id),
            title=(result.title or "").strip(),
            abstract=(result.summary or "").strip(),
            authors=[str(a) for a in (result.authors or [])],
            url=str(result.entry_id),
            pdf_url=str(result.pdf_url),
            date=result.published.strftime("%Y-%m-%d") if result.published else "",
            categories=list(result.categories or []),
        )
        return paper.as_dict()

    # ------------------------------------------------------------------ PDF

    def download_and_parse(self, arxiv_id: str) -> str:
        """Download the PDF for ``arxiv_id`` and return the extracted text."""
        if not arxiv_id:
            return ""
        meta = self.fetch_paper(arxiv_id)
        pdf_url = meta.get("pdf_url") or ""
        if not pdf_url:
            return ""
        try:
            import httpx
        except ImportError:  # pragma: no cover
            return ""
        try:
            r = httpx.get(pdf_url, follow_redirects=True, timeout=30.0)
            r.raise_for_status()
        except Exception as exc:
            logger.warning("arXiv PDF download failed for {}: {}", arxiv_id, exc)
            return ""

        from skills.pdf_processor import PDFProcessor

        if self.download_dir:
            local = self.download_dir / f"{_short_id(arxiv_id)}.pdf"
            local.write_bytes(r.content)
            return PDFProcessor().extract_text(str(local))
        # In-memory parse using pymupdf bytes interface.
        try:
            import fitz  # type: ignore
        except ImportError:  # pragma: no cover
            return ""
        doc = fitz.open(stream=r.content, filetype="pdf")
        try:
            return "\n\n".join(page.get_text("text") for page in doc).strip()
        finally:
            doc.close()


def _short_id(entry_id: str) -> str:
    """Return a bare arXiv id like '2401.12345' from a full URL or raw id."""
    if not entry_id:
        return ""
    eid = str(entry_id).strip()
    if "/abs/" in eid:
        eid = eid.split("/abs/")[-1]
    return eid.rsplit("v", 1)[0] if "v" in eid and eid.split("v")[-1].isdigit() else eid
