"""PDF text extraction and chunking.

Backed by :mod:`pymupdf` (``fitz``). Accepts either a local path or an
HTTP(S) URL — remote PDFs are fetched into a temp file first.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger


_WORD_BOUNDARY = re.compile(r"\s+")


class PDFProcessor:
    def __init__(self) -> None:
        pass

    # ---------------------------------------------------------------- extract

    def extract_text(self, path_or_url: str) -> str:
        if not path_or_url:
            return ""
        local_path = self._resolve_to_path(path_or_url)
        if local_path is None:
            return ""
        try:
            import fitz  # type: ignore
        except ImportError:  # pragma: no cover
            logger.error("pymupdf is not installed.")
            return ""
        try:
            doc = fitz.open(str(local_path))
        except Exception as exc:
            logger.warning("Failed to open PDF {}: {}", local_path, exc)
            return ""
        try:
            pages = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        text = "\n\n".join(pages)
        return _WORD_BOUNDARY.sub(" ", text).strip()

    def extract_metadata(self, path: str) -> dict[str, Any]:
        local_path = self._resolve_to_path(path)
        if local_path is None:
            return {}
        try:
            import fitz  # type: ignore
        except ImportError:  # pragma: no cover
            return {}
        try:
            doc = fitz.open(str(local_path))
        except Exception as exc:
            logger.warning("Failed to open PDF {}: {}", path, exc)
            return {}
        try:
            meta = dict(doc.metadata or {})
            meta["page_count"] = doc.page_count
        finally:
            doc.close()
        return meta

    # ------------------------------------------------------------------ chunk

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[str]:
        if not text:
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            overlap = max(0, min(overlap, chunk_size - 1))

        # Break on paragraphs first to keep chunks coherent.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        buffer = ""
        for para in paragraphs:
            if not buffer:
                buffer = para
                continue
            if len(buffer) + len(para) + 2 <= chunk_size:
                buffer = f"{buffer}\n\n{para}"
            else:
                chunks.append(buffer)
                # Start new buffer with overlap from previous chunk's tail.
                tail = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
                buffer = (tail + "\n\n" + para).strip() if tail else para
        if buffer:
            chunks.append(buffer)

        # If a single paragraph exceeds chunk_size, force-split it.
        out: list[str] = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                out.append(chunk)
                continue
            start = 0
            while start < len(chunk):
                end = min(start + chunk_size, len(chunk))
                out.append(chunk[start:end])
                if end == len(chunk):
                    break
                start = max(0, end - overlap)
        return out

    # ----------------------------------------------------------------- helpers

    def _resolve_to_path(self, path_or_url: str) -> Path | None:
        parsed = urlparse(path_or_url)
        if parsed.scheme in ("http", "https"):
            return self._download_to_temp(path_or_url)
        path = Path(path_or_url)
        if not path.exists():
            logger.warning("PDF not found: {}", path)
            return None
        return path

    def _download_to_temp(self, url: str) -> Path | None:
        try:
            import httpx
        except ImportError:  # pragma: no cover
            return None
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30.0)
            r.raise_for_status()
        except Exception as exc:
            logger.warning("PDF download failed {}: {}", url, exc)
            return None
        tmp = tempfile.NamedTemporaryFile(prefix="sb_pdf_", suffix=".pdf", delete=False)
        tmp.write(r.content)
        tmp.close()
        return Path(tmp.name)
