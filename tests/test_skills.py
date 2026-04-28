"""Tests for the skills layer.

Heavy dependencies are imported lazily inside individual tests; modules
that are unavailable cause the corresponding test to be skipped instead
of failing the whole suite.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# ============================================================== NoteWriter


def test_note_writer_creates_note_with_frontmatter(note_writer, vault_dir: Path):
    path = note_writer.create_note(
        folder="00_Inbox",
        title="My First Note",
        content="Body text.",
        metadata={"tags": ["alpha", "beta"], "type": "atomic"},
    )
    note_path = Path(path)
    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "title: My First Note" in content
    assert "tags:" in content
    assert "Body text." in content


def test_note_writer_sanitize_filename(note_writer):
    assert note_writer.sanitize_filename("Hello, world!") == "Hello_world"
    assert note_writer.sanitize_filename("") == "untitled"
    assert note_writer.sanitize_filename("Élégance & Café") == "Elegance_Cafe"


def test_note_writer_unique_paths_when_collision(note_writer):
    p1 = note_writer.create_note("00_Inbox", "Same Title", "first", {})
    p2 = note_writer.create_note("00_Inbox", "Same Title", "second", {})
    assert p1 != p2
    assert Path(p1).exists() and Path(p2).exists()


def test_note_writer_link_notes(note_writer):
    src = note_writer.create_note("00_Inbox", "Source Note", "Body.", {})
    tgt = note_writer.create_note("01_Atomic_Notes", "Target Note", "Body.", {})
    note_writer.link_notes(src, tgt)
    text = Path(src).read_text(encoding="utf-8")
    assert "[[Target_Note]]" in text


def test_note_writer_iter_notes(note_writer):
    note_writer.create_note("00_Inbox", "A", "x", {})
    note_writer.create_note("01_Atomic_Notes", "B", "y", {})
    titles = sorted(n.title for n in note_writer.iter_notes())
    assert "A" in titles and "B" in titles


def test_note_writer_move_note(note_writer):
    p = note_writer.create_note("00_Inbox", "Movable", "x", {})
    new_path = note_writer.move_note(p, "01_Atomic_Notes")
    assert Path(new_path).exists()
    assert not Path(p).exists()
    assert "01_Atomic_Notes" in new_path


# ============================================================== Summarizer


def test_summarizer_heuristic_summary_when_no_llm(monkeypatch):
    from skills.summarizer import Summarizer

    s = Summarizer()
    monkeypatch.setattr(s, "_check_lmstudio", lambda: False)
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    out = s.summarize(text)
    assert "First sentence" in out


def test_summarizer_heuristic_tags(monkeypatch):
    from skills.summarizer import Summarizer

    s = Summarizer()
    monkeypatch.setattr(s, "_check_lmstudio", lambda: False)
    tags = s.generate_tags("transformers attention mechanism transformer model attention")
    assert all(isinstance(t, str) for t in tags)
    assert tags  # heuristic should produce something


def test_summarizer_lmstudio_path_uses_http(monkeypatch):
    """The dispatcher must call _lmstudio_generate when the server is reachable."""
    from skills.summarizer import Summarizer

    s = Summarizer(base_url="http://localhost:1234/v1", model="test-model")
    monkeypatch.setattr(s, "_check_lmstudio", lambda: True)
    monkeypatch.setattr(s, "_lmstudio_generate", lambda prompt, system=None: "MOCKED REPLY")
    out = s.summarize("Some text to summarize.")
    assert out == "MOCKED REPLY"


def test_summarizer_resolves_model_priority():
    """Configured model wins over detected; detected wins over default."""
    from skills.summarizer import Summarizer

    s = Summarizer(base_url="http://localhost:1234/v1", model="explicit-model")
    assert s._resolve_model() == "explicit-model"

    s2 = Summarizer(base_url="http://localhost:1234/v1", model=None)
    s2._detected_model = "auto-detected-model"
    assert s2._resolve_model() == "auto-detected-model"

    s3 = Summarizer(base_url="http://localhost:1234/v1", model=None)
    assert s3._resolve_model() == "local-model"


def test_summarizer_parse_json_list_handles_messy_output():
    from skills.summarizer import _parse_json_list

    assert _parse_json_list('["a", "b", "c"]') == ["a", "b", "c"]
    assert _parse_json_list("Sure! Here you go: [\"x\", \"y\"]") == ["x", "y"]
    assert _parse_json_list("- one\n- two\n- three") == ["one", "two", "three"]
    assert _parse_json_list("") == []


# ============================================================== PDFProcessor


def test_pdf_chunk_text_basic():
    from skills.pdf_processor import PDFProcessor

    p = PDFProcessor()
    text = ("Paragraph one. " * 50) + "\n\n" + ("Paragraph two. " * 50)
    chunks = p.chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 300 for c in chunks)


def test_pdf_chunk_text_zero_input():
    from skills.pdf_processor import PDFProcessor

    assert PDFProcessor().chunk_text("") == []


def test_pdf_chunk_text_invalid_chunk_size():
    from skills.pdf_processor import PDFProcessor

    with pytest.raises(ValueError):
        PDFProcessor().chunk_text("hello", chunk_size=0)


# ============================================================== WebSearch


def test_web_search_extract_links_filters_same_host():
    from skills.web_search import _extract_links

    text = "See https://example.com/foo and https://other.org/bar and https://example.com/again"
    out = _extract_links(text, "https://example.com/")
    assert "https://other.org/bar" in out
    assert all("example.com" not in u for u in out)


# ============================================================== WebCrawler


def test_web_crawler_chunk_text_short_passthrough():
    from skills.web_crawler import chunk_text

    assert chunk_text("") == []
    assert chunk_text("short text") == ["short text"]


def test_web_crawler_chunk_text_splits_long_input():
    from skills.web_crawler import chunk_text

    paragraphs = "\n\n".join([f"Paragraph {i} " * 30 for i in range(8)])
    chunks = chunk_text(paragraphs, target_chars=400, overlap_chars=50)
    assert len(chunks) >= 3
    assert all(c.strip() for c in chunks)


def test_web_crawler_chunk_text_hard_slices_huge_paragraph():
    from skills.web_crawler import chunk_text

    huge = "a" * 5000
    chunks = chunk_text(huge, target_chars=1000, overlap_chars=100)
    assert len(chunks) >= 5
    assert all(len(c) <= 1000 for c in chunks)


def test_web_crawler_extract_links_keeps_same_domain_only():
    from skills.web_crawler import _extract_links, _same_domain

    html = (
        '<a href="/docs/foo">f</a>'
        '<a href="https://nextjs.org/docs/bar">b</a>'
        '<a href="https://other.com/x">x</a>'
        '<a href="mailto:hi@nextjs.org">m</a>'
        '<a href="javascript:void(0)">j</a>'
    )
    links = _extract_links(html, "https://nextjs.org/docs")
    same = [u for u in links if _same_domain(u, "nextjs.org")]
    assert "https://nextjs.org/docs/foo" in same
    assert "https://nextjs.org/docs/bar" in same
    assert all("other.com" not in u for u in same)
    assert all("mailto:" not in u and "javascript:" not in u for u in links)


def test_web_crawler_skips_assets():
    from skills.web_crawler import _looks_like_asset

    assert _looks_like_asset("https://x.com/img.png")
    assert _looks_like_asset("https://x.com/a/b/file.pdf")
    assert not _looks_like_asset("https://x.com/docs/intro")


def test_web_crawler_crawl_uses_stubbed_network(monkeypatch):
    from skills import web_crawler
    from skills.web_crawler import WebCrawler

    pages_html = {
        "https://example.com/docs": (
            "<html><head><title>Docs Home</title></head>"
            "<body><a href='/docs/intro'>Intro</a><a href='/docs/api'>API</a>"
            "<a href='https://other.com/x'>x</a>"
            "<p>" + ("welcome to the documentation home page " * 20) + "</p>"
            "</body></html>"
        ),
        "https://example.com/docs/intro": (
            "<html><head><title>Intro</title></head>"
            "<body><p>" + ("introduction body content here " * 20) + "</p></body></html>"
        ),
        "https://example.com/docs/api": (
            "<html><head><title>API</title></head>"
            "<body><p>" + ("api reference details here " * 20) + "</p></body></html>"
        ),
    }

    def fake_fetch(self, url):
        return pages_html.get(url)

    def fake_extract(self, html, url):
        # Pull <title> manually then use the visible text-ish content from html.
        import re as _re
        title_match = _re.search(r"<title>(.*?)</title>", html, _re.IGNORECASE | _re.DOTALL)
        title = (title_match.group(1).strip() if title_match else "").strip()
        body = _re.sub(r"<[^>]+>", " ", html)
        body = _re.sub(r"\s+", " ", body).strip()
        return title, body

    monkeypatch.setattr(WebCrawler, "_download", fake_fetch)
    monkeypatch.setattr(WebCrawler, "_extract", fake_extract)

    crawler = WebCrawler(request_delay=0.0)
    pages = crawler.crawl(
        "https://example.com/docs",
        max_pages=10,
        max_depth=2,
        allow_subdomains=False,
        path_prefix="/docs",
    )
    urls = {p.url for p in pages}
    assert "https://example.com/docs" in urls
    assert "https://example.com/docs/intro" in urls
    assert "https://example.com/docs/api" in urls
    assert all(p.url.startswith("https://example.com/") for p in pages)


def test_web_crawler_returns_empty_on_network_failure(monkeypatch):
    from skills.web_crawler import WebCrawler

    monkeypatch.setattr(WebCrawler, "_download", lambda self, url: None)
    crawler = WebCrawler(request_delay=0.0)
    assert crawler.crawl("https://example.com/", max_pages=5, max_depth=2) == []


# ============================================================== Youtube


def test_youtube_video_id_extraction():
    from skills.youtube_transcriber import _extract_video_id

    assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("") == ""


# ============================================================== ArxivFetcher


def test_arxiv_short_id_normalizes():
    from skills.arxiv_fetcher import _short_id

    assert _short_id("http://arxiv.org/abs/2401.12345v2") == "2401.12345"
    assert _short_id("2401.12345") == "2401.12345"
    assert _short_id("") == ""
