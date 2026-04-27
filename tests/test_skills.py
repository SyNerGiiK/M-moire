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

    s = Summarizer(provider="ollama", anthropic_api_key=None)
    monkeypatch.setattr(s, "_check_ollama", lambda: False)
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    out = s.summarize(text)
    assert "First sentence" in out


def test_summarizer_heuristic_tags(monkeypatch):
    from skills.summarizer import Summarizer

    s = Summarizer(provider="ollama", anthropic_api_key=None)
    monkeypatch.setattr(s, "_check_ollama", lambda: False)
    tags = s.generate_tags("transformers attention mechanism transformer model attention")
    assert all(isinstance(t, str) for t in tags)
    assert tags  # heuristic should produce something


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
