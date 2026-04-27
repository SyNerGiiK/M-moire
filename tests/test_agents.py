"""End-to-end tests for the agents using fake skills."""
from __future__ import annotations

from pathlib import Path

import pytest


# ============================================================== Researcher


def test_researcher_creates_inbox_notes(note_writer, fake_memory, fake_summarizer, fake_web_search):
    from agents.researcher_agent import ResearcherAgent

    agent = ResearcherAgent(
        note_writer=note_writer,
        memory=fake_memory,
        summarizer=fake_summarizer,
        web_search=fake_web_search,
        config={"max_sources_per_topic": 2, "search_depth": "quick"},
        topics_config={"settings": {"dedup_threshold": 0.99, "max_notes_per_run": 50}},
    )
    result = agent.run(topics=["LLM agents"])
    assert result.success
    assert result.notes_created > 0

    inbox_files = list((note_writer.vault_path / "00_Inbox").glob("*.md"))
    assert any("Page" in f.read_text(encoding="utf-8") for f in inbox_files)


def test_researcher_dedupes_existing_content(note_writer, fake_memory, fake_summarizer, fake_web_search):
    from agents.researcher_agent import ResearcherAgent

    # Pre-seed memory with the body the fake web search returns.
    body = "This is a long enough body of text describing the topic in detail. " * 20
    fake_memory.add(body, {"title": "Already there"})

    agent = ResearcherAgent(
        note_writer=note_writer,
        memory=fake_memory,
        summarizer=fake_summarizer,
        web_search=fake_web_search,
        config={"max_sources_per_topic": 2},
        topics_config={"settings": {"dedup_threshold": 0.5, "max_notes_per_run": 50}},
    )
    result = agent.run(topics=["X"])
    # All sources duplicate, so no notes should be created.
    assert result.notes_created == 0


def test_researcher_logs_to_vault(note_writer, fake_memory, fake_summarizer, fake_web_search):
    from agents.researcher_agent import ResearcherAgent

    agent = ResearcherAgent(
        note_writer=note_writer,
        memory=fake_memory,
        summarizer=fake_summarizer,
        web_search=fake_web_search,
        topics_config={"settings": {"dedup_threshold": 0.99, "max_notes_per_run": 5}},
    )
    agent.run(topics=["agents"])

    log_dir = note_writer.vault_path / "07_Agents_Log"
    logs = list(log_dir.glob("*.md"))
    assert logs
    assert any("Agent Log: researcher" in p.read_text(encoding="utf-8") for p in logs)


# ============================================================== Tagger


def test_tagger_tags_untagged_notes(note_writer, fake_summarizer):
    from agents.tagger_agent import TaggerAgent

    note_writer.create_note("00_Inbox", "Untagged Note", "Some body content here.", {"tags": []})
    note_writer.create_note("01_Atomic_Notes", "Tagged Note", "Other body.", {"tags": ["existing"]})

    tagger = TaggerAgent(note_writer=note_writer, summarizer=fake_summarizer)
    result = tagger.tag_all_untagged()
    assert result.notes_updated == 1


def test_tagger_normalizes_existing_tags(note_writer, fake_summarizer):
    from agents.tagger_agent import TaggerAgent

    note_writer.create_note(
        "00_Inbox",
        "Bad Tags",
        "body",
        {"tags": ["#FOO", "Bar Baz", "  qux  ", "FOO"]},
    )
    tagger = TaggerAgent(note_writer=note_writer, summarizer=fake_summarizer)
    tagger.normalize_tags()
    note = next(note_writer.iter_notes(folder="00_Inbox"))
    tags = note.metadata.get("tags") or []
    assert "foo" in tags and "bar-baz" in tags and "qux" in tags
    assert tags.count("foo") == 1


def test_tagger_tag_cloud(note_writer, fake_summarizer):
    from agents.tagger_agent import TaggerAgent

    note_writer.create_note("00_Inbox", "A", "x", {"tags": ["alpha", "beta"]})
    note_writer.create_note("01_Atomic_Notes", "B", "y", {"tags": ["alpha"]})
    cloud = TaggerAgent(note_writer=note_writer, summarizer=fake_summarizer).generate_tag_cloud()
    assert cloud.get("alpha") == 2
    assert cloud.get("beta") == 1


# ============================================================== Curator


def test_curator_process_inbox_moves_typed_notes(note_writer, fake_memory, fake_summarizer):
    from agents.curator_agent import CuratorAgent

    note_writer.create_note("00_Inbox", "Atom", "x", {"type": "atomic", "tags": ["test"]})
    note_writer.create_note("00_Inbox", "Capture", "x", {"type": "capture"})
    curator = CuratorAgent(note_writer=note_writer, memory=fake_memory, summarizer=fake_summarizer)
    result = curator.process_inbox()
    assert result.notes_updated >= 1
    assert (note_writer.vault_path / "01_Atomic_Notes").rglob("Atom*.md")


def test_curator_update_mocs(note_writer, fake_memory, fake_summarizer):
    from agents.curator_agent import CuratorAgent

    note_writer.create_note("01_Atomic_Notes", "N1", "body", {"tags": ["python"], "type": "atomic"})
    note_writer.create_note("01_Atomic_Notes", "N2", "body", {"tags": ["python"], "type": "atomic"})
    note_writer.create_note("01_Atomic_Notes", "N3", "body", {"tags": ["rust"], "type": "atomic"})

    curator = CuratorAgent(note_writer=note_writer, memory=fake_memory, summarizer=fake_summarizer)
    result = curator.update_mocs()
    assert result.success
    moc_dir = note_writer.vault_path / "02_MOC"
    mocs = list(moc_dir.glob("*.md"))
    assert any("Python" in m.read_text(encoding="utf-8") for m in mocs)


def test_curator_generate_connections(note_writer, fake_memory, fake_summarizer):
    from agents.curator_agent import CuratorAgent

    p1 = note_writer.create_note(
        "01_Atomic_Notes", "Topic A", "alpha beta gamma", {"type": "atomic", "tags": ["a"]}
    )
    p2 = note_writer.create_note(
        "01_Atomic_Notes", "Topic B", "alpha beta delta", {"type": "atomic", "tags": ["a"]}
    )
    fake_memory.add("alpha beta gamma", {"title": "Topic A"})
    fake_memory.add("alpha beta delta", {"title": "Topic B"})

    curator = CuratorAgent(note_writer=note_writer, memory=fake_memory, summarizer=fake_summarizer)
    result = curator.generate_connections(threshold=0.0)
    assert result.success


# ============================================================== ArxivAgent


def test_arxiv_agent_writes_paper_note(note_writer, fake_memory, fake_summarizer):
    from agents.arxiv_agent import ArxivAgent

    class _FakeArxiv:
        def search(self, query, max_results=5, days_back=None):
            return [
                {
                    "id": "2401.12345",
                    "title": "A Test Paper",
                    "abstract": "We describe a novel approach to testing.",
                    "authors": ["Alice", "Bob"],
                    "url": "http://arxiv.org/abs/2401.12345",
                    "pdf_url": "http://arxiv.org/pdf/2401.12345",
                    "date": "2024-01-15",
                    "categories": ["cs.AI"],
                }
            ]

        def fetch_paper(self, arxiv_id):
            return self.search("")[0]

    agent = ArxivAgent(
        note_writer=note_writer,
        memory=fake_memory,
        summarizer=fake_summarizer,
        arxiv_fetcher=_FakeArxiv(),
        config={"days_back": 7, "max_papers_per_topic": 1},
        topics_config={"domains": [{"name": "AI", "sources": {"arxiv": True}, "subtopics": ["ML"]}]},
    )
    result = agent.monitor_topics()
    assert result.notes_created == 1
    papers = list((note_writer.vault_path / "04_Resources/Papers").glob("*.md"))
    assert papers
    assert "2401.12345" in papers[0].read_text(encoding="utf-8")


# ============================================================== Orchestrator


def test_orchestrator_status(monkeypatch, tmp_path):
    """Verifies the orchestrator can be constructed and reports a status dict."""
    from agents.orchestrator import Orchestrator
    from config.settings import Settings

    settings = Settings(
        vault_path=tmp_path / "vault",
        chroma_dir=tmp_path / "chroma",
        embed_cache_dir=tmp_path / "cache",
    )
    settings.ensure_dirs()
    (settings.vault_path / "00_Inbox").mkdir(parents=True, exist_ok=True)

    pytest.importorskip("chromadb", reason="chromadb not installed")
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

    orch = Orchestrator(settings=settings)
    status = orch.get_status()
    assert "vault_path" in status
    assert "memory_stats" in status
    assert status["memory_stats"]["count"] == 0
