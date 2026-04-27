"""Tests for the in-memory fake and the public VectorMemory shape.

The real ChromaDB-backed implementation is exercised end-to-end by
``test_skills_chromadb`` only when chromadb is importable. The fake
fixture preserves the same contract so agent tests don't need a real
vector store.
"""
from __future__ import annotations

import pytest


def test_fake_memory_add_and_search(fake_memory):
    fake_memory.add("the quick brown fox jumps", {"source": "src1", "title": "Fox"})
    fake_memory.add("a slow green turtle walks", {"source": "src2", "title": "Turtle"})
    hits = fake_memory.search("brown fox", n_results=2)
    assert hits
    assert hits[0]["metadata"]["title"] == "Fox"


def test_fake_memory_filter_by_source(fake_memory):
    fake_memory.add("alpha", {"source": "A", "title": "T1"})
    fake_memory.add("beta", {"source": "B", "title": "T2"})
    hits = fake_memory.search("alpha", filter={"source": "A"})
    assert all(h["metadata"]["source"] == "A" for h in hits)


def test_fake_memory_dedup(fake_memory):
    fake_memory.add("one two three four", {"title": "A"})
    fake_memory.add("one two three four", {"title": "A2"})
    fake_memory.add("nine ten eleven twelve", {"title": "B"})
    removed = fake_memory.deduplicate(threshold=0.99)
    assert removed == 1
    assert fake_memory.get_collection_stats()["count"] == 2


def test_fake_memory_is_duplicate(fake_memory):
    fake_memory.add("alpha beta gamma", {"title": "ABG"})
    dup = fake_memory.is_duplicate("alpha beta gamma", threshold=0.5)
    assert dup is not None and dup["metadata"]["title"] == "ABG"
    none_dup = fake_memory.is_duplicate("totally unrelated content", threshold=0.9)
    assert none_dup is None


def test_fake_memory_delete(fake_memory):
    fake_memory.add("hello", {"title": "Hello"})
    ids = fake_memory.all_ids()
    assert ids
    fake_memory.delete(ids[0])
    assert fake_memory.all_ids() == []


# -------- Real ChromaDB smoke test (skipped when not installed) --------


@pytest.mark.skipif(
    pytest.importorskip("chromadb", reason="chromadb not installed") is None,
    reason="chromadb not installed",
)
def test_real_vector_memory_roundtrip(tmp_path):
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
    from skills.vector_memory import VectorMemory

    vm = VectorMemory(persist_dir=tmp_path / "chroma", collection_name="test_collection")
    vm.add("solar panel installation guide", {"title": "Solar"})
    vm.add("introduction to quantum mechanics", {"title": "Quantum"})

    hits = vm.search("photovoltaic solar")
    assert hits
    assert hits[0]["metadata"]["title"] in {"Solar", "Quantum"}

    stats = vm.get_collection_stats()
    assert stats["count"] >= 2
