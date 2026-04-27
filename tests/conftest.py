"""Shared fixtures.

The whole test suite avoids real network/LLM calls. Skills that hit the
internet are exercised through small fakes that match their public
interface. Heavy dependencies (chromadb, sentence_transformers) are
optional — tests that need them are skipped if the import fails.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Make project modules importable when running pytest from any cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------- vault


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    for sub in (
        "00_Inbox",
        "01_Atomic_Notes",
        "02_MOC",
        "03_Projects",
        "04_Resources/Papers",
        "05_Archive",
        "06_Templates",
        "07_Agents_Log",
    ):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


# --------------------------------------------------------------------- skills


@pytest.fixture
def note_writer(vault_dir: Path):
    from skills.note_writer import NoteWriter

    return NoteWriter(vault_dir)


@pytest.fixture
def fake_summarizer():
    """Deterministic stand-in for :class:`Summarizer` — no LLM calls."""

    class _Fake:
        def summarize(self, text: str, style: str = "atomic") -> str:
            text = (text or "").strip()
            return text[:280] or "(empty)"

        def extract_key_concepts(self, text: str) -> list[str]:
            words = sorted({w.lower() for w in text.split() if len(w) > 4})
            return words[:5]

        def generate_tags(self, text: str, existing_tags: list[str] | None = None) -> list[str]:
            return ["test", "auto"]

        def find_connections(self, content: str, vault_notes: list[dict[str, Any]], max_connections: int = 5) -> list[str]:
            return [n.get("title") for n in vault_notes[:max_connections] if n.get("title")]

    return _Fake()


@pytest.fixture
def fake_web_search():
    class _FakeWS:
        def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
            return [
                {"title": f"Result for {query}", "url": "https://example.com/a", "snippet": query},
                {"title": f"Other for {query}", "url": "https://example.com/b", "snippet": query},
            ][:max_results]

        def fetch_page(self, url: str) -> dict[str, Any]:
            return {
                "url": url,
                "title": f"Page {url}",
                "content": (
                    "This is a long enough body of text describing the topic in detail. " * 20
                ),
                "date": None,
                "word_count": 200,
            }

        def deep_search(self, query: str, depth: int = 2, max_per_level: int = 5) -> list[dict[str, Any]]:
            return [self.fetch_page(h["url"]) for h in self.search(query, max_per_level)]

    return _FakeWS()


@pytest.fixture
def fake_memory():
    """In-memory stand-in for VectorMemory — exposes the same public API."""

    class _FakeMemory:
        def __init__(self) -> None:
            self.entries: dict[str, dict[str, Any]] = {}

        def add(self, text: str, metadata: dict[str, Any] | None = None, doc_id: str | None = None) -> str:
            doc_id = doc_id or f"id-{len(self.entries) + 1}"
            self.entries[doc_id] = {"id": doc_id, "document": text, "metadata": dict(metadata or {})}
            return doc_id

        def add_many(self, items):
            return [self.add(i["text"], i.get("metadata"), i.get("id")) for i in items]

        def search(self, query: str, n_results: int = 10, filter: dict[str, Any] | None = None):
            results = []
            for entry in self.entries.values():
                if filter:
                    meta = entry["metadata"]
                    if not all(meta.get(k) == v for k, v in filter.items()):
                        continue
                # Trivial similarity: token overlap.
                q_tokens = set(query.lower().split())
                d_tokens = set(entry["document"].lower().split())
                overlap = len(q_tokens & d_tokens)
                results.append(
                    {
                        "id": entry["id"],
                        "document": entry["document"],
                        "metadata": entry["metadata"],
                        "distance": 1.0 - (overlap / max(1, len(q_tokens))),
                        "similarity": overlap / max(1, len(q_tokens)),
                    }
                )
            results.sort(key=lambda r: -r["similarity"])
            return results[:n_results]

        def get(self, doc_id: str):
            return self.entries.get(doc_id)

        def all_ids(self) -> list[str]:
            return list(self.entries)

        def delete(self, doc_id):
            ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
            for i in ids:
                self.entries.pop(i, None)

        def reset(self) -> None:
            self.entries.clear()

        def get_collection_stats(self) -> dict[str, Any]:
            return {
                "collection": "test",
                "count": len(self.entries),
                "persist_dir": "(memory)",
                "embedding_model": "fake",
            }

        def is_duplicate(self, text: str, threshold: float = 0.9):
            hits = self.search(text, n_results=1)
            if not hits:
                return None
            return hits[0] if hits[0]["similarity"] >= threshold else None

        def deduplicate(self, threshold: float = 0.95) -> int:
            seen: list[tuple[str, set[str]]] = []
            removed = 0
            to_remove: list[str] = []
            for entry in list(self.entries.values()):
                tokens = set(entry["document"].lower().split())
                duplicate = False
                for _, prev in seen:
                    if not tokens or not prev:
                        continue
                    overlap = len(tokens & prev) / max(len(tokens | prev), 1)
                    if overlap >= threshold:
                        duplicate = True
                        break
                if duplicate:
                    to_remove.append(entry["id"])
                    removed += 1
                else:
                    seen.append((entry["id"], tokens))
            for i in to_remove:
                self.entries.pop(i, None)
            return removed

    return _FakeMemory()
