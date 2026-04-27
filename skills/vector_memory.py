"""ChromaDB-backed semantic memory.

The :class:`VectorMemory` class wraps ChromaDB and a sentence-transformers
embedding model. It is intentionally narrow: callers store text + metadata
and query by similarity. All higher-level concerns (deduplication policy,
note linking, etc.) live in the agents.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from loguru import logger


@dataclass
class MemoryStats:
    collection: str
    count: int
    persist_dir: str
    embedding_model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "count": self.count,
            "persist_dir": self.persist_dir,
            "embedding_model": self.embedding_model,
        }


def _stable_id(text: str, metadata: dict[str, Any] | None = None) -> str:
    """Deterministic ID derived from content + key metadata fields."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    if metadata:
        for key in ("source", "url", "type", "agent"):
            v = metadata.get(key)
            if v is not None:
                h.update(b"\x00")
                h.update(str(v).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:32]


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values. Coerce lists/dicts to JSON."""
    if not metadata:
        return {}
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False, default=str)
    return clean


def _denormalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_normalize_metadata` on read paths (best-effort)."""
    out: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, str) and value and value[0] in "[{":
            try:
                out[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        out[key] = value
    return out


class VectorMemory:
    """Persistent semantic memory backed by ChromaDB."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "second_brain",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Lazy imports keep CLI startup fast and let tests stub these out.
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "VectorMemory ready (collection={}, dir={}, count={})",
            collection_name,
            self.persist_dir,
            self._collection.count(),
        )

    # ------------------------------------------------------------------ writes

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Store ``text`` and return the document ID used."""
        if not text or not text.strip():
            raise ValueError("VectorMemory.add: text must be non-empty")
        meta = dict(metadata or {})
        meta.setdefault("created_at", int(time.time()))
        clean_meta = _normalize_metadata(meta)
        the_id = doc_id or _stable_id(text, meta)
        try:
            self._collection.upsert(
                ids=[the_id],
                documents=[text],
                metadatas=[clean_meta],
            )
        except Exception as exc:  # pragma: no cover — defensive logging
            logger.error("VectorMemory.add failed for id={}: {}", the_id, exc)
            raise
        return the_id

    def add_many(self, items: Iterable[dict[str, Any]]) -> list[str]:
        """Bulk insert. Each item: {text, metadata?, id?}."""
        ids, docs, metas = [], [], []
        for item in items:
            text = item.get("text") or ""
            if not text.strip():
                continue
            meta = dict(item.get("metadata") or {})
            meta.setdefault("created_at", int(time.time()))
            ids.append(item.get("id") or _stable_id(text, meta))
            docs.append(text)
            metas.append(_normalize_metadata(meta))
        if not ids:
            return []
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return ids

    # ------------------------------------------------------------------ reads

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``n_results`` semantically similar documents."""
        if not query or not query.strip():
            return []
        n_results = max(1, int(n_results))
        try:
            res = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("VectorMemory.search failed: {}", exc)
            return []

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: list[dict[str, Any]] = []
        for i in range(len(ids)):
            distance = float(dists[i]) if i < len(dists) else 0.0
            similarity = max(0.0, 1.0 - distance)  # cosine distance -> similarity
            out.append(
                {
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": _denormalize_metadata(metas[i] if i < len(metas) else {}),
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return out

    def get(self, doc_id: str) -> dict[str, Any] | None:
        try:
            res = self._collection.get(ids=[doc_id])
        except Exception:
            return None
        ids = res.get("ids") or []
        if not ids:
            return None
        return {
            "id": ids[0],
            "document": (res.get("documents") or [""])[0],
            "metadata": _denormalize_metadata((res.get("metadatas") or [{}])[0]),
        }

    def all_ids(self) -> list[str]:
        res = self._collection.get(include=[])
        return list(res.get("ids") or [])

    # ----------------------------------------------------------------- deletes

    def delete(self, doc_id: str | list[str]) -> None:
        ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not ids:
            return
        self._collection.delete(ids=ids)

    def reset(self) -> None:
        """Drop the collection. Mostly useful for tests."""
        self._client.delete_collection(self.collection_name)
        from chromadb.utils import embedding_functions

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
            ),
            metadata={"hnsw:space": "cosine"},
        )

    # ---------------------------------------------------------------- analysis

    def get_collection_stats(self) -> dict[str, Any]:
        return MemoryStats(
            collection=self.collection_name,
            count=self._collection.count(),
            persist_dir=str(self.persist_dir),
            embedding_model=self.embedding_model_name,
        ).as_dict()

    def is_duplicate(self, text: str, threshold: float = 0.90) -> dict[str, Any] | None:
        """Return the closest match if its similarity >= threshold, else None."""
        hits = self.search(text, n_results=1)
        if not hits:
            return None
        top = hits[0]
        return top if top["similarity"] >= threshold else None

    def deduplicate(self, threshold: float = 0.95) -> int:
        """Remove near-duplicate documents and return how many were dropped.

        Iterates documents in insertion order (as returned by Chroma), keeping
        the first occurrence of any cluster of items with similarity >=
        ``threshold``. Safe to run repeatedly.
        """
        bundle = self._collection.get(include=["documents", "metadatas"])
        ids = bundle.get("ids") or []
        docs = bundle.get("documents") or []
        if len(ids) <= 1:
            return 0

        kept: list[str] = []
        to_remove: list[str] = []
        for doc_id, document in zip(ids, docs):
            if not document:
                continue
            duplicate_of = None
            # Compare against already-kept ids only (cheap when collection grows).
            if kept:
                hits = self._collection.query(
                    query_texts=[document],
                    n_results=min(5, len(kept)),
                    where={"$and": [{"_dedup_marker": {"$ne": "__none__"}}]} if False else None,
                )
                hit_ids = (hits.get("ids") or [[]])[0]
                hit_dists = (hits.get("distances") or [[]])[0]
                for hid, hdist in zip(hit_ids, hit_dists):
                    if hid == doc_id:
                        continue
                    sim = 1.0 - float(hdist)
                    if hid in kept and sim >= threshold:
                        duplicate_of = hid
                        break
            if duplicate_of is None:
                kept.append(doc_id)
            else:
                to_remove.append(doc_id)

        if to_remove:
            logger.info("Deduplicating {} entries from collection", len(to_remove))
            self._collection.delete(ids=to_remove)
        return len(to_remove)


def fresh_id() -> str:
    """Convenience: opaque unique id for callers that don't have content yet."""
    return uuid.uuid4().hex
