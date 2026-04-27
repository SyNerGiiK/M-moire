"""Curator agent: organize, dedup, link, and maintain MOC indexes."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from agents.base import AgentResult, BaseAgent
from skills.note_writer import Note, NoteWriter
from skills.summarizer import Summarizer
from skills.vector_memory import VectorMemory


_TYPE_TO_FOLDER = {
    "paper": "04_Resources/Papers",
    "moc": "02_MOC",
    "log": "07_Agents_Log",
    "atomic": "01_Atomic_Notes",
    "summary": "01_Atomic_Notes",
    "capture": "00_Inbox",
}


class CuratorAgent(BaseAgent):
    name = "curator"

    def __init__(
        self,
        note_writer: NoteWriter,
        memory: VectorMemory,
        summarizer: Summarizer,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(note_writer)
        self.memory = memory
        self.summarizer = summarizer
        self.config = config or {}

    # ============================================================== inbox

    def process_inbox(self) -> AgentResult:
        result = self._new_result()
        moved: list[dict[str, Any]] = []
        for note in self.note_writer.iter_notes(folder="00_Inbox"):
            target_folder = self._classify(note)
            if target_folder is None or target_folder.startswith("00_Inbox"):
                continue
            try:
                new_path = self.note_writer.move_note(note.path, target_folder)
                result.notes_updated += 1
                moved.append({"title": note.title, "path": new_path})
            except Exception as exc:
                result.add_error(f"move({note.path.name}): {exc}")
        summary = f"Moved {result.notes_updated} note(s) out of the inbox."
        self.write_log(result, summary=summary, created_notes=moved)
        return result

    def _classify(self, note: Note) -> str | None:
        ntype = (note.metadata.get("type") or "").lower()
        if ntype in _TYPE_TO_FOLDER:
            return _TYPE_TO_FOLDER[ntype]
        # Heuristic: title containing 'MOC' -> moc folder.
        title = (note.metadata.get("title") or note.path.stem).lower()
        if title.startswith("moc:") or " moc " in f" {title} ":
            return "02_MOC"
        # Default: promote tagged content to atomic notes.
        if note.metadata.get("tags") or note.metadata.get("agent_generated"):
            return "01_Atomic_Notes"
        return None

    # ============================================================== promote

    def promote_note(self, note_path: str, target_folder: str) -> str:
        return self.note_writer.move_note(note_path, target_folder)

    # ============================================================== dedup

    def deduplicate_vault(self, threshold: float | None = None) -> AgentResult:
        result = self._new_result()
        if threshold is None:
            threshold = float(self.config.get("dedup_threshold", 0.95))

        # We dedup at the memory level (semantic) — kept entry wins.
        try:
            removed = self.memory.deduplicate(threshold=threshold)
        except Exception as exc:
            result.add_error(f"memory.deduplicate: {exc}")
            removed = 0
        result.extra["memory_entries_removed"] = removed

        # Vault-level dedup: identical titles in the inbox -> archive duplicates.
        seen_titles: dict[str, Path] = {}
        duplicates: list[Path] = []
        for note in self.note_writer.iter_notes(folder="00_Inbox"):
            title = (note.metadata.get("title") or note.path.stem).strip().lower()
            if title in seen_titles:
                duplicates.append(note.path)
            else:
                seen_titles[title] = note.path
        for path in duplicates:
            try:
                self.note_writer.move_note(path, "05_Archive")
                result.notes_updated += 1
            except Exception as exc:
                result.add_error(f"archive({path.name}): {exc}")

        summary = (
            f"Removed {removed} memory duplicate(s); archived {len(duplicates)} "
            f"duplicate inbox note(s)."
        )
        self.write_log(result, summary=summary)
        return result

    # =========================================================== connections

    def generate_connections(self, max_per_note: int = 5, threshold: float = 0.3) -> AgentResult:
        result = self._new_result()
        link_count = 0
        notes = list(self.note_writer.iter_notes())
        note_index = [
            {
                "title": n.metadata.get("title") or n.path.stem,
                "path": str(n.path),
                "snippet": n.body[:400],
            }
            for n in notes
        ]
        for note in notes:
            if (note.metadata.get("type") or "").lower() == "log":
                continue
            text = f"{note.metadata.get('title', '')}\n\n{note.body}"
            try:
                hits = self.memory.search(text, n_results=max_per_note + 1)
            except Exception as exc:
                result.add_error(f"memory.search({note.path.name}): {exc}")
                continue
            existing_links = set(note.metadata.get("links") or [])
            added = 0
            for hit in hits:
                if added >= max_per_note:
                    break
                if hit["similarity"] < threshold:
                    continue
                target_title = hit["metadata"].get("title")
                if not target_title or target_title == note.metadata.get("title"):
                    continue
                if target_title in existing_links:
                    continue
                target_path = _find_note_by_title(note_index, target_title)
                if target_path is None:
                    continue
                try:
                    self.note_writer.link_notes(note.path, target_path)
                    existing_links.add(target_title)
                    added += 1
                    link_count += 1
                except Exception as exc:
                    result.add_error(f"link_notes({note.path.name}->{target_title}): {exc}")
            if added:
                result.notes_updated += 1
        result.extra["links_added"] = link_count
        self.write_log(
            result,
            summary=f"Added {link_count} wiki-link(s) across {result.notes_updated} note(s).",
        )
        return result

    # =============================================================== mocs

    def update_mocs(self) -> AgentResult:
        """Group atomic notes by their first tag and (re)generate a MOC per group."""
        result = self._new_result()
        groups: dict[str, list[Note]] = defaultdict(list)
        for note in self.note_writer.iter_notes(folder="01_Atomic_Notes"):
            tags = note.metadata.get("tags") or []
            if not tags:
                continue
            primary_tag = _normalize(tags[0])
            if primary_tag:
                groups[primary_tag].append(note)

        for tag, members in groups.items():
            if len(members) < 2:
                continue
            title = f"MOC: {tag.replace('-', ' ').title()}"
            body = _render_moc_body(title, members)
            metadata = {
                "title": title,
                "type": "moc",
                "domain": tag,
                "tags": [tag, "moc"],
                "agent_generated": True,
                "agent": self.name,
            }
            existing = self.note_writer.get_note(title, folder="02_MOC")
            try:
                if existing is None:
                    self.note_writer.create_note(
                        folder="02_MOC",
                        title=title,
                        content=body,
                        metadata=metadata,
                    )
                    result.notes_created += 1
                else:
                    self.note_writer.update_note(existing.path, content=body, metadata=metadata)
                    result.notes_updated += 1
            except Exception as exc:
                result.add_error(f"moc({tag}): {exc}")
        self.write_log(
            result,
            summary=(
                f"MOC sync: created {result.notes_created}, updated {result.notes_updated}."
            ),
        )
        return result


# ====================================================================
# Helpers
# ====================================================================


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "-", str(value).lower()).strip("-")


def _render_moc_body(title: str, notes: list[Note]) -> str:
    parts = [f"# {title}", "", "## Overview", ""]
    parts.append(f"This Map of Content lists every atomic note tagged with the primary domain.")
    parts.append("")
    parts.append("## Notes")
    for note in sorted(notes, key=lambda n: (n.metadata.get("title") or n.path.stem).lower()):
        link_title = note.path.stem
        display = note.metadata.get("title") or link_title
        parts.append(f"- [[{link_title}|{display}]]")
    parts.extend(
        [
            "",
            "## Sub-domains",
            "",
            "## Resources",
            "",
            f"_Auto-generated by curator agent on {datetime.now().strftime('%Y-%m-%d %H:%M')}._",
            "",
        ]
    )
    return "\n".join(parts)


def _find_note_by_title(index: list[dict[str, Any]], title: str) -> str | None:
    for entry in index:
        if entry["title"] == title:
            return entry["path"]
    return None
