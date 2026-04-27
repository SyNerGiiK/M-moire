"""Auto-tagging agent.

Iterates the vault, finds notes with no/empty tags, and asks the
:class:`Summarizer` to propose tags. Also normalizes existing tag
spelling (lowercase, hyphenated, no '#' prefix).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from loguru import logger

from agents.base import AgentResult, BaseAgent
from skills.note_writer import NoteWriter
from skills.summarizer import Summarizer


_TAG_INVALID = re.compile(r"[^a-z0-9\-]+")


def _normalize_tag(tag: str) -> str:
    tag = (tag or "").strip().lstrip("#").lower()
    tag = _TAG_INVALID.sub("-", tag).strip("-")
    return tag[:40]


class TaggerAgent(BaseAgent):
    name = "tagger"

    def __init__(
        self,
        note_writer: NoteWriter,
        summarizer: Summarizer,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(note_writer)
        self.summarizer = summarizer
        self.config = config or {}

    # ------------------------------------------------------------------ API

    def suggest_tags(self, note_content: str, existing_tags: list[str] | None = None) -> list[str]:
        try:
            tags = self.summarizer.generate_tags(note_content, existing_tags=existing_tags) or []
        except Exception as exc:
            logger.warning("Tag generation failed: {}", exc)
            tags = []
        return [t for t in (_normalize_tag(t) for t in tags) if t][:8]

    def tag_all_untagged(self) -> AgentResult:
        result = self._new_result()
        existing = self.generate_tag_cloud()
        existing_list = [t for t, _ in sorted(existing.items(), key=lambda kv: -kv[1])][:50]

        for note in self.note_writer.iter_notes():
            if (note.metadata.get("type") or "") == "log":
                continue
            current_tags = note.metadata.get("tags") or []
            if current_tags:
                continue
            if not note.body.strip():
                continue
            tags = self.suggest_tags(
                f"{note.metadata.get('title', '')}\n\n{note.body}",
                existing_tags=existing_list,
            )
            if not tags:
                continue
            try:
                self.note_writer.update_note(
                    note.path,
                    metadata={**note.metadata, "tags": tags},
                )
                result.notes_updated += 1
            except Exception as exc:
                result.add_error(f"update({note.path.name}): {exc}")

        summary = f"Auto-tagged {result.notes_updated} previously untagged note(s)."
        self.write_log(result, summary=summary)
        return result

    def normalize_tags(self) -> AgentResult:
        result = self._new_result()
        for note in self.note_writer.iter_notes():
            current = note.metadata.get("tags") or []
            if not current:
                continue
            normalized: list[str] = []
            seen: set[str] = set()
            for tag in current:
                norm = _normalize_tag(str(tag))
                if norm and norm not in seen:
                    seen.add(norm)
                    normalized.append(norm)
            if normalized != current:
                try:
                    self.note_writer.update_note(
                        note.path,
                        metadata={**note.metadata, "tags": normalized},
                    )
                    result.notes_updated += 1
                except Exception as exc:
                    result.add_error(f"normalize({note.path.name}): {exc}")
        self.write_log(
            result,
            summary=f"Normalized tags on {result.notes_updated} note(s).",
        )
        return result

    def generate_tag_cloud(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for note in self.note_writer.iter_notes():
            for tag in note.metadata.get("tags") or []:
                norm = _normalize_tag(str(tag))
                if norm:
                    counter[norm] += 1
        return dict(counter)
