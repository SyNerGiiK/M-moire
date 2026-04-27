"""Read/write Markdown notes inside the Obsidian vault.

The :class:`NoteWriter` is the only component that touches the filesystem
inside ``vault/``. It enforces:

* YAML frontmatter for every note (``--- ... ---``).
* Stable, filesystem-safe filenames.
* Deterministic timestamp-based IDs.

It does **not** know about ChromaDB or LLMs; agents combine the two.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from loguru import logger


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_FILENAME_INVALID = re.compile(r"[^a-zA-Z0-9._\- ]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class Note:
    path: Path
    metadata: dict[str, Any]
    body: str

    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or self.path.stem)

    def to_markdown(self) -> str:
        return _render(self.metadata, self.body)


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _now_iso(minute: bool = True) -> str:
    fmt = "%Y-%m-%d %H:%M" if minute else "%Y-%m-%d"
    return datetime.now().strftime(fmt)


def _render(metadata: dict[str, Any], body: str) -> str:
    fm = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
    body = body.rstrip() + "\n"
    return f"---\n{fm}\n---\n\n{body}"


def _parse(content: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
        if not isinstance(metadata, dict):
            metadata = {}
    except yaml.YAMLError:
        metadata = {}
    body = content[match.end() :]
    return metadata, body


class NoteWriter:
    """Writes and reads Markdown notes inside an Obsidian vault."""

    def __init__(self, vault_path: str | Path) -> None:
        self.vault_path = Path(vault_path).resolve()
        self.vault_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def folder(self, name: str) -> Path:
        path = (self.vault_path / name).resolve()
        if not str(path).startswith(str(self.vault_path)):
            raise ValueError(f"Folder {name!r} escapes vault root")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sanitize_filename(self, title: str) -> str:
        if not title:
            return "untitled"
        # Strip accents -> ASCII for portable filenames while keeping the visible title.
        normalized = unicodedata.normalize("NFKD", title)
        ascii_title = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = _FILENAME_INVALID.sub(" ", ascii_title)
        cleaned = _WHITESPACE.sub(" ", cleaned).strip()
        cleaned = cleaned.replace(" ", "_")
        if not cleaned:
            cleaned = "untitled"
        return cleaned[:120]

    def _unique_path(self, folder: Path, base_name: str, ext: str = ".md") -> Path:
        candidate = folder / f"{base_name}{ext}"
        if not candidate.exists():
            return candidate
        i = 2
        while True:
            candidate = folder / f"{base_name}_{i}{ext}"
            if not candidate.exists():
                return candidate
            i += 1

    # ------------------------------------------------------------------ writes

    def create_note(
        self,
        folder: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new note inside ``folder`` and return its path."""
        meta = dict(metadata or {})
        meta.setdefault("id", _now_id())
        meta.setdefault("title", title)
        meta.setdefault("created", _now_iso())
        meta.setdefault("updated", meta["created"])
        meta.setdefault("tags", meta.get("tags") or [])
        meta.setdefault("type", meta.get("type", "atomic"))
        meta.setdefault("agent_generated", meta.get("agent_generated", False))

        target_folder = self.folder(folder)
        base_name = self.sanitize_filename(title)
        path = self._unique_path(target_folder, base_name)
        path.write_text(_render(meta, content), encoding="utf-8")
        logger.debug("Created note: {}", path.relative_to(self.vault_path))
        return str(path)

    def update_note(
        self,
        note_path: str | Path,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Update either body or frontmatter (or both) of an existing note."""
        path = Path(note_path)
        if not path.is_absolute():
            path = (self.vault_path / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        existing_meta, existing_body = _parse(path.read_text(encoding="utf-8"))
        new_meta = {**existing_meta, **(metadata or {})}
        new_meta["updated"] = _now_iso()
        new_body = content if content is not None else existing_body
        path.write_text(_render(new_meta, new_body), encoding="utf-8")
        logger.debug("Updated note: {}", path.relative_to(self.vault_path))
        return str(path)

    def append_to_inbox(self, title: str, content: str, source: str = "") -> str:
        """Quick-capture helper: drops a Markdown note into ``00_Inbox/``."""
        return self.create_note(
            folder="00_Inbox",
            title=title,
            content=content,
            metadata={
                "source": source,
                "type": "capture",
                "tags": ["inbox"],
            },
        )

    def link_notes(self, source_note: str | Path, target_note: str | Path) -> None:
        """Append a ``[[wikilink]]`` to ``source_note`` pointing at ``target_note``."""
        src = Path(source_note)
        tgt = Path(target_note)
        if not src.is_absolute():
            src = (self.vault_path / src).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        target_title = tgt.stem
        meta, body = _parse(src.read_text(encoding="utf-8"))
        wikilink = f"[[{target_title}]]"
        if wikilink in body:
            return
        # Update the YAML 'links' list as well.
        links = list(meta.get("links") or [])
        if target_title not in links:
            links.append(target_title)
            meta["links"] = links
        if "## Connections" in body:
            body = body.replace(
                "## Connections",
                f"## Connections\n- Related: {wikilink}",
                1,
            )
        else:
            body = body.rstrip() + f"\n\n## Connections\n- Related: {wikilink}\n"
        meta["updated"] = _now_iso()
        src.write_text(_render(meta, body), encoding="utf-8")

    # ------------------------------------------------------------------- reads

    def note_exists(self, title: str, folder: str | None = None) -> bool:
        return self.get_note(title, folder=folder) is not None

    def get_note(self, title: str, folder: str | None = None) -> Note | None:
        base_name = self.sanitize_filename(title)
        roots = [self.folder(folder)] if folder else [self.vault_path]
        for root in roots:
            for candidate in root.rglob(f"{base_name}*.md"):
                meta, body = _parse(candidate.read_text(encoding="utf-8"))
                if (meta.get("title") or candidate.stem).strip() == title.strip():
                    return Note(path=candidate, metadata=meta, body=body)
                if candidate.stem == base_name:
                    return Note(path=candidate, metadata=meta, body=body)
        return None

    def read_note(self, note_path: str | Path) -> Note:
        path = Path(note_path)
        if not path.is_absolute():
            path = (self.vault_path / path).resolve()
        meta, body = _parse(path.read_text(encoding="utf-8"))
        return Note(path=path, metadata=meta, body=body)

    def iter_notes(self, folder: str | None = None) -> Iterable[Note]:
        root = self.folder(folder) if folder else self.vault_path
        for candidate in root.rglob("*.md"):
            if candidate.is_file() and ".obsidian" not in candidate.parts:
                meta, body = _parse(candidate.read_text(encoding="utf-8"))
                yield Note(path=candidate, metadata=meta, body=body)

    def move_note(self, note_path: str | Path, target_folder: str) -> str:
        src = Path(note_path)
        if not src.is_absolute():
            src = (self.vault_path / src).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        target = self.folder(target_folder)
        dest = self._unique_path(target, src.stem)
        src.rename(dest)
        return str(dest)

    # ----------------------------------------------------------------- helpers

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for note in self.iter_notes():
            rel = note.path.relative_to(self.vault_path)
            top = rel.parts[0] if rel.parts else "_root"
            counts[top] = counts.get(top, 0) + 1
        total = sum(counts.values())
        return {"total": total, "by_folder": counts, "vault": str(self.vault_path)}
