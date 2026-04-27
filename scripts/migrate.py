"""Memory migration utilities.

Sub-commands:
    rebuild-index   Rebuild the ChromaDB collection from every Markdown
                    note in the vault.
    export-json     Dump every memory entry to ``memory/export.json``.
    import-json     Re-ingest a previously exported JSON file.
    drop            Delete the entire collection (irreversible — prompts
                    unless ``--yes`` is provided).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from skills.note_writer import NoteWriter  # noqa: E402
from skills.vector_memory import VectorMemory  # noqa: E402


def _memory() -> VectorMemory:
    s = get_settings()
    return VectorMemory(
        persist_dir=s.chroma_dir,
        collection_name="second_brain",
        embedding_model=s.embed_model,
    )


def cmd_rebuild_index(args: argparse.Namespace) -> int:
    s = get_settings()
    note_writer = NoteWriter(s.vault_path)
    memory = _memory()

    if args.reset:
        memory.reset()

    count = 0
    for note in note_writer.iter_notes():
        if (note.metadata.get("type") or "") == "log":
            continue
        text = note.body.strip()
        if not text:
            continue
        memory.add(
            text=text,
            metadata={
                "title": note.metadata.get("title") or note.path.stem,
                "type": note.metadata.get("type") or "atomic",
                "tags": note.metadata.get("tags") or [],
                "source": note.metadata.get("source") or "",
                "url": note.metadata.get("source") or "",
                "agent": note.metadata.get("agent") or "migration",
                "confidence": float(note.metadata.get("confidence") or 0.6),
                "note_path": str(note.path.relative_to(s.vault_path)),
            },
        )
        count += 1
    print(f"Rebuilt index from {count} note(s).")
    print(memory.get_collection_stats())
    return 0


def cmd_export_json(args: argparse.Namespace) -> int:
    memory = _memory()
    s = get_settings()
    target = Path(args.output) if args.output else (s.project_root / "memory" / "export.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    ids = memory.all_ids()
    rows = []
    for doc_id in ids:
        entry = memory.get(doc_id)
        if entry:
            rows.append(entry)
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(rows)} entries -> {target}")
    return 0


def cmd_import_json(args: argparse.Namespace) -> int:
    memory = _memory()
    src = Path(args.input)
    rows = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        print("Input must be a JSON list", file=sys.stderr)
        return 1
    items = []
    for row in rows:
        text = row.get("document") or row.get("text")
        if not text:
            continue
        items.append({"text": text, "metadata": row.get("metadata") or {}, "id": row.get("id")})
    n = len(memory.add_many(items))
    print(f"Imported {n} entries from {src}")
    return 0


def cmd_drop(args: argparse.Namespace) -> int:
    if not args.yes:
        ans = input("Drop the entire memory collection? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 1
    memory = _memory()
    memory.reset()
    print("Collection dropped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="migrate", description="Second Brain memory utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rebuild = sub.add_parser("rebuild-index", help="Rebuild ChromaDB from the vault")
    p_rebuild.add_argument("--reset", action="store_true", help="Drop the collection before rebuilding")
    p_rebuild.set_defaults(func=cmd_rebuild_index)

    p_export = sub.add_parser("export-json", help="Export the collection to JSON")
    p_export.add_argument("-o", "--output", default=None)
    p_export.set_defaults(func=cmd_export_json)

    p_import = sub.add_parser("import-json", help="Import a JSON dump")
    p_import.add_argument("input")
    p_import.set_defaults(func=cmd_import_json)

    p_drop = sub.add_parser("drop", help="Drop the entire collection")
    p_drop.add_argument("--yes", action="store_true", help="Skip confirmation")
    p_drop.set_defaults(func=cmd_drop)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
