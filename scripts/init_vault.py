"""Create the canonical vault folder structure and seed the templates.

Idempotent: re-running it never destroys existing notes — it only adds
folders, ``.gitkeep`` markers, and the template files in ``06_Templates``
when they are missing.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow `python scripts/init_vault.py` from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402

VAULT_FOLDERS = [
    "00_Inbox",
    "01_Atomic_Notes",
    "02_MOC",
    "03_Projects",
    "04_Resources/Papers",
    "04_Resources/Attachments",
    "05_Archive",
    "06_Templates",
    "07_Agents_Log",
]

OBSIDIAN_FOLDERS = [
    ".obsidian",
    ".obsidian/plugins/dataview",
]


def _seed_template(repo_template_dir: Path, vault_dir: Path, name: str) -> bool:
    src = repo_template_dir / name
    dst = vault_dir / "06_Templates" / name
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _seed_obsidian_config(repo_obsidian: Path, vault_obsidian: Path) -> int:
    if not repo_obsidian.exists():
        return 0
    n = 0
    for src in repo_obsidian.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(repo_obsidian)
        dst = vault_obsidian / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def main() -> int:
    settings = get_settings()
    vault = settings.vault_path
    vault.mkdir(parents=True, exist_ok=True)

    print(f"Initialising vault at: {vault}")

    for folder in VAULT_FOLDERS:
        path = vault / folder
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.touch()
    for folder in OBSIDIAN_FOLDERS:
        (vault / folder).mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    repo_templates = repo_root / "vault" / "06_Templates"
    repo_obsidian = repo_root / "vault" / ".obsidian"

    seeded = 0
    if repo_templates.exists() and repo_templates.resolve() != (vault / "06_Templates").resolve():
        for tpl in repo_templates.glob("*.md"):
            if _seed_template(repo_templates, vault, tpl.name):
                seeded += 1

    if repo_obsidian.resolve() != (vault / ".obsidian").resolve():
        seeded += _seed_obsidian_config(repo_obsidian, vault / ".obsidian")

    print(f"Vault ready. Seeded {seeded} template/config file(s).")
    print("Folders:")
    for folder in VAULT_FOLDERS:
        print(f"  - {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
