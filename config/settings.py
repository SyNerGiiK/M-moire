"""Global settings for the Second Brain.

Settings are loaded from environment variables (with sensible defaults) and
exposed via a :class:`Settings` dataclass. ``get_settings()`` returns a cached
instance so importing modules see a consistent configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env at import time so anything that reads os.environ later sees the values.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


@dataclass
class Settings:
    """Runtime configuration for every component of the Second Brain."""

    # Paths
    project_root: Path = PROJECT_ROOT
    vault_path: Path = field(default_factory=lambda: _resolve(_env("SECOND_BRAIN_VAULT_PATH", "./vault")))
    chroma_dir: Path = field(default_factory=lambda: _resolve(_env("SECOND_BRAIN_CHROMA_DIR", "./memory/chroma_db")))
    embed_cache_dir: Path = field(default_factory=lambda: _resolve(_env("SECOND_BRAIN_EMBED_CACHE", "./memory/embeddings_cache")))
    config_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "config")

    # LLM
    llm_provider: str = field(default_factory=lambda: _env("SECOND_BRAIN_LLM_PROVIDER", "ollama") or "ollama")
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434")
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "mistral") or "mistral")
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-6") or "claude-sonnet-4-6")

    # Embeddings
    embed_model: str = field(default_factory=lambda: _env("SECOND_BRAIN_EMBED_MODEL", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2")

    # Behaviour
    language: str = field(default_factory=lambda: _env("SECOND_BRAIN_LANG", "fr") or "fr")
    log_level: str = field(default_factory=lambda: (_env("SECOND_BRAIN_LOG_LEVEL", "INFO") or "INFO").upper())

    # Scheduler
    full_cycle_hours: int = field(default_factory=lambda: _env_int("SECOND_BRAIN_FULL_CYCLE_HOURS", 6))

    # API
    api_host: str = field(default_factory=lambda: _env("SECOND_BRAIN_API_HOST", "127.0.0.1") or "127.0.0.1")
    api_port: int = field(default_factory=lambda: _env_int("SECOND_BRAIN_API_PORT", 8765))

    # Optional providers
    serpapi_api_key: str | None = field(default_factory=lambda: _env("SERPAPI_API_KEY"))

    def ensure_dirs(self) -> None:
        """Create paths that downstream code expects to exist."""
        for path in (self.vault_path, self.chroma_dir, self.embed_cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    # --- YAML config loaders ------------------------------------------------

    def load_topics(self) -> dict[str, Any]:
        return _load_yaml(self.config_dir / "topics.yaml")

    def load_agents(self) -> dict[str, Any]:
        return _load_yaml(self.config_dir / "agents.yaml")

    def load_templates(self) -> dict[str, Any]:
        return _load_yaml(self.config_dir / "templates.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
