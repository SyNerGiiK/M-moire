"""LLM-powered summarization, tagging, and connection finding.

The :class:`Summarizer` defaults to Ollama (local, no API key) and falls
back to the Anthropic API when Ollama is unavailable and an API key is
configured. All methods degrade gracefully — if no LLM is reachable the
caller still gets a deterministic, useful (if simpler) result.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from loguru import logger


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "by", "with", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "from", "into", "about", "we",
    "you", "they", "i", "he", "she", "their", "his", "her", "our", "your",
    "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "donc", "car",
    "que", "qui", "quoi", "dont", "où", "à", "au", "aux", "de", "du", "en",
    "sur", "sous", "par", "pour", "avec", "sans", "dans", "ce", "cette", "ces",
    "il", "elle", "ils", "elles", "nous", "vous", "je", "tu", "se", "son",
    "sa", "ses", "notre", "votre", "leur", "leurs", "est", "sont", "été",
}

_STYLE_PROMPTS = {
    "atomic": (
        "Summarize the text below into ONE atomic note: a single, self-contained "
        "idea expressed in 3-6 sentences. Use clear declarative language. Do "
        "not add bullet points unless quoting a list. Output Markdown only."
    ),
    "detailed": (
        "Produce a thorough Markdown summary of the text below: a brief 'Overview' "
        "paragraph, a 'Key Points' bullet list (5-10 items), and a 'Notable Quotes' "
        "section with direct quotes (3 max)."
    ),
    "bullet_points": (
        "Summarize the text below as 5-10 concise Markdown bullet points. Each "
        "bullet is a single sentence."
    ),
    "tldr": (
        "Write a TL;DR for the text below in a single sentence (max 35 words)."
    ),
}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


def _truncate(text: str, max_chars: int = 12_000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n[...truncated...]\n\n{tail}"


class Summarizer:
    """LLM front-end with graceful degradation."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str | None = None,
        ollama_host: str = "http://localhost:11434",
        anthropic_api_key: str | None = None,
        anthropic_model: str = "claude-sonnet-4-6",
        language: str = "en",
        temperature: float = 0.3,
    ) -> None:
        self.provider = (provider or "ollama").lower()
        self.model = model or ("mistral" if self.provider == "ollama" else anthropic_model)
        self.ollama_host = ollama_host
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_model = anthropic_model
        self.language = language
        self.temperature = temperature
        self._ollama_ready: bool | None = None
        self._anthropic_client: Any | None = None

    # ----------------------------------------------------- provider helpers

    def _check_ollama(self) -> bool:
        if self._ollama_ready is not None:
            return self._ollama_ready
        try:
            import httpx

            r = httpx.get(f"{self.ollama_host}/api/tags", timeout=2.0)
            self._ollama_ready = r.status_code == 200
        except Exception:
            self._ollama_ready = False
        if not self._ollama_ready:
            logger.warning("Ollama is not reachable at {}.", self.ollama_host)
        return self._ollama_ready

    def _ollama_generate(self, prompt: str, system: str | None = None) -> str:
        import ollama

        client = ollama.Client(host=self.ollama_host)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return (response.get("message") or {}).get("content", "").strip()

    def _anthropic_client_lazy(self) -> Any | None:
        if self._anthropic_client is not None:
            return self._anthropic_client
        if not self.anthropic_api_key:
            return None
        try:
            import anthropic

            self._anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            return self._anthropic_client
        except Exception as exc:  # pragma: no cover
            logger.warning("Anthropic client unavailable: {}", exc)
            return None

    def _anthropic_generate(self, prompt: str, system: str | None = None) -> str:
        client = self._anthropic_client_lazy()
        if client is None:
            return ""
        message = client.messages.create(
            model=self.anthropic_model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system or "You are a precise, concise knowledge curator.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate text parts.
        parts = []
        for block in message.content or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    def _generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        # Primary: Ollama (if configured and reachable).
        if self.provider == "ollama" and self._check_ollama():
            try:
                text = self._ollama_generate(prompt, system=system)
                if text:
                    return LLMResponse(text=text, provider="ollama", model=self.model)
            except Exception as exc:
                logger.warning("Ollama generation failed: {}", exc)
        # Fallback / primary: Anthropic.
        if self.anthropic_api_key:
            try:
                text = self._anthropic_generate(prompt, system=system)
                if text:
                    return LLMResponse(text=text, provider="anthropic", model=self.anthropic_model)
            except Exception as exc:
                logger.warning("Anthropic generation failed: {}", exc)
        # Last resort: empty string. Callers handle this as "no LLM available".
        return LLMResponse(text="", provider="none", model="none")

    # ---------------------------------------------------------------- public

    def summarize(self, text: str, style: str = "atomic") -> str:
        if not text or not text.strip():
            return ""
        style_key = style if style in _STYLE_PROMPTS else "atomic"
        instruction = _STYLE_PROMPTS[style_key]
        prompt = (
            f"{instruction}\n\n"
            f"Respond in {self.language!r}.\n\n"
            f"--- TEXT START ---\n{_truncate(text)}\n--- TEXT END ---\n"
        )
        result = self._generate(prompt, system="You are a precise knowledge curator.")
        if result.text:
            return result.text
        # Deterministic fallback: extract first 3 sentences.
        return _heuristic_summary(text, sentences=3)

    def extract_key_concepts(self, text: str, max_concepts: int = 8) -> list[str]:
        if not text or not text.strip():
            return []
        prompt = (
            f"List the {max_concepts} most important concepts in the text below. "
            f"Output a JSON array of short noun phrases (1-4 words each). No prose, no markdown.\n\n"
            f"--- TEXT ---\n{_truncate(text, 8000)}\n--- END ---"
        )
        result = self._generate(prompt)
        concepts = _parse_json_list(result.text)
        if concepts:
            return concepts[:max_concepts]
        return _heuristic_concepts(text, max_concepts)

    def generate_tags(
        self,
        text: str,
        existing_tags: list[str] | None = None,
        max_tags: int = 6,
    ) -> list[str]:
        if not text or not text.strip():
            return []
        existing = ", ".join(existing_tags or [])[:500]
        prompt = (
            f"Generate at most {max_tags} concise tags for the text below. "
            f"Tags must be lowercase, single words or hyphenated (no spaces), no '#' prefix. "
            f"Reuse any of these existing tags when relevant: [{existing}]. "
            f"Output a JSON array. No prose.\n\n"
            f"--- TEXT ---\n{_truncate(text, 6000)}\n--- END ---"
        )
        result = self._generate(prompt)
        tags = _parse_json_list(result.text)
        if not tags:
            tags = _heuristic_tags(text, max_tags)
        return [_normalize_tag(t) for t in tags if t][:max_tags]

    def find_connections(
        self,
        note_content: str,
        vault_notes: list[dict[str, Any]],
        max_connections: int = 5,
    ) -> list[str]:
        """Return titles from ``vault_notes`` likely related to ``note_content``."""
        if not note_content.strip() or not vault_notes:
            return []
        sample = []
        for n in vault_notes[:50]:
            title = n.get("title") or ""
            snippet = (n.get("snippet") or n.get("content") or "")[:200]
            if title:
                sample.append(f"- {title}: {snippet}")
        if not sample:
            return []
        joined = "\n".join(sample)
        prompt = (
            f"From the candidate notes below, pick the {max_connections} most "
            f"semantically related to the target. Output a JSON array of the "
            f"chosen titles (exact match), no prose.\n\n"
            f"TARGET:\n{_truncate(note_content, 3000)}\n\n"
            f"CANDIDATES:\n{joined}\n"
        )
        result = self._generate(prompt)
        picks = _parse_json_list(result.text)
        if picks:
            valid = {n.get("title") for n in vault_notes}
            return [p for p in picks if p in valid][:max_connections]
        # Fallback: naive keyword overlap.
        return _heuristic_connections(note_content, vault_notes, max_connections)


# ====================================================================
# Deterministic heuristics — used when no LLM is reachable.
# ====================================================================

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÉÈÀÎÔÙ])")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-]+")


def _heuristic_summary(text: str, sentences: int = 3) -> str:
    parts = _SENT_SPLIT.split(text.strip())
    return " ".join(parts[:sentences]).strip()


def _heuristic_concepts(text: str, max_concepts: int) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    counts = Counter(w for w in words if len(w) > 4 and w not in _STOPWORDS)
    return [w for w, _ in counts.most_common(max_concepts)]


def _heuristic_tags(text: str, max_tags: int) -> list[str]:
    return _heuristic_concepts(text, max_tags)


def _heuristic_connections(
    note_content: str,
    candidates: list[dict[str, Any]],
    max_connections: int,
) -> list[str]:
    target_words = set(_heuristic_concepts(note_content, 30))
    scored: list[tuple[float, str]] = []
    for note in candidates:
        title = note.get("title") or ""
        body = note.get("snippet") or note.get("content") or ""
        words = set(_heuristic_concepts(f"{title} {body}", 30))
        if not words:
            continue
        overlap = len(target_words & words)
        if overlap:
            scored.append((overlap, title))
    scored.sort(reverse=True)
    return [title for _, title in scored[:max_connections]]


def _normalize_tag(tag: str) -> str:
    tag = (tag or "").strip().lstrip("#").lower()
    tag = re.sub(r"[^a-z0-9\-]+", "-", tag).strip("-")
    return tag[:40]


def _parse_json_list(text: str) -> list[str]:
    if not text:
        return []
    # Try strict JSON, then bracket extraction.
    import json

    candidate = text.strip()
    for chunk in (candidate, _extract_bracketed(candidate)):
        if not chunk:
            continue
        try:
            data = json.loads(chunk)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            continue
    # Fallback: line-delimited list.
    lines = [
        re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip().strip('"').strip("'")
        for line in candidate.splitlines()
    ]
    return [l for l in lines if l]


def _extract_bracketed(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start : end + 1]
