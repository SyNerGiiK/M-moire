"""Verify the connection to LM Studio.

Pings ``LMSTUDIO_BASE_URL/models``, prints the loaded model(s), then
runs a tiny generation to confirm the full chain works. Exits with a
non-zero status when LM Studio is unreachable so it can be wired into
shell scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import get_settings


def _list_models(base_url: str, timeout: float = 3.0) -> list[str]:
    r = httpx.get(f"{base_url}/models", timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    out: list[str] = []
    for entry in payload.get("data") or []:
        if isinstance(entry, dict) and entry.get("id"):
            out.append(str(entry["id"]))
    return out


def _generate(base_url: str, model: str, timeout: float = 30.0) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise assistant."},
            {"role": "user", "content": "Reply with exactly one word: hello."},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
    }
    r = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def main() -> int:
    console = Console()
    settings = get_settings()
    base_url = settings.lmstudio_base_url.rstrip("/")
    configured_model = settings.lmstudio_model

    table = Table(title="LM Studio — connection check", show_header=False, header_style="bold cyan")
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value")
    table.add_row("Base URL", base_url)
    table.add_row("Model (env)", configured_model or "(auto-detect)")
    console.print(table)

    # 1. Health check: GET /v1/models
    try:
        models = _list_models(base_url)
    except httpx.HTTPError as exc:
        console.print(
            Panel.fit(
                (
                    f"[red]LM Studio is NOT reachable at[/red] [bold]{base_url}[/bold]\n\n"
                    f"Error: {exc}\n\n"
                    "Checklist:\n"
                    "  1. Open LM Studio.\n"
                    "  2. Download an instruct model (Llama 3.1 8B Instruct, Qwen 2.5 7B...).\n"
                    "  3. Load the model.\n"
                    "  4. Go to the [bold]Local Server[/bold] (or [bold]Developer[/bold]) tab.\n"
                    "  5. Click [bold]Start Server[/bold].\n"
                    "  6. Confirm the URL matches LMSTUDIO_BASE_URL in your .env.\n"
                ),
                title="Cannot reach LM Studio",
                border_style="red",
            )
        )
        return 1

    if not models:
        console.print(
            Panel.fit(
                "[yellow]LM Studio is running but no model is loaded.[/yellow]\n"
                "Open LM Studio, load a model, then re-run [bold]make check-llm[/bold].",
                title="No model loaded",
                border_style="yellow",
            )
        )
        return 1

    models_table = Table(title="Loaded models", show_header=True, header_style="bold green")
    models_table.add_column("#", justify="right", style="dim")
    models_table.add_column("model id")
    for idx, model_id in enumerate(models, start=1):
        models_table.add_row(str(idx), model_id)
    console.print(models_table)

    # 2. Pick a model: configured > first loaded.
    model_to_use = configured_model or models[0]
    console.print(f"[bold]Using model:[/bold] {model_to_use}")

    # 3. Generation smoke test.
    try:
        reply = _generate(base_url, model_to_use)
    except httpx.HTTPError as exc:
        console.print(
            Panel.fit(
                f"[red]Generation request failed:[/red] {exc}",
                title="Generation error",
                border_style="red",
            )
        )
        return 1

    if not reply:
        console.print(
            Panel.fit(
                "[yellow]LM Studio replied with an empty completion.[/yellow]\n"
                "The server is alive but the model produced no text. Try a different model.",
                title="Empty reply",
                border_style="yellow",
            )
        )
        return 1

    console.print(
        Panel.fit(
            f"[green]OK[/green] — LM Studio answered:\n\n[bold]{reply}[/bold]",
            title="Connection healthy",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
