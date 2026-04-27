"""Master agent coordinator.

Wires every component together (skills + agents + config) and exposes a
small API the CLI / scheduler / API server consume.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

from agents.arxiv_agent import ArxivAgent
from agents.base import AgentResult
from agents.curator_agent import CuratorAgent
from agents.researcher_agent import ResearcherAgent
from agents.tagger_agent import TaggerAgent
from config.settings import Settings, get_settings
from skills.arxiv_fetcher import ArxivFetcher
from skills.note_writer import NoteWriter
from skills.summarizer import Summarizer
from skills.vector_memory import VectorMemory
from skills.web_search import WebSearch


@dataclass
class _Bundle:
    settings: Settings
    note_writer: NoteWriter
    memory: VectorMemory
    summarizer: Summarizer
    web_search: WebSearch
    arxiv_fetcher: ArxivFetcher
    agents_config: dict[str, Any] = field(default_factory=dict)
    topics_config: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Wire skills + agents and run them in coordinated order."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.console = Console()
        self._bundle: _Bundle | None = None
        self._agents: dict[str, Any] = {}
        self._last_run: dict[str, AgentResult] = {}

    # ----------------------------------------------------- lazy initialization

    def _init_bundle(self) -> _Bundle:
        if self._bundle is not None:
            return self._bundle
        s = self.settings
        agents_config = s.load_agents()
        topics_config = s.load_topics()
        memory_cfg = agents_config.get("memory") or {}
        llm_cfg = agents_config.get("llm") or {}

        note_writer = NoteWriter(s.vault_path)
        memory = VectorMemory(
            persist_dir=s.chroma_dir,
            collection_name=memory_cfg.get("collection", "second_brain"),
            embedding_model=memory_cfg.get("embedding_model", s.embed_model),
        )
        summarizer = Summarizer(
            base_url=llm_cfg.get("base_url", s.lmstudio_base_url),
            model=llm_cfg.get("model") or s.lmstudio_model,
            language=s.language,
            temperature=float(llm_cfg.get("temperature", 0.3)),
            max_tokens=int(llm_cfg.get("max_tokens", 1024)),
        )
        bundle = _Bundle(
            settings=s,
            note_writer=note_writer,
            memory=memory,
            summarizer=summarizer,
            web_search=WebSearch(),
            arxiv_fetcher=ArxivFetcher(download_dir=s.embed_cache_dir / "arxiv"),
            agents_config=agents_config,
            topics_config=topics_config,
        )
        self._bundle = bundle
        return bundle

    def _agent(self, name: str) -> Any:
        if name in self._agents:
            return self._agents[name]
        b = self._init_bundle()
        cfg = b.agents_config.get(name) or {}
        if name == "researcher":
            agent = ResearcherAgent(
                note_writer=b.note_writer,
                memory=b.memory,
                summarizer=b.summarizer,
                web_search=b.web_search,
                config=cfg,
                topics_config=b.topics_config,
            )
        elif name == "arxiv":
            agent = ArxivAgent(
                note_writer=b.note_writer,
                memory=b.memory,
                summarizer=b.summarizer,
                arxiv_fetcher=b.arxiv_fetcher,
                config=cfg,
                topics_config=b.topics_config,
            )
        elif name == "curator":
            agent = CuratorAgent(
                note_writer=b.note_writer,
                memory=b.memory,
                summarizer=b.summarizer,
                config={**cfg, **(b.agents_config.get("memory") or {})},
            )
        elif name == "tagger":
            agent = TaggerAgent(
                note_writer=b.note_writer,
                summarizer=b.summarizer,
                config=cfg,
            )
        else:
            raise KeyError(f"Unknown agent: {name!r}")
        self._agents[name] = agent
        return agent

    # ------------------------------------------------------------------ run

    def run_agent(self, agent_name: str, **kwargs: Any) -> AgentResult:
        """Run a single agent's main entry point and stash the result."""
        agent = self._agent(agent_name)
        cfg = self._init_bundle().agents_config.get(agent_name) or {}
        if cfg.get("enabled", True) is False:
            logger.info("Agent {} disabled via config; skipping.", agent_name)
            result = AgentResult(agent=agent_name, success=True)
            result.extra["skipped"] = True
            self._last_run[agent_name] = result
            return result

        logger.info("[orchestrator] running {}", agent_name)
        if agent_name == "researcher":
            result = agent.run(**kwargs)
        elif agent_name == "arxiv":
            result = agent.monitor_topics(**kwargs)
        elif agent_name == "curator":
            sub = kwargs.pop("subcommand", "process_inbox")
            method = {
                "process_inbox": agent.process_inbox,
                "deduplicate_vault": agent.deduplicate_vault,
                "generate_connections": agent.generate_connections,
                "update_mocs": agent.update_mocs,
            }.get(sub)
            if method is None:
                raise ValueError(f"Unknown curator subcommand: {sub!r}")
            result = method(**kwargs)
        elif agent_name == "tagger":
            sub = kwargs.pop("subcommand", "tag_all_untagged")
            method = {
                "tag_all_untagged": agent.tag_all_untagged,
                "normalize_tags": agent.normalize_tags,
            }.get(sub)
            if method is None:
                raise ValueError(f"Unknown tagger subcommand: {sub!r}")
            result = method(**kwargs)
        else:
            raise ValueError(f"Unknown agent {agent_name!r}")

        self._last_run[agent_name] = result
        return result

    def run_full_cycle(self) -> dict[str, AgentResult]:
        """Run every enabled agent in a sensible order and return results."""
        order = [
            ("researcher", {}),
            ("arxiv", {}),
            ("tagger", {"subcommand": "tag_all_untagged"}),
            ("curator", {"subcommand": "process_inbox"}),
            ("curator", {"subcommand": "generate_connections"}),
            ("curator", {"subcommand": "update_mocs"}),
            ("curator", {"subcommand": "deduplicate_vault"}),
        ]
        results: dict[str, AgentResult] = {}
        for name, kwargs in order:
            key = name + (":" + str(kwargs.get("subcommand")) if kwargs.get("subcommand") else "")
            try:
                results[key] = self.run_agent(name, **kwargs)
            except Exception as exc:
                logger.exception("Agent {} crashed", name)
                err = AgentResult(agent=name, success=False)
                err.add_error(str(exc))
                results[key] = err
        return results

    # ------------------------------------------------------------- introspect

    def get_status(self) -> dict[str, Any]:
        b = self._init_bundle()
        return {
            "vault_path": str(b.settings.vault_path),
            "chroma_dir": str(b.settings.chroma_dir),
            "vault_stats": b.note_writer.stats(),
            "memory_stats": b.memory.get_collection_stats(),
            "agents_enabled": {
                name: bool((b.agents_config.get(name) or {}).get("enabled", True))
                for name in ("researcher", "arxiv", "curator", "tagger")
            },
            "last_run": {k: v.as_dict() for k, v in self._last_run.items()},
            "llm_base_url": b.settings.lmstudio_base_url,
            "llm_model": b.settings.lmstudio_model or "(auto)",
        }

    # ----------------------------------------------------------- pretty print

    def print_status(self) -> None:
        status = self.get_status()
        table = Table(title="Second Brain — Status", show_header=True, header_style="bold cyan")
        table.add_column("Component")
        table.add_column("Value")
        table.add_row("Vault", status["vault_path"])
        table.add_row("Memory dir", status["chroma_dir"])
        table.add_row("Notes", str(status["vault_stats"]["total"]))
        table.add_row("Memory entries", str(status["memory_stats"]["count"]))
        table.add_row("LLM", f"LM Studio @ {status['llm_base_url']} (model: {status['llm_model']})")
        for agent, enabled in status["agents_enabled"].items():
            table.add_row(f"agent:{agent}", "enabled" if enabled else "disabled")
        self.console.print(table)

    def print_results(self, results: dict[str, AgentResult]) -> None:
        table = Table(title="Run Summary", show_header=True, header_style="bold green")
        table.add_column("Agent")
        table.add_column("Created", justify="right")
        table.add_column("Updated", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Status")
        for key, res in results.items():
            table.add_row(
                key,
                str(res.notes_created),
                str(res.notes_updated),
                str(len(res.errors)),
                "[green]OK[/green]" if res.success else "[red]FAIL[/red]",
            )
        self.console.print(table)

    def schedule_cycle(self, interval_hours: int = 6) -> None:
        """Convenience entry — start an APScheduler running ``run_full_cycle``."""
        from scheduler.run_scheduler import run as scheduler_run

        scheduler_run(interval_hours=interval_hours)


# ====================================================================
# CLI entry point
# ====================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="second-brain", description="Second Brain CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show vault & memory stats")

    p_run = sub.add_parser("run", help="Run a full agent cycle")
    p_run.add_argument("--quiet", action="store_true")

    p_agent = sub.add_parser("agent", help="Run a single agent")
    p_agent.add_argument("name", choices=["researcher", "arxiv", "curator", "tagger"])
    p_agent.add_argument(
        "--subcommand",
        default=None,
        help="Curator/tagger subcommand (e.g. process_inbox, generate_connections)",
    )

    sub.add_parser("inbox", help="Process inbox only (curator.process_inbox)")
    sub.add_parser("research", help="Run researcher agent only")
    sub.add_parser("arxiv", help="Run arxiv agent only")
    sub.add_parser("curate", help="Run all curator subcommands")
    sub.add_parser("stats", help="Detailed memory + vault stats")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    orchestrator = Orchestrator()

    cmd = args.command or "status"
    try:
        if cmd == "status":
            orchestrator.print_status()
        elif cmd == "run":
            results = orchestrator.run_full_cycle()
            if not getattr(args, "quiet", False):
                orchestrator.print_results(results)
        elif cmd == "agent":
            kwargs: dict[str, Any] = {}
            if args.subcommand:
                kwargs["subcommand"] = args.subcommand
            result = orchestrator.run_agent(args.name, **kwargs)
            orchestrator.print_results({args.name: result})
        elif cmd == "inbox":
            result = orchestrator.run_agent("curator", subcommand="process_inbox")
            orchestrator.print_results({"curator:process_inbox": result})
        elif cmd == "research":
            result = orchestrator.run_agent("researcher")
            orchestrator.print_results({"researcher": result})
        elif cmd == "arxiv":
            result = orchestrator.run_agent("arxiv")
            orchestrator.print_results({"arxiv": result})
        elif cmd == "curate":
            results: dict[str, AgentResult] = {}
            for sub in ("process_inbox", "generate_connections", "update_mocs", "deduplicate_vault"):
                results[f"curator:{sub}"] = orchestrator.run_agent("curator", subcommand=sub)
            orchestrator.print_results(results)
        elif cmd == "stats":
            orchestrator.print_status()
        else:  # pragma: no cover
            parser.print_help()
            return 1
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("CLI failure")
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
