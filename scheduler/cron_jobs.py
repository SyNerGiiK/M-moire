"""APScheduler job registration.

Each function here is a thin wrapper around the orchestrator. Keeping
them in a dedicated module makes mocking and testing straightforward.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from agents.orchestrator import Orchestrator


def job_researcher(orchestrator: Orchestrator) -> None:
    logger.info("[cron] researcher start")
    orchestrator.run_agent("researcher")


def job_arxiv(orchestrator: Orchestrator) -> None:
    logger.info("[cron] arxiv start")
    orchestrator.run_agent("arxiv")


def job_tagger(orchestrator: Orchestrator) -> None:
    logger.info("[cron] tagger start")
    orchestrator.run_agent("tagger", subcommand="tag_all_untagged")


def job_curator_process_inbox(orchestrator: Orchestrator) -> None:
    logger.info("[cron] curator process_inbox start")
    orchestrator.run_agent("curator", subcommand="process_inbox")


def job_curator_connect(orchestrator: Orchestrator) -> None:
    logger.info("[cron] curator generate_connections start")
    orchestrator.run_agent("curator", subcommand="generate_connections")


def job_curator_mocs(orchestrator: Orchestrator) -> None:
    logger.info("[cron] curator update_mocs start")
    orchestrator.run_agent("curator", subcommand="update_mocs")


def job_curator_dedup(orchestrator: Orchestrator) -> None:
    logger.info("[cron] curator deduplicate_vault start")
    orchestrator.run_agent("curator", subcommand="deduplicate_vault")


def job_full_cycle(orchestrator: Orchestrator) -> None:
    logger.info("[cron] full cycle start")
    orchestrator.run_full_cycle()


def register_jobs(scheduler: Any, orchestrator: Orchestrator, config: dict[str, Any]) -> None:
    """Register each agent's job in ``scheduler`` (an APScheduler instance)."""
    researcher_cfg = config.get("researcher") or {}
    if researcher_cfg.get("enabled", True):
        scheduler.add_job(
            job_researcher,
            "interval",
            hours=int(researcher_cfg.get("schedule_hours", 6)),
            args=[orchestrator],
            id="researcher",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    arxiv_cfg = config.get("arxiv") or {}
    if arxiv_cfg.get("enabled", True):
        scheduler.add_job(
            job_arxiv,
            "interval",
            hours=int(arxiv_cfg.get("schedule_hours", 24)),
            args=[orchestrator],
            id="arxiv",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    tagger_cfg = config.get("tagger") or {}
    if tagger_cfg.get("enabled", True):
        scheduler.add_job(
            job_tagger,
            "interval",
            hours=int(tagger_cfg.get("schedule_hours", 2)),
            args=[orchestrator],
            id="tagger",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    curator_cfg = config.get("curator") or {}
    if curator_cfg.get("enabled", True):
        hours = int(curator_cfg.get("schedule_hours", 1))
        scheduler.add_job(
            job_curator_process_inbox,
            "interval",
            hours=hours,
            args=[orchestrator],
            id="curator_inbox",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        scheduler.add_job(
            job_curator_connect,
            "interval",
            hours=hours * 2,
            args=[orchestrator],
            id="curator_connect",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        scheduler.add_job(
            job_curator_mocs,
            "interval",
            hours=hours * 6,
            args=[orchestrator],
            id="curator_mocs",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        scheduler.add_job(
            job_curator_dedup,
            "interval",
            hours=hours * 24,
            args=[orchestrator],
            id="curator_dedup",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
