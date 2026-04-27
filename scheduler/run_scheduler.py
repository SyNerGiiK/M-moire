"""Entry point for the background daemon.

Runs an APScheduler ``BlockingScheduler`` with persistent jobs read from
``config/agents.yaml``. Logs to stderr and to the vault (via the agent
log writer used by each agent).
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from agents.orchestrator import Orchestrator
from config.settings import get_settings
from scheduler.cron_jobs import register_jobs


def _configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)


def run(interval_hours: int | None = None) -> None:
    """Start the blocking scheduler. Runs until SIGTERM/SIGINT."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.jobstores.memory import MemoryJobStore

    settings = get_settings()
    _configure_logging(settings.log_level)

    orchestrator = Orchestrator(settings=settings)
    agents_config = settings.load_agents()

    scheduler = BlockingScheduler(
        jobstores={"default": MemoryJobStore()},
        timezone="UTC",
    )
    register_jobs(scheduler, orchestrator, agents_config)

    # Optional master cycle.
    hours = interval_hours or settings.full_cycle_hours
    if hours > 0:
        from scheduler.cron_jobs import job_full_cycle

        scheduler.add_job(
            job_full_cycle,
            "interval",
            hours=int(hours),
            args=[orchestrator],
            id="full_cycle",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    def _shutdown(signum: int, _frame: Any) -> None:
        logger.warning("Received signal {}; shutting down scheduler", signum)
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("Scheduler starting (vault={}, full_cycle_hours={})", settings.vault_path, hours)
    for job in scheduler.get_jobs():
        logger.info("  job={} trigger={}", job.id, job.trigger)
    scheduler.start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="second-brain-daemon")
    parser.add_argument("--interval-hours", type=int, default=None,
                        help="Override the full-cycle interval (hours)")
    parser.add_argument("--once", action="store_true",
                        help="Run a full cycle once and exit (useful for cron)")
    args = parser.parse_args(argv)

    if args.once:
        _configure_logging(get_settings().log_level)
        Orchestrator().run_full_cycle()
        return 0

    try:
        run(interval_hours=args.interval_hours)
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
