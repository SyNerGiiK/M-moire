.PHONY: help setup run daemon inbox research arxiv curate tag status stats test test-verbose api check-llm migrate-rebuild migrate-export clean

PYTHON ?= python3
VENV   ?= .venv

ifeq ($(shell test -d $(VENV) && echo ok),ok)
	PY := $(VENV)/bin/python
else
	PY := $(PYTHON)
endif

help:
	@echo "Second Brain - common commands"
	@echo ""
	@echo "  make setup            One-shot setup (venv + deps + vault + chroma)"
	@echo "  make run              Run a full agent cycle once"
	@echo "  make daemon           Start the background scheduler (blocking)"
	@echo "  make inbox            Process the inbox (curator.process_inbox)"
	@echo "  make research         Run the researcher agent"
	@echo "  make arxiv            Run the arxiv agent"
	@echo "  make curate           Run all curator subcommands"
	@echo "  make tag              Run the tagger agent"
	@echo "  make status           Show vault & memory stats"
	@echo "  make stats            Same as status"
	@echo "  make test             Run the unit tests"
	@echo "  make api              Start the FastAPI control plane"
	@echo "  make check-llm        Verify LM Studio is reachable + run a test generation"
	@echo "  make migrate-rebuild  Rebuild ChromaDB from the vault"
	@echo "  make migrate-export   Export memory to JSON"
	@echo "  make clean            Remove caches (.pytest_cache, __pycache__, etc.)"

setup:
	bash scripts/setup.sh

run:
	$(PY) -m agents.orchestrator run

daemon:
	$(PY) -m scheduler.run_scheduler

inbox:
	$(PY) -m agents.orchestrator inbox

research:
	$(PY) -m agents.orchestrator research

arxiv:
	$(PY) -m agents.orchestrator arxiv

curate:
	$(PY) -m agents.orchestrator curate

tag:
	$(PY) -m agents.orchestrator agent tagger

status:
	$(PY) -m agents.orchestrator status

stats: status

api:
	$(PY) -m api.main

check-llm:
	$(PY) scripts/check_llm.py

test:
	$(PY) -m pytest

test-verbose:
	$(PY) -m pytest -vv

migrate-rebuild:
	$(PY) scripts/migrate.py rebuild-index

migrate-export:
	$(PY) scripts/migrate.py export-json

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage
