#!/usr/bin/env bash
# =====================================================================
# Second Brain — One-shot setup script.
# Idempotent: re-running it should be safe.
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"

say() { printf "${GREEN}==>${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}!! ${NC} %s\n" "$*"; }
fail() { printf "${RED}xx ${NC} %s\n" "$*" >&2; exit 1; }

# 1. Python venv ----------------------------------------------------------
if [[ ! -d ".venv" ]]; then
  say "Creating Python virtualenv at .venv"
  python3 -m venv .venv
else
  say "Using existing virtualenv at .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null

# 2. Install requirements ------------------------------------------------
say "Installing Python dependencies (this can take a few minutes)"
python -m pip install -r requirements.txt

# 3. .env ----------------------------------------------------------------
if [[ ! -f ".env" ]]; then
  say "Copying .env.example to .env"
  cp .env.example .env
else
  say ".env already exists; leaving untouched"
fi

# 4. Initialise the vault ------------------------------------------------
say "Initialising vault structure"
python scripts/init_vault.py

# 5. Initialise ChromaDB (idempotent) ------------------------------------
say "Bootstrapping ChromaDB collection"
python -c "
from skills.vector_memory import VectorMemory
from config.settings import get_settings
s = get_settings()
mem = VectorMemory(persist_dir=s.chroma_dir, collection_name='second_brain', embedding_model=s.embed_model)
print('ChromaDB ready:', mem.get_collection_stats())
"

# 6. Pull Ollama model (best-effort) -------------------------------------
if command -v ollama >/dev/null 2>&1; then
  MODEL="${OLLAMA_MODEL:-mistral}"
  if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    say "Ollama model ${MODEL} already present"
  else
    say "Pulling Ollama model: ${MODEL}"
    ollama pull "${MODEL}" || warn "Could not pull model ${MODEL}; configure OLLAMA_MODEL in .env"
  fi
else
  warn "Ollama not found in PATH. Install it from https://ollama.com or set SECOND_BRAIN_LLM_PROVIDER=anthropic in .env"
fi

# 7. Done ---------------------------------------------------------------
cat <<EOF

${GREEN}Setup complete.${NC}

Next steps:
  1. Edit .env and config/topics.yaml to your taste.
  2. Open the ./vault folder in Obsidian.
  3. Try a single agent run:           make research
  4. Or a full cycle:                  make run
  5. Or start the daemon:              make daemon

EOF
