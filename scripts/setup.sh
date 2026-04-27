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

# 6. Verify LM Studio connection (best-effort) ---------------------------
say "Checking LM Studio connection (it must be started manually)"
python scripts/check_llm.py || warn "LM Studio is not reachable yet. Start it, load a model, click 'Start Server', then run: make check-llm"

# 7. Done ---------------------------------------------------------------
cat <<EOF

${GREEN}Setup complete.${NC}

Next steps:
  1. Open LM Studio, load a model, click 'Start Server'.
  2. Verify the connection:            make check-llm
  3. Edit .env and config/topics.yaml to your taste.
  4. Open the ./vault folder in Obsidian.
  5. Try a single agent run:           make research
  6. Or a full cycle:                  make run
  7. Or start the daemon:              make daemon

EOF
