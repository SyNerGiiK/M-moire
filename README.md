# Second Brain

A local-first, fully autonomous "Second Brain" that turns an Obsidian
vault into a living, self-curating knowledge base.

- **Obsidian vault** is the human-readable interface.
- **ChromaDB** is the semantic memory backend.
- **Python agents** research, summarize, tag, and link notes.
- **APScheduler** keeps everything running in the background.
- **Ollama** (default) or **Anthropic** powers the LLM layer.

---

## Architecture

```
                       +-----------------------------+
                       |        Obsidian (you)        |
                       +--------------+--------------+
                                      |
                                      v
+----------------------+      +-------+--------+      +---------------------+
|  config/topics.yaml  +----->+   Orchestrator  +----->+   Agent logs (.md)  |
|  config/agents.yaml  |      +-------+--------+      |   in 07_Agents_Log/  |
+----------------------+              |               +---------------------+
                                      |
        +-----------------------------+-----------------------------+
        |                |                |                  |
        v                v                v                  v
+----------------+ +-------------+ +--------------+ +--------------------+
|  Researcher    | |   ArXiv     | |   Curator    | |     Tagger         |
|  (web search)  | |  (papers)   | | (organize +  | | (auto-tag, normalize|
|                | |             | |  link + dedup|)| |    tags)           |
+-------+--------+ +-----+-------+ +------+-------+ +---------+----------+
        |                |                |                  |
        v                v                v                  v
+--------------------------------------------------------------------------+
|                              Skills Layer                                 |
|  vector_memory  note_writer  summarizer  web_search  arxiv_fetcher       |
|                 youtube_transcriber  pdf_processor                       |
+--------------------------+--------------------------+--------------------+
                           |                          |
                           v                          v
                  +----------------+         +----------------+
                  |   ChromaDB     |         |  Obsidian Vault |
                  | (vectors+meta) |         | (Markdown notes)|
                  +----------------+         +----------------+
```

## Quick start

```bash
# 1. clone & enter
git clone <this-repo> && cd second-brain

# 2. install + initialise everything (venv, deps, vault, chromadb, ollama model)
make setup

# 3. run a single cycle
make run
```

That's it. Open the `./vault` folder in Obsidian and watch the notes appear.

## Repository layout

```
second-brain/
├── vault/              # Obsidian vault (the human UI)
│   ├── 00_Inbox/                # raw captures, unprocessed
│   ├── 01_Atomic_Notes/         # Zettelkasten atomic notes
│   ├── 02_MOC/                  # Maps of Content (auto-generated)
│   ├── 03_Projects/             # active project workspaces
│   ├── 04_Resources/Papers/     # arXiv papers (auto)
│   ├── 04_Resources/Attachments/
│   ├── 05_Archive/              # archived / duplicates
│   ├── 06_Templates/            # Markdown templates
│   ├── 07_Agents_Log/           # agent run logs (auto)
│   └── .obsidian/               # Obsidian configuration
├── agents/             # autonomous agents
├── skills/             # reusable building blocks
├── memory/chroma_db/   # ChromaDB persistent storage
├── config/             # topics.yaml, agents.yaml, settings.py
├── scheduler/          # APScheduler daemon
├── api/                # FastAPI control plane
├── tests/              # pytest suite
├── scripts/            # setup.sh, init_vault.py, migrate.py
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

## How each agent works

| Agent | Purpose | Output |
|-------|---------|--------|
| **researcher** | DuckDuckGo search + page extraction + summarization | Atomic notes in `00_Inbox/` |
| **arxiv** | Fetches recent arXiv papers per topic | Paper notes in `04_Resources/Papers/` |
| **tagger** | Auto-generates and normalizes tags | Updated frontmatter |
| **curator** | Moves notes to typed folders, dedupes, links, builds MOCs | Vault structure & MOC notes |
| **orchestrator** | Wires everything; runs full cycles | Stats + per-agent logs |

Every run produces a Markdown log under `vault/07_Agents_Log/` with: notes
created, topics processed, sources used, errors.

## Configuration

### `config/topics.yaml`

Declarative list of domains and subtopics the system monitors. Each domain
can opt-in/out of `arxiv`, `web`, `youtube` sources and pick an
`update_frequency`.

```yaml
domains:
  - name: "Artificial Intelligence"
    subtopics: ["LLMs", "AI Agents", "RAG"]
    sources: { arxiv: true, web: true, youtube: false }
    update_frequency: "daily"

settings:
  max_notes_per_run: 50
  dedup_threshold: 0.90
  language: "fr"
```

### `config/agents.yaml`

Per-agent flags + the LLM provider:

```yaml
researcher: { enabled: true, max_sources_per_topic: 10, schedule_hours: 6 }
arxiv:      { enabled: true, days_back: 7, schedule_hours: 24 }
curator:    { enabled: true, schedule_hours: 1 }
tagger:     { enabled: true, schedule_hours: 2 }
llm:        { provider: ollama, model: mistral, fallback: claude-sonnet-4-6 }
```

### `.env`

See `.env.example` for the full list. The most relevant variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SECOND_BRAIN_VAULT_PATH` | `./vault` | Obsidian vault location |
| `SECOND_BRAIN_LLM_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `ANTHROPIC_API_KEY` | _(empty)_ | Required only if provider=anthropic |
| `SECOND_BRAIN_LANG` | `fr` | Preferred summary language |

## CLI

```bash
make run            # full agent cycle
make daemon         # start the background scheduler (blocking)
make research       # run only the researcher
make arxiv          # run only the arxiv agent
make curate         # run all curator subcommands
make tag            # run the tagger
make inbox          # process inbox -> typed folders
make status         # vault & memory stats
make test           # pytest suite
make api            # FastAPI on :8765
```

Or invoke the Python entry points directly:

```bash
python -m agents.orchestrator status
python -m agents.orchestrator agent researcher
python -m agents.orchestrator agent curator --subcommand update_mocs
python -m scheduler.run_scheduler --once
python scripts/migrate.py rebuild-index --reset
```

## REST API (optional)

```bash
make api
# in another shell:
curl -s http://127.0.0.1:8765/status | jq .
curl -s -X POST http://127.0.0.1:8765/agents/run \
  -H 'content-type: application/json' \
  -d '{"agent":"researcher"}' | jq .
curl -s -X POST http://127.0.0.1:8765/memory/search \
  -H 'content-type: application/json' \
  -d '{"query":"retrieval augmented generation","n_results":5}' | jq .
```

## Adding a new skill

1. Create `skills/my_skill.py` with a single class exposing the public methods.
2. Re-export it from `skills/__init__.py`.
3. Inject it into whichever agent needs it (constructor argument).
4. Add unit tests in `tests/test_skills.py`.

Skills must:

- be importable in isolation (no global state),
- accept their dependencies via constructor,
- degrade gracefully when their network calls fail (return empty list / dict).

## Adding a new agent

1. Create `agents/my_agent.py` subclassing `agents.base.BaseAgent`.
2. Implement a single public entry point that returns an `AgentResult`.
3. Register it in `agents/__init__.py`.
4. Wire it inside `agents/orchestrator.py` (`_agent` and `run_agent`).
5. Optionally add a cron job in `scheduler/cron_jobs.py`.
6. Add tests using the fakes in `tests/conftest.py`.

## Obsidian setup

Open the `vault/` directory as an Obsidian vault. Recommended community
plugins (declared in `vault/.obsidian/community-plugins.json`):

| Plugin | Why |
|--------|-----|
| **Dataview** | SQL-like queries against your notes (used by MOC templates) |
| **Templater** | Dynamic note templates from `06_Templates/` |
| **Smart Connections** | AI-powered note linking on top of the agent links |
| **Tag Wrangler** | Bulk tag management |
| **Tasks** | Task queries across notes |
| **Kanban** | Project boards in `03_Projects/` |
| **Calendar** | Daily-notes calendar view |

Install them via *Settings → Community plugins*.

## FAQ

**Does this require an internet connection?**
Only the researcher and arXiv agents need network access. With Ollama and
local data sources you can run it fully offline.

**Where are my notes stored?**
In `vault/` as Markdown files with YAML frontmatter — fully owned by you,
no proprietary format. Every agent action is reversible (notes are moved
to `05_Archive/`, never deleted, except for explicit memory dedup).

**How do I reset the system?**

```bash
python scripts/migrate.py drop --yes      # wipe the vector store
rm -rf vault/00_Inbox/*.md                # clear the inbox
make run                                  # fresh cycle
```

**Can I bring my own vault?**
Set `SECOND_BRAIN_VAULT_PATH` in `.env` and run `python scripts/init_vault.py`
once — it only creates missing folders and templates, never overwrites
existing notes.

**Is anything sent to a third party?**
By default, no — Ollama runs locally. Only if you set
`SECOND_BRAIN_LLM_PROVIDER=anthropic` (or set `ANTHROPIC_API_KEY` and let
the fallback kick in) do summaries leave your machine.

**How does deduplication work?**
Two layers: (1) ChromaDB cosine similarity ≥ `dedup_threshold` skips
ingestion of near-duplicate sources; (2) the curator periodically prunes
exact-title duplicates from the inbox into `05_Archive/`.

## License

MIT.
