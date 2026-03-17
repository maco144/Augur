# Augur — Claude Instructions

## Project Overview
Multi-specialist ensemble forecasting — fan probabilistic questions to domain experts in parallel and aggregate calibrated predictions via confidence-weighted averaging.

See `PROJECT_INDEX.md` for full structure.

## Package Structure
```
augur/__init__.py              # Package init, version
augur/engine.py                # Core engine: specialist calls, aggregation, synthesis
augur/api.py                   # FastAPI routes + Pydantic models
augur/server.py                # FastAPI app entrypoint
augur/specialists/             # TOML playbook manifests (one per specialist)
  reasoner.toml                #   9-phase structured reasoning
  intelligence_analyst.toml    #   ACH methodology, NATO source grading
  market_analyst.toml          #   Technical + fundamental analysis
  researcher.toml              #   CRAAP test, cross-validation
  data_scientist.toml          #   Bayesian reasoning, base rates
tests/test_engine.py           # Unit tests (weighted_average, consensus_label)
tests/test_api.py              # Integration tests (mocked Anthropic client)
```

## Running

```bash
# Server
ANTHROPIC_API_KEY=sk-ant-... uvicorn augur.server:app --reload

# Default port override
python -m augur.server  # starts on :8200

# Tests
pytest

# Lint
ruff check augur/ tests/
```

## Installing Dependencies

```bash
pip install -e ".[dev]"
```

Dependencies: `anthropic>=0.25`, `fastapi>=0.111`, `uvicorn[standard]>=0.29`, `pydantic>=2.5`
Dev: `pytest>=8`, `pytest-asyncio>=0.23`, `httpx>=0.27`, `ruff>=0.4`

## Key Conventions

- **Specialist model**: All specialist LLM calls use `claude-sonnet-4-6`. Synthesis uses `claude-haiku-4-5-20251001`.
- **TOML manifests**: Each specialist is defined by a TOML file in `augur/specialists/`. The `[playbook]` section (phases + principles) is injected as the system prompt for forecasting. The `[model]` section in the TOML is metadata from the AIOS kernel — Augur overrides the actual model used.
- **Structured JSON responses**: Specialists return `{probability, confidence, reasoning, key_assumptions, key_uncertainties, would_change_if}`. Markdown fences are stripped automatically.
- **Aggregation**: Confidence-weighted average: `P = sum(p_i * c_i) / sum(c_i)`. Failed specialists return neutral (p=0.5, confidence=0) and are excluded from aggregation.
- **Consensus labels**: `strongly_yes` (≥0.80), `lean_yes` (≥0.60), `uncertain` (≥0.40), `lean_no` (≥0.20), `strongly_no` (<0.20).
- **Graceful degradation**: Timeouts and API errors per specialist do not block others (`asyncio.gather` with independent timeouts).
- **asyncio throughout**: All specialist calls and synthesis are async. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- **In-memory history**: Forecast history is per-process, max 200 entries, not persisted.
- **Environment**: `ANTHROPIC_API_KEY` required. `AUGUR_SPECIALISTS_DIR` optionally overrides specialist TOML location.
- **Config**: `pyproject.toml` — hatch build, ruff target py311 line-length 120, pytest asyncio_mode auto.

## Proprietary Boundaries

**Internal (OK to reference in this repo):**
- AIOS kernel specialist system, theta router, specialist TOML manifests
- Ensemble aggregation methodology, confidence calibration, specialist disagreement patterns
- All forecasts are probabilistic analysis, not predictions

**Public (any public-facing content, README, docs):**
- NEVER reference Rising Sun, AIOS, eudaimonia, kernel internals, servers, IPs, ports, container names
- NEVER reference other portfolio companies (AAASP, Nullcone, Wire.Band, Pickle, Emergent Languages, FNA, Concord)
- NEVER present forecasts as financial advice or trade signals
- ALWAYS include probabilistic framing: "our ensemble assigns X% probability" not "X will happen"
- Position as an independent forecasting platform powered by multi-specialist ensemble AI
- Do NOT publish internal specialist names, routing logic, or model identifiers

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/forecast` | Run ensemble forecast |
| `GET` | `/v1/forecast/specialists` | List available specialists |
| `GET` | `/v1/forecast/history` | Recent forecast history |
| `GET` | `/health` | Health check |

## AIOS Portfolio Integration

Augur is part of the **Rising Sun** portfolio managed by the AIOS autonomous CEO engine at `eudaimonia.win`. Work items (research, content, outreach tasks) are dispatched here from the CEO cycle.

### Work Queue

Pending tasks live on the AIOS rising server. Fetch at session start:

```bash
# List pending tasks
curl http://eudaimonia.win:8000/api/v1/portfolio/companies/augur/work?status=pending&limit=50

# Compact view
curl -s 'http://eudaimonia.win:8000/api/v1/portfolio/companies/augur/work?status=pending&limit=50' \
  | python3 -c "import sys,json; [print(i['id'][:8], i['category'], '-', i['title'][:80]) for i in json.load(sys.stdin)['items']]"

# Mark a task done
curl -X POST "http://eudaimonia.win:8000/api/v1/company/work-queue/{item_id}/complete?note=what+was+done"
```

Work through tasks one at a time. Mark done immediately after each one is complete.

**Categories:** `infra` (packaging, deployment), `kernel-core` (engine code), `research` (documentation, competitive analysis, forecast methodology), `testing` (benchmarks, calibration), `observability` (metrics, engagement)

### Rising (Production)

- **AIOS API:** `http://eudaimonia.win:8000`
- **Augur company memory DB:** `/data/eudaimonia/company_memory_augur.db` (on rising)
- **Departments active:** engineering, research (content and outreach disabled — build-first, advertise later)

### Portfolio Config

- `companies/augur/company.toml` in the AIOS repo — identity anchors, competitive landscape, content topics
- `portfolio.toml` in the AIOS repo — fleet registry (`active = true` to enable daily CEO cycle)
- Codebase path used by the CEO cycle: `/home/alex/augur` (git log for engineering context)

## Shared Work Queue (PostgreSQL)

This project's work queue (`company_id="augur"`) lives in the **shared PostgreSQL** on rising — not a local SQLite file and not behind the kernel REST API at `:8000`.

**Check pending tasks:**
```bash
ssh rising "docker exec eudaimonia-eudaimonia-postgres-1 psql -U eudaimonia -c \
  \"SELECT id, title, status, priority FROM work_items WHERE company_id='augur' AND status='pending' ORDER BY priority DESC\""
```

**Mark a task done:**
```bash
ssh rising "docker exec eudaimonia-eudaimonia-postgres-1 psql -U eudaimonia -c \
  \"UPDATE work_items SET status='done', completion_note='<note>' WHERE id='<uuid>'\""
```

**Do NOT** rely on `http://eudaimonia.win:8000` for work queue access — the kernel restarts frequently during upgrades and the API will timeout. Use PostgreSQL directly.
