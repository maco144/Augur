# Augur — Claude Instructions

## Project Overview
Multi-specialist ensemble forecasting — fan probabilistic questions to domain experts in parallel and aggregate calibrated predictions via confidence-weighted averaging.

See `PROJECT_INDEX.md` for full structure.

## Package Structure
```
augur/__init__.py              # Package init, version
augur/engine.py                # Core engine: specialist calls, aggregation, synthesis
augur/router.py                # Multi-provider model router (Anthropic, OpenAI, Google, OpenRouter)
augur/response_parser.py       # Extract structured JSON from heterogeneous LLM outputs
augur/calibration.py           # Calibration tracking: bucket accuracy, Brier scores, Wilson CIs
augur/base_rates.py            # Reference class playbook library (12+ domain categories)
augur/divergence.py            # Specialist divergence detection & webhook notifications
augur/api.py                   # FastAPI routes + Pydantic models (ensemble forecasting)
augur/submissions.py           # Dual-pool open submission system (questions, submissions, leaderboard)
augur/scoring.py               # Scoring engine: Brier, novelty, pool multipliers, reputation
augur/structured.py            # Structured prediction book: instruments, arb scanner, correlations
augur/server.py                # FastAPI app entrypoint (mounts forecast, submissions, structured routers)
augur/specialists/             # TOML playbook manifests (one per specialist)
  reasoner.toml                #   9-phase structured reasoning
  intelligence_analyst.toml    #   ACH methodology, NATO source grading
  market_analyst.toml          #   Technical + fundamental analysis
  researcher.toml              #   CRAAP test, cross-validation
  data_scientist.toml          #   Bayesian reasoning, base rates
tests/test_engine.py           # Unit tests (weighted_average, consensus_label)
tests/test_api.py              # Integration tests (mocked Anthropic client)
tests/test_calibration.py      # Calibration math tests
tests/test_base_rates.py       # Base rate lookup tests
tests/test_divergence.py       # Divergence detection tests
tests/test_response_parser.py  # Response parsing edge cases
tests/test_router.py           # Multi-provider routing tests
tests/test_scoring.py          # Scoring math tests (Brier, novelty, multipliers)
tests/test_submissions.py      # Dual-pool submission lifecycle tests
tests/test_structured.py       # Structured book: instruments, MTM, arb scanner
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

Dependencies: `anthropic>=0.25`, `openai>=1.30`, `google-genai>=1.0`, `fastapi>=0.111`, `uvicorn[standard]>=0.29`, `pydantic>=2.5`
Dev: `pytest>=8`, `pytest-asyncio>=0.23`, `httpx>=0.27`, `ruff>=0.4`

## Key Conventions

- **Multi-provider routing**: Specialist LLM calls route through `router.py` based on model string — supports Anthropic (Claude), OpenAI (GPT), Google (Gemini), OpenRouter (DeepSeek, Llama, Mistral). Default specialist model: `claude-sonnet-4-6`. Synthesis: `claude-haiku-4-5-20251001`.
- **TOML manifests**: Each specialist is defined by a TOML file in `augur/specialists/`. The `[playbook]` section (phases + principles) is injected as the system prompt for forecasting. The `[model]` section declares which provider/model to route through.
- **Structured JSON responses**: Specialists return `{probability, confidence, reasoning, key_assumptions, key_uncertainties, would_change_if}`. Response parser (`response_parser.py`) normalizes markdown fences, XML tags, and raw text.
- **Aggregation**: Confidence-weighted average: `P = sum(p_i * c_i) / sum(c_i)`. Failed specialists return neutral (p=0.5, confidence=0) and are excluded from aggregation.
- **Consensus labels**: `strongly_yes` (≥0.80), `lean_yes` (≥0.60), `uncertain` (≥0.40), `lean_no` (≥0.20), `strongly_no` (<0.20).
- **Graceful degradation**: Timeouts and API errors per specialist do not block others (`asyncio.gather` with independent timeouts).
- **asyncio throughout**: All specialist calls and synthesis are async. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- **In-memory history**: Forecast history is per-process, max 200 entries, not persisted.
- **Calibration tracking**: `calibration.py` tracks forecast outcomes — per-bucket accuracy, Brier scores, quarterly aggregation with Wilson score CIs.
- **Base rates**: `base_rates.py` provides reference class playbook library (12+ domain categories) for empirical anchoring.
- **Divergence detection**: `divergence.py` flags forecasts where specialists disagree substantially, with optional webhook notifications.
- **Structured prediction book**: `structured.py` provides multi-leg instruments (spreads, conditionals, baskets, calendars) with live mark-to-market P&L. Includes arbitrage scanner (Dutch books, exhaustive violations, correlation dislocations).
- **Environment**: `ANTHROPIC_API_KEY` required. Optional: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`. `AUGUR_SPECIALISTS_DIR` overrides specialist TOML location.
- **Config**: `pyproject.toml` — hatch build, ruff target py311 line-length 120, pytest asyncio_mode auto.
- **Dual-pool submissions**: Every question gets open + dark pools. Open pool is visible; dark pool is sealed until deadline or resolution. Scoring uses pool multipliers: dark = `1 + confidence` (conviction), open = `1 + time_remaining_pct` (early mover). Combined score = `(quality * α + novelty * (1-α)) * multiplier`. The spread between pool aggregates measures herding/anchoring.
- **Participant reputation**: Rolling combined score across resolved submissions feeds back as aggregation weight (min 5 resolved, floor 0.1).

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

### Ensemble Forecasting (internal specialists)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/forecast` | Run ensemble forecast |
| `GET` | `/v1/forecast/specialists` | List available specialists |
| `GET` | `/v1/forecast/history` | Recent forecast history |
| `POST` | `/v1/forecast/resolve` | Mark forecast as resolved with actual outcome |
| `GET` | `/v1/forecast/calibration` | Calibration curve data with confidence intervals |
| `GET` | `/health` | Health check |

### Open Submission System (dual-pool)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/questions` | Register a question (auto-creates open + dark pools) |
| `GET` | `/v1/questions` | List questions (`?status=open\|resolved\|all`) |
| `GET` | `/v1/questions/{id}` | Question + per-pool aggregates + spread |
| `POST` | `/v1/questions/{id}/submit` | Submit prediction (`"pool": "open"\|"dark"`) |
| `GET` | `/v1/questions/{id}/submissions` | List submissions (`?pool=open\|dark`) |
| `POST` | `/v1/questions/{id}/resolve` | Resolve + score all submissions |
| `POST` | `/v1/participants` | Register participant (agent/human/ensemble) |
| `GET` | `/v1/participants/{id}` | Participant profile + rolling stats |
| `GET` | `/v1/leaderboard` | Ranked by combined score |

### Structured Prediction Book

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/book/instruments` | Create instrument (spread, conditional, basket, calendar) |
| `GET` | `/v1/book/instruments/{id}` | Get instrument with live mark-to-market |
| `POST` | `/v1/book/instruments/{id}/close` | Close instrument and lock in P&L |
| `GET` | `/v1/book/positions/{participant_id}` | Full book with live P&L |
| `GET` | `/v1/book/arbitrage/scan` | Scan for arbitrage opportunities |
| `GET` | `/v1/book/correlations` | Cross-question correlation map |

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
