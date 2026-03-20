# Project Index: Augur

Generated: 2026-03-20

## Project Structure

```
augur/
  __init__.py              # Package init, version
  engine.py                # Core forecasting engine (specialist calls, aggregation, synthesis)
  router.py                # Multi-provider model router (Anthropic, OpenAI, Google, OpenRouter)
  response_parser.py       # Extract structured JSON from heterogeneous LLM outputs
  calibration.py           # Calibration tracking: bucket accuracy, Brier scores, Wilson CIs
  base_rates.py            # Reference class playbook library (12+ domain categories)
  divergence.py            # Specialist divergence detection & webhook notifications
  api.py                   # FastAPI forecast routes + Pydantic models
  submissions.py           # Dual-pool open submission system (questions, submissions, leaderboard)
  scoring.py               # Scoring engine: Brier, novelty, pool multipliers, reputation
  structured.py            # Structured prediction book: instruments, arb scanner, correlations
  server.py                # FastAPI app entrypoint (mounts all routers)
  specialists/             # TOML playbook manifests
    reasoner.toml
    intelligence_analyst.toml
    market_analyst.toml
    data_scientist.toml
    researcher.toml
  docs/
    citation_audit_2025_03.md
tests/
  test_engine.py           # Unit tests (weighted_average, consensus_label)
  test_api.py              # Integration tests (mocked Anthropic client)
  test_calibration.py      # Calibration math tests
  test_base_rates.py       # Base rate lookup tests
  test_divergence.py       # Divergence detection tests
  test_response_parser.py  # Response parsing edge cases
  test_router.py           # Multi-provider routing tests
  test_scoring.py          # Scoring math tests (Brier, novelty, multipliers)
  test_submissions.py      # Dual-pool submission lifecycle tests
  test_structured.py       # Structured book: instruments, MTM, arb scanner
pyproject.toml             # Hatch build, dependencies, tool config
README.md                  # Docs, quick start, API reference
```

## Entry Points

- **Server**: `augur/server.py` — `uvicorn augur.server:app --reload` (port 8200 default)
- **App object**: `augur.server:app` (FastAPI instance, mounts 3 routers)
- **Tests**: `pytest` (asyncio_mode=auto)

## Core Modules

### `augur/engine.py` — Forecasting Engine
- `load_toml(name)` — Load/cache specialist TOML manifest
- `build_system_prompt(name, config)` — Construct specialist system prompt from TOML playbook
- `call_specialist(name, question, context, timeout, api_key)` — Async LLM call per specialist via multi-provider router
- `weighted_average(estimates)` — Confidence-weighted probability aggregation
- `consensus_label(p)` — Map probability to strongly_yes/lean_yes/uncertain/lean_no/strongly_no
- `synthesize(question, estimates, ensemble_prob, api_key)` — Final Haiku call explaining ensemble result
- `DEFAULT_SPECIALISTS` — [reasoner, intelligence_analyst, market_analyst, researcher, data_scientist]

### `augur/router.py` — Multi-Provider Model Router
- Routes LLM calls to correct provider SDK based on model string
- Providers: Anthropic (Claude), OpenAI (GPT), Google (Gemini), OpenRouter (DeepSeek, Llama, Mistral)
- API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`

### `augur/response_parser.py` — Response Parser
- Normalizes heterogeneous LLM outputs (markdown fences, XML tags, raw JSON, surrounding text)
- Falls back to neutral estimate on parse failure (p=0.5, confidence=0)

### `augur/calibration.py` — Calibration Tracking
- Per-bucket confidence-vs-resolution rates
- Brier scores with quarterly aggregation
- Wilson score confidence intervals

### `augur/base_rates.py` — Reference Class Library
- 12+ domain categories of historical base rates
- Structured templates: category, subcategory, base_rate, source, time_period, notes
- `get_base_rates()`, `get_anchor()`, `list_categories()`, `search_base_rates()`

### `augur/divergence.py` — Divergence Detection
- Flags forecasts where 3+ specialists produce substantially different estimates
- Configurable threshold (percentage points) and minimum divergent count
- Optional webhook notifications via httpx

### `augur/api.py` — Forecast API Routes
- `POST /v1/forecast` — Run ensemble forecast
- `GET /v1/forecast/specialists` — List available specialists
- `GET /v1/forecast/history` — Recent forecast history
- `POST /v1/forecast/resolve` — Mark forecast as resolved
- `GET /v1/forecast/calibration` — Calibration curve data

### `augur/submissions.py` — Dual-Pool Submission System
- `POST /v1/questions` — Register question (auto-creates open + dark pools)
- `GET /v1/questions` — List questions (filter by status)
- `GET /v1/questions/{id}` — Question + per-pool aggregates + spread
- `POST /v1/questions/{id}/submit` — Submit prediction to open or dark pool
- `GET /v1/questions/{id}/submissions` — List submissions by pool
- `POST /v1/questions/{id}/resolve` — Resolve + score all submissions
- `POST /v1/participants` — Register participant
- `GET /v1/participants/{id}` — Participant profile + stats
- `GET /v1/leaderboard` — Ranked by combined score

### `augur/scoring.py` — Scoring Engine
- `brier(probability, outcome)` — Squared error score
- Novelty scoring: directional divergence from consensus
- Pool multipliers: dark = `1 + confidence`, open = `1 + time_remaining_pct`
- Combined score feeds into participant reputation

### `augur/structured.py` — Structured Prediction Book
- Instrument types: spread, conditional, basket, calendar
- Multi-leg positions across correlated questions with live mark-to-market P&L
- Arbitrage scanner: Dutch books (pool divergence), exhaustive violations (probability axioms), correlation dislocations
- `POST /v1/book/instruments` — Create instrument
- `GET /v1/book/instruments/{id}` — Get with live MTM
- `POST /v1/book/instruments/{id}/close` — Close and lock P&L
- `GET /v1/book/positions/{participant_id}` — Full book with live P&L
- `GET /v1/book/arbitrage/scan` — Scan for arbitrage opportunities
- `GET /v1/book/correlations` — Cross-question correlation map

## Specialists (TOML Playbooks)

| Name | Domain | Phases | Key Method |
|------|--------|--------|------------|
| reasoner | Logic, deduction | 9-phase reasoning | Steelman counter-arguments |
| intelligence_analyst | OSINT, geopolitics | 8-phase intel cycle | ACH, NATO source grading |
| market_analyst | Equities, crypto, macro | 8-phase market analysis | Technical + fundamental |
| researcher | Broad research | 9-phase research | CRAAP test, cross-validation |
| data_scientist | Quantitative, statistics | 9-phase data science | Bayesian, base rates |

## Configuration

- `pyproject.toml` — Build (hatch), deps, pytest (asyncio_mode=auto), ruff (py311, line-length=120)
- `ANTHROPIC_API_KEY` — Required (Claude models)
- `OPENAI_API_KEY` — Optional (GPT models)
- `GOOGLE_API_KEY` — Optional (Gemini models)
- `OPENROUTER_API_KEY` — Optional (OpenRouter models)
- `AUGUR_SPECIALISTS_DIR` — Override specialist TOML location

## Key Dependencies

- `anthropic>=0.25` — Claude API client
- `openai>=1.30` — OpenAI/OpenRouter client
- `google-genai>=1.0` — Google Gemini client
- `fastapi>=0.111` — Web framework
- `uvicorn[standard]>=0.29` — ASGI server
- `pydantic>=2.5` — Request/response models

## Test Coverage

- Unit tests: 9 files — engine, calibration, base_rates, divergence, response_parser, router, scoring, submissions, structured
- Integration tests: 1 file — API endpoints with mocked LLM client
- Run: `pytest` (uses pytest-asyncio, httpx for ASGI transport)

## Quick Start

1. `pip install -e ".[dev]"`
2. `ANTHROPIC_API_KEY=sk-ant-... uvicorn augur.server:app --reload`
3. `curl -X POST http://localhost:8000/v1/forecast -H "Content-Type: application/json" -d '{"question": "Will X happen?"}'`
4. `pytest` to run tests

## Architecture Notes

- Specialists run in parallel via `asyncio.gather` with independent timeouts
- Failed specialists return neutral (p=0.5, confidence=0) — graceful degradation
- Aggregation: confidence-weighted average `P = sum(p_i * c_i) / sum(c_i)`
- Synthesis: optional final Haiku call explaining agreement/disagreement
- Multi-provider routing: model string determines SDK (Claude, GPT, Gemini, OpenRouter)
- Dual-pool submissions: open (visible) + dark (sealed) with spread analysis
- Structured book: multi-leg instruments with live MTM, arbitrage scanning
- In-memory stores (per-process, not persisted)
- License: FSL-1.1-Apache-2.0
