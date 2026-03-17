# Project Index: Augur

Generated: 2026-03-17

## Project Structure

```
augur/
  __init__.py            # Package init, version
  engine.py              # Core forecasting engine (specialist calls, aggregation, synthesis)
  api.py                 # FastAPI routes and Pydantic models
  server.py              # FastAPI app entrypoint
  specialists/           # TOML playbook manifests
    reasoner.toml
    intelligence_analyst.toml
    market_analyst.toml
    data_scientist.toml
    researcher.toml
tests/
  test_engine.py         # Unit tests for weighted_average, consensus_label
  test_api.py            # Integration tests for API endpoints (mocked LLM)
pyproject.toml           # Hatch build, dependencies, tool config
README.md                # Docs, quick start, API reference
```

## Entry Points

- **Server**: `augur/server.py` — `uvicorn augur.server:app --reload` (port 8200 default)
- **App object**: `augur.server:app` (FastAPI instance)
- **Tests**: `pytest` (asyncio_mode=auto)

## Core Modules

### `augur/engine.py` — Forecasting Engine
- `load_toml(name)` — Load/cache specialist TOML manifest
- `build_system_prompt(name, config)` — Construct specialist system prompt from TOML playbook
- `call_specialist(name, question, context, timeout, api_key)` — Async LLM call per specialist (Anthropic API, claude-sonnet-4-6)
- `weighted_average(estimates)` — Confidence-weighted probability aggregation
- `consensus_label(p)` — Map probability to strongly_yes/lean_yes/uncertain/lean_no/strongly_no
- `synthesize(question, estimates, ensemble_prob, api_key)` — Final Haiku call explaining ensemble result
- `DEFAULT_SPECIALISTS` — [reasoner, intelligence_analyst, market_analyst, researcher, data_scientist]

### `augur/api.py` — FastAPI Routes
- `POST /v1/forecast` — Run ensemble forecast (fans out to N specialists in parallel)
- `GET /v1/forecast/specialists` — List available specialists with metadata
- `GET /v1/forecast/history` — Recent forecasts (in-memory, max 200)
- Pydantic models: `ForecastRequest`, `ForecastResponse`, `SpecialistEstimate`

### `augur/server.py` — App Factory
- `GET /health` — Health check
- Mounts `api.router` at `/v1/forecast`

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
- `ANTHROPIC_API_KEY` env var required at runtime
- `AUGUR_SPECIALISTS_DIR` env var overrides specialist TOML location

## Key Dependencies

- `anthropic>=0.25` — Claude API client
- `fastapi>=0.111` — Web framework
- `uvicorn[standard]>=0.29` — ASGI server
- `pydantic>=2.5` — Request/response models

## Test Coverage

- Unit tests: 1 file (`test_engine.py`) — weighted_average, consensus_label
- Integration tests: 1 file (`test_api.py`) — API endpoints with mocked Anthropic client
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
- In-memory forecast history (per-process, not persisted)
- Model: claude-sonnet-4-6 for specialists, claude-haiku-4-5 for synthesis
- License: FSL-1.1-Apache-2.0
