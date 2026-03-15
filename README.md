# Augur

**Multi-specialist ensemble forecasting. Fan probabilistic questions to domain experts and aggregate calibrated predictions.**

```bash
pip install augur
```

---

## Why Augur?

Single-model forecasts are brittle. Augur fans every question to 5+ domain specialists in parallel — each with its own analytical playbook — and aggregates their independent probability estimates via confidence-weighted averaging. The result: calibrated ensemble forecasts with full reasoning chains.

| Feature | Augur | Single LLM call | Prediction markets |
|---------|:-----:|:----------------:|:------------------:|
| Multiple independent perspectives | Yes | No | Yes (crowd) |
| Full reasoning chains | Yes | Partial | No |
| Confidence-weighted aggregation | Yes | N/A | Price-weighted |
| Sub-minute latency | Yes | Yes | No (market time) |
| Domain-specific playbooks | Yes | No | No |

---

## Quick start

### Start the server

```bash
ANTHROPIC_API_KEY=sk-ant-... uvicorn augur.server:app --reload
```

### Run a forecast

```bash
curl -X POST http://localhost:8000/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"question": "Will the Fed cut rates before September 2026?"}'
```

Response:
```json
{
  "question": "Will the Fed cut rates before September 2026?",
  "ensemble_probability": 0.68,
  "ensemble_confidence": 0.72,
  "consensus": "lean_yes",
  "synthesis": "Specialists broadly agreed on a likely cut, driven by...",
  "specialists": [
    {
      "specialist": "reasoner",
      "probability": 0.65,
      "confidence": 0.75,
      "reasoning": "Step-by-step analysis...",
      "status": "success"
    }
  ],
  "successful": 5,
  "failed": 0
}
```

---

## Default specialists

| Specialist | Domain | Playbook |
|-----------|--------|----------|
| **Reasoner** | Logic, deduction, structured analysis | 9-phase reasoning framework |
| **Intelligence Analyst** | OSINT, geopolitics, threat assessment | ACH methodology, NATO source grading |
| **Market Analyst** | Equities, crypto, macro | Technical + fundamental analysis |
| **Researcher** | Broad research, source validation | CRAAP test, cross-validation |
| **Data Scientist** | Quantitative analysis, statistics | Bayesian reasoning, base rates |

### Custom specialists

Add TOML manifests to `augur/specialists/` or set `AUGUR_SPECIALISTS_DIR`:

```bash
AUGUR_SPECIALISTS_DIR=/path/to/my/specialists uvicorn augur.server:app
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/forecast` | Run ensemble forecast |
| `GET` | `/v1/forecast/specialists` | List available specialists |
| `GET` | `/v1/forecast/history` | Recent forecast history |

---

## How it works

1. **Fan out** — question sent to N specialists in parallel via `asyncio.gather`
2. **Independent analysis** — each specialist applies its TOML-defined playbook
3. **Structured response** — probability, confidence, reasoning, assumptions, uncertainties
4. **Aggregate** — confidence-weighted average: `P = Sum(p_i * c_i) / Sum(c_i)`
5. **Synthesize** — optional Haiku pass explaining agreement/disagreement

Graceful degradation: timeouts and errors return neutral (0.5, confidence=0) and don't block other specialists.

---

## License

[FSL-1.1-Apache-2.0](https://fsl.software). Source-available for non-competing products. Converts to Apache 2.0 two years after each release.
