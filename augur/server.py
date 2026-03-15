"""
Augur server — standalone FastAPI application.

Usage:
    uvicorn augur.server:app --reload
    # or
    python -m augur.server
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .api import router as forecast_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Augur",
    description="Multi-specialist ensemble forecasting — probabilistic analysis powered by domain experts",
    version="0.1.0",
)

app.include_router(forecast_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "augur"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("augur.server:app", host="0.0.0.0", port=8200, reload=True)
