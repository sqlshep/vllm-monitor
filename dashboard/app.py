"""Remote monitoring dashboard.

Runs OFF the instance (your laptop / another box). Polls the on-instance agent
over its public IP using a bearer token, and serves an auto-refreshing web UI.

The token never reaches the browser: the frontend calls /api/status on this
server, which proxies to the agent.

Config via env:
  AGENT_URL    e.g. http://<PUBLIC_IP>:9100   (required)
  AGENT_TOKEN  shared secret set on the agent  (required)
  DASH_PORT    port to serve the dashboard on  (default 8080)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


class _SkipStatusPolls(logging.Filter):
    """Drop the frequent /api/status access-log lines (dashboard polling);
    other requests (/, /api/analyze, errors) still log normally."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_SkipStatusPolls())

AGENT_URL = os.environ.get("AGENT_URL", "").rstrip("/")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="vLLM Monitor Dashboard", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    if not AGENT_URL or not AGENT_TOKEN:
        return JSONResponse(
            {"ok": False, "error": "AGENT_URL / AGENT_TOKEN not configured on the dashboard."},
            status_code=500,
        )
    headers = {"Authorization": f"Bearer {AGENT_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{AGENT_URL}/api/all", headers=headers)
        if resp.status_code == 401:
            return JSONResponse(
                {"ok": False, "error": "Agent rejected the token (401)."}, status_code=502
            )
        resp.raise_for_status()
        data = resp.json()
        data["ok"] = True
        return JSONResponse(data)
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"ok": False, "error": f"Could not reach agent at {AGENT_URL}: {exc}"},
            status_code=502,
        )


@app.post("/api/analyze")
async def api_analyze() -> JSONResponse:
    if not AGENT_URL or not AGENT_TOKEN:
        return JSONResponse(
            {"ok": False, "error": "AGENT_URL / AGENT_TOKEN not configured on the dashboard."},
            status_code=500,
        )
    headers = {"Authorization": f"Bearer {AGENT_TOKEN}"}
    try:
        # Generation can take a while on a busy / cold model.
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(f"{AGENT_URL}/api/analyze", headers=headers)
        if resp.status_code == 401:
            return JSONResponse(
                {"ok": False, "error": "Agent rejected the token (401)."}, status_code=502
            )
        resp.raise_for_status()
        data = resp.json()
        data["ok"] = True
        return JSONResponse(data)
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"ok": False, "error": f"Could not reach agent at {AGENT_URL}: {exc}"},
            status_code=502,
        )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())
