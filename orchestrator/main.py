import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import models, prompts, mcp_servers, experiments, config, runs

app = FastAPI(title="XP Framework Test Platform")

app.include_router(models.router)
app.include_router(prompts.router)
app.include_router(mcp_servers.router)
app.include_router(experiments.router)
app.include_router(config.router)
app.include_router(runs.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the built React frontend as static files.
# The frontend is built at image build time; the dist/ directory is copied to static/.
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
