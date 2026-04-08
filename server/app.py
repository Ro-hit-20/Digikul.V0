"""
DigiKul-v0 — FastAPI Server Application.

Creates the OpenEnv-compatible HTTP server that exposes
/reset, /step, and /state endpoints via the OpenEnv framework.
"""

from __future__ import annotations

import os
import sys

# Ensure package root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

from models import DigiKulAction, DigiKulObservation, DigiKulState
from server.digikul_environment import DigiKulEnvironment

# ── Configuration via environment variable ──
TASK = os.environ.get("DIGIKUL_TASK", "medium")
SEED = os.environ.get("DIGIKUL_SEED", None)
seed_val = int(SEED) if SEED is not None else None

# ── Create environment instance ──
env = DigiKulEnvironment(task=TASK, seed=seed_val)

# ── FastAPI app ──
app = FastAPI(
    title="DigiKul-v0 Environment",
    description=(
        "A Meta OpenEnv RL environment for remote education bandwidth allocation. "
        "The agent distributes a limited server bandwidth across rural classrooms "
        "to maximise educational quality."
    ),
    version="0.1.0",
)

# CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──
@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "digikul-v0", "task": TASK}


# ── OpenEnv Endpoints ──

@app.post("/reset", response_model=DigiKulObservation)
async def reset():
    """Reset the environment and return the initial observation."""
    obs = env.reset()
    return obs


@app.post("/step", response_model=DigiKulObservation)
async def step(action: DigiKulAction):
    """Execute one step with the given action."""
    try:
        obs = env.step(action)
        return obs
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state", response_model=DigiKulState)
async def state():
    """Return the current episode state."""
    return env.state


# ── Info endpoint (non-standard but useful) ──

class EnvInfo(BaseModel):
    name: str
    task: str
    num_nodes: int
    server_bandwidth: float
    episode_length: int
    quality_levels: Dict[str, float]


@app.get("/info", response_model=EnvInfo)
async def info():
    """Return static environment configuration info."""
    from models import QUALITY_BW_MAP, QUALITY_LABELS
    return EnvInfo(
        name="digikul-v0",
        task=TASK,
        num_nodes=env.num_nodes,
        server_bandwidth=env._cfg.server_bandwidth,
        episode_length=env._cfg.episode_length,
        quality_levels={
            QUALITY_LABELS[k]: v for k, v in QUALITY_BW_MAP.items()
        },
    )


# ── Entrypoint for direct execution ──
def main():
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
