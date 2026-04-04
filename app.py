"""
FastAPI server for the OpenEnv Email Triage environment.

Exposes the standard OpenEnv HTTP API:
    POST /reset  → ResetResult
    POST /step   → StepResult
    GET  /state  → StateResult
    GET  /health → {"status": "ok"}
    GET  /tasks  → list of task definitions
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import (
    EmailAction,
    ResetResult,
    StateResult,
    StepResult,
    TaskDefinition,
)
from env.tasks import ALL_TASKS

print("[App] Starting OpenEnv Email Triage server...")

# ── Global environment registry (one per task_id) ────────────────────────────
_envs: Dict[str, EmailTriageEnv] = {}


def get_env(task_id: str = "task_classify") -> EmailTriageEnv:
    if task_id not in ALL_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id={task_id!r}. Valid: {sorted(ALL_TASKS.keys())}",
        )
    if task_id not in _envs:
        seed = int(os.getenv("ENV_SEED", "42"))
        print(f"[App] Creating new env for task_id={task_id}")
        _envs[task_id] = EmailTriageEnv(task_id=task_id, seed=seed)
    return _envs[task_id]


# ── Lifespan (pre-warm all envs at startup) ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[App] Lifespan startup — pre-warming environments...")
    for task_id in ALL_TASKS:
        env = get_env(task_id)
        env.reset()
    print("[App] All environments ready.")
    yield
    print("[App] Lifespan shutdown.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv Email Triage",
    description="Real-world email triage environment for AI agent training.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_classify"
    email_index: Optional[int] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    task_id: str = "task_classify"
    action: EmailAction


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check — returns 200 if the server is running."""
    print("[App] /health called")
    return {"status": "ok", "environment": "email-triage-env", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all available tasks with their metadata."""
    print("[App] /tasks called")
    return {
        "tasks": [task.model_dump() for task in ALL_TASKS.values()],
        "count": len(ALL_TASKS),
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> TaskDefinition:
    """Get details for a specific task."""
    if task_id not in ALL_TASKS:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return ALL_TASKS[task_id]


@app.post("/reset", response_model=ResetResult)
async def reset_environment(request: ResetRequest) -> ResetResult:
    """
    Reset the environment for a given task_id and return the first observation.
    Call this before starting a new episode.
    """
    print(f"[App] POST /reset — task_id={request.task_id}")
    env = get_env(request.task_id)
    if request.seed is not None:
        env.seed = request.seed
    return env.reset(email_index=request.email_index)


@app.post("/step", response_model=StepResult)
async def step_environment(request: StepRequest) -> StepResult:
    """
    Submit an action and receive next observation, reward, done flag, and info.
    The episode ends when done=True (action_type='done' or max_steps reached).
    """
    print(f"[App] POST /step — task_id={request.task_id}, action_type={request.action.action_type}")
    env = get_env(request.task_id)
    try:
        return env.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=StateResult)
async def get_state(task_id: str = Query(default="task_classify")) -> StateResult:
    """Return the current internal state of the environment (for debugging)."""
    print(f"[App] GET /state — task_id={task_id}")
    env = get_env(task_id)
    return env.state()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[App] Starting uvicorn on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)
