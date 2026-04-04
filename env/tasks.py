"""Task definitions and dataset management for each difficulty level."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from env.email_generator import generate_emails
from env.models import (
    ActionType,
    DepartmentType,
    EmailGroundTruth,
    TaskDefinition,
)


# ─────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────

TASK_CLASSIFY = TaskDefinition(
    task_id="task_classify",
    name="Basic Email Classification",
    difficulty="easy",
    description=(
        "Classify each incoming email into one of 5 categories: "
        "spam, inquiry, complaint, order, or support. "
        "Use the 'classify' action with the appropriate classification field. "
        "Finish with 'done' when classification is complete."
    ),
    required_actions=["classify", "done"],
    max_steps=3,
    target_score=0.8,
    email_count=20,
)

TASK_PRIORITY_ROUTING = TaskDefinition(
    task_id="task_priority_routing",
    name="Priority Assignment and Routing",
    difficulty="medium",
    description=(
        "For each email, you must: (1) classify the email type, "
        "(2) assign a priority level (urgent/high/normal/low), "
        "(3) route it to the correct department. "
        "All three actions are required before calling 'done'."
    ),
    required_actions=["classify", "prioritize", "route", "done"],
    max_steps=5,
    target_score=0.7,
    email_count=20,
)

TASK_FULL_TRIAGE = TaskDefinition(
    task_id="task_full_triage",
    name="Full Triage with Response Generation",
    difficulty="hard",
    description=(
        "Perform complete email triage: classify, assign priority, route to department, "
        "AND either generate a reply or escalate with a reason. "
        "Urgent/complaint emails typically need escalation; others need a reply. "
        "All required actions must be taken before calling 'done'."
    ),
    required_actions=["classify", "prioritize", "route", "reply", "done"],
    max_steps=8,
    target_score=0.6,
    email_count=20,
)

ALL_TASKS: Dict[str, TaskDefinition] = {
    "task_classify": TASK_CLASSIFY,
    "task_priority_routing": TASK_PRIORITY_ROUTING,
    "task_full_triage": TASK_FULL_TRIAGE,
}

# ── department context shown to the agent ─────────────────────────────────────

DEPARTMENT_CONTEXT: Dict[str, str] = {
    "sales": "Handles new business, pricing inquiries, quotes, bulk orders, enterprise deals",
    "support": "Technical help, product issues, how-to questions, account access problems",
    "billing": "Invoice questions, payment processing, subscription changes, refunds",
    "technical": "Complex technical issues, API/integration problems, developer queries",
    "management": "Escalated complaints, executive contacts, serious business issues",
    "hr": "Employment, recruitment, employee relations",
    "legal": "Legal threats, compliance, contracts, IP issues",
}


# ─────────────────────────────────────────────
# Dataset Storage (in-memory, loaded once)
# ─────────────────────────────────────────────

_DATASETS: Dict[str, List[Tuple[dict, EmailGroundTruth]]] = {}


def get_dataset(task_id: str, seed: int = 42) -> List[Tuple[dict, EmailGroundTruth]]:
    """Return (and lazily generate) the email dataset for a given task."""
    cache_key = f"{task_id}_{seed}"
    if cache_key not in _DATASETS:
        task = ALL_TASKS[task_id]
        print(f"[Tasks] Loading dataset for task_id={task_id}, difficulty={task.difficulty}")
        _DATASETS[cache_key] = generate_emails(
            task_difficulty=task.difficulty,
            count=task.email_count,
            seed=seed,
        )
        print(f"[Tasks] Dataset ready: {len(_DATASETS[cache_key])} emails for {task_id}")
    return _DATASETS[cache_key]


def get_email_by_index(
    task_id: str,
    index: int,
    seed: int = 42,
) -> Optional[Tuple[dict, EmailGroundTruth]]:
    dataset = get_dataset(task_id, seed)
    if index < 0 or index >= len(dataset):
        return None
    return dataset[index]


def get_task_definition(task_id: str) -> TaskDefinition:
    if task_id not in ALL_TASKS:
        raise ValueError(f"Unknown task_id={task_id!r}. Valid: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[task_id]
