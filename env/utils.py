"""Utility helpers for the OpenEnv Email Triage environment."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from env.models import (
    DepartmentType,
    EmailGroundTruth,
    EmailObservation,
    EmailReward,
)
from env.tasks import DEPARTMENT_CONTEXT


def build_observation(
    email_dict: dict,
    task_id: str,
    task_description: str,
    step_number: int,
    max_steps: int,
    previous_actions: Optional[List[str]] = None,
    last_action_error: Optional[str] = None,
) -> EmailObservation:
    """Construct an EmailObservation from a raw email dict."""
    ts_raw = email_dict["timestamp"]
    if isinstance(ts_raw, str):
        ts = datetime.fromisoformat(ts_raw)
    else:
        ts = ts_raw

    return EmailObservation(
        email_id=email_dict["email_id"],
        sender=email_dict["sender"],
        subject=email_dict["subject"],
        body=email_dict["body"],
        timestamp=ts,
        attachments=email_dict.get("attachments", []),
        previous_actions=previous_actions or [],
        department_context=DEPARTMENT_CONTEXT,
        urgency_indicators=email_dict.get("urgency_indicators", []),
        task_id=task_id,
        task_description=task_description,
        step_number=step_number,
        max_steps=max_steps,
        last_action_error=last_action_error,
    )


def step_reward_for_action(
    action_type: str,
    task_id: str,
    actions_so_far: Dict[str, Optional[str]],
    ground_truth: EmailGroundTruth,
) -> float:
    """
    Intermediate reward given after each individual action (before episode ends).
    Provides partial progress signal during the trajectory.
    """
    from env.grader import (
        _score_classification,
        _score_priority,
        _score_department,
        _score_reply,
    )

    partial_reward = 0.0

    if action_type == "classify":
        cls_score = _score_classification(
            actions_so_far.get("classification"), ground_truth.classification
        )
        # Scale to a smaller intermediate signal so terminal reward still matters
        partial_reward = cls_score * 0.3

    elif action_type == "prioritize":
        pri_score = _score_priority(
            actions_so_far.get("priority"), ground_truth.priority
        )
        partial_reward = pri_score * 0.2

    elif action_type == "route":
        dep_score = _score_department(
            actions_so_far.get("department"), ground_truth.department
        )
        partial_reward = dep_score * 0.2

    elif action_type == "reply":
        reply_score = _score_reply(
            actions_so_far.get("reply_text"),
            ground_truth.expected_reply_keywords,
        )
        partial_reward = reply_score * 0.2

    elif action_type == "escalate":
        # Small positive signal for recognising escalation need
        if ground_truth.should_escalate:
            partial_reward = 0.15
        else:
            partial_reward = -0.1  # penalty for unnecessary escalation

    print(f"[Reward] Intermediate reward for action_type={action_type}: {partial_reward:.3f}")
    return partial_reward


def format_reward_summary(reward: EmailReward) -> str:
    lines = [
        "─── Reward Breakdown ───",
        f"  Total:          {reward.total_score:.3f}",
        f"  Classification: {reward.classification_score:.3f}",
        f"  Priority:       {reward.priority_score:.3f}",
        f"  Routing:        {reward.routing_score:.3f}",
        f"  Response:       {reward.response_quality_score:.3f}",
        f"  Efficiency:     +{reward.efficiency_bonus:.3f}",
        f"  Penalty:        -{reward.penalty:.3f}",
        f"  Feedback: {reward.feedback}",
        "────────────────────────",
    ]
    return "\n".join(lines)
