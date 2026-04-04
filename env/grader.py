"""
Grading logic for all three tasks.

Each grader returns an EmailReward with decomposed scores (0.0–1.0).
All graders are deterministic given the same inputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from env.models import (
    ClassificationType,
    DepartmentType,
    EmailAction,
    EmailGroundTruth,
    EmailReward,
    PriorityType,
)

# ── priority adjacency (partial credit map) ───────────────────────────────────
#  Key → value: if predicted=key and correct=value, partial score applies
_PRIORITY_PARTIAL_CREDIT: Dict[str, Dict[str, float]] = {
    "urgent": {"high": 0.5, "normal": 0.1, "low": 0.0},
    "high":   {"urgent": 0.5, "normal": 0.5, "low": 0.1},
    "normal": {"high": 0.5, "urgent": 0.2, "low": 0.5},
    "low":    {"normal": 0.5, "high": 0.1, "urgent": 0.0},
}

# ── classification adjacency ───────────────────────────────────────────────────
_CLASSIFICATION_PARTIAL: Dict[str, Set[str]] = {
    "inquiry":   {"order", "feedback"},
    "order":     {"inquiry", "billing"},
    "complaint": {"support", "billing"},
    "support":   {"complaint"},
    "billing":   {"complaint", "order"},
    "feedback":  {"inquiry"},
    "spam":      set(),
}

# ── department adjacency ──────────────────────────────────────────────────────
_DEPARTMENT_PARTIAL: Dict[str, Set[str]] = {
    "sales":      {"technical", "management"},
    "support":    {"technical", "management"},
    "billing":    {"management", "sales"},
    "technical":  {"support", "sales"},
    "management": {"sales", "support", "billing", "legal"},
    "legal":      {"management"},
    "hr":         {"management"},
}


# ─────────────────────────────────────────────
# Helper scorers
# ─────────────────────────────────────────────

def _score_classification(predicted: Optional[str], correct: ClassificationType) -> float:
    if predicted is None:
        return 0.0
    if predicted == correct:
        return 1.0
    if predicted in _CLASSIFICATION_PARTIAL.get(correct, set()):
        return 0.4  # adjacent category partial credit
    return 0.0


def _score_priority(predicted: Optional[str], correct: PriorityType) -> float:
    if predicted is None:
        return 0.0
    if predicted == correct:
        return 1.0
    return _PRIORITY_PARTIAL_CREDIT.get(correct, {}).get(predicted, 0.0)


def _score_department(predicted: Optional[str], correct: DepartmentType) -> float:
    if predicted is None:
        return 0.0
    if predicted == correct:
        return 1.0
    if predicted in _DEPARTMENT_PARTIAL.get(correct, set()):
        return 0.4
    return 0.0


def _score_reply(reply_text: Optional[str], expected_keywords: List[str]) -> float:
    """Score reply quality based on keyword overlap — deterministic, no model needed."""
    if not reply_text or not expected_keywords:
        return 0.0
    reply_lower = reply_text.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in reply_lower)
    score = matches / len(expected_keywords)
    # Penalise extremely short replies (< 20 chars) as low quality
    if len(reply_text.strip()) < 20:
        score *= 0.3
    return min(score, 1.0)


def _score_escalation(
    action: EmailAction,
    ground_truth: EmailGroundTruth,
) -> float:
    """Score escalation decision: +1 if correct, -0.3 if wrong direction."""
    agent_escalated = action.action_type == "escalate"
    if ground_truth.should_escalate and agent_escalated:
        reason = (action.escalation_reason or "").lower()
        trigger_hit = any(t.lower() in reason for t in ground_truth.escalation_triggers)
        return 1.0 if trigger_hit else 0.7  # escalated but no clear reason
    if ground_truth.should_escalate and not agent_escalated:
        return 0.0  # should have escalated
    if not ground_truth.should_escalate and agent_escalated:
        return 0.0  # unnecessary escalation
    return 1.0  # correctly did not escalate (only applies to reply scoring)


# ─────────────────────────────────────────────
# Harmful action detection
# ─────────────────────────────────────────────

def _compute_penalty(
    actions_taken: Dict[str, Optional[str]],
    ground_truth: EmailGroundTruth,
) -> float:
    """Return a penalty (0.0–1.0) for clearly harmful decisions."""
    penalty = 0.0

    # Marking an urgent non-spam as spam
    classification = actions_taken.get("classification")
    if classification == "spam" and ground_truth.classification != "spam":
        if ground_truth.priority in ("urgent", "high"):
            penalty += 0.5  # critical misclassification

    # Routing a complaint directly to billing when it needs management
    department = actions_taken.get("department")
    if (ground_truth.classification == "complaint"
            and ground_truth.should_escalate
            and department == "billing"):
        penalty += 0.2

    return min(penalty, 1.0)


# ─────────────────────────────────────────────
# Per-task graders
# ─────────────────────────────────────────────

def grade_task_classify(
    actions_taken: Dict[str, Optional[str]],
    ground_truth: EmailGroundTruth,
    steps_used: int,
    max_steps: int,
) -> EmailReward:
    """
    Task 1: Basic Classification
    Weights: classification=1.0 (only dimension)
    """
    print(f"[Grader] grade_task_classify | predicted={actions_taken.get('classification')} | correct={ground_truth.classification}")

    cls_score = _score_classification(actions_taken.get("classification"), ground_truth.classification)
    penalty = _compute_penalty(actions_taken, ground_truth)

    # Efficiency bonus if done in 1 step (minimum possible)
    efficiency = 0.1 if steps_used <= 1 else 0.0

    total = max(0.0, cls_score + efficiency - penalty)
    total = min(total, 1.0)

    feedback = (
        f"Classification: {'✓' if cls_score == 1.0 else ('~' if cls_score > 0 else '✗')} "
        f"(predicted={actions_taken.get('classification')!r}, correct={ground_truth.classification!r})"
    )

    return EmailReward(
        total_score=total,
        classification_score=cls_score,
        priority_score=0.0,
        routing_score=0.0,
        response_quality_score=0.0,
        efficiency_bonus=efficiency,
        penalty=penalty,
        feedback=feedback,
    )


def grade_task_priority_routing(
    actions_taken: Dict[str, Optional[str]],
    ground_truth: EmailGroundTruth,
    steps_used: int,
    max_steps: int,
) -> EmailReward:
    """
    Task 2: Priority Assignment + Routing
    Weights: classification=0.4, priority=0.3, routing=0.3
    """
    print(f"[Grader] grade_task_priority_routing | actions={actions_taken} | truth=({ground_truth.classification},{ground_truth.priority},{ground_truth.department})")

    cls_score = _score_classification(actions_taken.get("classification"), ground_truth.classification)
    pri_score = _score_priority(actions_taken.get("priority"), ground_truth.priority)
    dep_score = _score_department(actions_taken.get("department"), ground_truth.department)
    penalty = _compute_penalty(actions_taken, ground_truth)

    weighted = cls_score * 0.4 + pri_score * 0.3 + dep_score * 0.3
    efficiency = 0.1 if steps_used <= 3 else 0.0

    total = max(0.0, weighted + efficiency - penalty)
    total = min(total, 1.0)

    feedback = (
        f"Class: {'✓' if cls_score==1.0 else '~' if cls_score>0 else '✗'} "
        f"({actions_taken.get('classification')!r}→{ground_truth.classification!r}), "
        f"Priority: {'✓' if pri_score==1.0 else '~' if pri_score>0 else '✗'} "
        f"({actions_taken.get('priority')!r}→{ground_truth.priority!r}), "
        f"Dept: {'✓' if dep_score==1.0 else '~' if dep_score>0 else '✗'} "
        f"({actions_taken.get('department')!r}→{ground_truth.department!r})"
    )

    return EmailReward(
        total_score=total,
        classification_score=cls_score,
        priority_score=pri_score,
        routing_score=dep_score,
        response_quality_score=0.0,
        efficiency_bonus=efficiency,
        penalty=penalty,
        feedback=feedback,
    )


def grade_task_full_triage(
    actions_taken: Dict[str, Optional[str]],
    ground_truth: EmailGroundTruth,
    steps_used: int,
    max_steps: int,
    action_type_used: Optional[str] = None,
) -> EmailReward:
    """
    Task 3: Full Triage (classify + prioritize + route + reply/escalate)
    Weights: classification=0.25, priority=0.2, routing=0.2, response=0.35
    """
    print(f"[Grader] grade_task_full_triage | actions={actions_taken} | should_escalate={ground_truth.should_escalate}")

    cls_score = _score_classification(actions_taken.get("classification"), ground_truth.classification)
    pri_score = _score_priority(actions_taken.get("priority"), ground_truth.priority)
    dep_score = _score_department(actions_taken.get("department"), ground_truth.department)

    # Response scoring
    if ground_truth.should_escalate:
        escalation_action = EmailAction(
            action_type="escalate",
            escalation_reason=actions_taken.get("escalation_reason"),
        )
        response_score = _score_escalation(escalation_action, ground_truth)
        # If agent replied instead of escalating, penalise
        if action_type_used == "reply" and ground_truth.should_escalate:
            response_score *= 0.3
    else:
        response_score = _score_reply(
            actions_taken.get("reply_text"),
            ground_truth.expected_reply_keywords,
        )
        # If agent escalated unnecessarily, penalise
        if action_type_used == "escalate" and not ground_truth.should_escalate:
            response_score *= 0.4

    penalty = _compute_penalty(actions_taken, ground_truth)

    weighted = (
        cls_score * 0.25
        + pri_score * 0.20
        + dep_score * 0.20
        + response_score * 0.35
    )
    efficiency = 0.1 if steps_used <= 4 else 0.0

    total = max(0.0, weighted + efficiency - penalty)
    total = min(total, 1.0)

    response_label = "escalate" if ground_truth.should_escalate else "reply"
    feedback = (
        f"Class: {'✓' if cls_score==1.0 else '~' if cls_score>0 else '✗'} "
        f"| Priority: {'✓' if pri_score==1.0 else '~' if pri_score>0 else '✗'} "
        f"| Dept: {'✓' if dep_score==1.0 else '~' if dep_score>0 else '✗'} "
        f"| {response_label}: {'✓' if response_score>=0.8 else '~' if response_score>0.3 else '✗'} "
        f"(score={response_score:.2f})"
    )

    return EmailReward(
        total_score=total,
        classification_score=cls_score,
        priority_score=pri_score,
        routing_score=dep_score,
        response_quality_score=response_score,
        efficiency_bonus=efficiency,
        penalty=penalty,
        feedback=feedback,
    )


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

def grade(
    task_id: str,
    actions_taken: Dict[str, Optional[str]],
    ground_truth: EmailGroundTruth,
    steps_used: int,
    max_steps: int,
    last_action_type: Optional[str] = None,
) -> EmailReward:
    """Dispatch to the correct grader based on task_id."""
    print(f"[Grader] Grading task_id={task_id}, steps_used={steps_used}")
    if task_id == "task_classify":
        return grade_task_classify(actions_taken, ground_truth, steps_used, max_steps)
    if task_id == "task_priority_routing":
        return grade_task_priority_routing(actions_taken, ground_truth, steps_used, max_steps)
    if task_id == "task_full_triage":
        return grade_task_full_triage(
            actions_taken, ground_truth, steps_used, max_steps, last_action_type
        )
    raise ValueError(f"Unknown task_id: {task_id!r}")
