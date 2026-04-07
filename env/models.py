"""Pydantic models for the OpenEnv Email Triage environment."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class EmailObservation(BaseModel):
    """What the agent sees at each step."""

    email_id: str = Field(..., description="Unique email identifier")
    sender: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body text")
    timestamp: datetime = Field(..., description="Email received timestamp")
    attachments: List[str] = Field(default_factory=list, description="Attachment filenames")
    previous_actions: List[str] = Field(
        default_factory=list,
        description="Actions already taken on this email",
    )
    department_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Available department routing options and descriptions",
    )
    urgency_indicators: List[str] = Field(
        default_factory=list,
        description="Keywords signalling urgency detected in email",
    )
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Human-readable task description")
    step_number: int = Field(default=0, description="Current step within episode")
    max_steps: int = Field(default=5, description="Maximum allowed steps per episode")
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message from last action, if any",
    )


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

ActionType = Literal["classify", "prioritize", "route", "reply", "escalate", "done"]
ClassificationType = Literal["spam", "inquiry", "complaint", "order", "support", "feedback", "billing"]
PriorityType = Literal["urgent", "high", "normal", "low"]
DepartmentType = Literal["sales", "support", "billing", "technical", "management", "hr", "legal"]


class EmailAction(BaseModel):
    """Action the agent wants to perform on the current email."""

    action_type: ActionType = Field(..., description="Type of action to perform")
    classification: Optional[ClassificationType] = Field(
        default=None,
        description="Email category (required for classify action)",
    )
    priority: Optional[PriorityType] = Field(
        default=None,
        description="Priority level (required for prioritize action)",
    )
    department: Optional[DepartmentType] = Field(
        default=None,
        description="Target department (required for route action)",
    )
    reply_text: Optional[str] = Field(
        default=None,
        description="Response text (required for reply action)",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Escalation justification (required for escalate action)",
    )

    @field_validator("classification", mode="before")
    @classmethod
    def validate_classification(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.lower().strip()
        return v

    @field_validator("priority", mode="before")
    @classmethod
    def validate_priority(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.lower().strip()
        return v

    @field_validator("department", mode="before")
    @classmethod
    def validate_department(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            # Reject comma-separated departments - force LLM to choose one
            if "," in v:
                raise ValueError(f"Multiple departments not allowed: '{v}'. Choose exactly ONE department.")
            return v.lower().strip()
        return v


# ─────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────

class EmailReward(BaseModel):
    """Decomposed reward signal returned after each action."""

    total_score: float = Field(..., ge=0.0, le=1.0, description="Overall reward (0.0–1.0)")
    classification_score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    routing_score: float = Field(default=0.0, ge=0.0, le=1.0)
    response_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency_bonus: float = Field(default=0.0, ge=0.0, le=0.2)
    penalty: float = Field(default=0.0, ge=0.0, le=1.0, description="Penalty for harmful actions")
    feedback: str = Field(default="", description="Human-readable reward explanation")


# ─────────────────────────────────────────────
# API Response Wrappers
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """Response returned by POST /step."""

    observation: EmailObservation
    reward: float = Field(..., ge=-1.0, le=1.0)
    reward_details: EmailReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Response returned by POST /reset."""

    observation: EmailObservation
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Response returned by GET /state."""

    task_id: str
    step_number: int
    max_steps: int
    current_email_id: Optional[str]
    classification_done: bool
    priority_done: bool
    routing_done: bool
    reply_done: bool
    episode_reward: float
    is_terminal: bool
    actions_taken: List[str]


# ─────────────────────────────────────────────
# Task Definition
# ─────────────────────────────────────────────

class TaskDefinition(BaseModel):
    """Metadata for a single task."""

    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    required_actions: List[ActionType]
    max_steps: int
    target_score: float
    email_count: int


# ─────────────────────────────────────────────
# Ground Truth (internal, not exposed to agent)
# ─────────────────────────────────────────────

class EmailGroundTruth(BaseModel):
    """Hidden ground truth for a single email used by the grader."""

    email_id: str
    classification: ClassificationType
    priority: PriorityType
    department: DepartmentType
    expected_reply_keywords: List[str] = Field(default_factory=list)
    should_escalate: bool = False
    escalation_triggers: List[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "easy"
