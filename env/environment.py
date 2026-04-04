"""
Core OpenEnv Email Triage Environment.

Implements the three required API methods:
    reset()  → ResetResult
    step()   → StepResult
    state()  → StateResult
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env.grader import grade
from env.models import (
    EmailAction,
    EmailGroundTruth,
    EmailObservation,
    EmailReward,
    ResetResult,
    StateResult,
    StepResult,
)
from env.tasks import ALL_TASKS, get_email_by_index, get_task_definition
from env.utils import build_observation, step_reward_for_action


class EmailTriageEnv:
    """
    Main environment class.

    One episode = triaging a single email.
    At the end of the episode (done=True), the final grader score is returned.
    Intermediate steps return partial rewards to guide the agent.
    """

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self, task_id: str = "task_classify", seed: int = 42) -> None:
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task_id={task_id!r}. Valid: {list(ALL_TASKS.keys())}")

        self.task_id = task_id
        self.seed = seed
        self._task_def = get_task_definition(task_id)

        # Episode state (reset on every reset() call)
        self._email_index: int = 0
        self._current_email: Optional[dict] = None
        self._current_gt: Optional[EmailGroundTruth] = None
        self._step_number: int = 0
        self._done: bool = False
        self._episode_reward: float = 0.0
        self._actions_taken: Dict[str, Optional[str]] = {}
        self._action_log: List[str] = []
        self._last_action_error: Optional[str] = None
        self._last_action_type: Optional[str] = None

        # Flags for required-action tracking
        self._classify_done: bool = False
        self._priority_done: bool = False
        self._routing_done: bool = False
        self._reply_done: bool = False

        print(f"[Env] Created EmailTriageEnv task_id={task_id}, seed={seed}")

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, email_index: Optional[int] = None) -> ResetResult:
        """
        Reset the environment and return the initial observation.
        If email_index is None, episodes cycle through the dataset sequentially.
        """
        print(f"[Env] reset() called — task_id={self.task_id}, email_index={email_index}")

        # Advance through dataset
        idx = email_index if email_index is not None else self._email_index
        entry = get_email_by_index(self.task_id, idx, self.seed)
        if entry is None:
            # Wrap around to beginning of dataset
            idx = 0
            entry = get_email_by_index(self.task_id, 0, self.seed)

        assert entry is not None
        self._current_email, self._current_gt = entry
        self._email_index = idx + 1  # advance for next episode

        # Reset per-episode state
        self._step_number = 0
        self._done = False
        self._episode_reward = 0.0
        self._actions_taken = {}
        self._action_log = []
        self._last_action_error = None
        self._last_action_type = None
        self._classify_done = False
        self._priority_done = False
        self._routing_done = False
        self._reply_done = False

        obs = build_observation(
            email_dict=self._current_email,
            task_id=self.task_id,
            task_description=self._task_def.description,
            step_number=self._step_number,
            max_steps=self._task_def.max_steps,
        )
        print(f"[Env] reset() → email_id={obs.email_id}, subject={obs.subject[:60]!r}")
        return ResetResult(
            observation=obs,
            done=False,
            info={"task_id": self.task_id, "email_index": idx},
        )

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: EmailAction) -> StepResult:
        """
        Apply an action to the current email and return the result.
        Returns partial reward mid-episode and full graded reward on done.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")
        if self._current_email is None or self._current_gt is None:
            raise RuntimeError("No active episode. Call reset() first.")

        print(f"[Env] step() — action_type={action.action_type}, step={self._step_number + 1}/{self._task_def.max_steps}")

        self._step_number += 1
        self._last_action_error = None
        immediate_reward = 0.0
        done = False

        # ── Validate action for task requirements ──────────────────────────────
        error = self._validate_action(action)
        if error:
            self._last_action_error = error
            print(f"[Env] Action validation error: {error}")
            obs = self._make_observation()
            return StepResult(
                observation=obs,
                reward=0.0,
                reward_details=EmailReward(
                    total_score=0.0,
                    feedback=f"Invalid action: {error}",
                ),
                done=False,
                info={"error": error},
            )

        # ── Apply action ───────────────────────────────────────────────────────
        action_str = self._apply_action(action)
        self._action_log.append(action_str)
        self._last_action_type = action.action_type

        # Intermediate reward (not final graded score)
        if action.action_type != "done":
            immediate_reward = step_reward_for_action(
                action_type=action.action_type,
                task_id=self.task_id,
                actions_so_far=self._actions_taken,
                ground_truth=self._current_gt,
            )
            self._episode_reward += immediate_reward

        # ── Check terminal conditions ──────────────────────────────────────────
        if action.action_type == "done":
            done = True
        elif self._step_number >= self._task_def.max_steps:
            done = True
            print(f"[Env] Max steps reached ({self._task_def.max_steps}), ending episode")

        # ── Final grading on episode end ───────────────────────────────────────
        if done:
            final_reward = grade(
                task_id=self.task_id,
                actions_taken=self._actions_taken,
                ground_truth=self._current_gt,
                steps_used=self._step_number,
                max_steps=self._task_def.max_steps,
                last_action_type=self._last_action_type,
            )
            self._done = True
            self._episode_reward += final_reward.total_score
            print(f"[Env] Episode done — final_score={final_reward.total_score:.3f}, episode_reward={self._episode_reward:.3f}")
            obs = self._make_observation()
            return StepResult(
                observation=obs,
                reward=final_reward.total_score,
                reward_details=final_reward,
                done=True,
                info={
                    "episode_reward": self._episode_reward,
                    "steps_used": self._step_number,
                    "actions_taken": self._actions_taken,
                    "ground_truth_classification": self._current_gt.classification,
                    "ground_truth_priority": self._current_gt.priority,
                    "ground_truth_department": self._current_gt.department,
                },
            )

        # ── Intermediate step ──────────────────────────────────────────────────
        obs = self._make_observation()
        intermediate_reward_details = EmailReward(
            total_score=immediate_reward,
            classification_score=immediate_reward if action.action_type == "classify" else 0.0,
            priority_score=immediate_reward if action.action_type == "prioritize" else 0.0,
            routing_score=immediate_reward if action.action_type == "route" else 0.0,
            response_quality_score=immediate_reward if action.action_type in ("reply", "escalate") else 0.0,
            feedback=f"Intermediate reward for {action.action_type}: {immediate_reward:+.3f}",
        )
        return StepResult(
            observation=obs,
            reward=immediate_reward,
            reward_details=intermediate_reward_details,
            done=False,
            info={"step": self._step_number, "cumulative_reward": self._episode_reward},
        )

    # ── state ─────────────────────────────────────────────────────────────────

    def state(self) -> StateResult:
        """Return current internal state (useful for debugging/monitoring)."""
        print(f"[Env] state() called — step={self._step_number}, done={self._done}")
        return StateResult(
            task_id=self.task_id,
            step_number=self._step_number,
            max_steps=self._task_def.max_steps,
            current_email_id=self._current_email["email_id"] if self._current_email else None,
            classification_done=self._classify_done,
            priority_done=self._priority_done,
            routing_done=self._routing_done,
            reply_done=self._reply_done,
            episode_reward=self._episode_reward,
            is_terminal=self._done,
            actions_taken=self._action_log,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _validate_action(self, action: EmailAction) -> Optional[str]:
        """Return an error string if the action is invalid, else None."""
        at = action.action_type

        if at == "classify" and action.classification is None:
            return "classify action requires 'classification' field"
        if at == "prioritize" and action.priority is None:
            return "prioritize action requires 'priority' field"
        if at == "route" and action.department is None:
            return "route action requires 'department' field"
        if at == "reply" and (not action.reply_text or len(action.reply_text.strip()) < 5):
            return "reply action requires non-empty 'reply_text' (min 5 chars)"
        if at == "escalate" and (not action.escalation_reason or len(action.escalation_reason.strip()) < 5):
            return "escalate action requires non-empty 'escalation_reason' (min 5 chars)"

        # Prevent duplicate actions (except done)
        if at == "classify" and self._classify_done:
            return "classify action already performed; cannot repeat"
        if at == "prioritize" and self._priority_done:
            return "prioritize action already performed; cannot repeat"
        if at == "route" and self._routing_done:
            return "route action already performed; cannot repeat"
        if at in ("reply", "escalate") and self._reply_done:
            return "reply/escalate action already performed; cannot repeat"

        return None

    def _apply_action(self, action: EmailAction) -> str:
        """Persist action to internal state, return human-readable log entry."""
        at = action.action_type

        if at == "classify":
            self._actions_taken["classification"] = action.classification
            self._classify_done = True
            return f"classify({action.classification})"

        if at == "prioritize":
            self._actions_taken["priority"] = action.priority
            self._priority_done = True
            return f"prioritize({action.priority})"

        if at == "route":
            self._actions_taken["department"] = action.department
            self._routing_done = True
            return f"route({action.department})"

        if at == "reply":
            self._actions_taken["reply_text"] = action.reply_text
            self._reply_done = True
            return f"reply('{(action.reply_text or '')[:50]}...')"

        if at == "escalate":
            self._actions_taken["escalation_reason"] = action.escalation_reason
            self._reply_done = True
            return f"escalate('{(action.escalation_reason or '')[:50]}...')"

        return f"{at}()"

    def _make_observation(self) -> EmailObservation:
        assert self._current_email is not None
        return build_observation(
            email_dict=self._current_email,
            task_id=self.task_id,
            task_description=self._task_def.description,
            step_number=self._step_number,
            max_steps=self._task_def.max_steps,
            previous_actions=list(self._action_log),
            last_action_error=self._last_action_error,
        )

    # ── convenience for scripts ───────────────────────────────────────────────

    def close(self) -> None:
        """Clean up resources (no-op for this env)."""
        print(f"[Env] close() called for task_id={self.task_id}")
