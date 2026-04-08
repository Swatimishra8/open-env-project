"""
Baseline inference script for the OpenEnv Email Triage environment.

Runs an LLM agent against all three tasks using the OpenAI client.
Reads credentials from environment variables:
    API_BASE_URL  — LLM API endpoint (provided by validator)
    API_KEY       — API key (provided by validator, fallback to OPENAI_API_KEY)
    MODEL_NAME    — Model identifier
    HF_TOKEN      — Hugging Face token (fallback if no API_KEY)
    ENV_BASE_URL  — Base URL of the running OpenEnv server (default: http://localhost:7860)

Usage:
    python inference.py
    python inference.py --task task_classify
    python inference.py --task all --emails 5
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from config import USE_OPENAI, USE_HF, API_KEY, HF_TOKEN, API_BASE_URL, MODEL_NAME, ENV_BASE_URL
from env.models import EmailAction

# Initialize the appropriate client based on config
DEMO_MODE = API_KEY == "demo-mode"

if DEMO_MODE:
    client = None  # Mock client
    print(f"[Inference] Using DEMO MODE with model: {MODEL_NAME}")
    print("[Inference] ⚠️  This will simulate realistic agent responses for demonstration")
elif USE_OPENAI:
    from openai import OpenAI
    # Use validator's API_KEY and API_BASE_URL if provided
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    print(f"[Inference] Using OpenAI client with model: {MODEL_NAME}")
    print(f"[Inference] API_BASE_URL: {API_BASE_URL}")
    print(f"[Inference] Using API_KEY: {'***' + API_KEY[-4:] if len(API_KEY) > 4 else 'provided'}")
elif USE_HF:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)
    print(f"[Inference] Using Hugging Face client with model: {MODEL_NAME}")
else:
    print("[Inference] ERROR: No API key found. Set API_KEY or HF_TOKEN in .env")
    sys.exit(1)

MAX_STEPS: int = 10
TEMPERATURE: float = 0.0  # deterministic for reproducibility
MAX_TOKENS: int = 512
DEBUG: bool = os.getenv("DEBUG", "0") == "1"
EMAILS_PER_TASK: int = int(os.getenv("EMAILS_PER_TASK", "5"))  # keep runtime < 20 min

TASKS = ["task_classify", "task_priority_routing", "task_full_triage"]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage agent. Your job is to process incoming emails and take the correct actions.

For each email you will receive its content (sender, subject, body, urgency indicators) and the current task requirements.

You must respond with a JSON action object. The possible actions are:

1. classify — Classify the email category
   {"action_type": "classify", "classification": "<spam|inquiry|complaint|order|support|feedback|billing>"}

2. prioritize — Assign priority level
   {"action_type": "prioritize", "priority": "<urgent|high|normal|low>"}

3. route — Route to the correct department
   {"action_type": "route", "department": "<sales|support|billing|technical|management|hr|legal>"}

4. reply — Generate a response to the email
   {"action_type": "reply", "reply_text": "<your reply here>"}

5. escalate — Escalate the email with a reason
   {"action_type": "escalate", "escalation_reason": "<reason for escalation>"}

6. done — Signal that you have completed the task
   {"action_type": "done"}

CRITICAL REQUIREMENTS:
- Output MUST be valid JSON
- Do NOT include explanations  
- Do NOT include markdown
- Use ONLY lowercase values for classification, priority, department
- Choose EXACTLY ONE value per field - NO comma-separated lists
- Example: {"action_type": "route", "department": "support"} NOT "support, billing"
- Example: {"action_type": "classify", "classification": "spam"} NOT "Spam"

Classification guide:
- spam: unsolicited bulk email, phishing, scams
- inquiry: questions about products, services, pricing
- complaint: expressing dissatisfaction, reporting problems
- order: purchase requests, order modifications, bulk orders
- support: technical help, how-to questions, account issues
- feedback: product feedback, feature requests, suggestions
- billing: invoice questions, payment issues, subscription changes

Priority guide:
- urgent: immediate action needed (data loss, legal threat, 24hr deadline, system outage)
- high: important issue needing prompt attention (damaged goods, overcharge, week deadline)
- normal: standard requests with no time pressure
- low: feedback, suggestions, informational only

Department guide (choose EXACTLY ONE):
- sales: pricing, quotes, new business, bulk orders
- support: technical issues, account problems, how-to help
- billing: invoices, payments, refunds, subscriptions
- technical: API/integration issues, developer questions
- management: escalated complaints, executive contacts
- legal: legal threats, contracts, compliance issues
- hr: employment, recruitment

IMPORTANT: Each email goes to ONE department only. Choose the MOST appropriate single department."""

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(
    step: int,
    observation: dict,
    history: List[str],
    task_id: str,
) -> str:
    task_guidance = {
        "task_classify": (
            "TASK: Basic Email Classification\n"
            "You must: (1) call 'classify' with the correct category, then (2) call 'done'."
        ),
        "task_priority_routing": (
            "TASK: Priority Assignment and Routing\n"
            "You must: (1) call 'classify', (2) call 'prioritize', (3) call 'route', then (4) call 'done'."
        ),
        "task_full_triage": (
            "TASK: Full Triage with Response Generation\n"
            "You must: (1) call 'classify', (2) call 'prioritize', (3) call 'route', "
            "(4) call 'reply' (or 'escalate' if urgent/complaint), then (5) call 'done'."
        ),
    }

    previous_actions_str = ""
    if history:
        previous_actions_str = "\nPrevious actions this episode:\n" + "\n".join(
            f"  {h}" for h in history[-5:]
        )

    urgency_str = ""
    if observation.get("urgency_indicators"):
        urgency_str = f"\nUrgency indicators: {', '.join(observation['urgency_indicators'])}"

    attachments_str = ""
    if observation.get("attachments"):
        attachments_str = f"\nAttachments: {', '.join(observation['attachments'])}"

    return f"""{task_guidance.get(task_id, '')}

Step {step} of {observation.get('max_steps', MAX_STEPS)}

--- EMAIL ---
From: {observation['sender']}
Subject: {observation['subject']}
{urgency_str}{attachments_str}

{observation['body']}
--- END EMAIL ---
{previous_actions_str}

What is your next action? Respond with JSON only."""


def parse_action(response_text: str) -> Optional[dict]:
    import json
    import re

    text = response_text.strip()

    # Remove markdown
    text = text.replace("```json", "").replace("```", "").strip()

    # 🔥 Extract ALL JSON objects
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)  # ✅ return FIRST valid JSON
        except json.JSONDecodeError:
            continue

    return None


# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    # Required structured output for validator
    print(f"[START] task={task}", flush=True)
    
    # Additional logging for debugging
    print(f"\n{'='*60}")
    print(f"  Task:  {task}")
    print(f"  Model: {model}")
    print(f"  ENV:   {ENV_BASE_URL}")
    print(f"{'='*60}")


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    # Required structured output for validator
    print(f"[STEP] step={step} reward={reward}", flush=True)
    
    # Additional logging for debugging
    status = "DONE" if done else f"reward={reward:+.3f}"
    error_str = f" [ERROR: {error}]" if error else ""
    print(f"  Step {step:2d} | {action:<50} | {status}{error_str}")


def log_end(success: bool, steps: int, rewards: List[float], final_score: float, task: str = "") -> None:
    # Required structured output for validator
    print(f"[END] task={task} score={final_score} steps={steps}", flush=True)
    
    # Additional logging for debugging
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n  Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Steps:  {steps}")
    print(f"  Final score: {final_score:.3f}")
    print(f"  Avg step reward: {avg:.3f}")
    print(f"{'='*60}\n")


# ── HTTP client helpers ───────────────────────────────────────────────────────

def api_reset(http: httpx.Client, task_id: str) -> dict:
    resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_step(http: httpx.Client, task_id: str, action: dict) -> dict:
    resp = http.post(
        f"{ENV_BASE_URL}/step",
        json={"task_id": task_id, "action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(
    client: OpenAI,
    http: httpx.Client,
    task_id: str,
    emails_count: int = EMAILS_PER_TASK,
) -> Dict[str, Any]:
    """Run the agent on `emails_count` emails for a given task. Returns aggregate scores."""

    print(f"\n[Inference] Starting task={task_id} for {emails_count} emails", flush=True)
    all_scores: List[float] = []
    success_count = 0

    for email_idx in range(emails_count):
        print(f"\n[Inference] Email {email_idx + 1}/{emails_count}")
        log_start(task=task_id, model=MODEL_NAME)

        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        success = False
        final_score = 0.0

        # Reset environment
        try:
            reset_result = api_reset(http, task_id)
            observation = reset_result["observation"]
            done = reset_result.get("done", False)
        except Exception as exc:
            print(f"[Inference] ERROR: reset() failed: {exc}")
            all_scores.append(0.0)
            continue

        # Run episode
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_user_prompt(step, observation, history, task_id)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM
            try:
                if DEMO_MODE:
                    # Smart demo responses based on email content and previous actions
                    email_subject = observation.get("subject", "").lower()
                    email_body = observation.get("body", "").lower()
                    previous_actions = observation.get("previous_actions", [])
                    
                    # If already classified, simulate multi-department routing (for testing)
                    if any("classify(" in action for action in previous_actions):
                        if step == 2:  # Second step after classification
                            response_text = '{"action_type": "route", "department": "billing, support"}'
                        else:
                            response_text = '{"action_type": "done"}'
                    # Otherwise, classify based on content
                    elif "congratulations" in email_subject or "won" in email_subject:
                        response_text = '{"action_type": "classify", "classification": "spam"}'
                    elif "unacceptable" in email_body or "complaint" in email_body:
                        response_text = '{"action_type": "classify", "classification": "complaint"}'
                    elif "question" in email_subject or "inquiry" in email_subject:
                        response_text = '{"action_type": "classify", "classification": "inquiry"}'
                    elif "order" in email_body or "purchase" in email_body:
                        response_text = '{"action_type": "classify", "classification": "order"}'
                    elif "help" in email_body or "support" in email_body or "crash" in email_body:
                        response_text = '{"action_type": "classify", "classification": "support"}'
                    elif "feedback" in email_subject or "update" in email_subject:
                        response_text = '{"action_type": "classify", "classification": "feedback"}'
                    elif "billing" in email_body or "invoice" in email_body:
                        response_text = '{"action_type": "classify", "classification": "billing"}'
                    else:
                        # Simulate the multi-department issue for testing
                        if step == 2 and any("classify(" in action for action in previous_actions):
                            response_text = '{"action_type": "route", "department": "billing, support"}'
                        else:
                            response_text = '{"action_type": "classify", "classification": "inquiry"}'
                elif USE_OPENAI:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                elif USE_HF:
                    # Convert messages to a simple prompt for HF text generation
                    prompt_parts = []
                    for msg in messages:
                        if msg["role"] == "system":
                            prompt_parts.append(f"System: {msg['content']}")
                        elif msg["role"] == "user":
                            prompt_parts.append(f"User: {msg['content']}")
                    
                    full_prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
                    
                    # Use a simple text generation model that works
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    print(f"[Inference] Response text: {response_text}")
                else:
                    response_text = '{"action_type": "done"}'  # fallback
            except Exception as exc:
                print(f"[Inference] LLM request failed at step {step}: {exc}")
                # Fallback: call done
                response_text = '{"action_type": "done"}'

            if DEBUG:
                print(f"[DEBUG] LLM response: {response_text[:200]}")

            # Parse action
            action_dict = parse_action(response_text)
            print(f"[Inference] Action dict: {action_dict}")
            if action_dict is None:
                print(f"[Inference] Invalid JSON → fallback to done")
                action_dict = {"action_type": "done"}

            # ✅ Validate using Pydantic EmailAction model
            try:
                action_obj = EmailAction(**action_dict)
                action_dict = action_obj.model_dump()  # clean + normalized
                if DEBUG:
                    print(f"[DEBUG] Pydantic validation passed: {action_dict}")
            except Exception as e:
                print(f"[Inference] Pydantic validation failed: {e}")
                print(f"[Inference] Raw action was: {action_dict}")
                action_dict = {"action_type": "done"}

            # Step environment
            try:
                step_result = api_step(http, task_id, action_dict)
            except Exception as exc:
                print(f"[Inference] ERROR: step() failed: {exc}")
                break

            observation = step_result["observation"]
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            error = observation.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action_dict)
            log_step(step=step, action=action_str[:50], reward=reward, done=done, error=error)

            history_line = f"Step {step}: {action_str[:60]} → reward {reward:+.3f}"
            if error:
                history_line += " [ERROR]"
            history.append(history_line)

            if done:
                final_score = reward
                success = reward >= 0.5
                break
        else:
            # Exhausted steps without done — force final step
            try:
                step_result = api_step(http, task_id, {"action_type": "done"})
                final_score = step_result.get("reward", 0.0)
                success = final_score >= 0.5
            except Exception:
                final_score = 0.0
                success = False

        all_scores.append(final_score)
        if success:
            success_count += 1

        log_end(success=success, steps=steps_taken, rewards=rewards, final_score=final_score, task=task_id)

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[Inference] Task {task_id} complete — avg_score={avg_score:.3f}, success_rate={success_count}/{emails_count}", flush=True)

    return {
        "task_id": task_id,
        "emails_evaluated": emails_count,
        "scores": all_scores,
        "avg_score": avg_score,
        "success_count": success_count,
        "success_rate": success_count / emails_count if emails_count > 0 else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OpenEnv Email Triage baseline inference")
    parser.add_argument("--task", default="all", help="Task ID or 'all'")
    parser.add_argument("--emails", type=int, default=EMAILS_PER_TASK, help="Emails per task")
    parser.add_argument("--env-url", default=ENV_BASE_URL, help="Environment server URL")
    args = parser.parse_args()

    # Override from args (use local variables)
    env_base_url = args.env_url
    emails_per_task = args.emails

    # API key validation already done at module level

    api_provider = "OpenAI" if USE_OPENAI else "HuggingFace" if USE_HF else "None"
    print(f"[Inference] API Provider: {api_provider}")
    print(f"[Inference] API_BASE_URL={API_BASE_URL}")
    print(f"[Inference] MODEL_NAME={MODEL_NAME}")
    print(f"[Inference] ENV_BASE_URL={env_base_url}")
    print(f"[Inference] EMAILS_PER_TASK={emails_per_task}")

    # Client already initialized at module level

    # Check environment health
    with httpx.Client() as http:
        try:
            health = http.get(f"{env_base_url}/health", timeout=10)
            health.raise_for_status()
            print(f"[Inference] Environment health: {health.json()}")
        except Exception as exc:
            print(f"[Inference] WARNING: Could not reach environment at {env_base_url}: {exc}")
            print("[Inference] Make sure the server is running: python app.py")
            sys.exit(1)

        # Determine tasks to run
        tasks_to_run = TASKS if args.task == "all" else [args.task]

        all_results: List[Dict] = []
        start_time = time.time()

        for task_id in tasks_to_run:
            result = run_task(client, http, task_id, emails_count=emails_per_task)
            all_results.append(result)

        elapsed = time.time() - start_time

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Runtime: {elapsed:.1f}s")
    print()

    for r in all_results:
        print(f"  Task: {r['task_id']}")
        print(f"    Emails evaluated: {r['emails_evaluated']}")
        print(f"    Avg score:        {r['avg_score']:.3f}")
        print(f"    Success rate:     {r['success_count']}/{r['emails_evaluated']} ({r['success_rate']*100:.1f}%)")
        print(f"    All scores:       {[f'{s:.2f}' for s in r['scores']]}")
        print()

    overall_avg = sum(r["avg_score"] for r in all_results) / len(all_results)
    print(f"  Overall average score: {overall_avg:.3f}")
    print("=" * 60)

    # Save results to JSON for reproducibility
    results_path = "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "emails_per_task": EMAILS_PER_TASK,
                "runtime_seconds": elapsed,
                "overall_avg_score": overall_avg,
                "task_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n[Inference] Results saved to {results_path}")


if __name__ == "__main__":
    main()
