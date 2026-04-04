"""Generates realistic synthetic email datasets for each task difficulty level."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from env.models import ClassificationType, DepartmentType, EmailGroundTruth, PriorityType

# ── random seed helpers ────────────────────────────────────────────────────────
SEED = 42
_rng = random.Random(SEED)

# ── name / company pools ───────────────────────────────────────────────────────
FIRST_NAMES = ["Alice", "Bob", "Carlos", "Diana", "Ethan", "Fatima", "George",
               "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina",
               "Oscar", "Priya", "Quinn", "Rachel", "Sam", "Tina"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Martinez", "Hernandez", "Lopez", "Wilson",
              "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
COMPANIES = ["Acme Corp", "TechStart", "DataWorks", "GlobalSales", "NexGen",
             "BlueSky", "Innovate Ltd", "QuantumLeap", "ZenithCo", "PrimeSoft"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.org",
                 "business.net", "enterprise.io"]
PRODUCTS = ["Pro Subscription", "Enterprise Suite", "Analytics Dashboard",
            "API Access Package", "Developer Tool Kit", "Premium Support Plan"]
VERSIONS = ["3.2.1", "4.0.0", "2.9.5", "5.1.0", "3.8.2"]
OS_NAMES = ["Windows 11", "macOS Ventura", "Ubuntu 22.04", "Windows 10"]


def _rand_date(days_back: int = 60) -> datetime:
    offset = timedelta(days=_rng.randint(0, days_back),
                       hours=_rng.randint(0, 23),
                       minutes=_rng.randint(0, 59))
    return datetime.utcnow() - offset


def _substitute(template: str) -> str:
    """Fill template placeholders with random realistic values."""
    first = _rng.choice(FIRST_NAMES)
    last = _rng.choice(LAST_NAMES)
    company = _rng.choice(COMPANIES)
    domain = _rng.choice(EMAIL_DOMAINS)

    substitutions = {
        "{sender_name}": f"{first} {last}",
        "{first}": first.lower(),
        "{last}": last.lower(),
        "{company}": company.replace(" ", "").lower(),
        "{Company}": company,
        "{email_domain}": domain,
        "{account_email}": f"{first.lower()}.{last.lower()}@{domain}",
        "{order_id}": str(_rng.randint(10000, 99999)),
        "{invoice_id}": f"INV-{_rng.randint(1000, 9999)}",
        "{po_number}": f"PO-{_rng.randint(100, 999)}",
        "{account_id}": str(_rng.randint(100000, 999999)),
        "{price_per}": str(_rng.randint(50, 500)),
        "{total_price}": str(_rng.randint(1000, 50000)),
        "{amount}": str(_rng.randint(100, 5000)),
        "{wrong_amount}": str(_rng.randint(150, 600)),
        "{correct_amount}": str(_rng.randint(50, 149)),
        "{missing_amount}": f"{_rng.randint(1, 4)} hours",
        "{version}": _rng.choice(VERSIONS),
        "{os_name}": _rng.choice(OS_NAMES),
        "{product_name}": _rng.choice(PRODUCTS),
        "{product_1}": _rng.choice(PRODUCTS),
        "{product_2}": _rng.choice(PRODUCTS),
        "{product_3}": _rng.choice(PRODUCTS),
        "{item_name}": _rng.choice(PRODUCTS),
        "{current_item}": _rng.choice(PRODUCTS),
        "{new_item}": _rng.choice(PRODUCTS),
        "{current_qty}": str(_rng.randint(1, 20)),
        "{new_qty}": str(_rng.randint(1, 20)),
        "{years}": str(_rng.randint(1, 10)),
        "{title}": _rng.choice(["Manager", "Director", "VP", "Analyst", "Engineer"]),
        "{address}": f"{_rng.randint(1, 999)} Business Ave, Suite {_rng.randint(100, 999)}",
        "{charge_date}": _rand_date(30).strftime("%Y-%m-%d"),
        "{invoice_date}": _rand_date(30).strftime("%Y-%m-%d"),
        "{order_date}": _rand_date(30).strftime("%Y-%m-%d"),
        "{expected_date}": (_rand_date(30) - timedelta(days=7)).strftime("%Y-%m-%d"),
        "{today_date}": datetime.utcnow().strftime("%Y-%m-%d"),
        "{deadline_date}": (datetime.utcnow() + timedelta(days=_rng.randint(7, 30))).strftime("%Y-%m-%d"),
        "{last_save}": f"{_rng.randint(1, 60)} minutes ago",
        "{occasion}": _rng.choice(["birthday", "anniversary", "graduation", "holiday"]),
    }

    for key, val in substitutions.items():
        template = template.replace(key, val)
    return template


def _make_sender(template_sender: str) -> str:
    first = _rng.choice(FIRST_NAMES).lower()
    last = _rng.choice(LAST_NAMES).lower()
    company = _rng.choice(COMPANIES).replace(" ", "").lower()
    domain = _rng.choice(EMAIL_DOMAINS)
    return (template_sender
            .replace("{first}", first)
            .replace("{last}", last)
            .replace("{company}", company)
            .replace("{email_domain}", domain))


def load_templates() -> dict:
    template_path = Path(__file__).parent.parent / "data" / "email_templates.json"
    with open(template_path) as f:
        return json.load(f)


def load_department_rules() -> dict:
    rules_path = Path(__file__).parent.parent / "data" / "department_rules.json"
    with open(rules_path) as f:
        return json.load(f)


# ── ground truth helpers ───────────────────────────────────────────────────────

_CLASSIFICATION_TO_DEPARTMENT: dict[ClassificationType, DepartmentType] = {
    "spam": "support",         # spam → support triage
    "inquiry": "sales",
    "complaint": "management",
    "order": "sales",
    "support": "support",
    "feedback": "management",
    "billing": "billing",
}

_CLASSIFICATION_TO_PRIORITY: dict[ClassificationType, PriorityType] = {
    "spam": "low",
    "inquiry": "normal",
    "complaint": "high",
    "order": "normal",
    "support": "normal",
    "feedback": "low",
    "billing": "normal",
}

_URGENCY_PRIORITY_OVERRIDE: dict[str, PriorityType] = {
    "immediately": "urgent",
    "urgent": "urgent",
    "asap": "urgent",
    "emergency": "urgent",
    "critical": "urgent",
    "24 hours": "urgent",
    "data loss": "urgent",
    "legal": "urgent",
    "lawsuit": "urgent",
    "cancel": "high",
    "dispute": "high",
    "negative review": "high",
    "damaged": "high",
    "overcharge": "high",
    "refund": "high",
    "billing error": "high",
    "deadline": "high",
    "before it ships": "high",
}

_EXPECTED_REPLY_KEYWORDS: dict[ClassificationType, list[str]] = {
    "spam": ["spam", "marked", "reported", "block"],
    "inquiry": ["thank you", "information", "here are", "please find", "happy to help"],
    "complaint": ["sorry", "apologize", "understand", "resolve", "immediately", "refund"],
    "order": ["confirmed", "order", "processing", "shipped", "tracking"],
    "support": ["help", "steps", "solution", "fixed", "resolved", "issue"],
    "feedback": ["thank you", "feedback", "appreciate", "consider", "team"],
    "billing": ["invoice", "payment", "account", "processed", "billing team"],
}


def _infer_priority(classification: ClassificationType, urgency_indicators: list[str]) -> PriorityType:
    urgency_lower = [u.lower() for u in urgency_indicators]
    for indicator in urgency_lower:
        for trigger, prio in _URGENCY_PRIORITY_OVERRIDE.items():
            if trigger in indicator:
                # bump up but don't exceed current
                existing = _CLASSIFICATION_TO_PRIORITY[classification]
                prio_order = ["low", "normal", "high", "urgent"]
                if prio_order.index(prio) > prio_order.index(existing):
                    return prio
                return existing
    return _CLASSIFICATION_TO_PRIORITY[classification]


def _infer_department(classification: ClassificationType, urgency_indicators: list[str]) -> DepartmentType:
    urgency_lower = " ".join(u.lower() for u in urgency_indicators)
    legal_triggers = ["legal", "lawsuit", "attorney", "sue", "court"]
    if any(t in urgency_lower for t in legal_triggers):
        return "legal"
    complaint_triggers = ["immediately", "unacceptable", "manager", "supervisor", "escalate"]
    if classification == "complaint" and any(t in urgency_lower for t in complaint_triggers):
        return "management"
    billing_triggers = ["billing", "invoice", "refund", "overcharge", "cancel"]
    if any(t in urgency_lower for t in billing_triggers):
        return "billing"
    return _CLASSIFICATION_TO_DEPARTMENT[classification]


# ── public API ─────────────────────────────────────────────────────────────────

def generate_emails(
    task_difficulty: str = "easy",
    count: int = 20,
    seed: int = SEED,
) -> List[Tuple[dict, EmailGroundTruth]]:
    """
    Returns a list of (email_dict, ground_truth) tuples.
    email_dict keys: email_id, sender, subject, body, timestamp, attachments, urgency_indicators
    """
    print(f"[EmailGenerator] Generating {count} emails for difficulty={task_difficulty}, seed={seed}")
    local_rng = random.Random(seed)
    templates = load_templates()

    categories: List[ClassificationType]
    if task_difficulty == "easy":
        # Only very clear-cut categories, no ambiguity
        categories = ["spam", "inquiry", "complaint", "order", "support"]
    elif task_difficulty == "medium":
        # Include billing and feedback too, and add more urgency indicators
        categories = ["spam", "inquiry", "complaint", "order", "support", "feedback", "billing"]
    else:  # hard
        # All categories, some with overlapping signals to increase ambiguity
        categories = ["spam", "inquiry", "complaint", "order", "support", "feedback", "billing"]

    results: List[Tuple[dict, EmailGroundTruth]] = []

    for i in range(count):
        classification: ClassificationType = local_rng.choice(categories)
        template_list = templates.get(classification, templates["inquiry"])
        template = local_rng.choice(template_list)

        # For hard difficulty, randomly inject cross-category confusion
        if task_difficulty == "hard" and local_rng.random() < 0.3:
            # Mix urgency signals from complaint into other categories
            complaint_urgency = ["URGENT: please help", "I need this resolved NOW",
                                  "This is completely unacceptable", "I will escalate this further"]
            extra_body = f"\n\nNOTE: {local_rng.choice(complaint_urgency)}"
            body = _substitute(template["body"]) + extra_body
        else:
            body = _substitute(template["body"])

        subject = _substitute(template["subject"])
        sender = _make_sender(template["sender"])
        urgency_indicators = list(template.get("urgency_indicators", []))
        attachments = list(template.get("attachments", []))

        # For hard difficulty, sometimes add misleading urgency indicators
        if task_difficulty == "hard" and local_rng.random() < 0.2:
            urgency_indicators.append("time-sensitive")

        priority = _infer_priority(classification, urgency_indicators)
        department = _infer_department(classification, urgency_indicators)
        should_escalate = (
            priority == "urgent"
            or department in ["management", "legal"]
            or (task_difficulty == "hard" and local_rng.random() < 0.15)
        )

        email_id = f"email_{i:04d}_{uuid.uuid4().hex[:8]}"
        ts = _rand_date(60)
        ts = ts.replace(tzinfo=None)  # naive UTC for JSON simplicity

        email_dict = {
            "email_id": email_id,
            "sender": sender,
            "subject": subject,
            "body": body,
            "timestamp": ts.isoformat(),
            "attachments": attachments,
            "urgency_indicators": urgency_indicators,
        }

        ground_truth = EmailGroundTruth(
            email_id=email_id,
            classification=classification,
            priority=priority,
            department=department,
            expected_reply_keywords=_EXPECTED_REPLY_KEYWORDS[classification],
            should_escalate=should_escalate,
            escalation_triggers=[u for u in urgency_indicators if any(
                t in u.lower() for t in ["immediately", "urgent", "legal", "cancel", "crisis"]
            )],
            difficulty=task_difficulty,
        )

        results.append((email_dict, ground_truth))

    print(f"[EmailGenerator] Generated {len(results)} emails successfully")
    return results
