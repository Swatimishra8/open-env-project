# OpenEnv Email Triage Environment

A real-world email triage environment for training and evaluating AI agents. Agents learn to classify, prioritize, route, and respond to realistic corporate emails through a standard `step()` / `reset()` / `state()` API.

---

## Motivation

Email management is one of the most universally-performed knowledge-work tasks. A skilled human assistant processes hundreds of emails daily by quickly classifying their type, assessing urgency, routing to the right team, and drafting appropriate responses.

This environment teaches AI agents the same skill through three progressive tasks with meaningful, decomposed reward signals — making it an ideal real-world benchmark for language-model agents.

---

## Environment Description

The environment simulates a corporate inbox receiving realistic synthetic emails across 7 categories:

| Category    | Description                                        |
|-------------|---------------------------------------------------|
| `spam`      | Unsolicited bulk email, phishing attempts         |
| `inquiry`   | Product questions, pricing requests               |
| `complaint` | Customer dissatisfaction, service issues          |
| `order`     | Purchase requests, order modifications            |
| `support`   | Technical help, account access problems           |
| `feedback`  | Product feedback, feature requests                |
| `billing`   | Invoice questions, payment issues, subscriptions  |

---

## Action Space

Each step the agent submits one of the following actions:

```json
// Classify the email type
{"action_type": "classify", "classification": "<spam|inquiry|complaint|order|support|feedback|billing>"}

// Assign priority level
{"action_type": "prioritize", "priority": "<urgent|high|normal|low>"}

// Route to the correct department
{"action_type": "route", "department": "<sales|support|billing|technical|management|hr|legal>"}

// Write a reply
{"action_type": "reply", "reply_text": "Dear customer, thank you for reaching out..."}

// Escalate the email
{"action_type": "escalate", "escalation_reason": "Legal threat requires immediate management attention"}

// Signal task completion
{"action_type": "done"}
```

---

## Observation Space

After each action, the agent receives:

```json
{
  "email_id": "email_0003_a1b2c3d4",
  "sender": "john.smith@acmecorp.com",
  "subject": "Unacceptable service - Order #45231 still not received",
  "body": "To the Customer Service Team, I placed order #45231 over THREE WEEKS AGO...",
  "timestamp": "2026-03-15T14:30:00",
  "attachments": [],
  "previous_actions": ["classify(complaint)", "prioritize(high)"],
  "department_context": {
    "sales": "Handles new business, pricing inquiries, quotes, bulk orders",
    "support": "Technical help, product issues, how-to questions, account access",
    ...
  },
  "urgency_indicators": ["IMMEDIATELY", "unacceptable", "dispute", "24 hours"],
  "task_id": "task_priority_routing",
  "task_description": "Classify, prioritize, and route each email...",
  "step_number": 3,
  "max_steps": 5,
  "last_action_error": null
}
```

---

## Tasks

### Task 1: Basic Email Classification (Easy)

- **ID**: `task_classify`
- **Objective**: Classify each email into the correct category
- **Required actions**: `classify` → `done`
- **Max steps**: 3
- **Target score**: 0.8
- **Reward**: +1.0 correct, +0.4 adjacent category, 0.0 wrong, +0.1 efficiency bonus

### Task 2: Priority Assignment and Routing (Medium)

- **ID**: `task_priority_routing`
- **Objective**: Classify + assign priority + route to department
- **Required actions**: `classify` → `prioritize` → `route` → `done`
- **Max steps**: 5
- **Target score**: 0.7
- **Reward weights**: classification 40%, priority 30%, routing 30%, +0.1 efficiency

### Task 3: Full Triage with Response Generation (Hard)

- **ID**: `task_full_triage`
- **Objective**: Complete triage + generate reply or escalation
- **Required actions**: `classify` → `prioritize` → `route` → `reply`/`escalate` → `done`
- **Max steps**: 8
- **Target score**: 0.6
- **Reward weights**: classification 25%, priority 20%, routing 20%, response 35%, +0.1 efficiency

---

## Reward Function

Rewards are given at **every step** (partial progress) as well as at episode end (final graded score).

| Signal                  | When                          | Amount            |
|-------------------------|-------------------------------|-------------------|
| Correct classification  | After `classify` action       | +0.30 (partial)   |
| Correct priority        | After `prioritize` action     | +0.20 (partial)   |
| Correct routing         | After `route` action          | +0.20 (partial)   |
| Reply quality           | After `reply` action          | up to +0.20       |
| Correct escalation      | After `escalate` action       | +0.15             |
| Unnecessary escalation  | After `escalate` action       | -0.10             |
| Efficiency bonus        | Episode end (fast completion) | +0.10             |
| Critical misclassify    | Episode end penalty           | -0.50             |
| Final graded score      | Episode end                   | 0.0–1.0           |

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### Local Installation

```bash
git clone https://github.com/your-username/openenv-email-triage
cd openenv-email-triage

pip install fastapi "uvicorn[standard]" pydantic pydantic-settings openai httpx python-dotenv pyyaml numpy rich typer
```

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=your_hugging_face_token
ENV_BASE_URL=http://localhost:7860
```

---

## Usage

### Start the Environment Server

```bash
python app.py
# Server starts at http://localhost:7860
```

### Test the API

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset for task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_classify"}'

# Take a classify action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_classify",
    "action": {
      "action_type": "classify",
      "classification": "complaint"
    }
  }'

# Signal done
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_classify", "action": {"action_type": "done"}}'
```

### Run Baseline Inference

```bash
# Run all tasks (5 emails each)
python inference.py

# Run specific task
python inference.py --task task_classify --emails 10

# With custom model
MODEL_NAME=gpt-4o python inference.py --emails 5
```

---

## Docker

### Build

```bash
docker build -t openenv-email-triage .
```

### Run

```bash
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  openenv-email-triage
```

---

## Hugging Face Spaces Deployment

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Add the tag `openenv` to the Space
4. Push this repository or upload files
5. Set the following Space secrets:
   - `OPENAI_API_KEY`
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`

The Space will automatically build and deploy using the Dockerfile. It will expose port 7860.

---

## API Reference

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/health`        | Health check, returns `{"status": "ok"}` |
| GET    | `/tasks`         | List all tasks with metadata             |
| GET    | `/tasks/{id}`    | Get specific task details                |
| POST   | `/reset`         | Reset environment, get first observation |
| POST   | `/step`          | Submit action, get next observation      |
| GET    | `/state`         | Get current environment state            |

---

## Baseline Scores

Baseline scores using `gpt-4o-mini` on 5 emails per task:

| Task                          | Difficulty | Avg Score | Success Rate |
|-------------------------------|------------|-----------|--------------|
| Basic Email Classification    | Easy       | ~0.82     | ~80%         |
| Priority Assignment & Routing | Medium     | ~0.68     | ~60%         |
| Full Triage with Response     | Hard       | ~0.55     | ~50%         |

*Scores are approximate and depend on model version and temperature.*

---

## Project Structure

```
openenv-project/
├── env/
│   ├── __init__.py          # Package exports
│   ├── environment.py       # Core OpenEnv interface (step/reset/state)
│   ├── models.py            # Pydantic typed models
│   ├── tasks.py             # Task definitions and dataset management
│   ├── grader.py            # Scoring logic with partial credit
│   ├── email_generator.py   # Synthetic email creation
│   └── utils.py             # Reward helpers and observation builders
├── data/
│   ├── email_templates.json # Email templates by category
│   └── department_rules.json# Routing and priority rules
├── openenv.yaml             # OpenEnv metadata specification
├── app.py                   # FastAPI server
├── inference.py             # Baseline inference script
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Disqualification Checks

- [x] Environment deploys and responds to `/health`
- [x] `openenv.yaml` is present with full metadata
- [x] Typed Pydantic models for Observation, Action, Reward
- [x] `step()`, `reset()`, `state()` endpoints implemented
- [x] Dockerfile builds successfully
- [x] `inference.py` in root directory using OpenAI client
- [x] 3+ tasks with programmatic graders (scores 0.0–1.0)
- [x] Graders return different scores (deterministic, not constant)
- [x] Environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`

---

## License

MIT License. See LICENSE for details.
