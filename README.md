---
title: OpenEnv Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
license: apache-2.0
---

# 📧 OpenEnv Email Triage Environment

A complete, real-world OpenEnv environment for training AI agents on email triage tasks.

## 🎯 What is this?

This is a **production-ready OpenEnv environment** that simulates realistic email triage scenarios. AI agents can learn to:

- **Classify emails** (spam, urgent, normal, promotional)
- **Set priorities** (high, medium, low) 
- **Route to departments** (sales, support, billing, technical, management, hr, legal)
- **Generate appropriate responses**
- **Escalate when needed**

## 🚀 API Endpoints

The FastAPI server exposes these OpenEnv-compliant endpoints:

- `GET /health` - Health check
- `GET /tasks` - List available tasks
- `POST /reset` - Reset environment to start new episode
- `POST /step` - Take an action in the environment
- `GET /state` - Get current environment state

## 📊 Available Tasks

1. **task_classify** - Basic email classification (Easy)
   - Learn to categorize emails by type and urgency
   
2. **task_priority_routing** - Priority + department routing (Medium) 
   - Assign priorities and route to correct departments
   
3. **task_full_triage** - Complete triage with responses (Hard)
   - Full email handling including response generation

## 🏆 Key Features

- **Progressive Difficulty**: 3 tasks with increasing complexity
- **Realistic Data**: Synthetic emails based on real-world patterns
- **Comprehensive Scoring**: Detailed feedback with partial credit
- **Type Safety**: Pydantic models for all data structures
- **Production Ready**: FastAPI server with proper error handling
- **Docker Deployment**: Containerized for easy deployment

## 🔧 Technical Stack

- **Framework**: FastAPI + Pydantic + OpenEnv specification
- **Language**: Python 3.11+
- **Deployment**: Docker container
- **API**: RESTful endpoints following OpenEnv standard

## 📈 Performance Metrics

The environment provides detailed scoring across multiple dimensions:
- Classification accuracy
- Priority assignment precision  
- Department routing correctness
- Response quality evaluation
- Efficiency bonuses and penalties

## 🚀 Quick Start

```python
import requests

# Health check
response = requests.get("https://huggingface.co/spaces/Swatimishra/open-env/health")

# Start new episode
reset_response = requests.post("https://huggingface.co/spaces/Swatimishra/open-env/reset", 
                              json={"task_id": "task_classify"})
observation = reset_response.json()

# Take an action
action = {
    "action_type": "classify",
    "classification": "urgent", 
    "priority": "high",
    "department": "support"
}
step_response = requests.post("https://huggingface.co/spaces/Swatimishra/open-env/step", 
                             json=action)
result = step_response.json()