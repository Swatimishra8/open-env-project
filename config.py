import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Validator provides API_KEY, fallback to OPENAI_API_KEY for local testing
API_KEY: str = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# Use validator's API_KEY if provided, otherwise check our keys
USE_OPENAI = bool(API_KEY)
USE_HF = not USE_OPENAI and bool(HF_TOKEN)

if USE_HF and MODEL_NAME == "gpt-4o-mini":
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")