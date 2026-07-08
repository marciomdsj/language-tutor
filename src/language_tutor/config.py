"""Application settings loaded from environment variables and sensible defaults.

This module centralizes all configuration so that no other module needs to know
about .env files or environment variables.  Every setting has a default that
works out of the box — just set a GROQ_API_KEY in .env and you're ready.

LLM provider cascade:
    1. Groq   (primary — fast, free tier: 14,400 req/day)
    2. Gemini (fallback — free tier: 1,500 req/day)
    3. Ollama (offline — always available, no internet needed)
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Load .env from project root (if it exists)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = _PROJECT_ROOT / "data"
DB_PATH: Path = DATA_DIR / "tutor.db"

# ---------------------------------------------------------------------------
# LLM — provider cascade via LiteLLM
# ---------------------------------------------------------------------------
# Each entry is a LiteLLM model string: "provider/model-name"
# The system tries them in order; if one fails (rate limit, timeout, offline),
# it falls through to the next.
LLM_MODELS: list[str] = []

# Build the cascade based on which API keys are available
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

if GROQ_API_KEY:
    LLM_MODELS.append(os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"))
if GEMINI_API_KEY:
    LLM_MODELS.append(os.getenv("GEMINI_MODEL", "gemini/gemini-2.0-flash"))
# Ollama is always available as last resort (local, no API key)
LLM_MODELS.append(f"ollama/{OLLAMA_MODEL}")

# The primary model (first in cascade) — used for display
PRIMARY_MODEL: str = LLM_MODELS[0]

# ---------------------------------------------------------------------------
# Language & learner profile
# ---------------------------------------------------------------------------
TARGET_LANGUAGE: str = "en"
LEARNER_LEVEL: str = "C1"
USER_PROFILE_PATH: Path = DATA_DIR / "user_profile.json"

# ---------------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------------
SESSION_DURATION_MINUTES: int = int(os.getenv("SESSION_DURATION", "15"))
MAX_DUE_CARDS_PER_TURN: int = 5

# ---------------------------------------------------------------------------
# TTS provider: "edge" | "openai" | "piper"
# ---------------------------------------------------------------------------
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "edge")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# STT provider: "vosk" | "deepgram"
# ---------------------------------------------------------------------------
STT_PROVIDER: str = os.getenv("STT_PROVIDER", "vosk")
DEEPGRAM_API_KEY: str | None = os.getenv("DEEPGRAM_API_KEY")
