"""Application settings loaded from environment variables and sensible defaults.

This module centralizes all configuration so that no other module needs to know
about .env files or environment variables. Every setting has a default that works
out of the box for local development — just have Ollama running with qwen3:8b.
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
# LLM (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Language & learner profile
# ---------------------------------------------------------------------------
TARGET_LANGUAGE: str = "en"
LEARNER_LEVEL: str = "C1"

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
