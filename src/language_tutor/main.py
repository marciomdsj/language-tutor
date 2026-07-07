"""CLI entry point for the language tutor.

Usage:
    tutor           Run a session (text + voice, TTS enabled)
    tutor --debug   Run with verbose output (latency, STT confidence, etc.)

In a session:
    - Type text + Enter → text input
    - Just press Enter  → voice recording (Enter again to stop)
    - "mute"            → disable TTS
    - "unmute"          → re-enable TTS
    - "quit" / "exit"   → end session (triggers final review)
    - Ctrl+C            → end session (triggers final review)
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from language_tutor import config
from language_tutor.session import SessionRunner

console = Console()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tutor",
        description="Personal language tutor — LLM conversation + spaced repetition",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show verbose output (latency, STT details, tool calls)",
    )
    args = parser.parse_args()

    try:
        runner = SessionRunner(debug=args.debug)
        runner.run()
    except ConnectionError as e:
        console.print(
            f"[bold red]Error:[/bold red] All LLM providers failed.\n"
            f"Tried: {', '.join(config.LLM_MODELS)}\n"
            f"Check: API keys in .env, internet connection, or Ollama running."
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
