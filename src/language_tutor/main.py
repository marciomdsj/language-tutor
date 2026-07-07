"""CLI entry point for the language tutor.

Usage:
    tutor            Run a session (text + voice, TTS enabled)
    tutor --stats    Show learning analytics report
    tutor --debug    Run with verbose output

In a session:
    - Type text + Enter → text input
    - Just press Enter  → voice recording (Enter again to stop)
    - "mute"/"unmute"   → toggle TTS
    - "stats"           → show analytics mid-session
    - "quit"/"exit"     → end session (triggers final review)
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
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show learning analytics report and exit",
    )
    args = parser.parse_args()

    if args.stats:
        _show_stats()
        return

    try:
        runner = SessionRunner(debug=args.debug)
        runner.run()
    except ConnectionError:
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


def _show_stats() -> None:
    """Show the learning analytics report."""
    from language_tutor import analytics, db

    conn = db.get_connection()
    report = analytics.generate_report(conn)
    report.display()
    conn.close()


if __name__ == "__main__":
    main()
