"""Activity planner — suggests the best activity based on learning data.

The planner looks at:
    1. Which skills haven't been practiced recently
    2. How many cards are due for review
    3. Recent error patterns
    4. Activity variety (avoid repeating the same type)

It then presents 2-3 options to the learner and lets them choose.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter

from rich.console import Console
from rich.table import Table

from language_tutor import db
from language_tutor.activities.article_summary import ArticleSummary
from language_tutor.activities.error_correction import ErrorCorrection
from language_tutor.activities.free_conversation import FreeConversation
from language_tutor.activities.writing_prompt import WritingPrompt

console = Console()

# All available activity classes, keyed by type
ACTIVITY_REGISTRY = {
    "free_conversation": FreeConversation,
    "writing_prompt": WritingPrompt,
    "article_summary": ArticleSummary,
    "error_correction": ErrorCorrection,
}


def suggest_activities(conn: sqlite3.Connection) -> list[str]:
    """Analyze learning data and suggest 3 ranked activity types.

    The suggestion logic:
    1. Check which skills are under-practiced (writing, reading, grammar)
    2. Check if there are many due cards (→ free_conversation to review them)
    3. Check recent error density (→ error_correction if many grammar errors)
    4. Avoid repeating the last session's activity
    5. Always include free_conversation as an option

    Args:
        conn: Database connection.

    Returns:
        List of 3 activity_type strings, best suggestion first.
    """
    recent_sessions = db.get_recent_sessions(conn, limit=7)
    recent_errors = db.get_recent_errors(conn, limit=20)
    due_cards = db.get_due_cards(conn)

    # Count skills practiced in the last 7 sessions
    skill_counts: Counter[str] = Counter()
    activity_counts: Counter[str] = Counter()
    for session in recent_sessions:
        skills = json.loads(session.get("skills_practiced", "[]"))
        skill_counts.update(skills)
        activity_counts[session["activity_type"]] += 1

    # Count error types
    error_types: Counter[str] = Counter()
    for error in recent_errors:
        error_types[error.get("error_type", "other")] += 1

    last_activity = recent_sessions[0]["activity_type"] if recent_sessions else None

    # Score each activity
    scores: dict[str, float] = {}

    # Free conversation: good when many cards are due
    scores["free_conversation"] = 5.0
    if len(due_cards) >= 3:
        scores["free_conversation"] += 3.0

    # Writing prompt: good when writing is under-practiced
    scores["writing_prompt"] = 5.0
    if skill_counts.get("writing", 0) < 2:
        scores["writing_prompt"] += 4.0

    # Article summary: good when reading is under-practiced
    scores["article_summary"] = 5.0
    if skill_counts.get("reading", 0) < 1:
        scores["article_summary"] += 4.0

    # Error correction: good when many grammar errors recently
    scores["error_correction"] = 3.0
    grammar_errors = sum(
        v for k, v in error_types.items()
        if k in ("grammar", "preposition", "article", "word_order", "collocation")
    )
    if grammar_errors >= 5:
        scores["error_correction"] += 5.0

    # Penalize repeating the last activity
    if last_activity and last_activity in scores:
        scores[last_activity] -= 3.0

    # Penalize over-practiced activities
    for activity_type, count in activity_counts.items():
        if activity_type in scores and count >= 3:
            scores[activity_type] -= 2.0

    # Sort by score descending, return top 3
    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return ranked[:3]


def present_choices(suggestions: list[str]) -> str:
    """Display activity options and let the learner choose.

    Args:
        suggestions: Ranked list of activity_type strings.

    Returns:
        The chosen activity_type.
    """
    console.print("\n[bold cyan]What would you like to practice today?[/bold cyan]\n")

    table = Table(border_style="cyan", show_header=False)
    table.add_column("", style="bold cyan", width=3)
    table.add_column("Activity", style="bold")
    table.add_column("Description", style="dim")

    for i, activity_type in enumerate(suggestions, 1):
        cls = ACTIVITY_REGISTRY[activity_type]
        table.add_row(str(i), cls.name, cls.description)

    console.print(table)
    console.print()

    while True:
        try:
            raw = console.input(
                f"[dim]Choose (1-{len(suggestions)}, or type name):[/dim] "
            ).strip()
        except EOFError:
            return suggestions[0]

        if not raw:
            return suggestions[0]

        # Number choice
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(suggestions):
                return suggestions[idx]

        # Name match (partial)
        raw_lower = raw.lower()
        for activity_type in suggestions:
            cls = ACTIVITY_REGISTRY[activity_type]
            if raw_lower in cls.name.lower() or raw_lower in activity_type:
                return activity_type

        console.print(f"[dim]Invalid choice. Try 1-{len(suggestions)}.[/dim]")
