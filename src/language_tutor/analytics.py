"""Learning analytics — metrics, insights, and progress reports.

This module provides two layers of analysis:

1. **Quantitative metrics (SQL)**: accuracy rates, card distribution,
   error patterns, streaks, leech detection — all from SQLite queries.

2. **Qualitative insights (LLM)**: narrative analysis of learning progress,
   pattern detection, and study recommendations — generated on demand
   by sending accumulated session data to the LLM.

Usage:
    from language_tutor.analytics import generate_report
    report = generate_report(conn)
    # report.display() shows the formatted report in the terminal
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from language_tutor import db, llm

console = Console()


@dataclass
class LearningMetrics:
    """Quantitative metrics computed from SQL queries."""

    total_sessions: int = 0
    total_turns: int = 0
    total_corrections: int = 0
    total_cards: int = 0
    cards_by_status: dict[str, int] = field(default_factory=dict)
    accuracy_by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    top_errors: list[dict] = field(default_factory=list)
    leeches: list[dict] = field(default_factory=list)
    activities_distribution: dict[str, int] = field(default_factory=dict)
    skills_distribution: dict[str, int] = field(default_factory=dict)
    study_streak: int = 0
    words_mastered: int = 0  # cards in review with interval > 21 days


@dataclass
class LearningReport:
    """Complete learning report with metrics and LLM insights."""

    metrics: LearningMetrics
    insights: str = ""

    def display(self) -> None:
        """Render the full report in the terminal."""
        _display_metrics(self.metrics)
        if self.insights:
            console.print()
            console.print(
                Panel(self.insights, title="AI Insights", border_style="magenta")
            )


def generate_report(
    conn: sqlite3.Connection,
    include_insights: bool = True,
) -> LearningReport:
    """Generate a complete learning report with metrics and optional LLM insights.

    Args:
        conn: Database connection.
        include_insights: Whether to call the LLM for qualitative analysis.

    Returns:
        A LearningReport ready for display.
    """
    metrics = _compute_metrics(conn)
    insights = ""

    if include_insights and metrics.total_sessions > 0:
        with console.status(
            "[dim]Generating insights...[/dim]", spinner="dots"
        ):
            insights = _generate_insights(conn, metrics)

    return LearningReport(metrics=metrics, insights=insights)


def _compute_metrics(conn: sqlite3.Connection) -> LearningMetrics:
    """Compute all quantitative metrics from the database.

    Args:
        conn: Database connection.

    Returns:
        Populated LearningMetrics.
    """
    m = LearningMetrics()

    # Session stats
    row = conn.execute(
        """SELECT COUNT(*) as total,
                  COALESCE(SUM(total_turns), 0) as turns,
                  COALESCE(SUM(errors_found), 0) as errors
           FROM sessions WHERE ended_at IS NOT NULL"""
    ).fetchone()
    m.total_sessions = row["total"]
    m.total_turns = row["turns"]
    m.total_corrections = row["errors"]

    # Card stats by status
    m.cards_by_status = db.get_card_stats(conn)
    m.total_cards = m.cards_by_status.pop("total", 0)

    # Accuracy by error type
    error_rows = conn.execute(
        """SELECT error_type, COUNT(*) as count
           FROM corrections
           WHERE error_type IS NOT NULL
           GROUP BY error_type
           ORDER BY count DESC"""
    ).fetchall()
    for row in error_rows:
        m.accuracy_by_type[row["error_type"]] = {"errors": row["count"]}

    # Top repeated errors (same mistake made multiple times)
    top_rows = conn.execute(
        """SELECT user_said, corrected, error_type, COUNT(*) as count
           FROM corrections
           GROUP BY LOWER(user_said), LOWER(corrected)
           HAVING count > 1
           ORDER BY count DESC
           LIMIT 10"""
    ).fetchall()
    m.top_errors = [dict(r) for r in top_rows]

    # Leeches (cards with high leech_count)
    leech_rows = conn.execute(
        """SELECT front, type, leech_count, times_seen, times_correct
           FROM cards
           WHERE leech_count >= 3
           ORDER BY leech_count DESC
           LIMIT 10"""
    ).fetchall()
    m.leeches = [dict(r) for r in leech_rows]

    # Activity distribution
    act_rows = conn.execute(
        """SELECT activity_type, COUNT(*) as count
           FROM sessions WHERE ended_at IS NOT NULL
           GROUP BY activity_type"""
    ).fetchall()
    m.activities_distribution = {r["activity_type"]: r["count"] for r in act_rows}

    # Skills distribution
    skill_rows = conn.execute(
        """SELECT skills_practiced FROM sessions
           WHERE ended_at IS NOT NULL AND skills_practiced != '[]'"""
    ).fetchall()
    from collections import Counter
    skill_counter: Counter[str] = Counter()
    for row in skill_rows:
        skills = json.loads(row["skills_practiced"])
        skill_counter.update(skills)
    m.skills_distribution = dict(skill_counter)

    # Study streak (consecutive days with at least one session)
    day_rows = conn.execute(
        """SELECT DISTINCT DATE(started_at) as day
           FROM sessions WHERE ended_at IS NOT NULL
           ORDER BY day DESC"""
    ).fetchall()
    m.study_streak = _calculate_streak([r["day"] for r in day_rows])

    # Words mastered (review status with interval > 21 days)
    mastered = conn.execute(
        "SELECT COUNT(*) as count FROM cards WHERE status = 'review' AND interval > 21"
    ).fetchone()
    m.words_mastered = mastered["count"]

    return m


def _calculate_streak(days: list[str]) -> int:
    """Calculate consecutive day streak from a list of date strings.

    Args:
        days: Dates in 'YYYY-MM-DD' format, newest first.

    Returns:
        Number of consecutive days including today (or yesterday).
    """
    if not days:
        return 0

    from datetime import date, timedelta

    streak = 0
    expected = date.today()

    for day_str in days:
        day = date.fromisoformat(day_str)
        if day == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif day == expected - timedelta(days=1):
            # Allow starting from yesterday
            streak += 1
            expected = day - timedelta(days=1)
        else:
            break

    return streak


def _generate_insights(
    conn: sqlite3.Connection,
    metrics: LearningMetrics,
) -> str:
    """Ask the LLM to analyze the learning data and generate insights.

    Sends a summary of metrics and recent sessions to the LLM for
    qualitative analysis. Costs ~$0.01 per call (Claude Haiku level).

    Args:
        conn: Database connection.
        metrics: Pre-computed quantitative metrics.

    Returns:
        Narrative text with insights and recommendations.
    """
    # Gather recent session summaries
    recent = db.get_recent_sessions(conn, limit=10)
    recent_errors = db.get_recent_errors(conn, limit=20)

    sessions_text = "\n".join(
        f"- {s['activity_type']}: {s['total_turns']} turns, "
        f"{s['errors_found']} errors ({s['started_at'][:10]})"
        for s in recent
    )

    errors_text = "\n".join(
        f"- \"{e['user_said']}\" → \"{e['corrected']}\" ({e.get('error_type', '')})"
        for e in recent_errors[:10]
    )

    leeches_text = "\n".join(
        f"- \"{l['front']}\" (failed {l['leech_count']}x, "
        f"accuracy: {l['times_correct']}/{l['times_seen']})"
        for l in metrics.leeches
    )

    prompt = f"""Analyze this English learner's progress data and provide a brief,
encouraging report (3-5 paragraphs). Include:
1. Overall progress assessment
2. Strongest and weakest areas
3. Specific recommendations for improvement
4. Motivation and encouragement

LEARNER DATA:
- Level: C1
- Total sessions: {metrics.total_sessions}
- Total conversation turns: {metrics.total_turns}
- Total errors corrected: {metrics.total_corrections}
- Cards in deck: {metrics.total_cards}
- Words mastered (interval > 21 days): {metrics.words_mastered}
- Study streak: {metrics.study_streak} day(s)
- Card status: {json.dumps(metrics.cards_by_status)}
- Error types: {json.dumps(metrics.accuracy_by_type)}
- Skills practiced: {json.dumps(metrics.skills_distribution)}
- Activities: {json.dumps(metrics.activities_distribution)}

RECENT SESSIONS:
{sessions_text or "(no sessions yet)"}

RECENT ERRORS:
{errors_text or "(no errors recorded)"}

PROBLEM WORDS (leeches):
{leeches_text or "(none)"}

Write the report in English. Be specific — reference actual words and error
patterns from the data. Keep it concise and actionable."""

    from language_tutor.llm import _call_llm, _get_content, _strip_tool_artifacts
    response = _call_llm(messages=[{"role": "user", "content": prompt}])
    return _strip_tool_artifacts(_get_content(response))


def _display_metrics(metrics: LearningMetrics) -> None:
    """Render quantitative metrics as Rich tables and panels."""
    # Overview
    console.print(
        Panel(
            f"[bold]Sessions:[/bold] {metrics.total_sessions}  |  "
            f"[bold]Turns:[/bold] {metrics.total_turns}  |  "
            f"[bold]Errors:[/bold] {metrics.total_corrections}  |  "
            f"[bold]Streak:[/bold] {metrics.study_streak} day(s)\n"
            f"[bold]Cards:[/bold] {metrics.total_cards}  |  "
            f"[bold]Mastered:[/bold] {metrics.words_mastered}",
            title="Learning Overview",
            border_style="cyan",
        )
    )

    # Card status distribution
    if metrics.cards_by_status:
        table = Table(title="Card Status", border_style="yellow")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        for status, count in sorted(metrics.cards_by_status.items()):
            table.add_row(status, str(count))
        console.print(table)

    # Error types
    if metrics.accuracy_by_type:
        table = Table(title="Error Distribution", border_style="red")
        table.add_column("Type", style="bold")
        table.add_column("Count", justify="right")
        for err_type, data in sorted(
            metrics.accuracy_by_type.items(),
            key=lambda x: x[1]["errors"],
            reverse=True,
        ):
            table.add_row(err_type, str(data["errors"]))
        console.print(table)

    # Top repeated errors
    if metrics.top_errors:
        table = Table(title="Most Repeated Mistakes", border_style="red")
        table.add_column("You said")
        table.add_column("Correct")
        table.add_column("Times", justify="right")
        for err in metrics.top_errors[:5]:
            table.add_row(
                err["user_said"], err["corrected"], str(err["count"])
            )
        console.print(table)

    # Leeches
    if metrics.leeches:
        table = Table(title="Problem Words (Leeches)", border_style="yellow")
        table.add_column("Word", style="bold")
        table.add_column("Failures", justify="right")
        table.add_column("Accuracy", justify="right")
        for l in metrics.leeches:
            acc = f"{l['times_correct']}/{l['times_seen']}" if l["times_seen"] else "0/0"
            table.add_row(l["front"], str(l["leech_count"]), acc)
        console.print(table)

    # Skills & Activities
    if metrics.skills_distribution or metrics.activities_distribution:
        table = Table(title="Practice Distribution", border_style="green")
        table.add_column("Category", style="bold")
        table.add_column("Item")
        table.add_column("Count", justify="right")
        for skill, count in sorted(
            metrics.skills_distribution.items(), key=lambda x: x[1], reverse=True
        ):
            table.add_row("Skill", skill, str(count))
        for act, count in sorted(
            metrics.activities_distribution.items(), key=lambda x: x[1], reverse=True
        ):
            table.add_row("Activity", act, str(count))
        console.print(table)
