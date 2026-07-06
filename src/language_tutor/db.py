"""SQLite database layer — schema creation and CRUD operations.

This module owns the database connection and provides functions to create,
read, update, and query cards, sessions, and corrections.

Card lifecycle (mirrors Anki):
    new → learning → review ⇄ relearning → (suspended if leech)

    - **new**: just created, never reviewed
    - **learning**: needs N correct uses to graduate to review
    - **review**: SM-2 algorithm controls intervals
    - **relearning**: failed during review, needs 1 correct use to re-enter
    - **suspended**: paused (manually or auto-leech)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from language_tutor import config

# ---------------------------------------------------------------------------
# Type alias for a database row returned as a dictionary
# ---------------------------------------------------------------------------
Row = dict[str, Any]

# Number of consecutive failures before a card is flagged as a leech
LEECH_THRESHOLD = 8

# Number of correct uses required to graduate from learning to review
LEARNING_STEPS = 2

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lang          TEXT    NOT NULL DEFAULT 'en',
    type          TEXT    NOT NULL CHECK(type IN ('word', 'phrase', 'grammar')),
    front         TEXT    NOT NULL,
    back          TEXT,
    context       TEXT,
    tags          TEXT    DEFAULT '[]',
    -- Card lifecycle: new → learning → review ⇄ relearning | suspended
    status        TEXT    NOT NULL DEFAULT 'new'
                  CHECK(status IN ('new','learning','review','relearning','suspended')),
    learning_step INTEGER NOT NULL DEFAULT 0,
    -- SRS fields (SM-2 algorithm, active only in 'review' status)
    interval      REAL    NOT NULL DEFAULT 1.0,
    ease_factor   REAL    NOT NULL DEFAULT 2.5,
    repetitions   INTEGER NOT NULL DEFAULT 0,
    next_review   TEXT    NOT NULL DEFAULT (datetime('now')),
    -- Stats
    times_seen    INTEGER NOT NULL DEFAULT 0,
    times_correct INTEGER NOT NULL DEFAULT 0,
    leech_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at       TEXT,
    lang           TEXT    NOT NULL DEFAULT 'en',
    topic          TEXT,
    total_turns    INTEGER NOT NULL DEFAULT 0,
    errors_found   INTEGER NOT NULL DEFAULT 0,
    cards_reviewed INTEGER NOT NULL DEFAULT 0,
    summary        TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    card_id    INTEGER REFERENCES cards(id),
    user_said  TEXT    NOT NULL,
    corrected  TEXT    NOT NULL,
    error_type TEXT,
    explanation TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema exists.

    Args:
        db_path: Override path for testing. Uses config.DB_PATH by default.

    Returns:
        A sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Cards CRUD
# ---------------------------------------------------------------------------

def create_card(
    conn: sqlite3.Connection,
    front: str,
    card_type: str = "word",
    back: str | None = None,
    context: str | None = None,
    tags: list[str] | None = None,
    lang: str = "en",
) -> int:
    """Insert a new card and return its id.

    Args:
        conn: Database connection.
        front: The word, phrase, or grammar structure.
        card_type: One of 'word', 'phrase', 'grammar'.
        back: Definition, translation, or correct usage example.
        context: Sentence where the item appeared.
        tags: Topic tags (e.g. ["technology", "grammar"]).
        lang: ISO language code.

    Returns:
        The auto-generated card id.
    """
    tags_json = json.dumps(tags or [])
    cur = conn.execute(
        """INSERT INTO cards (front, type, back, context, tags, lang)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (front, card_type, back, context, tags_json, lang),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_card(conn: sqlite3.Connection, card_id: int) -> Row | None:
    """Fetch a single card by id.

    Args:
        conn: Database connection.
        card_id: The card's primary key.

    Returns:
        A dict-like Row or None if not found.
    """
    row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    return dict(row) if row else None


def get_due_cards(
    conn: sqlite3.Connection,
    lang: str = "en",
    limit: int | None = None,
) -> list[Row]:
    """Return reviewable cards whose next_review is now or in the past.

    Only includes cards in 'new', 'learning', 'review', or 'relearning'
    status.  Suspended cards are excluded.

    Args:
        conn: Database connection.
        lang: Filter by language.
        limit: Max number of cards to return.

    Returns:
        List of card rows ordered by next_review (oldest first).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    limit = limit or config.MAX_DUE_CARDS_PER_TURN
    rows = conn.execute(
        """SELECT * FROM cards
           WHERE lang = ?
             AND status != 'suspended'
             AND next_review <= ?
           ORDER BY
             -- Prioritize: relearning > learning > new > review
             CASE status
               WHEN 'relearning' THEN 0
               WHEN 'learning'   THEN 1
               WHEN 'new'        THEN 2
               WHEN 'review'     THEN 3
             END,
             next_review ASC
           LIMIT ?""",
        (lang, now, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def update_card_srs(
    conn: sqlite3.Connection,
    card_id: int,
    *,
    interval: float,
    ease_factor: float,
    repetitions: int,
    next_review: str,
    status: str,
    learning_step: int,
    correct: bool,
    leech_count: int,
) -> None:
    """Update a card's SRS fields and lifecycle status after a review.

    Args:
        conn: Database connection.
        card_id: The card to update.
        interval: New interval in days.
        ease_factor: New ease factor.
        repetitions: New consecutive correct count.
        next_review: ISO datetime of next review.
        status: New lifecycle status.
        learning_step: Current learning step (0-based).
        correct: Whether the learner got it right (updates stats).
        leech_count: Cumulative failure count for leech detection.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """UPDATE cards
           SET interval      = ?,
               ease_factor   = ?,
               repetitions   = ?,
               next_review   = ?,
               status        = ?,
               learning_step = ?,
               times_seen    = times_seen + 1,
               times_correct = times_correct + ?,
               leech_count   = ?,
               updated_at    = ?
           WHERE id = ?""",
        (
            interval, ease_factor, repetitions, next_review,
            status, learning_step, int(correct), leech_count,
            now, card_id,
        ),
    )
    conn.commit()


def suspend_card(conn: sqlite3.Connection, card_id: int) -> None:
    """Suspend a card (manually or due to leech detection).

    Args:
        conn: Database connection.
        card_id: The card to suspend.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE cards SET status = 'suspended', updated_at = ? WHERE id = ?",
        (now, card_id),
    )
    conn.commit()


def unsuspend_card(conn: sqlite3.Connection, card_id: int) -> None:
    """Reactivate a suspended card back to 'new' status.

    Args:
        conn: Database connection.
        card_id: The card to unsuspend.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """UPDATE cards
           SET status = 'new', learning_step = 0, repetitions = 0,
               interval = 1.0, updated_at = ?
           WHERE id = ? AND status = 'suspended'""",
        (now, card_id),
    )
    conn.commit()


def get_card_stats(conn: sqlite3.Connection, lang: str = "en") -> Row:
    """Return summary stats about the card deck.

    Args:
        conn: Database connection.
        lang: Filter by language.

    Returns:
        Dict with counts per status and total.
    """
    rows = conn.execute(
        """SELECT status, COUNT(*) as count
           FROM cards WHERE lang = ?
           GROUP BY status""",
        (lang,),
    ).fetchall()
    stats = {row["status"]: row["count"] for row in rows}
    stats["total"] = sum(stats.values())
    return stats


# ---------------------------------------------------------------------------
# Sessions CRUD
# ---------------------------------------------------------------------------

def create_session(conn: sqlite3.Connection, lang: str = "en") -> int:
    """Start a new conversation session.

    Args:
        conn: Database connection.
        lang: Session language.

    Returns:
        The new session id.
    """
    cur = conn.execute(
        "INSERT INTO sessions (lang) VALUES (?)",
        (lang,),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def end_session(
    conn: sqlite3.Connection,
    session_id: int,
    total_turns: int,
    errors_found: int,
    cards_reviewed: int,
    summary: str | None = None,
) -> None:
    """Close a session with final stats.

    Args:
        conn: Database connection.
        session_id: The session to close.
        total_turns: Number of conversation turns.
        errors_found: Number of errors the LLM identified.
        cards_reviewed: Number of SRS cards that appeared.
        summary: Optional end-of-session summary text.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """UPDATE sessions
           SET ended_at       = ?,
               total_turns    = ?,
               errors_found   = ?,
               cards_reviewed = ?,
               summary        = ?
           WHERE id = ?""",
        (now, total_turns, errors_found, cards_reviewed, summary, session_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Corrections CRUD
# ---------------------------------------------------------------------------

def create_correction(
    conn: sqlite3.Connection,
    session_id: int,
    user_said: str,
    corrected: str,
    error_type: str | None = None,
    explanation: str | None = None,
    card_id: int | None = None,
) -> int:
    """Record an error the LLM found in the learner's text.

    Args:
        conn: Database connection.
        session_id: Which session this occurred in.
        user_said: What the learner actually said/wrote.
        corrected: The correct version.
        error_type: Category (e.g. 'grammar', 'vocabulary', 'preposition').
        explanation: Brief explanation of the error.
        card_id: Link to an existing card, if applicable.

    Returns:
        The new correction id.
    """
    cur = conn.execute(
        """INSERT INTO corrections
           (session_id, card_id, user_said, corrected, error_type, explanation)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, card_id, user_said, corrected, error_type, explanation),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_recent_errors(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> list[Row]:
    """Return the most recent corrections across all sessions.

    Used to feed the system prompt so the tutor knows what the learner
    has been struggling with and can steer the conversation accordingly.

    Args:
        conn: Database connection.
        limit: Max number of corrections to return.

    Returns:
        List of correction rows, newest first.
    """
    rows = conn.execute(
        """SELECT user_said, corrected, error_type, explanation
           FROM corrections
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
