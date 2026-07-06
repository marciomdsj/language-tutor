"""Shared fixtures for the test suite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from language_tutor import db


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Provide a fresh database for each test.

    Uses tmp_path so each test gets an isolated SQLite file that is
    automatically cleaned up after the test finishes.
    """
    db_path = tmp_path / "test_tutor.db"
    connection = db.get_connection(db_path)
    yield connection
    connection.close()


@pytest.fixture
def new_card(conn: sqlite3.Connection) -> int:
    """Create a card in 'new' status and return its id."""
    return db.create_card(
        conn, front="thoroughly", card_type="word",
        back="completely; in detail", context="Review it thoroughly.",
    )


@pytest.fixture
def learning_card(conn: sqlite3.Connection) -> int:
    """Create a card in 'learning' status (step 1 of 2) and return its id."""
    card_id = db.create_card(
        conn, front="would have been", card_type="grammar",
        back="third conditional structure",
    )
    # Simulate one correct use → learning step 1
    conn.execute(
        """UPDATE cards SET status = 'learning', learning_step = 1
           WHERE id = ?""",
        (card_id,),
    )
    conn.commit()
    return card_id


@pytest.fixture
def review_card_fixture(conn: sqlite3.Connection) -> int:
    """Create a card in 'review' status with some history and return its id."""
    card_id = db.create_card(
        conn, front="ubiquitous", card_type="word",
        back="present everywhere",
    )
    conn.execute(
        """UPDATE cards SET status = 'review', repetitions = 3,
           interval = 15.0, ease_factor = 2.6, times_seen = 5,
           times_correct = 4
           WHERE id = ?""",
        (card_id,),
    )
    conn.commit()
    return card_id
