"""Tests for the database module (schema, CRUD, queries)."""

from __future__ import annotations

import sqlite3

from language_tutor import db


class TestCardsCRUD:
    """Tests for card creation and retrieval."""

    def test_create_card_returns_valid_id(self, conn: sqlite3.Connection) -> None:
        # Arrange & Act
        card_id = db.create_card(conn, front="however", card_type="word")

        # Assert
        assert card_id is not None
        assert card_id > 0

    def test_get_card_returns_all_fields(self, conn: sqlite3.Connection) -> None:
        # Arrange
        card_id = db.create_card(
            conn,
            front="make a decision",
            card_type="phrase",
            back="to decide; to choose",
            context="I need to make a decision.",
            tags=["business", "collocations"],
        )

        # Act
        card = db.get_card(conn, card_id)

        # Assert
        assert card is not None
        assert card["front"] == "make a decision"
        assert card["type"] == "phrase"
        assert card["back"] == "to decide; to choose"
        assert card["status"] == "new"
        assert card["learning_step"] == 0
        assert card["ease_factor"] == 2.5
        assert card["leech_count"] == 0
        assert '"business"' in card["tags"]

    def test_get_card_returns_none_for_missing_id(
        self, conn: sqlite3.Connection
    ) -> None:
        # Act
        card = db.get_card(conn, 9999)

        # Assert
        assert card is None

    def test_create_card_with_invalid_type_raises(
        self, conn: sqlite3.Connection
    ) -> None:
        # Act & Assert
        try:
            db.create_card(conn, front="test", card_type="invalid_type")
            assert False, "Should have raised an error"
        except sqlite3.IntegrityError:
            pass


class TestDueCards:
    """Tests for the get_due_cards query."""

    def test_new_card_is_immediately_due(self, conn: sqlite3.Connection) -> None:
        # Arrange
        db.create_card(conn, front="ubiquitous", card_type="word")

        # Act
        due = db.get_due_cards(conn)

        # Assert
        assert len(due) == 1
        assert due[0]["front"] == "ubiquitous"

    def test_suspended_card_is_not_due(self, conn: sqlite3.Connection) -> None:
        # Arrange
        card_id = db.create_card(conn, front="suspended_word", card_type="word")
        db.suspend_card(conn, card_id)

        # Act
        due = db.get_due_cards(conn)

        # Assert
        assert len(due) == 0

    def test_due_cards_respects_limit(self, conn: sqlite3.Connection) -> None:
        # Arrange
        for word in ["one", "two", "three", "four", "five", "six"]:
            db.create_card(conn, front=word, card_type="word")

        # Act
        due = db.get_due_cards(conn, limit=3)

        # Assert
        assert len(due) == 3

    def test_due_cards_prioritizes_relearning(
        self, conn: sqlite3.Connection
    ) -> None:
        # Arrange
        id_new = db.create_card(conn, front="new_word", card_type="word")
        id_relearn = db.create_card(conn, front="relearn_word", card_type="word")
        conn.execute(
            "UPDATE cards SET status = 'relearning' WHERE id = ?", (id_relearn,)
        )
        conn.commit()

        # Act
        due = db.get_due_cards(conn)

        # Assert — relearning should come first
        assert due[0]["front"] == "relearn_word"
        assert due[1]["front"] == "new_word"


class TestSuspend:
    """Tests for suspend/unsuspend functionality."""

    def test_suspend_and_unsuspend_card(self, conn: sqlite3.Connection) -> None:
        # Arrange
        card_id = db.create_card(conn, front="test", card_type="word")

        # Act — suspend
        db.suspend_card(conn, card_id)
        card = db.get_card(conn, card_id)
        assert card["status"] == "suspended"

        # Act — unsuspend
        db.unsuspend_card(conn, card_id)
        card = db.get_card(conn, card_id)
        assert card["status"] == "new"
        assert card["learning_step"] == 0


class TestSessionsCRUD:
    """Tests for session lifecycle."""

    def test_create_and_end_session(self, conn: sqlite3.Connection) -> None:
        # Arrange
        session_id = db.create_session(conn)

        # Act
        db.end_session(
            conn,
            session_id=session_id,
            total_turns=10,
            errors_found=3,
            cards_reviewed=5,
            summary="Good session overall.",
        )

        # Assert
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        assert row is not None
        assert row["total_turns"] == 10
        assert row["errors_found"] == 3
        assert row["ended_at"] is not None


class TestCorrectionsCRUD:
    """Tests for correction records."""

    def test_create_correction_links_to_session(
        self, conn: sqlite3.Connection
    ) -> None:
        # Arrange
        session_id = db.create_session(conn)

        # Act
        correction_id = db.create_correction(
            conn,
            session_id=session_id,
            user_said="I goed to the store",
            corrected="I went to the store",
            error_type="grammar",
            explanation="Irregular past tense of 'go'.",
        )

        # Assert
        row = conn.execute(
            "SELECT * FROM corrections WHERE id = ?", (correction_id,)
        ).fetchone()
        assert row is not None
        assert row["session_id"] == session_id
        assert row["error_type"] == "grammar"
