"""Tests for the SM-2 spaced repetition algorithm and card lifecycle."""

from __future__ import annotations

import sqlite3

import pytest

from language_tutor import db
from language_tutor.srs import (
    QUALITY_AGAIN,
    QUALITY_EASY,
    QUALITY_GOOD,
    QUALITY_HARD,
    calculate_sm2,
    review_card,
)


class TestCalculateSM2:
    """Tests for the pure SM-2 calculation (Wozniak's real formula)."""

    def test_first_correct_good_sets_interval_to_1(self) -> None:
        # Arrange & Act
        result = calculate_sm2(
            repetitions=0, interval=1.0, ease_factor=2.5, quality=QUALITY_GOOD,
        )

        # Assert
        assert result["repetitions"] == 1
        assert result["interval"] == 1.0
        # EF: 2.5 + (0.1 - 2*0.12) = 2.5 - 0.14 = 2.36
        assert result["ease_factor"] == 2.36

    def test_first_correct_easy_boosts_ease(self) -> None:
        # Act
        result = calculate_sm2(
            repetitions=0, interval=1.0, ease_factor=2.5, quality=QUALITY_EASY,
        )

        # Assert
        assert result["repetitions"] == 1
        assert result["interval"] == 1.0
        # EF: 2.5 + 0.1 = 2.6
        assert result["ease_factor"] == 2.6

    def test_second_correct_sets_interval_to_6(self) -> None:
        # Act
        result = calculate_sm2(
            repetitions=1, interval=1.0, ease_factor=2.5, quality=QUALITY_GOOD,
        )

        # Assert
        assert result["repetitions"] == 2
        assert result["interval"] == 6.0

    def test_third_correct_multiplies_by_ease(self) -> None:
        # Act
        result = calculate_sm2(
            repetitions=2, interval=6.0, ease_factor=2.5, quality=QUALITY_GOOD,
        )

        # Assert
        assert result["repetitions"] == 3
        # interval = 6.0 * new_ease (2.36) = 14.16
        assert result["interval"] == 14.16

    def test_incorrect_resets_repetitions(self) -> None:
        # Act
        result = calculate_sm2(
            repetitions=5, interval=30.0, ease_factor=2.5, quality=QUALITY_AGAIN,
        )

        # Assert
        assert result["repetitions"] == 0
        assert result["interval"] == 1.0
        # EF: 2.5 + (0.1 - 5*0.18) = 2.5 - 0.8 = 1.7
        assert result["ease_factor"] == 1.7

    def test_hard_rating_resets_but_less_harsh(self) -> None:
        # Act
        result = calculate_sm2(
            repetitions=3, interval=15.0, ease_factor=2.5, quality=QUALITY_HARD,
        )

        # Assert — hard is still a failure (quality < 3)
        assert result["repetitions"] == 0
        assert result["interval"] == 1.0
        # EF: 2.5 + (0.1 - 3*0.14) = 2.5 - 0.32 = 2.18
        assert result["ease_factor"] == 2.18

    def test_ease_factor_never_drops_below_1_3(self) -> None:
        # Arrange — already at minimum
        # Act
        result = calculate_sm2(
            repetitions=0, interval=1.0, ease_factor=1.3, quality=QUALITY_AGAIN,
        )

        # Assert
        assert result["ease_factor"] == 1.3

    def test_long_success_streak_grows_interval(self) -> None:
        # Arrange & Act — 5 consecutive GOOD answers
        state = {"repetitions": 0, "interval": 1.0, "ease_factor": 2.5}
        for _ in range(5):
            state = calculate_sm2(**state, quality=QUALITY_GOOD)

        # Assert
        assert state["repetitions"] == 5
        assert state["interval"] > 30  # substantial growth
        assert state["ease_factor"] < 2.5  # GOOD (q=3) slowly reduces ease

    def test_easy_streak_grows_interval_faster(self) -> None:
        # Arrange & Act — 5 consecutive EASY answers
        state = {"repetitions": 0, "interval": 1.0, "ease_factor": 2.5}
        for _ in range(5):
            state = calculate_sm2(**state, quality=QUALITY_EASY)

        # Assert — EASY should grow faster than GOOD
        assert state["repetitions"] == 5
        assert state["interval"] > 80  # much larger than GOOD streak
        assert state["ease_factor"] > 2.5  # EASY increases ease


class TestReviewCardLifecycle:
    """Tests for review_card with card lifecycle transitions."""

    def test_new_card_first_correct_goes_to_learning(
        self, conn: sqlite3.Connection, new_card: int
    ) -> None:
        # Act
        result = review_card(conn, new_card, quality=QUALITY_GOOD)

        # Assert
        assert result.old_status == "new"
        assert result.new_status == "learning"  # needs 1 more correct use

        card = db.get_card(conn, new_card)
        assert card["learning_step"] == 1

    def test_learning_card_graduates_to_review(
        self, conn: sqlite3.Connection, learning_card: int
    ) -> None:
        # Arrange — card is at learning step 1 (needs 1 more)
        # Act
        result = review_card(conn, learning_card, quality=QUALITY_GOOD)

        # Assert
        assert result.old_status == "learning"
        assert result.new_status == "review"  # graduated!
        assert result.interval == 1.0  # first real SM-2 interval

    def test_learning_card_failure_resets_to_step_0(
        self, conn: sqlite3.Connection, learning_card: int
    ) -> None:
        # Act
        result = review_card(conn, learning_card, quality=QUALITY_AGAIN)

        # Assert
        assert result.new_status == "learning"
        card = db.get_card(conn, learning_card)
        assert card["learning_step"] == 0  # reset

    def test_review_card_failure_goes_to_relearning(
        self, conn: sqlite3.Connection, review_card_fixture: int
    ) -> None:
        # Act
        result = review_card(conn, review_card_fixture, quality=QUALITY_AGAIN)

        # Assert
        assert result.old_status == "review"
        assert result.new_status == "relearning"

    def test_review_card_success_stays_in_review(
        self, conn: sqlite3.Connection, review_card_fixture: int
    ) -> None:
        # Act
        result = review_card(conn, review_card_fixture, quality=QUALITY_GOOD)

        # Assert
        assert result.new_status == "review"
        assert result.interval > 1.0  # interval grew

    def test_relearning_card_success_returns_to_review(
        self, conn: sqlite3.Connection
    ) -> None:
        # Arrange — create a card in relearning status
        card_id = db.create_card(conn, front="test_relearn", card_type="word")
        conn.execute(
            """UPDATE cards SET status = 'relearning', interval = 10.0,
               ease_factor = 2.3 WHERE id = ?""",
            (card_id,),
        )
        conn.commit()

        # Act
        result = review_card(conn, card_id, quality=QUALITY_GOOD)

        # Assert
        assert result.old_status == "relearning"
        assert result.new_status == "review"
        assert result.interval == 5.0  # 10.0 * 0.5 — reduced interval

    def test_invalid_quality_raises(
        self, conn: sqlite3.Connection, new_card: int
    ) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid quality"):
            review_card(conn, new_card, quality=4)

    def test_nonexistent_card_raises(self, conn: sqlite3.Connection) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="Card 9999 not found"):
            review_card(conn, 9999, quality=QUALITY_GOOD)


class TestLeechDetection:
    """Tests for automatic leech suspension."""

    def test_card_becomes_leech_after_threshold_failures(
        self, conn: sqlite3.Connection
    ) -> None:
        # Arrange — card with leech_count just below threshold
        card_id = db.create_card(conn, front="leech_word", card_type="word")
        conn.execute(
            f"""UPDATE cards SET status = 'review', repetitions = 1,
                leech_count = {db.LEECH_THRESHOLD - 1}
                WHERE id = ?""",
            (card_id,),
        )
        conn.commit()

        # Act — one more failure tips it over
        result = review_card(conn, card_id, quality=QUALITY_AGAIN)

        # Assert
        assert result.is_leech is True
        assert result.new_status == "suspended"

        card = db.get_card(conn, card_id)
        assert card["status"] == "suspended"
        assert card["leech_count"] == db.LEECH_THRESHOLD

    def test_correct_answer_does_not_increment_leech(
        self, conn: sqlite3.Connection, new_card: int
    ) -> None:
        # Act
        result = review_card(conn, new_card, quality=QUALITY_GOOD)

        # Assert
        assert result.is_leech is False
        card = db.get_card(conn, new_card)
        assert card["leech_count"] == 0
