"""SM-2 spaced repetition algorithm with Anki-style card lifecycle.

This is the REAL SM-2 algorithm by Piotr Wozniak, not a simplification.
The key difference from our previous version: quality ratings 0-5 instead
of binary correct/incorrect, plus card lifecycle transitions.

Quality ratings (adapted for conversational context):
    0 (again)  — didn't use the word, or used it incorrectly
    2 (hard)   — used it, but awkwardly or with hesitation
    3 (good)   — used it correctly and naturally
    5 (easy)   — used it perfectly, immediate and confident

    (1 and 4 are reserved for finer granularity if needed later)

Card lifecycle:
    new → learning → review ⇄ relearning → (suspended if leech)

    - new:        first time seeing this card
    - learning:   must be used correctly LEARNING_STEPS times to graduate
    - review:     SM-2 controls intervals (1d, 6d, 15d, ...)
    - relearning: failed during review, needs 1 correct use to re-enter
    - suspended:  auto-suspended after LEECH_THRESHOLD failures

SM-2 ease factor formula:
    EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

    This adjusts the ease factor based on quality:
    q=5 → EF + 0.10 (gets much easier)
    q=4 → EF + 0.00 (unchanged)
    q=3 → EF - 0.14 (slightly harder)
    q=2 → EF - 0.32 (harder)
    q=0 → EF - 0.80 (much harder)
    Minimum EF is always 1.3.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from language_tutor import db

# Valid quality ratings that the user can give
QUALITY_AGAIN = 0
QUALITY_HARD = 2
QUALITY_GOOD = 3
QUALITY_EASY = 5


@dataclass
class ReviewResult:
    """The outcome of reviewing a card, returned for display/logging."""

    card_id: int
    quality: int
    old_status: str
    new_status: str
    interval: float
    ease_factor: float
    repetitions: int
    next_review: str
    is_leech: bool


def review_card(
    conn: sqlite3.Connection,
    card_id: int,
    quality: int,
) -> ReviewResult:
    """Apply SM-2 with lifecycle transitions and persist the result.

    This function handles the full review flow:
    1. Load the card's current state
    2. Determine the lifecycle transition based on quality and status
    3. Calculate new SM-2 values (if in review status)
    4. Check for leech condition
    5. Persist everything to the database

    Args:
        conn: Database connection.
        card_id: The card that was reviewed.
        quality: Rating 0-5 (use QUALITY_AGAIN/HARD/GOOD/EASY constants).

    Returns:
        ReviewResult with the full outcome for display.

    Raises:
        ValueError: If card_id does not exist or quality is invalid.
    """
    if quality not in (QUALITY_AGAIN, QUALITY_HARD, QUALITY_GOOD, QUALITY_EASY):
        raise ValueError(f"Invalid quality {quality}. Use 0, 2, 3, or 5.")

    card = db.get_card(conn, card_id)
    if card is None:
        raise ValueError(f"Card {card_id} not found")

    old_status = card["status"]
    correct = quality >= QUALITY_GOOD

    # Calculate new SRS state based on current lifecycle status
    new = _transition(card, quality)

    # Leech detection: increment on failure, check threshold
    leech_count = card["leech_count"] + (0 if correct else 1)
    is_leech = leech_count >= db.LEECH_THRESHOLD and not correct

    if is_leech:
        new["status"] = "suspended"

    # Calculate next review datetime
    now = datetime.now(timezone.utc)
    next_review = now + timedelta(days=new["interval"])
    next_review_str = next_review.strftime("%Y-%m-%d %H:%M:%S")

    # Persist to database
    db.update_card_srs(
        conn,
        card_id=card_id,
        interval=new["interval"],
        ease_factor=new["ease_factor"],
        repetitions=new["repetitions"],
        next_review=next_review_str,
        status=new["status"],
        learning_step=new["learning_step"],
        correct=correct,
        leech_count=leech_count,
    )

    return ReviewResult(
        card_id=card_id,
        quality=quality,
        old_status=old_status,
        new_status=new["status"],
        interval=new["interval"],
        ease_factor=new["ease_factor"],
        repetitions=new["repetitions"],
        next_review=next_review_str,
        is_leech=is_leech,
    )


def _transition(card: db.Row, quality: int) -> dict:
    """Determine new SRS state based on current status and quality.

    This is where the card lifecycle logic lives.  Each status has its own
    rules for what happens on success vs failure.

    Args:
        card: The current card state from the database.
        quality: The quality rating (0-5).

    Returns:
        Dict with keys: interval, ease_factor, repetitions, status, learning_step.
    """
    status = card["status"]
    correct = quality >= QUALITY_GOOD

    if status in ("new", "learning"):
        return _handle_learning(card, quality, correct)
    elif status == "review":
        return _handle_review(card, quality, correct)
    elif status == "relearning":
        return _handle_relearning(card, quality, correct)
    else:
        # Suspended cards shouldn't be reviewed, but handle gracefully
        return _handle_learning(card, quality, correct)


def _handle_learning(card: db.Row, quality: int, correct: bool) -> dict:
    """Handle a review for a card in 'new' or 'learning' status.

    Learning cards need LEARNING_STEPS correct uses to graduate.
    Each correct use advances the learning step.  A failure resets to step 0.

    Args:
        card: Current card state.
        quality: Quality rating.
        correct: Whether quality >= GOOD.

    Returns:
        New SRS state dict.
    """
    step = card["learning_step"]
    ease = card["ease_factor"]

    if correct:
        step += 1
        if step >= db.LEARNING_STEPS:
            # Graduate to review — first real SM-2 interval
            return {
                "interval": 1.0,
                "ease_factor": ease,
                "repetitions": 1,
                "status": "review",
                "learning_step": step,
            }
        else:
            # Still learning — short interval (review again soon)
            return {
                "interval": 0.0007,  # ~1 minute (for same-session re-exposure)
                "ease_factor": ease,
                "repetitions": 0,
                "status": "learning",
                "learning_step": step,
            }
    else:
        # Failed — reset to step 0
        return {
            "interval": 0.0007,
            "ease_factor": max(1.3, ease - 0.2),
            "repetitions": 0,
            "status": "learning",
            "learning_step": 0,
        }


def _handle_review(card: db.Row, quality: int, correct: bool) -> dict:
    """Handle a review for a card in 'review' status (SM-2 algorithm).

    This is where the real SM-2 math happens. On success: interval grows
    based on ease factor. On failure: card goes to relearning.

    Args:
        card: Current card state.
        quality: Quality rating.
        correct: Whether quality >= GOOD.

    Returns:
        New SRS state dict.
    """
    sm2 = calculate_sm2(
        repetitions=card["repetitions"],
        interval=card["interval"],
        ease_factor=card["ease_factor"],
        quality=quality,
    )

    if correct:
        return {
            **sm2,
            "status": "review",
            "learning_step": card["learning_step"],
        }
    else:
        # Failed review — goes to relearning
        return {
            **sm2,
            "status": "relearning",
            "learning_step": 0,
        }


def _handle_relearning(card: db.Row, quality: int, correct: bool) -> dict:
    """Handle a review for a card in 'relearning' status.

    One correct use sends it back to review. Failure keeps it in relearning
    with reduced ease.

    Args:
        card: Current card state.
        quality: Quality rating.
        correct: Whether quality >= GOOD.

    Returns:
        New SRS state dict.
    """
    ease = card["ease_factor"]

    if correct:
        # Back to review with a reduced interval (not full reset)
        new_interval = max(1.0, card["interval"] * 0.5)
        return {
            "interval": new_interval,
            "ease_factor": ease,
            "repetitions": 1,
            "status": "review",
            "learning_step": card["learning_step"],
        }
    else:
        return {
            "interval": 0.0007,
            "ease_factor": max(1.3, ease - 0.2),
            "repetitions": 0,
            "status": "relearning",
            "learning_step": 0,
        }


def calculate_sm2(
    repetitions: int,
    interval: float,
    ease_factor: float,
    quality: int,
) -> dict[str, float | int]:
    """Pure SM-2 calculation with the original Wozniak formula.

    The ease factor formula is:
        EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

    This is NOT the same as our previous "+0.1 / -0.2" simplification.
    The real formula produces graduated adjustments based on quality.

    Args:
        repetitions: Current consecutive correct count.
        interval: Current interval in days.
        ease_factor: Current ease factor (>= 1.3).
        quality: Rating from 0-5.

    Returns:
        Dict with keys: repetitions, interval, ease_factor.
    """
    # Ease factor adjustment (Wozniak's original formula)
    new_ease = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease = max(1.3, round(new_ease, 2))

    if quality >= QUALITY_GOOD:
        # Correct — advance the interval
        repetitions += 1
        if repetitions == 1:
            new_interval = 1.0
        elif repetitions == 2:
            new_interval = 6.0
        else:
            new_interval = interval * new_ease
    else:
        # Incorrect — reset
        repetitions = 0
        new_interval = 1.0

    return {
        "repetitions": repetitions,
        "interval": round(new_interval, 2),
        "ease_factor": new_ease,
    }
