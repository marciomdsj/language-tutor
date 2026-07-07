"""Base types for the activity system.

Activity is a Protocol — any class with a `run()` method that returns
an ActivityResult is a valid activity.  No inheritance needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from language_tutor import llm


@dataclass
class ActivityResult:
    """Outcome of a completed activity, used by session.py for SRS processing."""

    total_turns: int = 0
    corrections: list[dict[str, str]] = field(default_factory=list)
    card_assessments: list[llm.CardAssessment] = field(default_factory=list)
    new_word_suggestions: list[llm.NewWordSuggestion] = field(default_factory=list)
    skills_practiced: list[str] = field(default_factory=list)


@runtime_checkable
class Activity(Protocol):
    """Interface for activity types.

    Each activity manages its own interaction loop and returns
    an ActivityResult when complete.
    """

    @property
    def name(self) -> str:
        """Short name for display (e.g. 'Free Conversation')."""
        ...

    @property
    def activity_type(self) -> str:
        """Machine name for DB storage (e.g. 'free_conversation')."""
        ...

    @property
    def description(self) -> str:
        """One-line description for the planner menu."""
        ...

    def run(self) -> ActivityResult:
        """Execute the activity and return the result."""
        ...
