"""Unified session runner — orchestrates activities, SRS, and voice.

This module manages the full session lifecycle:
    1. Welcome + warmup
    2. Planner suggests activities, learner chooses
    3. Activity runs (free conversation, writing, article summary, etc.)
    4. SRS: auto-accept during activity, final review at session end
    5. Session summary

Each activity manages its own interaction loop and returns an
ActivityResult with corrections, card assessments, and new word
suggestions.  The session runner handles SRS persistence and
the final review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from language_tutor import config, db, llm, srs
from language_tutor.activities.base import ActivityResult
from language_tutor.activities.free_conversation import FreeConversation
from language_tutor.activities.writing_prompt import WritingPrompt
from language_tutor.activities.article_summary import ArticleSummary
from language_tutor.activities.error_correction import ErrorCorrection
from language_tutor.planner import ACTIVITY_REGISTRY, present_choices, suggest_activities
from language_tutor.stt import get_stt_provider
from language_tutor.tts import get_tts_provider

console = Console()

QUALITY_MAP = {
    "a": srs.QUALITY_AGAIN,
    "h": srs.QUALITY_HARD,
    "g": srs.QUALITY_GOOD,
    "e": srs.QUALITY_EASY,
    "s": None,
}

QUALITY_NAMES = {
    srs.QUALITY_AGAIN: "again",
    srs.QUALITY_HARD: "hard",
    srs.QUALITY_GOOD: "good",
    srs.QUALITY_EASY: "easy",
}


@dataclass
class PendingReview:
    """A card review auto-accepted during the activity, pending final confirmation."""

    card_id: int
    front: str
    quality: int
    reasoning: str
    times_seen_this_session: int = 1


class SessionRunner:
    """Runs a unified session with activity selection and SRS integration."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.conn = db.get_connection()
        self.tts = get_tts_provider()
        self.stt = get_stt_provider()
        self.tts_muted = False

        self.total_turns = 0
        self.total_errors = 0
        self.total_cards_reviewed = 0
        self.total_cards_created = 0

        self.pending_reviews: dict[int, PendingReview] = {}
        self.pending_new_words: list[llm.NewWordSuggestion] = []

    def run(self) -> None:
        """Run the full session."""
        due_cards = db.get_due_cards(self.conn)
        recent_errors = db.get_recent_errors(self.conn, limit=5)
        card_stats = db.get_card_stats(self.conn)

        self._display_welcome(card_stats)
        self._display_due_cards(due_cards)

        # Warm up LLM connection
        with console.status("[dim]Connecting to LLM...[/dim]", spinner="dots"):
            active_model = llm.warmup()
        console.print(f"[dim]Connected to {active_model}[/dim]\n")

        # Activity selection
        suggestions = suggest_activities(self.conn)
        chosen = present_choices(suggestions)
        console.print(
            f"\n[bold]Starting: {ACTIVITY_REGISTRY[chosen].name}[/bold]\n"
        )

        # Create session in DB
        session_id = db.create_session(
            self.conn, activity_type=chosen
        )

        # Build tutor
        tutor = llm.TutorLLM(due_cards=due_cards, recent_errors=recent_errors)

        # Tutor opening (only for free conversation)
        if chosen == "free_conversation":
            with console.status("[dim]Preparing...[/dim]", spinner="dots"):
                opening = tutor.generate_opening()
            console.print(f"[bold green]Tutor:[/bold green] {opening.message}")
            if not self.tts_muted:
                try:
                    from language_tutor import audio
                    tts_bytes = self.tts.synthesize(opening.message)
                    audio.play_audio_async(tts_bytes, self.tts.audio_format)
                except Exception:
                    pass
            console.print()

        # Run the activity
        activity = self._create_activity(chosen, tutor)

        try:
            result = activity.run()
        except KeyboardInterrupt:
            console.print("\n[dim]Session interrupted.[/dim]")
            result = ActivityResult()

        # Process SRS from activity result
        self._process_activity_result(result, session_id)

        # Final review
        self._final_review()

        # Close session
        db.end_session(
            self.conn,
            session_id=session_id,
            total_turns=self.total_turns,
            errors_found=self.total_errors,
            cards_reviewed=self.total_cards_reviewed,
            skills_practiced=result.skills_practiced,
        )
        self._display_summary()
        self.conn.close()

    def _create_activity(self, activity_type: str, tutor: llm.TutorLLM) -> object:
        """Instantiate the chosen activity with its dependencies.

        Args:
            activity_type: The activity type string.
            tutor: The TutorLLM instance.

        Returns:
            An activity instance ready to run.
        """
        if activity_type == "free_conversation":
            return FreeConversation(
                tutor=tutor, conn=self.conn, tts=self.tts,
                stt=self.stt, tts_muted=self.tts_muted, debug=self.debug,
            )
        elif activity_type == "writing_prompt":
            return WritingPrompt(tutor=tutor, conn=self.conn, debug=self.debug)
        elif activity_type == "article_summary":
            return ArticleSummary(tutor=tutor, conn=self.conn, debug=self.debug)
        elif activity_type == "error_correction":
            return ErrorCorrection(tutor=tutor, conn=self.conn, debug=self.debug)
        else:
            return FreeConversation(
                tutor=tutor, conn=self.conn, tts=self.tts,
                stt=self.stt, tts_muted=self.tts_muted, debug=self.debug,
            )

    def _process_activity_result(
        self, result: ActivityResult, session_id: int
    ) -> None:
        """Process the activity result: save corrections, accumulate SRS data.

        Args:
            result: The ActivityResult from the completed activity.
            session_id: Current session id.
        """
        self.total_turns = result.total_turns

        # Save corrections
        for c in result.corrections:
            db.create_correction(
                self.conn,
                session_id=session_id,
                user_said=c.get("user_said", ""),
                corrected=c.get("corrected", ""),
                error_type=c.get("error_type"),
                explanation=c.get("explanation"),
                card_id=llm.find_card_by_front(self.conn, c.get("corrected", "")),
            )
            self.total_errors += 1

        # Accumulate card assessments for final review
        for assessment in result.card_assessments:
            card_id = llm.find_card_by_front(self.conn, assessment.front)
            if not card_id:
                continue
            quality = _quality_from_suggestion(assessment.quality_suggestion)
            if card_id in self.pending_reviews:
                self.pending_reviews[card_id].quality = quality
                self.pending_reviews[card_id].times_seen_this_session += 1
            else:
                self.pending_reviews[card_id] = PendingReview(
                    card_id=card_id, front=assessment.front,
                    quality=quality, reasoning=assessment.reasoning,
                )

        # Accumulate new word suggestions
        for suggestion in result.new_word_suggestions:
            if not llm.find_card_by_front(self.conn, suggestion.word):
                self.pending_new_words.append(suggestion)

    def _final_review(self) -> None:
        """Final review — confirm card ratings and accept/reject new words."""
        if not self.pending_reviews and not self.pending_new_words:
            return

        console.print()
        console.print(
            Panel(
                "[bold]Final Review[/bold] — confirm your session results",
                border_style="yellow",
            )
        )

        if self.pending_reviews:
            console.print("\n[bold yellow]Card Ratings[/bold yellow]")
            for review in self.pending_reviews.values():
                suggested = QUALITY_NAMES.get(review.quality, "good")
                reasoning = f" — {review.reasoning}" if review.reasoning else ""
                seen = (
                    f", seen {review.times_seen_this_session}x"
                    if review.times_seen_this_session > 1
                    else ""
                )

                console.print(
                    f'  [bold]"{review.front}"[/bold] '
                    f"(auto: {suggested}{seen}{reasoning})"
                )

                rating = self._prompt_quality(default=suggested)
                if rating is None:
                    console.print("    [dim]Skipped[/dim]")
                    continue

                result = srs.review_card(self.conn, review.card_id, quality=rating)
                self.total_cards_reviewed += 1

                status_color = "green" if result.new_status == "review" else "yellow"
                console.print(
                    f"    [{status_color}]{result.old_status} → "
                    f"{result.new_status}[/{status_color}]"
                    f" | next: {result.next_review[:10]}"
                )
                if result.is_leech:
                    console.print("    [bold red]LEECH — card suspended[/bold red]")

        if self.pending_new_words:
            console.print("\n[bold green]New Vocabulary[/bold green]")
            for suggestion in self.pending_new_words:
                if llm.find_card_by_front(self.conn, suggestion.word):
                    continue
                tags_str = f" [{', '.join(suggestion.tags)}]" if suggestion.tags else ""
                console.print(
                    f'  [bold]"{suggestion.word}"[/bold] '
                    f"({suggestion.card_type}{tags_str})\n"
                    f"    Definition: {suggestion.back}"
                )
                if suggestion.context:
                    console.print(f"    Example: [italic]{suggestion.context}[/italic]")
                try:
                    answer = console.input(
                        "    Add to deck? [y]es / [n]o → "
                    ).strip().lower()
                except EOFError:
                    break
                if answer in ("y", "yes", ""):
                    db.create_card(
                        self.conn, front=suggestion.word,
                        card_type=suggestion.card_type, back=suggestion.back,
                        context=suggestion.context, tags=suggestion.tags,
                    )
                    self.total_cards_created += 1
                    console.print("    [green]Added![/green]")
                else:
                    console.print("    [dim]Skipped[/dim]")

    def _display_welcome(self, card_stats: db.Row) -> None:
        """Show the welcome banner."""
        stats_parts = [
            f"[cyan]{k}[/cyan]: {v}"
            for k, v in card_stats.items()
            if k != "total"
        ]
        stats_line = " | ".join(stats_parts) if stats_parts else "empty deck"
        console.print(
            Panel(
                "[bold cyan]Language Tutor[/bold cyan] — "
                f"Level: [yellow]{config.LEARNER_LEVEL}[/yellow] | "
                f"Model: [green]{config.PRIMARY_MODEL}[/green]\n"
                f"Deck: {card_stats.get('total', 0)} cards ({stats_line})\n\n"
                "[dim]Commands: [bold]mute[/bold] | [bold]unmute[/bold] | "
                "[bold]quit[/bold] | [bold]done[/bold][/dim]",
                title="Welcome",
                border_style="cyan",
            )
        )

    def _display_due_cards(self, due_cards: list[db.Row]) -> None:
        """Show which SRS cards are due."""
        if not due_cards:
            console.print("[dim]No cards due for review. Free practice![/dim]\n")
            return
        table = Table(title="Cards Due for Review", border_style="yellow")
        table.add_column("Word/Phrase", style="bold")
        table.add_column("Type")
        table.add_column("Status")
        for card in due_cards:
            table.add_row(card["front"], card["type"], card["status"])
        console.print(table)
        console.print()

    def _display_summary(self) -> None:
        """Show end-of-session stats."""
        console.print()
        console.print(
            Panel(
                f"[bold]Turns:[/bold] {self.total_turns}\n"
                f"[bold]Errors corrected:[/bold] {self.total_errors}\n"
                f"[bold]Cards reviewed:[/bold] {self.total_cards_reviewed}\n"
                f"[bold]New cards added:[/bold] {self.total_cards_created}",
                title="Session Summary",
                border_style="green",
            )
        )

    @staticmethod
    def _prompt_quality(default: str) -> int | None:
        """Prompt for a quality rating."""
        default_key = default[0].lower() if default else "g"
        try:
            raw = console.input(
                f"    [a]gain [h]ard [g]ood [e]asy [s]kip "
                f"(default: {default}) → "
            ).strip().lower()
        except EOFError:
            return None
        key = raw if raw else default_key
        return QUALITY_MAP.get(key, QUALITY_MAP.get(default_key))


def _quality_from_suggestion(suggestion: str) -> int:
    """Convert a quality suggestion string to its numeric value."""
    mapping = {
        "again": srs.QUALITY_AGAIN,
        "hard": srs.QUALITY_HARD,
        "good": srs.QUALITY_GOOD,
        "easy": srs.QUALITY_EASY,
    }
    return mapping.get(suggestion.lower(), srs.QUALITY_GOOD)
