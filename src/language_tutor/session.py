"""Unified session runner — text and voice in one experience.

This module contains the core session logic extracted from main.py.
It supports both text and voice input in a single session, with the
tutor always responding via TTS (when not muted) + terminal text.

Input modes (per turn, no switching needed):
    - Type text + Enter → text input
    - Just press Enter   → voice recording (Enter again to stop)

SRS during conversation (Option D — auto-accept):
    - Corrections: displayed, saved automatically
    - Card ratings: LLM suggests, auto-accepted, shown in compact line
    - New words: LLM suggests, auto-rejected (conservative)

SRS at session end (final review):
    - All auto-accepted ratings shown for confirmation/adjustment
    - Rejected new words shown for second chance
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from language_tutor import audio, config, db, llm, srs
from language_tutor.stt import STTProvider, get_stt_provider
from language_tutor.tts import TTSProvider, get_tts_provider

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
    """A card review auto-accepted during conversation, pending final confirmation."""

    card_id: int
    front: str
    quality: int
    reasoning: str
    times_seen_this_session: int = 1


@dataclass
class PendingNewWord:
    """A new word suggestion rejected during conversation, offered again at session end."""

    suggestion: llm.NewWordSuggestion


class SessionRunner:
    """Runs a unified conversation session with text + voice support.

    Manages the full lifecycle: welcome → conversation loop → final review → summary.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.conn = db.get_connection()
        self.session_id = db.create_session(self.conn)
        self.tts: TTSProvider = get_tts_provider()
        self.stt: STTProvider = get_stt_provider()
        self.tts_muted = False

        # Session counters
        self.total_turns = 0
        self.total_errors = 0
        self.total_cards_reviewed = 0
        self.total_cards_created = 0

        # Accumulated reviews for final confirmation
        self.pending_reviews: dict[int, PendingReview] = {}
        self.pending_new_words: list[PendingNewWord] = []

    def run(self) -> None:
        """Run the full session: welcome → opening → loop → final review → summary."""
        due_cards = db.get_due_cards(self.conn)
        recent_errors = db.get_recent_errors(self.conn, limit=5)
        card_stats = db.get_card_stats(self.conn)

        self._display_welcome(card_stats)
        self._display_due_cards(due_cards)

        tutor = llm.TutorLLM(due_cards=due_cards, recent_errors=recent_errors)

        # Tutor speaks first — proactive opening
        with console.status("[dim]Preparing session...[/dim]", spinner="dots"):
            opening = tutor.generate_opening()
        self._display_tutor_response(opening)
        console.print()

        try:
            self._conversation_loop(tutor)
        except KeyboardInterrupt:
            console.print("\n[dim]Session interrupted.[/dim]")

        self._final_review()

        db.end_session(
            self.conn,
            session_id=self.session_id,
            total_turns=self.total_turns,
            errors_found=self.total_errors,
            cards_reviewed=self.total_cards_reviewed,
        )
        self._display_summary()
        self.conn.close()

    def _conversation_loop(self, tutor: llm.TutorLLM) -> None:
        """Main conversation loop — text or voice input, TTS + text output."""
        while True:
            user_input = self._get_input()
            if user_input is None:
                break

            self.total_turns += 1

            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = tutor.chat(user_input)

            # Display + speak tutor response
            self._display_tutor_response(response)

            # Process SRS (auto-accept mode)
            self._process_turn_srs(response.metadata)

            console.print()

    def _get_input(self) -> str | None:
        """Get input from the user — text or voice.

        Uses a loop instead of recursion for command handling and retries.

        Returns:
            The user's message, or None to end the session.
        """
        while True:
            try:
                raw = console.input("[bold blue]You:[/bold blue] ").strip()
            except EOFError:
                return None

            # Commands
            if raw.lower() in ("quit", "exit"):
                return None
            if raw.lower() == "mute":
                self.tts_muted = True
                console.print("[dim]TTS muted.[/dim]")
                continue
            if raw.lower() == "unmute":
                self.tts_muted = False
                console.print("[dim]TTS unmuted.[/dim]")
                continue

            # Text input
            if raw:
                return raw

            # Voice input — empty Enter triggers recording
            console.print("  [yellow]🎤 Recording... press Enter to stop[/yellow]")
            audio_bytes = audio.record_until_enter()

            if not audio_bytes:
                console.print("  [dim]No audio captured.[/dim]")
                continue

            with console.status("[dim]Transcribing...[/dim]", spinner="dots"):
                text = self.stt.transcribe(audio_bytes)

            if not text:
                console.print("  [dim]Could not understand audio. Try again.[/dim]")
                continue

            console.print(f"  [dim]📝 {text}[/dim]")
            return text

    def _display_tutor_response(self, response: llm.TutorResponse) -> None:
        """Display tutor response as text and play TTS audio."""
        console.print(f"\n[bold green]Tutor:[/bold green] {response.message}")

        if not self.tts_muted:
            try:
                tts_bytes = self.tts.synthesize(response.message)
                audio.play_audio_async(tts_bytes, self.tts.audio_format)
            except Exception as e:
                if self.debug:
                    console.print(f"  [dim red]TTS error: {e}[/dim red]")

    def _process_turn_srs(self, metadata: llm.TurnMetadata) -> None:
        """Process SRS data from a turn with auto-accept.

        Corrections are saved immediately.  Card ratings are auto-accepted
        and accumulated for final review.  New words are accumulated for
        final review (auto-rejected during conversation).
        """
        # 1. Corrections — always save
        if metadata.corrections:
            self._display_corrections(metadata.corrections)
            self.total_errors += len(metadata.corrections)

        # 2. Card assessments — auto-accept
        if metadata.card_assessments:
            self._auto_accept_cards(metadata.card_assessments)

        # 3. New words — accumulate for final review
        for suggestion in metadata.new_word_suggestions:
            if not llm.find_card_by_front(self.conn, suggestion.word):
                self.pending_new_words.append(PendingNewWord(suggestion=suggestion))

        # Compact SRS status line
        self._display_compact_srs(metadata)

    def _display_corrections(self, corrections: list[dict[str, str]]) -> None:
        """Display and save corrections."""
        if self.debug:
            console.print("\n[bold red]Corrections[/bold red]")

        for c in corrections:
            error_type = c.get("error_type", "other")
            explanation = c.get("explanation", "")
            explanation_str = f" — {explanation}" if explanation else ""

            if self.debug:
                console.print(
                    f'  [red]✗[/red] "{c.get("user_said", "")}" → '
                    f'[green]"{c.get("corrected", "")}"[/green] '
                    f"[dim]({error_type}{explanation_str})[/dim]"
                )

            db.create_correction(
                self.conn,
                session_id=self.session_id,
                user_said=c.get("user_said", ""),
                corrected=c.get("corrected", ""),
                error_type=c.get("error_type"),
                explanation=c.get("explanation"),
                card_id=llm.find_card_by_front(self.conn, c.get("corrected", "")),
            )

    def _auto_accept_cards(self, assessments: list[llm.CardAssessment]) -> None:
        """Auto-accept card ratings, accumulate for final review."""
        for assessment in assessments:
            card_id = llm.find_card_by_front(self.conn, assessment.front)
            if not card_id:
                continue

            quality = _quality_from_suggestion(assessment.quality_suggestion)

            if card_id in self.pending_reviews:
                self.pending_reviews[card_id].quality = quality
                self.pending_reviews[card_id].times_seen_this_session += 1
                self.pending_reviews[card_id].reasoning = assessment.reasoning
            else:
                self.pending_reviews[card_id] = PendingReview(
                    card_id=card_id,
                    front=assessment.front,
                    quality=quality,
                    reasoning=assessment.reasoning,
                )

    def _display_compact_srs(self, metadata: llm.TurnMetadata) -> None:
        """Show a compact one-line SRS summary for clean mode."""
        parts = []

        if metadata.corrections:
            parts.append(f"[red]✗ {len(metadata.corrections)} error(s)[/red]")

        for a in metadata.card_assessments:
            q = a.quality_suggestion
            color = {"again": "red", "hard": "yellow", "good": "green", "easy": "cyan"}.get(q, "white")
            parts.append(f'[{color}]"{a.front}" → {q}[/{color}]')

        if self.pending_new_words:
            new_count = len(metadata.new_word_suggestions)
            if new_count:
                parts.append(f"[dim]+{new_count} new word(s) pending[/dim]")

        if parts:
            console.print(f"  {' | '.join(parts)}")

    def _final_review(self) -> None:
        """Final review — confirm card ratings and accept/reject new words.

        This is the human-in-the-loop safety net.  Everything auto-accepted
        during conversation is shown here for confirmation or adjustment.
        """
        if not self.pending_reviews and not self.pending_new_words:
            return

        console.print()
        console.print(
            Panel("[bold]Final Review[/bold] — confirm your session results",
                  border_style="yellow")
        )

        # Card ratings
        if self.pending_reviews:
            console.print("\n[bold yellow]Card Ratings[/bold yellow]")
            for review in self.pending_reviews.values():
                suggested = QUALITY_NAMES.get(review.quality, "good")
                reasoning = f" — {review.reasoning}" if review.reasoning else ""
                seen = f", seen {review.times_seen_this_session}x" if review.times_seen_this_session > 1 else ""

                console.print(
                    f'  [bold]"{review.front}"[/bold] (auto: {suggested}{seen}{reasoning})'
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
                    console.print(
                        "    [bold red]LEECH — card suspended[/bold red]"
                    )

        # New words
        if self.pending_new_words:
            console.print("\n[bold green]New Vocabulary[/bold green]")
            for pending in self.pending_new_words:
                s = pending.suggestion
                if llm.find_card_by_front(self.conn, s.word):
                    continue

                tags_str = f" [{', '.join(s.tags)}]" if s.tags else ""
                console.print(
                    f'  [bold]"{s.word}"[/bold] ({s.card_type}{tags_str})\n'
                    f"    Definition: {s.back}"
                )
                if s.context:
                    console.print(f"    Example: [italic]{s.context}[/italic]")

                try:
                    answer = console.input("    Add to deck? [y]es / [n]o → ").strip().lower()
                except EOFError:
                    break

                if answer in ("y", "yes", ""):
                    db.create_card(
                        self.conn,
                        front=s.word,
                        card_type=s.card_type,
                        back=s.back,
                        context=s.context,
                        tags=s.tags,
                    )
                    self.total_cards_created += 1
                    console.print("    [green]Added![/green]")
                else:
                    console.print("    [dim]Skipped[/dim]")

    def _display_welcome(self, card_stats: db.Row) -> None:
        """Show the welcome banner."""
        stats_parts = [f"[cyan]{k}[/cyan]: {v}" for k, v in card_stats.items() if k != "total"]
        stats_line = " | ".join(stats_parts) if stats_parts else "empty deck"

        console.print(
            Panel(
                "[bold cyan]Language Tutor[/bold cyan] — "
                f"Level: [yellow]{config.LEARNER_LEVEL}[/yellow] | "
                f"Model: [green]{config.PRIMARY_MODEL}[/green]\n"
                f"Deck: {card_stats.get('total', 0)} cards ({stats_line})\n\n"
                "[dim]Type in English, or press Enter to speak.\n"
                "Commands: [bold]mute[/bold] | [bold]unmute[/bold] | [bold]quit[/bold][/dim]",
                title="Welcome",
                border_style="cyan",
            )
        )

    def _display_due_cards(self, due_cards: list[db.Row]) -> None:
        """Show which SRS cards are due for review this session."""
        if not due_cards:
            console.print("[dim]No cards due for review. Free conversation![/dim]\n")
            return

        table = Table(title="Cards Due for Review", border_style="yellow")
        table.add_column("Word/Phrase", style="bold")
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Accuracy")

        for card in due_cards:
            seen = card["times_seen"]
            correct = card["times_correct"]
            accuracy = f"{correct}/{seen}" if seen > 0 else "new"
            table.add_row(card["front"], card["type"], card["status"], accuracy)

        console.print(table)
        console.print(
            "[dim]The tutor will weave these into the conversation naturally.[/dim]\n"
        )

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

    def _prompt_quality(self, default: str) -> int | None:
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
    """Convert a quality suggestion string to its numeric value.

    Args:
        suggestion: One of "again", "hard", "good", "easy".

    Returns:
        The SM-2 quality value (0, 2, 3, or 5).
    """
    mapping = {
        "again": srs.QUALITY_AGAIN,
        "hard": srs.QUALITY_HARD,
        "good": srs.QUALITY_GOOD,
        "easy": srs.QUALITY_EASY,
    }
    return mapping.get(suggestion.lower(), srs.QUALITY_GOOD)
