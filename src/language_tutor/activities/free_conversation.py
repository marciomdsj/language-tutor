"""Free conversation activity — open-ended chat with SRS integration.

This is the original activity from Phase 1+2, extracted into the
activity system.  The tutor leads a natural conversation, weaving
due cards into the discussion and detecting errors.
"""

from __future__ import annotations

import sqlite3

from rich.console import Console

from language_tutor import audio, db, llm
from language_tutor.activities.base import ActivityResult
from language_tutor.stt import STTProvider
from language_tutor.tts import TTSProvider

console = Console()


class FreeConversation:
    """Open-ended conversation with SRS card integration."""

    name = "Free Conversation"
    activity_type = "free_conversation"
    description = "Chat freely about any topic with vocabulary review"

    def __init__(
        self,
        tutor: llm.TutorLLM,
        conn: sqlite3.Connection,
        tts: TTSProvider,
        stt: STTProvider,
        tts_muted: bool = False,
        debug: bool = False,
    ) -> None:
        self.tutor = tutor
        self.conn = conn
        self.tts = tts
        self.stt = stt
        self.tts_muted = tts_muted
        self.debug = debug

    def run(self) -> ActivityResult:
        """Run a free conversation loop until the user quits.

        Returns:
            ActivityResult with accumulated corrections, assessments, etc.
        """
        result = ActivityResult(skills_practiced=["speaking", "listening"])

        while True:
            user_input = self._get_input()
            if user_input is None:
                break

            result.total_turns += 1

            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = self.tutor.chat(user_input)

            self._display_response(response)
            self._accumulate(result, response.metadata)
            console.print()

        return result

    def _get_input(self) -> str | None:
        """Get text or voice input from the learner."""
        while True:
            try:
                raw = console.input("[bold blue]You:[/bold blue] ").strip()
            except EOFError:
                return None

            if raw.lower() in ("quit", "exit", "done"):
                return None
            if raw.lower() == "mute":
                self.tts_muted = True
                console.print("[dim]TTS muted.[/dim]")
                continue
            if raw.lower() == "unmute":
                self.tts_muted = False
                console.print("[dim]TTS unmuted.[/dim]")
                continue

            if raw:
                return raw

            # Voice input
            console.print("  [yellow]🎤 Recording... press Enter to stop[/yellow]")
            audio_bytes = audio.record_until_enter()
            if not audio_bytes:
                console.print("  [dim]No audio captured.[/dim]")
                continue
            with console.status("[dim]Transcribing...[/dim]", spinner="dots"):
                text = self.stt.transcribe(audio_bytes)
            if not text:
                console.print("  [dim]Could not understand audio.[/dim]")
                continue
            console.print(f"  [dim]📝 {text}[/dim]")
            return text

    def _display_response(self, response: llm.TutorResponse) -> None:
        """Display and speak the tutor's response."""
        console.print(f"\n[bold green]Tutor:[/bold green] {response.message}")
        if not self.tts_muted:
            try:
                tts_bytes = self.tts.synthesize(response.message)
                audio.play_audio_async(tts_bytes, self.tts.audio_format)
            except Exception:
                pass

    @staticmethod
    def _accumulate(result: ActivityResult, metadata: llm.TurnMetadata) -> None:
        """Accumulate metadata into the activity result."""
        result.corrections.extend(metadata.corrections)
        result.card_assessments.extend(metadata.card_assessments)
        result.new_word_suggestions.extend(metadata.new_word_suggestions)
