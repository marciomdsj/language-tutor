"""Vocabulary drill activity — focused flashcard-style review of due cards.

Unlike free conversation where cards are woven into natural dialogue,
this activity is a direct, Anki-style drill: the tutor presents each
due card and asks the learner to use it in a sentence.  The tutor then
evaluates whether the usage was correct and natural.

Best used when there are many due cards to review quickly.
"""

from __future__ import annotations

import sqlite3

from rich.console import Console
from rich.panel import Panel

from language_tutor import db, llm
from language_tutor.activities.base import ActivityResult

console = Console()

_EVALUATE_PROMPT = """The learner was asked to use the word/phrase "{front}" in a sentence.
Its definition is: "{back}"

They wrote: "{sentence}"

Evaluate:
1. Did they use "{front}" correctly?
2. Is the sentence grammatically correct?
3. Is the usage natural for a C1 speaker?
If there are errors, list them. Be brief (1-2 sentences)."""


class VocabularyDrill:
    """Focused flashcard-style review of due cards."""

    name = "Vocabulary Drill"
    activity_type = "vocabulary_drill"
    description = "Review due cards Anki-style — use each word in a sentence"

    def __init__(
        self,
        tutor: llm.TutorLLM,
        conn: sqlite3.Connection,
        debug: bool = False,
    ) -> None:
        self.tutor = tutor
        self.conn = conn
        self.debug = debug

    def run(self) -> ActivityResult:
        """Run the vocabulary drill.

        Flow per card:
        1. Show the word/phrase and its definition
        2. Learner writes a sentence using it
        3. Tutor evaluates the usage
        4. Learner rates themselves (again/hard/good/easy)

        Returns:
            ActivityResult with assessments for each card.
        """
        result = ActivityResult(skills_practiced=["vocabulary", "writing"])

        due_cards = db.get_due_cards(self.conn, limit=10)
        if not due_cards:
            console.print("[dim]No cards due for review! Great job.[/dim]")
            return result

        console.print(
            Panel(
                f"[bold]{len(due_cards)} card(s) due for review[/bold]\n"
                "[dim]For each word, write a sentence using it. "
                "Type [bold]quit[/bold] to stop early.[/dim]",
                title="Vocabulary Drill",
                border_style="cyan",
            )
        )
        console.print()

        for i, card in enumerate(due_cards, 1):
            front = card["front"]
            back = card.get("back", "")
            card_type = card["type"]

            console.print(
                f"[bold cyan]({i}/{len(due_cards)})[/bold cyan] "
                f'[bold]"{front}"[/bold] [{card_type}]'
            )
            if back:
                console.print(f"  [dim]Definition: {back}[/dim]")
            console.print(f"  [dim]Use it in a sentence:[/dim]")

            try:
                sentence = console.input("  [blue]> [/blue]").strip()
            except EOFError:
                break

            if sentence.lower() in ("quit", "exit"):
                break

            if not sentence:
                console.print("  [dim]Skipped[/dim]\n")
                result.card_assessments.append(llm.CardAssessment(
                    front=front, used=False,
                    quality_suggestion="again", reasoning="skipped",
                ))
                continue

            result.total_turns += 1

            # Evaluate with LLM
            eval_prompt = _EVALUATE_PROMPT.format(
                front=front, back=back, sentence=sentence,
            )

            with console.status("[dim]Evaluating...[/dim]", spinner="dots"):
                response = self.tutor.chat(eval_prompt)

            console.print(f"  [green]Tutor:[/green] {response.message}")

            result.corrections.extend(response.metadata.corrections)
            result.card_assessments.extend(response.metadata.card_assessments)
            result.new_word_suggestions.extend(response.metadata.new_word_suggestions)

            # If the LLM didn't provide a card assessment, create one
            has_assessment = any(
                a.front.lower() == front.lower()
                for a in response.metadata.card_assessments
            )
            if not has_assessment:
                # No errors found = good usage
                quality = "good" if not response.metadata.corrections else "hard"
                result.card_assessments.append(llm.CardAssessment(
                    front=front, used=True, quality_suggestion=quality,
                    reasoning="auto-assessed from drill evaluation",
                ))

            console.print()

        return result
