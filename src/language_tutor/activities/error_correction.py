"""Error correction activity — find and fix errors in sentences.

The tutor generates sentences with deliberate errors (grammar, vocabulary,
prepositions, collocations) and the learner must identify and correct them.
Excellent for focused grammar practice without the pressure of free writing.
"""

from __future__ import annotations

import sqlite3

from rich.console import Console
from rich.panel import Panel

from language_tutor import db, llm
from language_tutor.activities.base import ActivityResult

console = Console()

_GENERATE_PROMPT = """Generate {count} English sentences with deliberate errors for a C1 learner to correct.
Each sentence should have exactly ONE error from these categories:
grammar, preposition, collocation, article, word order, vocabulary (false friends).

Format your response as a numbered list. Do NOT reveal the errors or corrections.
Example:
1. I depend of my team at work.
2. She made a big effort to do progress.

Do NOT call any tools. Just list the sentences."""

_EVALUATE_PROMPT = """The learner was asked to correct errors in these sentences:

Original sentences:
{originals}

Learner's corrections:
{corrections}

For each sentence, evaluate:
1. Did they find the error?
2. Is their correction correct?
3. If they missed the error or corrected it wrongly, explain what the actual error was.

Be specific for each sentence."""


class ErrorCorrection:
    """Find and fix errors in sentences — focused grammar practice."""

    name = "Error Correction"
    activity_type = "error_correction"
    description = "Find and fix errors in sentences — focused grammar practice"

    SENTENCES_PER_ROUND = 5

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
        """Run the error correction exercise.

        Flow:
        1. Tutor generates sentences with errors
        2. Learner types the corrected version for each
        3. Tutor evaluates all corrections at once

        Returns:
            ActivityResult with corrections found.
        """
        result = ActivityResult(skills_practiced=["grammar"])

        # Generate sentences
        prompt = _GENERATE_PROMPT.format(count=self.SENTENCES_PER_ROUND)
        with console.status(
            "[dim]Creating error correction exercise...[/dim]", spinner="dots"
        ):
            gen_response = self.tutor.chat(prompt)

        console.print(
            Panel(
                gen_response.message,
                title="Find and Fix the Errors",
                border_style="cyan",
            )
        )
        console.print(
            "\n[dim]Type the corrected version of each sentence.\n"
            "Type [bold]quit[/bold] to skip. Press Enter to skip a sentence.[/dim]\n"
        )

        # Collect corrections
        corrections = []
        for i in range(1, self.SENTENCES_PER_ROUND + 1):
            try:
                corrected = console.input(f"[blue]{i}.[/blue] ").strip()
            except EOFError:
                break
            if corrected.lower() == "quit":
                return result
            corrections.append(corrected if corrected else "(skipped)")

        if not corrections:
            return result

        result.total_turns = 1

        # Evaluate
        originals_text = gen_response.message
        corrections_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(corrections)
        )

        eval_prompt = _EVALUATE_PROMPT.format(
            originals=originals_text,
            corrections=corrections_text,
        )

        with console.status(
            "[dim]Checking your corrections...[/dim]", spinner="dots"
        ):
            evaluation = self.tutor.chat(eval_prompt)

        console.print(f"\n[bold green]Tutor:[/bold green] {evaluation.message}\n")

        result.corrections.extend(evaluation.metadata.corrections)
        result.card_assessments.extend(evaluation.metadata.card_assessments)
        result.new_word_suggestions.extend(evaluation.metadata.new_word_suggestions)

        return result
