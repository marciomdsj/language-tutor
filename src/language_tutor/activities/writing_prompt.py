"""Writing prompt activity — tutor gives a topic, learner writes, tutor evaluates.

The tutor generates a writing prompt appropriate for C1 level (e.g. "Write
a formal email requesting...", "Describe your opinion on..."), the learner
types a response, and the tutor evaluates grammar, vocabulary, structure,
and coherence.
"""

from __future__ import annotations

import sqlite3

from rich.console import Console
from rich.panel import Panel

from language_tutor import db, llm
from language_tutor.activities.base import ActivityResult

console = Console()

_WRITING_SYSTEM = """You are an English writing tutor for a C1-level learner.
You will evaluate their writing for: grammar, vocabulary, coherence, register,
and structure. Be encouraging but thorough. Point out EVERY error.
ALWAYS call the report_metadata tool with corrections and new vocabulary."""

_PROMPT_REQUEST = """Generate a writing exercise for a C1 English learner.
Choose ONE of these formats randomly:
- A formal email (complaint, request, application)
- A short opinion essay (150-200 words) on a current topic
- A summary of a given situation
- A letter (to a friend, to an editor, to a company)
- A report or review

State the task clearly in 2-3 sentences. Include any specific requirements
(word count, tone, format). Do NOT call any tools."""


class WritingPrompt:
    """Writing exercise: tutor gives a prompt, learner writes, tutor evaluates."""

    name = "Writing Exercise"
    activity_type = "writing_prompt"
    description = "Write a text on a given topic — tutor evaluates your writing"

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
        """Run the writing exercise.

        Flow:
        1. Tutor generates a writing prompt
        2. Learner writes their response
        3. Tutor evaluates and provides corrections

        Returns:
            ActivityResult with corrections from the evaluation.
        """
        result = ActivityResult(skills_practiced=["writing"])

        # Generate the prompt
        with console.status("[dim]Creating writing exercise...[/dim]", spinner="dots"):
            prompt_response = self.tutor.chat(_PROMPT_REQUEST)

        console.print(
            Panel(
                prompt_response.message,
                title="Writing Exercise",
                border_style="cyan",
            )
        )
        console.print(
            "[dim]Write your response below. Type [bold]done[/bold] on a new "
            "line when finished. Type [bold]quit[/bold] to skip.[/dim]\n"
        )

        # Collect learner's writing (multi-line)
        lines = []
        while True:
            try:
                line = console.input("[blue]> [/blue]").strip()
            except EOFError:
                break
            if line.lower() == "done":
                break
            if line.lower() == "quit":
                return result
            lines.append(line)

        if not lines:
            console.print("[dim]No text written. Skipping evaluation.[/dim]")
            return result

        learner_text = "\n".join(lines)
        result.total_turns = 1

        # Evaluate
        eval_prompt = (
            f"The learner wrote the following text in response to your prompt. "
            f"Evaluate it thoroughly — check grammar, vocabulary, coherence, "
            f"register, and structure. List ALL errors.\n\n"
            f"Learner's text:\n\"{learner_text}\""
        )

        with console.status("[dim]Evaluating your writing...[/dim]", spinner="dots"):
            evaluation = self.tutor.chat(eval_prompt)

        console.print(f"\n[bold green]Tutor:[/bold green] {evaluation.message}\n")

        result.corrections.extend(evaluation.metadata.corrections)
        result.card_assessments.extend(evaluation.metadata.card_assessments)
        result.new_word_suggestions.extend(evaluation.metadata.new_word_suggestions)

        return result
