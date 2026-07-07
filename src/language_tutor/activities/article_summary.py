"""Article summary activity — read a real article and write a summary.

The tutor fetches a real article from RSS feeds (tech, science), presents
it to the learner, and asks them to write a summary.  The tutor then
evaluates the summary for accuracy, grammar, and vocabulary.

This is the activity the user described as the "ideal tutor experience":
arriving one day and getting a real scientific article to summarize.
"""

from __future__ import annotations

import sqlite3

from rich.console import Console
from rich.panel import Panel

from language_tutor import content, db, llm
from language_tutor.activities.base import ActivityResult

console = Console()


class ArticleSummary:
    """Read a real article and write a summary — tutor evaluates."""

    name = "Article Summary"
    activity_type = "article_summary"
    description = "Read a real article and write a summary — tests reading and writing"

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
        """Run the article summary exercise.

        Flow:
        1. Fetch a real article from RSS feeds
        2. Display it to the learner
        3. Learner writes a summary
        4. Tutor evaluates the summary
        5. Optional: discuss the article topic

        Returns:
            ActivityResult with corrections and assessments.
        """
        result = ActivityResult(skills_practiced=["reading", "writing"])

        # Fetch article
        with console.status(
            "[dim]Finding an interesting article...[/dim]", spinner="dots"
        ):
            article = content.fetch_article(topic="technology")

        source_info = f" — [dim]{article.source}[/dim]" if article.source else ""
        console.print(
            Panel(
                f"[bold]{article.title}[/bold]{source_info}\n"
                f"[dim]({article.word_count} words)[/dim]\n\n"
                f"{article.content}",
                title="Article",
                border_style="cyan",
            )
        )
        console.print(
            "\n[dim]Read the article above, then write a 3-5 sentence summary.\n"
            "Type [bold]done[/bold] when finished. "
            "Type [bold]quit[/bold] to skip.[/dim]\n"
        )

        # Collect summary
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
            console.print("[dim]No summary written. Skipping.[/dim]")
            return result

        summary_text = "\n".join(lines)
        result.total_turns = 1

        # Evaluate summary
        eval_prompt = (
            f"The learner read this article:\n"
            f"Title: \"{article.title}\"\n"
            f"Content: \"{article.content[:500]}\"\n\n"
            f"They wrote this summary:\n\"{summary_text}\"\n\n"
            f"Evaluate: (1) Does the summary capture the main points? "
            f"(2) Grammar and vocabulary errors? (3) Suggest improvements. "
            f"Be specific about each error."
        )

        with console.status(
            "[dim]Evaluating your summary...[/dim]", spinner="dots"
        ):
            evaluation = self.tutor.chat(eval_prompt)

        console.print(f"\n[bold green]Tutor:[/bold green] {evaluation.message}\n")

        result.corrections.extend(evaluation.metadata.corrections)
        result.new_word_suggestions.extend(evaluation.metadata.new_word_suggestions)

        # Optional: discuss the article
        console.print(
            "[dim]Want to discuss the article? Type your thoughts, "
            "or [bold]done[/bold] to finish.[/dim]\n"
        )

        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in ("done", "quit", "exit"):
                break

            result.total_turns += 1
            result.skills_practiced = ["reading", "writing", "speaking"]

            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = self.tutor.chat(user_input)

            console.print(f"\n[bold green]Tutor:[/bold green] {response.message}\n")
            result.corrections.extend(response.metadata.corrections)
            result.card_assessments.extend(response.metadata.card_assessments)
            result.new_word_suggestions.extend(response.metadata.new_word_suggestions)

        return result
