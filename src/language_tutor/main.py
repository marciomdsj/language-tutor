"""Entry point — text-based conversation loop with human-in-the-loop SRS.

This is the CLI interface for the language tutor.  The key difference from
a simple chatbot: after each LLM response, the learner reviews and confirms
the SRS metadata before it's persisted.  This ensures accuracy — the LLM
suggests, but YOU are the judge of your own learning.

Flow per turn:
    1. You type a message in English
    2. The tutor responds and suggests metadata (corrections, card ratings)
    3. You review: confirm/adjust card ratings, accept/reject new words
    4. Confirmed data is persisted to SQLite (SRS updated, corrections saved)

Type 'quit' or 'exit' to end.  Ctrl+C also works.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from language_tutor import config, db, llm, srs

console = Console()

# Maps user input to SM-2 quality values
QUALITY_MAP = {
    "a": srs.QUALITY_AGAIN,  # 0
    "h": srs.QUALITY_HARD,   # 2
    "g": srs.QUALITY_GOOD,   # 3
    "e": srs.QUALITY_EASY,   # 5
    "s": None,               # skip — don't review this card
}


def display_welcome(card_stats: db.Row) -> None:
    """Show the welcome banner with deck stats."""
    stats_line = " | ".join(
        f"[cyan]{k}[/cyan]: {v}" for k, v in card_stats.items() if k != "total"
    )
    console.print(
        Panel(
            "[bold cyan]Language Tutor[/bold cyan] — "
            f"Level: [yellow]{config.LEARNER_LEVEL}[/yellow] | "
            f"Model: [green]{config.OLLAMA_MODEL}[/green]\n"
            f"Deck: {card_stats.get('total', 0)} cards ({stats_line})\n\n"
            "[dim]Type in English to start a conversation with your tutor.\n"
            "Type [bold]quit[/bold] or [bold]exit[/bold] to end the session.[/dim]",
            title="Welcome",
            border_style="cyan",
        )
    )


def display_due_cards(due_cards: list[db.Row]) -> None:
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


def confirm_card_assessments(
    conn: db.sqlite3.Connection,
    assessments: list[llm.CardAssessment],
) -> list[srs.ReviewResult]:
    """Let the learner confirm or adjust quality ratings for reviewed cards.

    The LLM suggests a rating, but the learner has final say.  This is
    the human-in-the-loop step that keeps the SRS accurate.

    Args:
        conn: Database connection.
        assessments: LLM's suggested card assessments.

    Returns:
        List of ReviewResults for cards that were actually reviewed.
    """
    if not assessments:
        return []

    results = []
    console.print("\n[bold yellow]Card Review[/bold yellow]")

    for assessment in assessments:
        card_id = llm.find_card_by_front(conn, assessment.front)
        if not card_id:
            continue

        suggestion = assessment.quality_suggestion
        reasoning = f" — {assessment.reasoning}" if assessment.reasoning else ""
        used = "used" if assessment.used else "not used"

        console.print(
            f'  [bold]"{assessment.front}"[/bold] ({used}{reasoning})\n'
            f"    Tutor suggests: [cyan]{suggestion}[/cyan]"
        )

        rating = _prompt_quality(default=suggestion)
        if rating is None:
            console.print("    [dim]Skipped[/dim]")
            continue

        result = srs.review_card(conn, card_id, quality=rating)
        results.append(result)

        status_color = "green" if result.new_status == "review" else "yellow"
        console.print(
            f"    [{status_color}]{result.old_status} → {result.new_status}[/{status_color}]"
            f" | next: {result.next_review[:10]}"
        )
        if result.is_leech:
            console.print(
                "    [bold red]LEECH — card suspended (too many failures)[/bold red]"
            )

    return results


def confirm_new_words(
    conn: db.sqlite3.Connection,
    suggestions: list[llm.NewWordSuggestion],
) -> int:
    """Let the learner accept or reject new card suggestions.

    The LLM suggests vocabulary to track, but the learner decides what
    actually enters their deck.  No unsolicited cards.

    Args:
        conn: Database connection.
        suggestions: LLM's new word suggestions.

    Returns:
        Number of cards actually created.
    """
    if not suggestions:
        return 0

    created = 0
    console.print("\n[bold green]New Vocabulary[/bold green]")

    for suggestion in suggestions:
        if llm.find_card_by_front(conn, suggestion.word):
            continue  # already exists

        tags_str = f" [{', '.join(suggestion.tags)}]" if suggestion.tags else ""
        console.print(
            f'  [bold]"{suggestion.word}"[/bold] ({suggestion.card_type}{tags_str})\n'
            f"    Definition: {suggestion.back}"
        )
        if suggestion.context:
            console.print(f"    Example: [italic]{suggestion.context}[/italic]")

        try:
            answer = console.input("    Add to deck? [y]es / [n]o → ").strip().lower()
        except EOFError:
            break

        if answer in ("y", "yes", ""):
            db.create_card(
                conn,
                front=suggestion.word,
                card_type=suggestion.card_type,
                back=suggestion.back,
                context=suggestion.context,
                tags=suggestion.tags,
            )
            created += 1
            console.print("    [green]Added![/green]")
        else:
            console.print("    [dim]Skipped[/dim]")

    return created


def save_corrections(
    conn: db.sqlite3.Connection,
    session_id: int,
    corrections: list[dict[str, str]],
) -> int:
    """Save corrections and display them to the learner.

    Corrections are always saved (the LLM identified real errors), but
    we show them so the learner is aware.

    Args:
        conn: Database connection.
        session_id: Current session id.
        corrections: List of correction dicts from the LLM.

    Returns:
        Number of corrections saved.
    """
    if not corrections:
        return 0

    console.print("\n[bold red]Corrections[/bold red]")
    for c in corrections:
        error_type = c.get("error_type", "other")
        explanation = c.get("explanation", "")
        explanation_str = f" — {explanation}" if explanation else ""
        console.print(
            f'  [red]✗[/red] "{c.get("user_said", "")}" → '
            f'[green]"{c.get("corrected", "")}"[/green] '
            f"[dim]({error_type}{explanation_str})[/dim]"
        )

        card_id = llm.find_card_by_front(conn, c.get("corrected", ""))
        db.create_correction(
            conn,
            session_id=session_id,
            user_said=c.get("user_said", ""),
            corrected=c.get("corrected", ""),
            error_type=c.get("error_type"),
            explanation=c.get("explanation"),
            card_id=card_id,
        )

    return len(corrections)


def display_session_summary(
    total_turns: int,
    errors_found: int,
    cards_reviewed: int,
    cards_created: int,
) -> None:
    """Show end-of-session stats."""
    console.print()
    console.print(
        Panel(
            f"[bold]Turns:[/bold] {total_turns}\n"
            f"[bold]Errors corrected:[/bold] {errors_found}\n"
            f"[bold]Cards reviewed:[/bold] {cards_reviewed}\n"
            f"[bold]New cards added:[/bold] {cards_created}",
            title="Session Summary",
            border_style="green",
        )
    )


def _prompt_quality(default: str) -> int | None:
    """Prompt the learner for a quality rating with a suggested default.

    Args:
        default: The LLM's suggested rating ("again", "hard", "good", "easy").

    Returns:
        Quality integer (0, 2, 3, 5) or None to skip.
    """
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


def run_session() -> None:
    """Run a single conversation session from start to finish.

    The core loop:
    1. Connect to DB, load due cards
    2. Build LLM with dynamic system prompt
    3. Loop: input → LLM → corrections → card review → new words → persist
    4. On exit: save session summary
    """
    conn = db.get_connection()
    session_id = db.create_session(conn)
    due_cards = db.get_due_cards(conn)
    card_stats = db.get_card_stats(conn)

    display_welcome(card_stats)
    display_due_cards(due_cards)

    tutor = llm.TutorLLM(due_cards=due_cards)

    total_turns = 0
    total_errors = 0
    total_cards_reviewed = 0
    total_cards_created = 0

    try:
        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            total_turns += 1

            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = tutor.chat(user_input)

            console.print(f"\n[bold green]Tutor:[/bold green] {response.message}")

            # --- Human-in-the-loop SRS ---

            # 1. Show corrections (always saved — LLM found real errors)
            errors = save_corrections(
                conn, session_id, response.metadata.corrections
            )
            total_errors += errors

            # 2. Card review — learner confirms/adjusts quality ratings
            review_results = confirm_card_assessments(
                conn, response.metadata.card_assessments
            )
            total_cards_reviewed += len(review_results)

            # 3. New words — learner accepts/rejects suggestions
            created = confirm_new_words(
                conn, response.metadata.new_word_suggestions
            )
            total_cards_created += created

            console.print()  # breathing room between turns

    except KeyboardInterrupt:
        console.print("\n[dim]Session interrupted.[/dim]")

    db.end_session(
        conn,
        session_id=session_id,
        total_turns=total_turns,
        errors_found=total_errors,
        cards_reviewed=total_cards_reviewed,
    )
    display_session_summary(
        total_turns, total_errors, total_cards_reviewed, total_cards_created
    )
    conn.close()


def main() -> None:
    """CLI entry point."""
    try:
        run_session()
    except ConnectionError:
        console.print(
            "[bold red]Error:[/bold red] Could not connect to Ollama. "
            f"Is it running at {config.OLLAMA_HOST}?\n"
            "Start it with: [bold]ollama serve[/bold]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
