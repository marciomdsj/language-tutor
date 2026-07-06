"""LLM integration via Ollama — prompt building, chat, and metadata extraction.

This module handles all communication with the local Qwen3-8B model through
Ollama.  The key design decisions:

1. **System prompt is dynamic**: before each session, we build a system prompt
   that includes the learner's level, SRS cards due for review, and session
   rules.  The LLM acts as a tutor, not just a chatbot.

2. **Metadata via tool use**: Qwen3-8B supports tool calling natively.  We
   define a tool called `report_metadata` that the model calls after each
   response to report corrections, card assessments, and new vocabulary
   suggestions.  The LEARNER then confirms or adjusts these (human-in-the-loop).

3. **Fallback to JSON**: if tool use fails (e.g. model doesn't call the tool),
   we attempt to parse a JSON block from the response as a safety net.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import ollama

from language_tutor import config, db

# ---------------------------------------------------------------------------
# Metadata schema — what we ask the LLM to report after each turn
# ---------------------------------------------------------------------------

METADATA_TOOL = {
    "type": "function",
    "function": {
        "name": "report_metadata",
        "description": (
            "After EVERY response, report: corrections found in the learner's "
            "message, assessment of how well they used the review cards, and "
            "any new vocabulary worth tracking. ALWAYS call this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "corrections": {
                    "type": "array",
                    "description": "Errors found in the learner's message",
                    "items": {
                        "type": "object",
                        "properties": {
                            "user_said": {
                                "type": "string",
                                "description": "The incorrect fragment",
                            },
                            "corrected": {
                                "type": "string",
                                "description": "The correct version",
                            },
                            "error_type": {
                                "type": "string",
                                "enum": [
                                    "grammar",
                                    "vocabulary",
                                    "preposition",
                                    "article",
                                    "word_order",
                                    "spelling",
                                    "collocation",
                                    "other",
                                ],
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of why it's wrong",
                            },
                        },
                        "required": ["user_said", "corrected", "error_type"],
                    },
                },
                "card_assessments": {
                    "type": "array",
                    "description": (
                        "For each review card the learner attempted to use, "
                        "assess how well they used it"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "front": {
                                "type": "string",
                                "description": "The card word/phrase",
                            },
                            "used": {
                                "type": "boolean",
                                "description": "Whether the learner used this card",
                            },
                            "quality_suggestion": {
                                "type": "string",
                                "enum": ["again", "hard", "good", "easy"],
                                "description": (
                                    "Suggested quality: again=didn't use or wrong, "
                                    "hard=awkward usage, good=correct, easy=perfect"
                                ),
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief reason for the assessment",
                            },
                        },
                        "required": ["front", "used", "quality_suggestion"],
                    },
                },
                "new_word_suggestions": {
                    "type": "array",
                    "description": (
                        "New vocabulary the learner encountered or should learn. "
                        "Include a definition/translation as 'back'."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["word", "phrase", "grammar"],
                            },
                            "back": {
                                "type": "string",
                                "description": (
                                    "Definition or correct usage example for the card"
                                ),
                            },
                            "context": {
                                "type": "string",
                                "description": "Example sentence using this word",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Topic tags (e.g. 'technology', 'business')",
                            },
                        },
                        "required": ["word", "type", "back"],
                    },
                },
            },
            "required": ["corrections", "card_assessments", "new_word_suggestions"],
        },
    },
}


@dataclass
class CardAssessment:
    """LLM's assessment of how well the learner used a review card."""

    front: str
    used: bool
    quality_suggestion: str  # "again", "hard", "good", "easy"
    reasoning: str = ""


@dataclass
class NewWordSuggestion:
    """LLM's suggestion for a new card to create."""

    word: str
    card_type: str  # "word", "phrase", "grammar"
    back: str  # definition / correct usage
    context: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class TurnMetadata:
    """Structured metadata extracted from a single LLM turn.

    The LLM suggests, the learner confirms — this is the raw suggestion
    before human-in-the-loop confirmation.
    """

    corrections: list[dict[str, str]] = field(default_factory=list)
    card_assessments: list[CardAssessment] = field(default_factory=list)
    new_word_suggestions: list[NewWordSuggestion] = field(default_factory=list)


@dataclass
class TutorResponse:
    """The LLM's response split into the visible message and hidden metadata."""

    message: str
    metadata: TurnMetadata


def build_system_prompt(due_cards: list[db.Row]) -> str:
    """Build the dynamic system prompt with learner profile and SRS cards.

    The system prompt tells the LLM WHO it is (tutor), WHO the learner is
    (level, target language), and WHAT to review (due cards injected from SRS).

    Args:
        due_cards: Cards that are due for review right now.

    Returns:
        The complete system prompt string.
    """
    cards_section = ""
    if due_cards:
        lines = []
        for card in due_cards:
            seen = card["times_seen"]
            correct = card["times_correct"]
            accuracy = f"{correct}/{seen}" if seen > 0 else "never seen"
            status = card["status"]
            lines.append(
                f'- "{card["front"]}" ({card["type"]}, {status}, accuracy: {accuracy})'
            )
        cards_section = (
            "\n\nACTIVE REVIEW (important):\n"
            "Try to NATURALLY include these words/expressions in the conversation,\n"
            "so the learner needs to use them in their response:\n"
            + "\n".join(lines)
            + "\n\nDo NOT list them as flashcards. Weave them into natural conversation.\n"
            "In your report_metadata call, assess how well the learner used each one."
        )

    return f"""You are a conversational English tutor for a {config.LEARNER_LEVEL}-level learner.

RULES:
- Speak ONLY in English. The learner is practicing English.
- Have a natural, engaging conversation on whatever topic the learner chooses.
- If the learner makes an error, correct it BRIEFLY inline and continue the conversation.
- Do NOT interrupt the flow with long grammar explanations.
- For a {config.LEARNER_LEVEL} learner, focus on: nuanced vocabulary, idiomatic expressions, \
subtle grammar (conditionals, subjunctive, collocations), and natural phrasing.
- Keep responses concise (2-4 sentences) to maintain a conversational rhythm.
- ALWAYS call the report_metadata tool after your response.
- This is CRITICAL: report ALL errors via the tool, even minor ones.
  Never skip the tool call. If there are no errors, call it with empty arrays.
- When correcting errors inline, still report them via the tool — the tool call
  is how the learning system tracks progress.
{cards_section}"""


# ---------------------------------------------------------------------------
# Focused error-checking tool — minimal schema, high accuracy
# ---------------------------------------------------------------------------

_ERROR_CHECK_SYSTEM = """You are an English grammar checker for C1-level learners.
Your ONLY job is to find errors in the text and call the report_errors tool.
Check for: grammar, prepositions, collocations (make/do), articles, word order,
subject-verb agreement, false friends, register, tense, relative pronouns.
Do NOT reply with any text — ONLY call the tool."""

_ERROR_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "report_errors",
        "description": "Report all errors found in the learner's text.",
        "parameters": {
            "type": "object",
            "properties": {
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "wrong": {"type": "string"},
                            "correct": {"type": "string"},
                            "type": {"type": "string"},
                        },
                        "required": ["wrong", "correct", "type"],
                    },
                }
            },
            "required": ["errors"],
        },
    },
}


class TutorLLM:
    """Manages the conversation state and LLM interaction.

    Uses a dual-pass architecture for optimal speed + accuracy:

    Pass 1 (conversation): the tutor responds naturally and tries to extract
    metadata (card assessments, new words).  think=False for speed (~5s).

    Pass 2 (error check): a SEPARATE, focused call with a minimal schema
    checks ONLY for errors.  think=False but with a simple prompt, giving
    12/12 accuracy in validation at ~5s.

    Both passes run in parallel using ThreadPoolExecutor, so total latency
    is max(pass1, pass2) ≈ 5-7s, not the sum.
    """

    def __init__(self, due_cards: list[db.Row] | None = None) -> None:
        self.model = config.OLLAMA_MODEL
        self.system_prompt = build_system_prompt(due_cards or [])
        self.history: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, user_message: str) -> TutorResponse:
        """Send a message and get the tutor's response with metadata.

        Runs two LLM passes in parallel:
        1. Conversation pass → natural response + card assessments + new words
        2. Error check pass → focused correction detection (12/12 accuracy)

        Args:
            user_message: What the learner said/typed.

        Returns:
            TutorResponse with the visible message and merged metadata.
        """
        self.history.append({"role": "user", "content": user_message})

        with ThreadPoolExecutor(max_workers=2) as pool:
            conv_future = pool.submit(self._conversation_pass)
            err_future = pool.submit(self._error_check_pass, user_message)

            conv_result = conv_future.result()
            err_result = err_future.result()

        reply_text = conv_result["reply"]
        self.history.append({"role": "assistant", "content": reply_text})

        # Merge: use error checker's corrections (more reliable),
        # conversation's card assessments and new word suggestions
        metadata = conv_result["metadata"]
        metadata.corrections = err_result

        return TutorResponse(message=reply_text, metadata=metadata)

    def _conversation_pass(self) -> dict:
        """Pass 1: natural conversation with card assessments and new words.

        Returns:
            Dict with 'reply' (str) and 'metadata' (TurnMetadata).
        """
        response = ollama.chat(
            model=self.model,
            messages=self.history,
            tools=[METADATA_TOOL],
            think=False,
        )

        assistant_msg = response.message
        reply_text = assistant_msg.content or ""
        metadata = self._extract_metadata(assistant_msg)

        return {"reply": reply_text, "metadata": metadata}

    def _error_check_pass(self, user_message: str) -> list[dict[str, str]]:
        """Pass 2: focused error detection with minimal schema.

        This pass runs independently with no conversation history — just the
        user's message and a focused grammar-checking prompt.  Validated at
        12/12 accuracy on C1 error scenarios.

        Args:
            user_message: The raw text to check for errors.

        Returns:
            List of correction dicts with keys: wrong, correct, type.
        """
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _ERROR_CHECK_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            tools=[_ERROR_CHECK_TOOL],
            think=False,
        )

        tool_calls = getattr(response.message, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                fn = getattr(call, "function", None)
                if fn and getattr(fn, "name", None) == "report_errors":
                    args = getattr(fn, "arguments", {}) or {}
                    raw_errors = args.get("errors", [])
                    return [
                        {
                            "user_said": e.get("wrong", ""),
                            "corrected": e.get("correct", ""),
                            "error_type": e.get("type", "other"),
                        }
                        for e in raw_errors
                    ]

        return []

    def _extract_metadata(self, assistant_msg: object) -> TurnMetadata:
        """Extract card assessments and new words from conversation pass.

        Corrections are handled by the error check pass, so this focuses
        on card_assessments and new_word_suggestions.

        Args:
            assistant_msg: The Message object from Ollama's response.

        Returns:
            Parsed TurnMetadata (corrections will be overwritten by error check).
        """
        tool_calls = getattr(assistant_msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                fn = getattr(call, "function", None)
                if fn and getattr(fn, "name", None) == "report_metadata":
                    args = getattr(fn, "arguments", {}) or {}
                    return _parse_metadata_args(args)

        content = getattr(assistant_msg, "content", "") or ""
        return self._parse_json_fallback(content)

    @staticmethod
    def _parse_json_fallback(text: str) -> TurnMetadata:
        """Try to find and parse a JSON object in the response text.

        Args:
            text: The full response text.

        Returns:
            Parsed TurnMetadata, empty if no valid JSON found.
        """
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            match = re.search(
                r"(\{[^{}]*\"corrections\"[^{}]*\})\s*$", text, re.DOTALL
            )

        if match:
            try:
                data = json.loads(match.group(1))
                return _parse_metadata_args(data)
            except (json.JSONDecodeError, AttributeError):
                pass

        return TurnMetadata()


def _parse_metadata_args(args: dict) -> TurnMetadata:
    """Parse raw metadata arguments into typed dataclasses.

    Handles both the new schema (card_assessments, new_word_suggestions)
    and legacy schema (cards_used_correctly, new_words) for robustness.

    Args:
        args: Raw dict from tool call arguments or parsed JSON.

    Returns:
        Structured TurnMetadata.
    """
    corrections = args.get("corrections", [])

    # Parse card assessments (new schema)
    raw_assessments = args.get("card_assessments", [])
    assessments = []
    for a in raw_assessments:
        assessments.append(CardAssessment(
            front=a.get("front", ""),
            used=a.get("used", False),
            quality_suggestion=a.get("quality_suggestion", "again"),
            reasoning=a.get("reasoning", ""),
        ))

    # Backwards compatibility: convert old cards_used_correctly to assessments
    if not raw_assessments:
        for front in args.get("cards_used_correctly", []):
            assessments.append(CardAssessment(
                front=front, used=True, quality_suggestion="good",
            ))

    # Parse new word suggestions (new schema)
    raw_words = args.get("new_word_suggestions", [])
    suggestions = []
    for w in raw_words:
        suggestions.append(NewWordSuggestion(
            word=w.get("word", ""),
            card_type=w.get("type", "word"),
            back=w.get("back", ""),
            context=w.get("context", ""),
            tags=w.get("tags", []),
        ))

    # Backwards compatibility: convert old new_words format
    if not raw_words:
        for w in args.get("new_words", []):
            suggestions.append(NewWordSuggestion(
                word=w.get("word", ""),
                card_type=w.get("type", "word"),
                back=w.get("context", ""),
                context=w.get("context", ""),
            ))

    return TurnMetadata(
        corrections=corrections,
        card_assessments=assessments,
        new_word_suggestions=suggestions,
    )


def find_card_by_front(conn: db.sqlite3.Connection, text: str) -> int | None:
    """Find a card whose front matches the given text (case-insensitive).

    Args:
        conn: Database connection.
        text: The text to search for.

    Returns:
        The card id if found, None otherwise.
    """
    if not text:
        return None
    row = conn.execute(
        "SELECT id FROM cards WHERE LOWER(front) = LOWER(?) LIMIT 1",
        (text.strip(),),
    ).fetchone()
    return row["id"] if row else None
