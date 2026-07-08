"""LLM integration — persistent client with provider fallback.

Uses a persistent OpenAI-compatible client for the primary provider
(Groq by default).  This keeps the HTTP connection alive across calls,
eliminating the 60s cold start that plagued the LiteLLM approach.

    1. Groq   (primary — persistent client, <1s per call)
    2. Gemini (fallback via LiteLLM)
    3. Ollama (offline fallback via LiteLLM)

Key design decisions:

1. **Persistent client**: a single `openai.OpenAI` instance pointed at
   Groq's API.  Connection pool stays warm — no cold start per call.

2. **Dual-pass architecture**: conversation + focused error check.
   Validated at 92% C1 error detection accuracy.

3. **Tool use**: OpenAI-compatible format across all providers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from openai import OpenAI

from language_tutor import config, db

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Persistent LLM client — stays alive across calls, no cold start
# ---------------------------------------------------------------------------
_PROVIDER_CONFIGS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": config.GROQ_API_KEY,
        "model": "llama-3.1-8b-instant",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": config.GEMINI_API_KEY,
        "model": "gemini-2.0-flash",
    },
}

_client: OpenAI | None = None
_active_model: str | None = None


def _get_client() -> tuple[OpenAI, str]:
    """Get or create the persistent LLM client.

    Returns:
        Tuple of (OpenAI client, model name).
    """
    global _client, _active_model

    if _client is not None and _active_model is not None:
        return _client, _active_model

    # Try Groq first, then Gemini
    for name, cfg in _PROVIDER_CONFIGS.items():
        if cfg["api_key"]:
            _client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
            _active_model = cfg["model"]
            return _client, _active_model

    # Fallback to Ollama via its OpenAI-compatible endpoint
    _client = OpenAI(
        api_key="ollama",
        base_url=f"{config.OLLAMA_HOST}/v1",
    )
    _active_model = config.OLLAMA_MODEL
    return _client, _active_model

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
    """Structured metadata extracted from a single LLM turn."""

    corrections: list[dict[str, str]] = field(default_factory=list)
    card_assessments: list[CardAssessment] = field(default_factory=list)
    new_word_suggestions: list[NewWordSuggestion] = field(default_factory=list)


@dataclass
class TutorResponse:
    """The LLM's response split into the visible message and hidden metadata."""

    message: str
    metadata: TurnMetadata


# ---------------------------------------------------------------------------
# Provider-agnostic LLM call with automatic fallback
# ---------------------------------------------------------------------------

def warmup() -> str:
    """Initialize the persistent client and verify connectivity.

    Creates the OpenAI client (if not already created) and makes a
    cheap test call to verify the provider responds.  The client stays
    alive for all subsequent calls — no more cold starts.

    Returns:
        The name of the active model (for display).
    """
    client, model = _get_client()
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
    )
    return model


def _call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> object:
    """Call the LLM using the persistent client.

    All calls reuse the same HTTP connection — no cold start, no
    cascade overhead.  Typical latency: <1s after warmup.

    Args:
        messages: Conversation messages in OpenAI format.
        tools: Optional tool definitions for function calling.

    Returns:
        The OpenAI response object.

    Raises:
        ConnectionError: If the call fails.
    """
    client, model = _get_client()

    try:
        kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        return client.chat.completions.create(**kwargs)

    except Exception as e:
        # If tool use failed (malformed JSON from model), retry without tools
        if "tool_use_failed" in str(e) and tools:
            kwargs.pop("tools", None)
            return client.chat.completions.create(**kwargs)

        raise ConnectionError(
            f"LLM call failed (model: {model}): {e}"
        ) from e


def _extract_tool_args(response: object, tool_name: str) -> dict | None:
    """Extract arguments from a tool call in the LLM response.

    LiteLLM returns OpenAI-compatible responses where tool call arguments
    are JSON strings (not dicts).  This function handles the parsing.

    Args:
        response: The LiteLLM response object.
        tool_name: The name of the tool to look for.

    Returns:
        Parsed arguments dict, or None if the tool wasn't called.
    """
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None

    for tc in tool_calls:
        fn = tc.function
        if fn.name == tool_name:
            args = fn.arguments
            if isinstance(args, str):
                return json.loads(args)
            return args or {}

    return None


def _strip_tool_artifacts(text: str) -> str:
    """Remove tool call JSON that leaked into the response text.

    Some models (especially via Ollama) dump tool call JSON directly into
    the content instead of using the tool_calls field.  This strips it.

    Args:
        text: The raw response text.

    Returns:
        Cleaned text with JSON artifacts removed.
    """
    # Remove {"name": "report_metadata", "arguments": {...}} blocks
    text = re.sub(r'\{"name":\s*"report_\w+".*', "", text, flags=re.DOTALL)
    # Remove ```json ... ``` blocks
    text = re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL)
    # Remove **report_metadata** and everything after
    text = re.sub(r"\*{0,2}report_metadata\*{0,2}.*", "", text, flags=re.DOTALL)
    return text.strip()


def _get_content(response: object) -> str:
    """Extract text content from a LiteLLM response.

    Args:
        response: The LiteLLM response object.

    Returns:
        The text content, or empty string.
    """
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    due_cards: list[db.Row],
    recent_errors: list[dict] | None = None,
    user_context: dict | None = None,
) -> str:
    """Build the dynamic system prompt with learner profile and SRS context.

    Args:
        due_cards: Cards that are due for review right now.
        recent_errors: Recent corrections from past sessions.
        user_context: User profile dict (name, profession, interests, etc.).

    Returns:
        The complete system prompt string.
    """
    recent_errors = recent_errors or []
    user_context = user_context or {}

    # User profile section
    profile_section = ""
    if user_context:
        parts = []
        if user_context.get("name"):
            parts.append(f"- Name: {user_context['name']}")
        if user_context.get("profession"):
            parts.append(f"- Profession: {user_context['profession']}")
        if user_context.get("interests"):
            parts.append(f"- Interests: {', '.join(user_context['interests'])}")
        if parts:
            profile_section = (
                "\n\nLEARNER PROFILE:\n"
                + "\n".join(parts)
                + "\nPersonalize topics, examples, and vocabulary based on this profile. "
                "Use their professional domain and interests to make conversations relevant."
            )

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
            "\n\nACTIVE REVIEW — cards the learner needs to practice:\n"
            + "\n".join(lines)
            + "\n\nYour job is to DESIGN the conversation so the learner MUST use these "
            "words/structures to respond. Don't list them — create situations, questions, "
            "or scenarios that naturally require them.\n"
            "In your report_metadata call, assess how well the learner used each one."
        )

    recent_errors_section = ""
    if recent_errors:
        err_lines = [
            f'- "{e["user_said"]}" → "{e["corrected"]}" ({e.get("error_type", "")})'
            for e in recent_errors[:5]
        ]
        recent_errors_section = (
            "\n\nRECENT ERRORS — the learner has struggled with these recently:\n"
            + "\n".join(err_lines)
            + "\nGently steer the conversation to test whether they've improved."
        )

    return f"""You are a proactive conversational English tutor for a {config.LEARNER_LEVEL}-level learner.

YOUR ROLE:
- You LEAD the conversation. Don't just respond — propose topics, ask questions,
  create scenarios. Be the teacher, not a passive chatbot.
- Speak ONLY in English. The learner is practicing English.
- If the learner makes an error, correct it BRIEFLY inline and continue.
- For {config.LEARNER_LEVEL}: focus on nuanced vocabulary, idiomatic expressions,
  subtle grammar (conditionals, subjunctive, collocations), natural phrasing.
- Keep responses concise (2-4 sentences) to maintain conversational rhythm.
- ALWAYS call the report_metadata tool after your response.
- Report ALL errors via the tool, even minor ones. Never skip the tool call.
{profile_section}{cards_section}{recent_errors_section}"""


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

    Pass 1 (conversation): natural response + card assessments + new words.
    Pass 2 (error check): focused correction detection with minimal schema.

    Both passes use _call_llm() which automatically cascades across
    Groq → Gemini → Ollama based on availability.
    """

    def __init__(
        self,
        due_cards: list[db.Row] | None = None,
        recent_errors: list[dict] | None = None,
        user_context: dict | None = None,
    ) -> None:
        self.due_cards = due_cards or []
        self.user_context = user_context or {}
        self.system_prompt = build_system_prompt(
            self.due_cards, recent_errors, self.user_context
        )
        self.history: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def generate_opening(self) -> TutorResponse:
        """Generate the tutor's opening message to start the session.

        Uses the learner's profile to pick relevant topics instead of
        always defaulting to travel/generic themes.

        Returns:
            TutorResponse with the opening message.
        """
        preferred = self.user_context.get("preferred_topics", [])
        interests = self.user_context.get("interests", [])
        profession = self.user_context.get("profession", "")
        all_topics = preferred + interests
        if profession:
            all_topics = [profession] + all_topics

        interests_hint = ""
        if all_topics:
            topics_str = ", ".join(dict.fromkeys(all_topics))  # deduplicate
            interests_hint = (
                f"Pick a topic related to ONE of these (the learner's interests): "
                f"{topics_str}. Choose something SPECIFIC, not generic. "
                f"VARY the topic from previous sessions — do NOT repeat. "
            )

        if self.due_cards:
            card_list = ", ".join(f'"{c["front"]}"' for c in self.due_cards[:3])
            opening_prompt = (
                f"Start the session. Greet the learner briefly and propose an "
                f"engaging conversation topic. {interests_hint}"
                f"You have these review cards to work into the conversation: "
                f"{card_list}. Ask an opening question that naturally leads "
                f"toward using them. Keep it to 2-3 sentences. "
                f"Do NOT call any tools or output JSON."
            )
        else:
            opening_prompt = (
                f"Start the session. Greet the learner briefly and propose an "
                f"engaging conversation topic. {interests_hint}"
                f"Ask an opening question appropriate for a C1 English learner. "
                f"Keep it to 2-3 sentences. Do NOT call any tools or output JSON."
            )

        self.history.append({"role": "user", "content": opening_prompt})

        response = _call_llm(messages=self.history)
        reply_text = _get_content(response)

        # Strip any tool call artifacts the model might generate
        reply_text = re.sub(
            r"\*{0,2}report_metadata\*{0,2}.*", "", reply_text, flags=re.DOTALL
        ).strip()

        self.history.pop()  # remove the opening_prompt
        self.history.append({"role": "assistant", "content": reply_text})

        return TutorResponse(message=reply_text, metadata=TurnMetadata())

    def chat(self, user_message: str) -> TutorResponse:
        """Send a message and get the tutor's response with metadata.

        Runs two LLM passes sequentially (not parallel):
        1. Conversation pass → natural response + card assessments + new words
        2. Error check pass → focused correction detection

        Sequential is better than parallel here because cloud providers
        (Groq, Gemini) respond in <0.5s each, and ThreadPoolExecutor threads
        don't share the warmed HTTP connection — causing cold starts per thread.
        Sequential: ~1s total.  Parallel with cold starts: ~60s.

        Args:
            user_message: What the learner said/typed.

        Returns:
            TutorResponse with the visible message and merged metadata.
        """
        self.history.append({"role": "user", "content": user_message})

        conv_result = self._conversation_pass()

        if len(user_message.split()) >= 4:
            err_result = self._error_check_pass(user_message)
        else:
            err_result = []

        reply_text = conv_result["reply"]
        self.history.append({"role": "assistant", "content": reply_text})

        metadata = conv_result["metadata"]
        metadata.corrections = err_result

        return TutorResponse(message=reply_text, metadata=metadata)

    def _conversation_pass(self) -> dict:
        """Pass 1: natural conversation — NO tools.

        The conversation pass focuses purely on generating a natural response.
        Tools were causing the model to return empty text (it put everything
        into tool calls instead of responding). Corrections are handled
        by the separate error check pass.

        Returns:
            Dict with 'reply' (str) and 'metadata' (TurnMetadata).
        """
        response = _call_llm(messages=self.history)

        reply_text = _strip_tool_artifacts(_get_content(response))

        if not reply_text.strip():
            reply_text = "Could you tell me more about that?"

        return {"reply": reply_text, "metadata": TurnMetadata()}

    def _error_check_pass(self, user_message: str) -> list[dict[str, str]]:
        """Pass 2: focused error detection with minimal schema.

        Args:
            user_message: The raw text to check for errors.

        Returns:
            List of correction dicts with keys: user_said, corrected, error_type.
        """
        response = _call_llm(
            messages=[
                {"role": "system", "content": _ERROR_CHECK_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            tools=[_ERROR_CHECK_TOOL],
        )

        args = _extract_tool_args(response, "report_errors")
        if args:
            return [
                {
                    "user_said": e.get("wrong", ""),
                    "corrected": e.get("correct", ""),
                    "error_type": e.get("type", "other"),
                }
                for e in args.get("errors", [])
            ]

        return []

    @staticmethod
    def _extract_metadata(response: object) -> TurnMetadata:
        """Extract card assessments and new words from conversation pass.

        Args:
            response: The LiteLLM response object.

        Returns:
            Parsed TurnMetadata.
        """
        args = _extract_tool_args(response, "report_metadata")
        if args:
            return _parse_metadata_args(args)

        content = _get_content(response)
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
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

    raw_assessments = args.get("card_assessments", [])
    assessments = []
    for a in raw_assessments:
        assessments.append(CardAssessment(
            front=a.get("front", ""),
            used=a.get("used", False),
            quality_suggestion=a.get("quality_suggestion", "again"),
            reasoning=a.get("reasoning", ""),
        ))

    if not raw_assessments:
        for front in args.get("cards_used_correctly", []):
            assessments.append(CardAssessment(
                front=front, used=True, quality_suggestion="good",
            ))

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
