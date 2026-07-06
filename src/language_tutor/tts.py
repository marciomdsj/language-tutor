"""Text-to-Speech abstraction layer with pluggable providers.

Architecture:
    TTSProvider is a Protocol (structural typing) — any class that implements
    synthesize(text) -> bytes is a valid provider.  No inheritance needed.

    Providers:
    - EdgeTTSProvider: Microsoft neural voices via edge-tts (free, requires internet)
    - PiperTTSProvider: Local VITS voices via piper-tts (free, offline) [future]
    - OpenAITTSProvider: OpenAI TTS API (paid, high quality) [future]

    The active provider is selected by config.TTS_PROVIDER and instantiated
    by get_tts_provider().
"""

from __future__ import annotations

import asyncio
import io
from typing import Protocol, runtime_checkable

from language_tutor import config


@runtime_checkable
class TTSProvider(Protocol):
    """Interface for text-to-speech providers.

    Any class implementing this protocol can be used as a TTS backend.
    The synthesize method returns raw audio bytes (MP3 or WAV).
    """

    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes.

        Args:
            text: The text to speak.

        Returns:
            Raw audio bytes (format depends on provider).
        """
        ...

    @property
    def audio_format(self) -> str:
        """The audio format returned by synthesize ('mp3' or 'wav')."""
        ...


class EdgeTTSProvider:
    """Microsoft neural TTS via the edge-tts library (free, no API key).

    Uses Microsoft Edge's Read Aloud voices — high quality neural voices
    at zero cost.  Requires internet connection.

    The edge-tts library is async, so we wrap calls with asyncio.run().
    """

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        self.voice = voice

    @property
    def audio_format(self) -> str:
        return "mp3"

    def synthesize(self, text: str) -> bytes:
        """Convert text to MP3 audio using edge-tts.

        Args:
            text: The text to speak.

        Returns:
            MP3 audio bytes.
        """
        return asyncio.run(self._synthesize_async(text))

    async def _synthesize_async(self, text: str) -> bytes:
        """Async implementation of TTS synthesis.

        Args:
            text: The text to speak.

        Returns:
            MP3 audio bytes.
        """
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

        return buffer.getvalue()


def get_tts_provider() -> TTSProvider:
    """Create and return the configured TTS provider.

    Uses config.TTS_PROVIDER to select the backend:
    - "edge" → EdgeTTSProvider (default)
    - "piper" → PiperTTSProvider (future)
    - "openai" → OpenAITTSProvider (future)

    Returns:
        An instance of the selected TTSProvider.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    provider = config.TTS_PROVIDER.lower()

    if provider == "edge":
        return EdgeTTSProvider()
    else:
        raise ValueError(
            f"Unsupported TTS provider: {provider!r}. "
            "Available: 'edge'"
        )
