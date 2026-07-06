"""Speech-to-Text abstraction layer with Vosk provider.

Architecture:
    STTProvider is a Protocol — any class implementing transcribe(audio_bytes)
    is a valid provider.

    Providers:
    - VoskSTTProvider: Local speech recognition via Vosk (free, offline)
    - DeepgramSTTProvider: Cloud API (paid, high accuracy) [future]

    On first use, VoskSTTProvider automatically downloads the English model
    (~50MB) to data/models/vosk/.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from language_tutor import config
from language_tutor.audio import SAMPLE_RATE

# Where Vosk models are stored
MODELS_DIR = config.DATA_DIR / "models" / "vosk"

# Small English model — good balance of speed and accuracy
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"


@runtime_checkable
class STTProvider(Protocol):
    """Interface for speech-to-text providers."""

    def transcribe(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to text.

        Args:
            audio_bytes: Raw PCM audio (16kHz, mono, int16).

        Returns:
            The transcribed text.
        """
        ...


class VoskSTTProvider:
    """Local speech recognition using Vosk (free, offline).

    On first instantiation, downloads the English model (~50MB) if not
    already present.  Subsequent uses load from disk instantly.

    Vosk runs entirely on CPU — no GPU needed, works great on Apple Silicon.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_path = MODELS_DIR / VOSK_MODEL_NAME

    def _ensure_model(self) -> Path:
        """Download the Vosk model if not present.

        Returns:
            Path to the model directory.
        """
        if self._model_path.exists():
            return self._model_path

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        from rich.progress import Progress
        import urllib.request

        zip_path = MODELS_DIR / f"{VOSK_MODEL_NAME}.zip"

        print(f"Downloading Vosk model ({VOSK_MODEL_NAME})...")
        with Progress() as progress:
            task = progress.add_task("Downloading...", total=None)

            def reporthook(block_num: int, block_size: int, total_size: int) -> None:
                if total_size > 0:
                    progress.update(task, total=total_size, completed=block_num * block_size)

            urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path, reporthook=reporthook)

        print("Extracting model...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(MODELS_DIR)

        zip_path.unlink()
        print("Model ready.")

        return self._model_path

    def _get_model(self) -> object:
        """Lazy-load the Vosk model.

        Returns:
            A vosk.Model instance.
        """
        if self._model is None:
            import vosk

            vosk.SetLogLevel(-1)  # Suppress Vosk's verbose logging
            model_path = self._ensure_model()
            self._model = vosk.Model(str(model_path))

        return self._model

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe raw PCM audio to text using Vosk.

        Args:
            audio_bytes: Raw PCM audio (16kHz, mono, int16).

        Returns:
            The transcribed text, or empty string if nothing recognized.
        """
        import vosk

        model = self._get_model()
        recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)

        # Feed audio in chunks for streaming recognition
        chunk_size = 4000
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            recognizer.AcceptWaveform(chunk)

        result = json.loads(recognizer.FinalResult())
        return result.get("text", "").strip()


def get_stt_provider() -> STTProvider:
    """Create and return the configured STT provider.

    Uses config.STT_PROVIDER to select the backend:
    - "vosk" → VoskSTTProvider (default)
    - "deepgram" → DeepgramSTTProvider (future)

    Returns:
        An instance of the selected STTProvider.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    provider = config.STT_PROVIDER.lower()

    if provider == "vosk":
        return VoskSTTProvider()
    else:
        raise ValueError(
            f"Unsupported STT provider: {provider!r}. "
            "Available: 'vosk'"
        )
