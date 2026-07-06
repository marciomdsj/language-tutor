"""Audio I/O — microphone recording and audio playback.

This module handles the low-level audio operations:
- Recording from the microphone until the user presses Enter
- Playing audio bytes (MP3 or WAV) through the speakers

Recording uses sounddevice to capture audio in a background thread,
while the main thread waits for Enter.  Playback uses miniaudio for
MP3 support without requiring ffmpeg.
"""

from __future__ import annotations

import io
import queue
import threading
from typing import Literal

import miniaudio
import numpy as np
import sounddevice as sd

# Audio recording settings
SAMPLE_RATE = 16000  # 16kHz — standard for speech recognition
CHANNELS = 1         # Mono
DTYPE = "int16"      # 16-bit PCM


def record_until_enter() -> bytes:
    """Record audio from the microphone until the user presses Enter.

    Starts recording in a background thread.  The main thread blocks
    on input() waiting for Enter.  When Enter is pressed, recording
    stops and the captured audio is returned as raw PCM bytes.

    Returns:
        Raw PCM audio bytes (16kHz, mono, int16).
    """
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block."""
        if not stop_event.is_set():
            audio_queue.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
        blocksize=1024,
    )

    stream.start()

    # Wait for Enter in main thread
    try:
        input()
    except EOFError:
        pass

    stop_event.set()
    stream.stop()
    stream.close()

    # Collect all recorded chunks
    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())

    if not chunks:
        return b""

    audio_data = np.concatenate(chunks)
    return audio_data.tobytes()


def play_audio(audio_bytes: bytes, audio_format: Literal["mp3", "wav"] = "mp3") -> None:
    """Play audio bytes through the speakers.

    Uses miniaudio for decoding (supports MP3 natively without ffmpeg)
    and sounddevice for playback.

    Args:
        audio_bytes: Raw audio file bytes (MP3 or WAV).
        audio_format: The format of the audio bytes.
    """
    if not audio_bytes:
        return

    if audio_format == "mp3":
        decoded = miniaudio.decode(audio_bytes, sample_rate=SAMPLE_RATE, nchannels=1)
        samples = np.array(decoded.samples, dtype=np.float32)
        # Normalize to [-1, 1] range for sounddevice
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
        sd.play(samples, samplerate=decoded.sample_rate)
        sd.wait()
    elif audio_format == "wav":
        decoded = miniaudio.decode(audio_bytes, sample_rate=SAMPLE_RATE, nchannels=1)
        samples = np.array(decoded.samples, dtype=np.float32)
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
        sd.play(samples, samplerate=decoded.sample_rate)
        sd.wait()


def play_audio_async(audio_bytes: bytes, audio_format: Literal["mp3", "wav"] = "mp3") -> threading.Thread:
    """Play audio in a background thread (non-blocking).

    Returns immediately — audio plays while the program continues.
    Useful for playing TTS while showing text in the terminal.

    Args:
        audio_bytes: Raw audio file bytes.
        audio_format: The format of the audio bytes.

    Returns:
        The playback thread (can be joined if needed).
    """
    thread = threading.Thread(
        target=play_audio,
        args=(audio_bytes, audio_format),
        daemon=True,
    )
    thread.start()
    return thread
