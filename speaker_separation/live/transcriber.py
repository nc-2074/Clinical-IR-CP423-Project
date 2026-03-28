"""
transcriber.py
--------------
LiveKit real-time transcription agent for live clinical interviews.

Uses:
- LiveKit Agents SDK for room connection and audio streaming
- Silero VAD for voice activity detection (speech chunking)
- Groq Whisper REST API for transcription (batch per speech chunk)

Flow:
    Both participants join LiveKit room via meet.livekit.io
    → Agent connects and subscribes to ALL participant audio tracks
    → Silero VAD detects when each participant starts/stops speaking
    → Each complete speech chunk is sent to Groq Whisper via REST
    → Role detection maps participant identities to PATIENT / CLINICIAN
    → Segments are written to a session JSON file in real-time
    → Flask /live/stop endpoint reads the file, indexes it, and runs MedGemma
"""

import os
import io
import json
import wave
import asyncio
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins.silero import VAD

Path("audio").mkdir(exist_ok=True)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live-transcriber")

# ── Preload models once at startup ────────────────────────────────────
print("Preloading models...")
_vad = VAD.load()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Models ready.")

# ── Session state ─────────────────────────────────────────────────────
_sessions: dict[str, list[dict]] = {}
_session_files: dict[str, str] = {}

SAMPLE_RATE = 16000
NUM_CHANNELS = 1


def get_role_for_identity(identity: str) -> str:
    identity_lower = identity.lower()
    if "patient" in identity_lower:
        return "PATIENT"
    elif "clinician" in identity_lower or "doctor" in identity_lower:
        return "CLINICIAN"
    else:
        return "PATIENT"


def frames_to_wav_bytes(frames: list, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a list of audio frames into WAV bytes for Groq Whisper."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.
