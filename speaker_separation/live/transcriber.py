"""
transcriber.py
--------------
LiveKit real-time transcription agent for live clinical interviews.

Uses:
- LiveKit Agents SDK for room connection and audio streaming
- Silero VAD for voice activity detection
- Groq Whisper for real-time speech-to-text

Flow:
    Both participants join LiveKit room via meet.livekit.io
    → Agent connects and subscribes to ALL participant audio tracks
    → Each participant's audio is transcribed separately (one coroutine per track)
    → Role detection maps participant identities to PATIENT / CLINICIAN
    → Segments are written to a session JSON file in real-time
    → Flask /live/stop endpoint reads the file, indexes it, and runs MedGemma

Speaker Identity:
    LiveKit participant identities are set when generating tokens.
    We generate tokens with identity="patient" and identity="clinician"
    so we always know which track belongs to which role — no LLM detection needed.
"""

import os
import json
import asyncio
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import groq as livekit_groq
from livekit.plugins.silero import VAD

Path("audio").mkdir(exist_ok=True)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live-transcriber")

# ── Preload models once at startup ────────────────────────────────────
print("Preloading models...")
_vad = VAD.load()
_stt = livekit_groq.STT()
print("Models ready.")

# ── Session state ─────────────────────────────────────────────────────
# Maps room_name → list of transcript segments collected so far
_sessions: dict[str, list[dict]] = {}
_session_files: dict[str, str] = {}  # room_name → output file path


def get_role_for_identity(identity: str) -> str:
    """
    Map LiveKit participant identity to clinical role.
    Tokens are generated with identity='patient' or identity='clinician'.
    """
    identity_lower = identity.lower()
    if "patient" in identity_lower:
        return "PATIENT"
    elif "clinician" in identity_lower or "doctor" in identity_lower:
        return "CLINICIAN"
    else:
        return "PATIENT"


async def transcribe_track(
    participant: rtc.RemoteParticipant,
    track: rtc.Track,
    room_name: str,
    start_time: float,
    output_path: str,
):
    """
    Transcribe a single participant's audio track continuously.
    One coroutine runs per participant, so both are transcribed in parallel.
    """
    identity = participant.identity
    role = get_role_for_identity(identity)
    logger.info(f"Started transcribing track for: {identity} ({role})")

    audio_stream = rtc.AudioStream(track)

    stt_stream = _stt.stream()

    async def push_audio():
        async for audio_frame_event in audio_stream:
            stt_stream.push_frame(audio_frame_event.frame)
        await stt_stream.aclose()

    asyncio.ensure_future(push_audio())

    async for event in stt_stream:
        if not hasattr(event, "alternatives") or not event.alternatives:
            continue
        text = event.alternatives[0].text.strip()
        if not text:
            continue

        elapsed = time.time() - start_time
        segment = {
            "speaker":    identity,
            "role":       role,
            "start":      round(elapsed, 3),
            "end":        round(elapsed + 2.0, 3),
            "start_time": round(elapsed, 3),
            "end_time":   round(elapsed + 2.0, 3),
            "text":       text,
        }

        _sessions[room_name].append(segment)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_sessions[room_name], f, indent=2)

        timestamp = time.strftime("%H:%M:%S")
        logger.info(f"[{timestamp}] [{role}] {text}")
        print(f"[{timestamp}] [{role:10s}] {text}")


async def entrypoint(ctx: JobContext):
    """Main entry point — called by LiveKit worker for each new room job."""
    room_name = ctx.room.name
    print(f"\n{'='*60}")
    print(f"LIVE CLINICAL TRANSCRIBER")
    print(f"Room: {room_name}")
    print(f"{'='*60}")

    required = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GROQ_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error(f"Missing env vars: {', '.join(missing)}")
        return

    # Initialise session storage for this room
    if room_name not in _sessions:
        _sessions[room_name] = []

    output_path = f"audio/{room_name}_live_transcript.json"
    _session_files[room_name] = output_path

    start_time = time.time()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to LiveKit room")

    async def handle_track(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Subscribe to an audio track and spawn a transcription coroutine for it."""
        if publication.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if publication.track is None:
            return
        asyncio.ensure_future(
            transcribe_track(participant, publication.track, room_name, start_time, output_path)
        )

    # Handle participants already in the room when the agent joins
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track is not None:
                await handle_track(publication, participant)

    # Handle tracks from participants who join or publish after the agent connects
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        asyncio.ensure_future(handle_track(publication, participant))

    logger.info("Transcription agent ready — listening to all participants...")
    print("Transcriber ready. Waiting for participants...\n")

    while True:
        await asyncio.sleep(1)


def get_session_transcript(room_name: str) -> list[dict]:
    """Return in-memory transcript segments for a room (used by Flask)."""
    return _sessions.get(room_name, [])


def get_session_file(room_name: str) -> str | None:
    """Return the output file path for a room's transcript."""
    return _session_files.get(room_name)


def start_worker():
    """Start the LiveKit worker. Call this from a background thread."""
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
