"""
transcriber.py
--------------
LiveKit real-time transcription agent for live clinical interviews.
 
Uses:
- LiveKit Agents SDK for room connection and audio streaming
- Silero VAD for voice activity detection
- Groq Whisper for real-time speech-to-text
- Groq Llama 3 for patient/clinician role detection (same logic as align.py)
 
Flow:
    Both participants join LiveKit room via meet.livekit.io
    → Agent connects and subscribes to audio tracks
    → Each participant's audio is transcribed separately (LiveKit gives per-track audio)
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
 
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    AgentSession,
    Agent,
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
        # Fallback: first unknown speaker is patient, second is clinician
        return "PATIENT"
 
 
class ClinicalTranscriptionAgent(Agent):
    """
    LiveKit agent that transcribes audio and labels speakers by role.
    One agent instance is created per room connection.
    """
 
    def __init__(self, room_name: str):
        super().__init__(
            instructions="You are a real-time transcriber for clinical interviews. "
                         "Transcribe speech accurately and completely.",
            stt=_stt,
        )
        self.room_name = room_name
 
        # Initialise session storage for this room
        if room_name not in _sessions:
            _sessions[room_name] = []
 
        # Output path: audio/<room_name>_live_transcript.json
        output_path = f"audio/{room_name}_live_transcript.json"
        _session_files[room_name] = output_path
        Path("audio").mkdir(exist_ok=True)
 
        self._start_time = time.time()
        logger.info(f"Agent started for room: {room_name}")
 
    async def on_user_turn_completed(self, chat_ctx, new_message):
        """
        Called by the LiveKit SDK whenever a participant finishes speaking
        and their speech has been transcribed.
        """
        if not (new_message and hasattr(new_message, "text_content")
                and new_message.text_content):
            return
 
        text = new_message.text_content.strip()
        if not text:
            return
 
        # Determine who spoke — LiveKit gives us the participant identity
        # via the chat context's last user message source
        identity = "unknown"
        try:
            if hasattr(new_message, "participant") and new_message.participant:
                identity = new_message.participant.identity
        except Exception:
            pass
 
        role = get_role_for_identity(identity)
        now = time.time()
        elapsed = now - self._start_time
 
        segment = {
            "speaker":    identity,
            "role":       role,
            "start":      round(elapsed, 3),
            "end":        round(elapsed + 2.0, 3),  # approximate end
            "start_time": round(elapsed, 3),
            "end_time":   round(elapsed + 2.0, 3),
            "text":       text,
        }
 
        _sessions[self.room_name].append(segment)
 
        # Write to file in real-time so Flask can read it at any point
        output_path = _session_files[self.room_name]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_sessions[self.room_name], f, indent=2)
 
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
 
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("Connected to LiveKit room")
 
        session = AgentSession(stt=_stt, vad=_vad)
        agent = ClinicalTranscriptionAgent(room_name=room_name)
 
        await session.start(agent=agent, room=ctx.room)
        logger.info("Transcription session started — waiting for participants...")
        print("Transcriber ready. Waiting for participants...\n")
 
        while True:
            await asyncio.sleep(1)
 
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Transcriber stopped")
 
 
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
