"""
transcribe.py
-------------
Step 2 of offline pipeline: transcribe the audio file using Whisper
via the Groq API (free tier — much faster than running Whisper locally).

Output example:
    [
        {"start": 0.0, "end": 4.1, "text": "I've been having headaches..."},
        {"start": 4.5, "end": 9.0, "text": "How long has this been going on?"},
        ...
    ]
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Groq free tier supports these Whisper models:
# "whisper-large-v3"         <- most accurate, use this
# "whisper-large-v3-turbo"   <- faster, slightly less accurate
WHISPER_MODEL = "whisper-large-v3"


def get_groq_client() -> Groq:
    """Create a Groq API client using the key from .env"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing from your .env file. "
            "Get a free key at https://console.groq.com"
        )
    return Groq(api_key=api_key)


def transcribe_audio(audio_path: str, client: Groq = None) -> list[dict]:
    """
    Transcribe an audio file using Whisper via Groq.

    Parameters
    ----------
    audio_path : str
        Path to .wav / .mp3 / .m4a file.
    client : Groq, optional
        Pre-initialized Groq client.

    Returns
    -------
    list[dict]
        Word/segment level timestamps:
        [{"start": float, "end": float, "text": str}, ...]
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if client is None:
        client = get_groq_client()

    print(f"Transcribing: {audio_path.name} with {WHISPER_MODEL}...")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, f.read()),
            model=WHISPER_MODEL,
            # "verbose_json" gives us timestamps per segment — essential for alignment
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="en",   # set to your language or remove for auto-detect
        )

    # Convert Groq response segments into plain dicts
    segments = []
    
    for seg in response.segments:
        # handle both object-style and dict-style responses
        if isinstance(seg, dict):
            segments.append({
                "start": round(seg["start"], 3),
                "end":   round(seg["end"],   3),
                "text":  seg["text"].strip(),
            })
        else:
            segments.append({
                "start": round(seg.start, 3),
                "end":   round(seg.end,   3),
                "text":  seg.text.strip(),
            })

    print(f"  Got {len(segments)} transcript segments.")
    return segments


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py path/to/audio.wav")
        sys.exit(1)

    result = transcribe_audio(sys.argv[1])
    print(json.dumps(result, indent=2))
