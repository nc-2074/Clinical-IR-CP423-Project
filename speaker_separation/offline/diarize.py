"""
diarize.py
----------
Step 1 of offline pipeline: take a raw audio file and produce
speaker-labeled time segments using Pyannote.

Output example:
    [
        {"speaker": "SPEAKER_0", "start": 0.0,  "end": 4.2},
        {"speaker": "SPEAKER_1", "start": 4.5,  "end": 9.1},
        ...
    ]
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

# ── How many speakers are in the audio? ──────────────────────────────
# Clinical interviews always have exactly 2: patient + clinician.
# Telling Pyannote this upfront makes it far more accurate.
NUM_SPEAKERS = 2


from huggingface_hub import login

def load_diarization_pipeline() -> Pipeline:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is missing from your .env file."
        )

    print("Loading Pyannote diarization model (downloads on first run ~300MB)...")
    
    # Login to HuggingFace first
    login(token=hf_token)
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    )

    print("Model loaded.")
    return pipeline


def diarize_audio(audio_path: str, pipeline: Pipeline = None) -> list[dict]:
    """
    Run speaker diarization on a single audio file.

    Parameters
    ----------
    audio_path : str
        Path to a .wav, .mp3, or .m4a file.
    pipeline : Pipeline, optional
        Pre-loaded Pyannote pipeline. If None, it will be loaded fresh.

    Returns
    -------
    list[dict]
        Sorted list of diarization segments:
        [{"speaker": "SPEAKER_0", "start": float, "end": float}, ...]
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if pipeline is None:
        pipeline = load_diarization_pipeline()

    print(f"Diarizing: {audio_path.name}")

    # num_speakers=2 tells Pyannote we know there are exactly 2 speakers.
    # This improves accuracy for clinical interviews.
    diarization = pipeline(str(audio_path), num_speakers=NUM_SPEAKERS)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start":   round(turn.start, 3),
            "end":     round(turn.end,   3),
        })

    # Sort by start time (should already be sorted, but just in case)
    segments.sort(key=lambda s: s["start"])

    print(f"  Found {len(segments)} segments across "
          f"{len({s['speaker'] for s in segments})} speakers.")
    return segments


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python diarize.py path/to/audio.wav")
        sys.exit(1)

    result = diarize_audio(sys.argv[1])
    print(json.dumps(result, indent=2))


