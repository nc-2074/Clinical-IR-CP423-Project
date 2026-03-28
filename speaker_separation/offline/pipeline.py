"""
pipeline.py
-----------
The main entry point for offline speaker separation.

Runs the full 3-step process:
    1. diarize.py   → who spoke when
    2. transcribe.py → what was said when
    3. align.py     → merge into speaker-labeled transcript

Usage from command line:
    python pipeline.py path/to/interview.wav

Usage as a module:
    from speaker_separation.offline.pipeline import run_offline_pipeline
    result = run_offline_pipeline("interview.wav")
"""

import json
import argparse
from pathlib import Path

from .diarize     import load_diarization_pipeline, diarize_audio
from .transcribe  import get_groq_client, transcribe_audio
from .align       import align




def run_offline_pipeline(
    audio_path: str,
    patient_speaker: str = None,
    output_path: str = None,
) -> list[dict]:
    # Default output path to the audio file name with .json extension
    if output_path is None:
        output_path = Path(audio_path).stem + "_transcript.json"

    """
    Run the complete offline speaker separation pipeline.

    Parameters
    ----------
    audio_path : str
        Path to the interview audio file.
    patient_speaker : str, optional
        Force "SPEAKER_0" or "SPEAKER_1" as the patient.
        If None, auto-detected based on speaking time.
    output_path : str, optional
        If provided, saves the JSON result to this path.

    Returns
    -------
    list[dict]
        Speaker-labeled transcript segments:
        [{"speaker", "role", "start", "end", "text"}, ...]
    """
    print("=" * 60)
    print("OFFLINE SPEAKER SEPARATION PIPELINE")
    print("=" * 60)

    # ── Step 1: Diarization ──────────────────────────────────────────
    print("\n[1/3] Running speaker diarization (Pyannote)...")
    diarization_pipeline = load_diarization_pipeline()
    diarization_segments = diarize_audio(audio_path, diarization_pipeline)

    # ── Step 2: Transcription ────────────────────────────────────────
    print("\n[2/3] Transcribing audio (Whisper via Groq)...")
    groq_client = get_groq_client()
    transcript_segments = transcribe_audio(audio_path, groq_client)

    # ── Step 3: Alignment ────────────────────────────────────────────
    print("\n[3/3] Aligning diarization with transcription...")
    labeled_transcript = align(
        transcript_segments,
        diarization_segments,
        patient_speaker=patient_speaker,
    )

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULT PREVIEW (first 5 segments):")
    print("=" * 60)
    for seg in labeled_transcript[:5]:
        print(f"  [{seg['role']:10s}] {seg['start']:6.1f}s – {seg['end']:6.1f}s  |  {seg['text']}")

    patient_count   = sum(1 for s in labeled_transcript if s["role"] == "PATIENT")
    clinician_count = sum(1 for s in labeled_transcript if s["role"] == "CLINICIAN")
    print(f"\n  Total segments : {len(labeled_transcript)}")
    print(f"  Patient        : {patient_count} segments")
    print(f"  Clinician      : {clinician_count} segments")

    # ── Save output ──────────────────────────────────────────────────
    if output_path:
        Path(output_path).write_text(json.dumps(labeled_transcript, indent=2))
        print(f"\n  Saved to: {output_path}")

    return labeled_transcript


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline clinical interview speaker separation"
    )
    parser.add_argument(
        "audio",
        help="Path to interview audio file (.wav / .mp3 / .m4a)"
    )
    parser.add_argument(
        "--patient",
        choices=["SPEAKER_0", "SPEAKER_1"],
        default=None,
        help="Force which speaker is the patient (optional, auto-detected if not set)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON output (optional)"
    )
    args = parser.parse_args()

    result = run_offline_pipeline(
        audio_path=args.audio,
        patient_speaker=args.patient,
        output_path=args.output,
    )
