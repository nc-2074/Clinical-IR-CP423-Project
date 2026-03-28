"""
align.py
--------
Step 3 of offline pipeline: merge diarization segments (who spoke when)
with transcription segments (what was said when) to produce a unified,
speaker-labeled transcript.

The core challenge: Pyannote and Whisper produce independent timelines.
We need to match them up.

Strategy — "maximum overlap" matching:
    For each Whisper transcript segment, find the diarization speaker
    whose time window overlaps most with that segment's window.

Role Detection Strategy:
    We send a sample of the transcript to an LLM (Llama 3 via Groq)
    and ask it to identify the patient based on conversational content.
    The patient describes symptoms and personal history; the clinician
    asks structured medical questions. This distinction is reliable and
    does not depend on superficial signals like speaking duration.

    If patient_speaker is passed explicitly (via --patient flag in the
    CLI), LLM detection is skipped entirely.

Output example:
    [
        {
            "speaker":  "SPEAKER_0",
            "role":     "PATIENT",
            "start":    0.0,
            "end":      4.2,
            "text":     "I've been having headaches for about a week."
        },
        ...
    ]
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Number of segments to send to the LLM for role detection.
# We expand this automatically if only one speaker appears in the first N segments.
ROLE_DETECTION_SAMPLE_SIZE = 10


def overlap_duration(seg_start: float, seg_end: float,
                     dia_start: float, dia_end: float) -> float:
    """
    Calculate how many seconds two time intervals overlap.
    Returns 0.0 if they don't overlap at all.
    """
    overlap_start = max(seg_start, dia_start)
    overlap_end   = min(seg_end,   dia_end)
    return max(0.0, overlap_end - overlap_start)


def assign_speaker_to_segment(
    transcript_seg: dict,
    diarization_segs: list[dict]
) -> str:
    """
    Find which diarization speaker overlaps most with a transcript segment.

    Parameters
    ----------
    transcript_seg : dict
        {"start": float, "end": float, "text": str}
    diarization_segs : list[dict]
        [{"speaker": str, "start": float, "end": float}, ...]

    Returns
    -------
    str
        Speaker label, e.g. "SPEAKER_0". Returns "UNKNOWN" if no overlap found.
    """
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for dia in diarization_segs:
        ov = overlap_duration(
            transcript_seg["start"], transcript_seg["end"],
            dia["start"],            dia["end"]
        )
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = dia["speaker"]

    return best_speaker


def detect_roles_with_llm(
    labeled_segments: list[dict],
    client: Groq
) -> str:
    """
    Use an LLM to identify which speaker is the patient based purely
    on the content of what each speaker is saying.

    Keeps expanding the transcript sample in batches until segments
    from both speakers are present, then sends it to Llama 3 via Groq.

    Parameters
    ----------
    labeled_segments : list[dict]
        Segments already assigned a speaker label but not yet a role.
    client : Groq
        Pre-initialized Groq client.

    Returns
    -------
    str
        "SPEAKER_0" or "SPEAKER_1" — whichever the LLM identifies as the patient.

    Raises
    ------
    ValueError
        If the entire transcript only contains one speaker, or if the
        LLM returns a response that cannot be mapped to a known speaker.
    """
    # Keep adding batches of segments until both speakers are represented
    sample = []
    speakers_seen = set()
    batch_size = ROLE_DETECTION_SAMPLE_SIZE
    cursor = 0

    while cursor < len(labeled_segments):
        batch = labeled_segments[cursor : cursor + batch_size]
        sample.extend(batch)
        speakers_seen.update(seg["speaker"] for seg in batch)
        cursor += batch_size

        if len(speakers_seen) >= 2:
            print(f"  Both speakers found after {len(sample)} segments.")
            break

        print(f"  Only one speaker in first {len(sample)} segments, "
              f"expanding sample...")

    # If we exhausted the entire transcript and still only have one speaker,
    # something is wrong with the diarization output itself
    if len(speakers_seen) < 2:
        raise ValueError(
            "Only one speaker found across the entire transcript. "
            "The diarization may have failed. "
            "Please specify --patient manually."
        )

    # Format the sample as a readable transcript for the LLM
    sample_text = "\n".join(
        f"{seg['speaker']} ({seg['start']:.1f}s): {seg['text']}"
        for seg in sample
    )

    print("  Sending transcript sample to LLM for role detection...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": (
                    "The following is a sample from a clinical interview transcript.\n"
                    "Two speakers are labeled SPEAKER_00 and SPEAKER_01.\n"
                    "Based on the content of what each person is saying, identify which "
                    "speaker is the PATIENT.\n\n"
                    "The PATIENT describes their symptoms, feelings, medical history, "
                    "and personal experiences.\n"
                    "The CLINICIAN asks structured medical questions, gives instructions, "
                    "and guides the conversation.\n\n"
                    "Reply with ONLY 'SPEAKER_00' or 'SPEAKER_01' and nothing else.\n\n"
                    f"Transcript:\n{sample_text}"
                )
            }
        ],
        max_tokens=10,
        temperature=0.0,
    )

    llm_answer = response.choices[0].message.content.strip()

    all_speakers = list({seg["speaker"] for seg in labeled_segments})
    if llm_answer in all_speakers:
        print(f"  LLM detected PATIENT as {llm_answer} based on transcript content.")
        return llm_answer

    raise ValueError(
        f"LLM returned unexpected value: '{llm_answer}'. "
        f"Expected one of {all_speakers}. "
        f"Please specify --patient manually."
    )


def map_speakers_to_roles(
    labeled_segments: list[dict],
    patient_speaker: str
) -> list[dict]:
    """
    Apply PATIENT / CLINICIAN role labels to each segment.

    Parameters
    ----------
    labeled_segments : list[dict]
    patient_speaker : str
        The speaker label identified as the patient (e.g. "SPEAKER_0").

    Returns
    -------
    list[dict]
        Same list with "role" field populated on every segment.
    """
    role_map = {}
    for spk in {seg["speaker"] for seg in labeled_segments}:
        role_map[spk] = "PATIENT" if spk == patient_speaker else "CLINICIAN"

    for seg in labeled_segments:
        seg["role"] = role_map[seg["speaker"]]

    return labeled_segments


def align(
    transcript_segments: list[dict],
    diarization_segments: list[dict],
    patient_speaker: str = None,
    groq_client: Groq = None,
) -> list[dict]:
    """
    Full alignment: merge Whisper transcript + Pyannote diarization,
    then detect patient/clinician roles using an LLM.

    Parameters
    ----------
    transcript_segments : list[dict]
        From transcribe.py — [{"start", "end", "text"}, ...]
    diarization_segments : list[dict]
        From diarize.py  — [{"speaker", "start", "end"}, ...]
    patient_speaker : str, optional
        Force a specific speaker to be PATIENT (e.g. "SPEAKER_0").
        If provided, LLM detection is skipped entirely.
    groq_client : Groq, optional
        Pre-initialized Groq client for LLM role detection.
        If None and patient_speaker is also None, a new client is created
        from the GROQ_API_KEY environment variable.

    Returns
    -------
    list[dict]
        [{"speaker", "role", "start", "end", "text"}, ...]
    """
    print("Aligning transcript segments with diarization...")

    # Step 1: assign a speaker label to every transcript segment
    labeled = []
    for seg in transcript_segments:
        speaker = assign_speaker_to_segment(seg, diarization_segments)
        labeled.append({
            "speaker": speaker,
            "role":    None,
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
        })

    print(f"  Alignment complete: {len(labeled)} segments labeled.")

    # Step 2: determine which speaker is the patient
    if patient_speaker is not None:
        print(f"  Using manually specified PATIENT: {patient_speaker}")
    else:
        if groq_client is None:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        patient_speaker = detect_roles_with_llm(labeled, groq_client)

    # Step 3: apply role labels to all segments
    labeled = map_speakers_to_roles(labeled, patient_speaker)

    return labeled


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    transcript = [
        {"start": 0.0,  "end": 3.5,  "text": "I've had a headache for days."},
        {"start": 4.0,  "end": 6.0,  "text": "Can you describe the pain?"},
        {"start": 6.5,  "end": 10.0, "text": "It's a throbbing pain on the left side."},
        {"start": 10.5, "end": 13.0, "text": "Have you taken any medication for it?"},
        {"start": 13.5, "end": 17.0, "text": "Just ibuprofen but it doesn't really help."},
    ]
    diarization = [
        {"speaker": "SPEAKER_0", "start": 0.0,  "end": 3.8},
        {"speaker": "SPEAKER_1", "start": 3.9,  "end": 6.1},
        {"speaker": "SPEAKER_0", "start": 6.2,  "end": 10.2},
        {"speaker": "SPEAKER_1", "start": 10.3, "end": 13.1},
        {"speaker": "SPEAKER_0", "start": 13.4, "end": 17.2},
    ]
    import json
    result = align(transcript, diarization)
    print(json.dumps(result, indent=2))