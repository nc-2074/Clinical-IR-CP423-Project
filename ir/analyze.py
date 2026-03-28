"""
analyze.py
----------
Step 6 of offline pipeline: use MedGemma 4B via MLX to analyze the
retrieved transcript segments and produce structured clinical outputs.

Five analysis modules:
1. Clinical Interview Summarization
   - Summarizes the full interview into a structured clinical note

2. Symptom Based Question Answering
   - Answers specific questions about the patient's symptoms
   - Grounded in retrieved patient segments

3. Automated Interview Analyzer
   - Analyzes the quality and completeness of the clinical interview
   - Grounded in retrieved clinician segments
   - Prompt constrained to 400 words to prevent output truncation

4. Referral Recommendation
   - Recommends a specialist referral only if clinically warranted
   - If no referral is needed, says so clearly

5. Follow-Up Questions
   - Suggests follow-up questions the clinician should have asked
   - Only if there are clear clinical gaps in the interview

Explainability:
   Modules 1 and 2 run a verification step after generation that checks
   each sentence in the output against the transcript segments using
   cosine similarity. Sentences that cannot be traced back to a segment
   are flagged as unverified. This provides a basic citation mechanism
   and satisfies the project's explainability requirements.

LLM Backend:
   Uses MedGemma 4B via MLX, optimized for Apple Silicon.
   MedGemma is a medical AI model from Google trained on clinical
   text, making it more appropriate for clinical interview analysis
   than a general purpose model like Llama 3.
"""

import os
import re
from dotenv import load_dotenv
from mlx_lm import load, generate
from .retrieve import retrieve
from sentence_transformers import SentenceTransformer, util

load_dotenv()

MEDGEMMA_MODEL  = "mlx-community/medgemma-4b-it-4bit"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Global model cache — load once, reuse across all five modules
_model     = None
_tokenizer = None


def get_medgemma():
    """
    Load MedGemma 4B via MLX.
    Model is cached globally so it only loads once per session.
    Downloads ~2.5GB on first run, cached for future runs.
    """
    global _model, _tokenizer
    if _model is None:
        print("Loading MedGemma 4B via MLX (downloads on first run ~2.5GB)...")
        _model, _tokenizer = load(
            MEDGEMMA_MODEL,
            tokenizer_config={"trust_remote_code": True},
        )
        print("MedGemma loaded.")
    return _model, _tokenizer


def medgemma_generate(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512
) -> str:
    """
    Generate a response from MedGemma given a system and user prompt.

    Parameters
    ----------
    system_prompt : str
        Instructions for how MedGemma should behave.
    user_prompt : str
        The actual question or task.
    max_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        MedGemma's response.
    """
    model, tokenizer = get_medgemma()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )

    return response


def format_segments_for_prompt(segments: list[dict]) -> str:
    """
    Format retrieved segments into a readable block for the prompt.

    Parameters
    ----------
    segments : list[dict]
        Retrieved segments from retrieve.py or the transcript directly.

    Returns
    -------
    str
        Formatted transcript block.
    """
    lines = []
    for seg in segments:
        start = seg.get("start", seg.get("start_time", 0))
        end   = seg.get("end",   seg.get("end_time",   0))
        lines.append(
            f"[{seg['role']}] ({start:.1f}s - {end:.1f}s): {seg['text']}"
        )
    return "\n".join(lines)


# ── Verification / Citation ───────────────────────────────────────────

def verify_output_against_transcript(
    output: str,
    segments: list[dict],
    embedding_model: SentenceTransformer = None,
) -> dict:
    """
    Verify that claims in the LLM output are traceable to transcript segments.
    Flags any sentence that does not match any segment closely enough.

    This is a lightweight citation check — it does not guarantee accuracy
    but provides a basic explainability layer by linking output claims
    to source segments using cosine similarity.

    Parameters
    ----------
    output : str
        The LLM generated output to verify.
    segments : list[dict]
        All transcript segments to check against.
    embedding_model : SentenceTransformer, optional
        Pre-loaded embedding model.

    Returns
    -------
    dict
        {
            "verified":   list of sentences traceable to a segment,
            "unverified": list of sentences that could not be traced,
            "citations":  dict mapping verified sentences to source segments
        }
    """
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Split output into sentences, skip very short ones
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+|\n{2,}', output)
        if len(s.strip()) > 20
    ]

    # Get all segment texts
    segment_texts = [seg["text"] for seg in segments]

    # Encode everything
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    segment_embeddings  = embedding_model.encode(segment_texts, convert_to_tensor=True)

    verified   = []
    unverified = []
    citations  = {}

    # Sentences with similarity above this threshold are considered traceable
    SIMILARITY_THRESHOLD = 0.4

    for i, sentence in enumerate(sentences):
        similarities = util.cos_sim(sentence_embeddings[i], segment_embeddings)[0]
        best_score   = float(similarities.max())
        best_idx     = int(similarities.argmax())

        if best_score >= SIMILARITY_THRESHOLD:
            verified.append(sentence)
            citations[sentence] = {
                "source":     segment_texts[best_idx],
                "role":       segments[best_idx]["role"],
                "start":      segments[best_idx].get(
                                  "start", segments[best_idx].get("start_time", 0)
                              ),
                "similarity": round(best_score, 3)
            }
        else:
            unverified.append(sentence)

    return {
        "verified":   verified,
        "unverified": unverified,
        "citations":  citations
    }


def print_verification_results(verification: dict) -> None:
    """Pretty print verification results to the terminal."""
    print(f"\n  Verified sentences:   {len(verification['verified'])}")
    print(f"  Unverified sentences: {len(verification['unverified'])}")

    if verification["unverified"]:
        print("\n  ⚠️  Unverified claims (could not be traced to transcript):")
        for s in verification["unverified"]:
            print(f"    - {s}")

    if verification["citations"]:
        print("\n  Citations:")
        for sentence, citation in verification["citations"].items():
            print(f"\n    Claim:  {sentence}")
            print(f"    Source: [{citation['role']}] {citation['source']}")
            print(f"    Score:  {citation['similarity']}")


# ── Module 1: Summarization ──────────────────────────────────────────

def summarize_interview(segments: list[dict]) -> str:
    """
    Summarize the full clinical interview into a structured clinical note.

    Parameters
    ----------
    segments : list[dict]
        All transcript segments from the interview.

    Returns
    -------
    str
        Structured clinical summary.
    """
    transcript_block = format_segments_for_prompt(segments)

    print("Generating clinical interview summary with MedGemma...")

    return medgemma_generate(
        system_prompt=(
            "You are a clinical documentation assistant. "
            "Your job is to summarize clinical interview transcripts "
            "into structured clinical notes. "
            "Always clearly separate patient reported symptoms from "
            "clinician observations and recommendations. "
            "Never make up information not present in the transcript. "
            "This is for educational purposes only and is not real medical advice."
        ),
        user_prompt=(
            "Please summarize the following clinical interview transcript "
            "into a structured clinical note with these sections:\n"
            "- Chief Complaint\n"
            "- Symptoms Reported\n"
            "- Duration\n"
            "- Clinician Findings\n"
            "- Treatment Plan\n\n"
            f"Transcript:\n{transcript_block}"
        ),
        max_tokens=512,
    )


# ── Module 2: Symptom QA ─────────────────────────────────────────────

def answer_symptom_question(
    question: str,
    all_segments: list[dict] = None,
    embedding_model: SentenceTransformer = None,
) -> str:
    """
    Answer a specific question about the patient's symptoms.

    For short transcripts (under 20 segments), uses all patient segments
    directly to avoid retrieval missing short context-free answers like
    "For two days." which score poorly in vector search.

    For longer transcripts, falls back to retrieval to avoid sending
    too much context to the LLM.

    Parameters
    ----------
    question : str
        A question about the patient's symptoms.
    all_segments : list[dict], optional
        The full transcript. If provided and short enough, all patient
        segments are used directly instead of retrieval.
    embedding_model : SentenceTransformer, optional
        Pre-loaded embedding model. Only used if falling back to retrieval.

    Returns
    -------
    str
        MedGemma answer grounded in patient segments.
    """
    if all_segments is not None and len(all_segments) <= 20:
        patient_segments = [s for s in all_segments if s["role"] == "PATIENT"]
        context = format_segments_for_prompt(patient_segments)
        print(f"Using all {len(patient_segments)} patient segments directly "
              f"(short transcript — retrieval skipped)")
    else:
        if embedding_model is None:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        retrieved = retrieve(
            question,
            k=5,
            mode="patient",
            embedding_model=embedding_model
        )
        context = format_segments_for_prompt(retrieved)
        print("Using retrieval for symptom QA (long transcript)")

    print(f"Answering symptom question with MedGemma: '{question}'")

    return medgemma_generate(
        system_prompt=(
            "You are a clinical information assistant. "
            "Answer questions about a patient's symptoms based only "
            "on the provided transcript segments. "
            "If the answer is not in the transcript, say so clearly. "
            "Never make up information. "
            "This is for educational purposes only and is not real medical advice."
        ),
        user_prompt=(
            f"Based on the following patient transcript segments, "
            f"please answer this question: {question}\n\n"
            f"Patient segments:\n{context}"
        ),
        max_tokens=256,
    )


# ── Module 3: Interview Analyzer ─────────────────────────────────────

def analyze_interview_quality(segments: list[dict]) -> str:
    """
    Analyze the quality and completeness of the clinical interview
    based on the clinician's questions and approach.

    The prompt explicitly constrains the response to 400 words to
    prevent output truncation at the max_tokens limit.

    Parameters
    ----------
    segments : list[dict]
        All transcript segments from the interview.

    Returns
    -------
    str
        Analysis of the interview quality.
    """
    clinician_segments = [s for s in segments if s["role"] == "CLINICIAN"]
    context = format_segments_for_prompt(clinician_segments)

    print("Analyzing interview quality with MedGemma...")

    return medgemma_generate(
        system_prompt=(
            "You are a clinical education supervisor. "
            "Your job is to analyze clinical interviews and provide "
            "constructive feedback on the clinician's questioning technique, "
            "thoroughness, and communication style. "
            "Keep your response concise and complete — do not exceed 400 words. "
            "Make sure to finish all sections before reaching the word limit. "
            "This is for educational purposes only."
        ),
        user_prompt=(
            "Please analyze the following clinician segments from a clinical "
            "interview and provide feedback on:\n"
            "- Thoroughness of questioning\n"
            "- Communication style\n"
            "- Areas for improvement\n"
            "- Overall assessment\n\n"
            f"Clinician segments:\n{context}"
        ),
        max_tokens=768,
    )


# ── Module 4: Referral Recommendation ────────────────────────────────

def recommend_referral(segments: list[dict]) -> str:
    """
    Analyze the transcript and recommend a referral only if the
    patient's symptoms warrant one. If no referral is needed, says so.

    Parameters
    ----------
    segments : list[dict]
        All transcript segments from the interview.

    Returns
    -------
    str
        Referral recommendation or confirmation that none is needed.
    """
    patient_segments = [s for s in segments if s["role"] == "PATIENT"]
    context = format_segments_for_prompt(patient_segments)

    print("Evaluating referral necessity with MedGemma...")

    return medgemma_generate(
        system_prompt=(
            "You are a clinical triage assistant reviewing a patient interview. "
            "Your job is to determine whether the patient's symptoms require "
            "a referral to a specialist. "
            "Only recommend a referral if there is a clear clinical reason — "
            "do not suggest referrals for minor or routine complaints that a "
            "general practitioner can handle. "
            "If no referral is needed, say so clearly and explain why. "
            "This is for educational purposes only and is not real medical advice."
        ),
        user_prompt=(
            "Based on the following patient transcript segments, determine whether "
            "a specialist referral is necessary.\n\n"
            "If a referral IS needed: specify the type of specialist and the "
            "clinical reason based only on what was said in the transcript.\n\n"
            "If a referral is NOT needed: explain why the symptoms can be managed "
            "at the current level of care.\n\n"
            f"Patient segments:\n{context}"
        ),
        max_tokens=256,
    )


# ── Module 5: Follow-Up Questions ────────────────────────────────────

def recommend_followup_questions(segments: list[dict]) -> str:
    """
    Suggest follow-up questions the clinician should have asked,
    but only if there are clear clinical gaps in the interview.
    If the interview was thorough enough, says so.

    Parameters
    ----------
    segments : list[dict]
        All transcript segments from the interview.

    Returns
    -------
    str
        Suggested follow-up questions or confirmation that none are needed.
    """
    transcript_block = format_segments_for_prompt(segments)

    print("Evaluating follow-up questions with MedGemma...")

    return medgemma_generate(
        system_prompt=(
            "You are a clinical education supervisor reviewing a clinical interview. "
            "Your job is to identify any important follow-up questions that were "
            "not asked but should have been given the patient's reported symptoms. "
            "Only suggest follow-up questions if there are clear clinical gaps — "
            "do not invent questions for the sake of it. "
            "If the interview was sufficiently thorough for the presenting complaint, "
            "say so clearly. "
            "This is for educational purposes only and is not real medical advice."
        ),
        user_prompt=(
            "Review the following clinical interview transcript and identify any "
            "important follow-up questions that were missed.\n\n"
            "Only suggest questions if they are clinically relevant to the "
            "patient's presenting complaint and were not already addressed.\n\n"
            "If no follow-up questions are needed, explain why the interview "
            "was sufficiently thorough.\n\n"
            f"Transcript:\n{transcript_block}"
        ),
        max_tokens=256,
    )


# ── Run all five modules ──────────────────────────────────────────────

def run_analysis(segments: list[dict]) -> None:
    """
    Run all five analysis modules on the transcript.
    MedGemma is loaded once and reused across all modules.

    Parameters
    ----------
    segments : list[dict]
        All transcript segments from the interview.
    """
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load MedGemma once upfront so it is not reloaded between modules
    get_medgemma()

    print("\n" + "=" * 60)
    print("MODULE 1: CLINICAL INTERVIEW SUMMARY")
    print("=" * 60)
    summary = summarize_interview(segments)
    print(summary)
    print("\n--- Verification ---")
    verification = verify_output_against_transcript(
        summary, segments, embedding_model
    )
    print_verification_results(verification)

    print("\n" + "=" * 60)
    print("MODULE 2: SYMPTOM QUESTION ANSWERING")
    print("=" * 60)
    answer = answer_symptom_question(
        "What symptoms does the patient have and how long have they had them?",
        all_segments=segments,
        embedding_model=embedding_model,
    )
    print(answer)
    print("\n--- Verification ---")
    verification = verify_output_against_transcript(
        answer, segments, embedding_model
    )
    print_verification_results(verification)

    print("\n" + "=" * 60)
    print("MODULE 3: INTERVIEW QUALITY ANALYSIS")
    print("=" * 60)
    analysis = analyze_interview_quality(segments)
    print(analysis)

    print("\n" + "=" * 60)
    print("MODULE 4: REFERRAL RECOMMENDATION")
    print("=" * 60)
    referral = recommend_referral(segments)
    print(referral)

    print("\n" + "=" * 60)
    print("MODULE 5: FOLLOW-UP QUESTIONS")
    print("=" * 60)
    followup = recommend_followup_questions(segments)
    print(followup)


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python analyze.py path/to/transcript.json")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])
    segments = json.loads(transcript_path.read_text())
    run_analysis(segments)