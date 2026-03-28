"""
server.py
---------
Flask server for LIVE MODE only — Clinical Interview IR System
Port: 5000

Handles:
    POST /api/generate-tokens   - generate real LiveKit JWT tokens
    GET  /api/transcript        - read live transcript written by transcriber agent
    GET  /api/analysis          - Groq Llama 3 clinical analysis (no MedGemma needed)
    POST /api/retrieve          - semantic search with Precision@K scores
    GET  /                      - serve the frontend

Run alongside app.py (port 5001) which handles offline mode.

Start live mode:
    # Terminal 1 — offline pipeline API
    python app.py

    # Terminal 2 — live mode API
    python server.py

    # Terminal 3 — LiveKit transcriber agent
    python speaker_separation/live/transcriber.py start
"""

import os
import re
import time
import json
import numpy as np
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from livekit import api
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = Flask(__name__)
CORS(app)

LIVEKIT_URL        = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")

# ── Lazy-load heavy models once ───────────────────────────────────────
_embedder    = None
_groq_client = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading sentence-transformer...")
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Embedder ready.")
    return _embedder


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── Transcript reading ─────────────────────────────────────────────────
def read_live_transcript(room_name: str) -> list[dict]:
    """
    Read the live transcript file written by the LiveKit transcriber agent.
    The agent writes to audio/{room_name}_live_transcript.json in real-time.

    Falls back to transcript_log.txt (plain text format) if the JSON file
    doesn't exist yet — useful during early testing.
    """
    # Primary: JSON file written by transcriber.py
    json_path = Path("audio") / f"{room_name}_live_transcript.json"
    if json_path.exists():
        try:
            segments = json.loads(json_path.read_text(encoding="utf-8"))
            return [
                {
                    "timestamp": time.strftime(
                        "%H:%M:%S",
                        time.localtime(seg.get("start", 0))
                    ),
                    "speaker": seg.get("role", seg.get("speaker", "UNKNOWN")),
                    "text":    seg.get("text", ""),
                }
                for seg in segments
                if seg.get("text", "").strip()
            ]
        except Exception as e:
            print(f"Warning: could not read {json_path}: {e}")

    # Fallback: plain text transcript_log.txt (from original livekit_transcriber.py)
    txt_path = Path("transcript_log.txt")
    if txt_path.exists():
        segments = []
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s*(PATIENT|CLINICIAN):\s*(.*)", line)
            if m:
                ts, speaker, text = m.groups()
                segments.append({"timestamp": ts, "speaker": speaker, "text": text})
            else:
                # Plain line without speaker label — treat as unknown
                m2 = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)", line)
                if m2:
                    ts, text = m2.groups()
                    segments.append({"timestamp": ts, "speaker": "UNKNOWN", "text": text})
        return segments

    return []


MEDICAL_TERMS = {
    "pain", "ache", "headache", "fever", "nausea", "vomit", "dizzy",
    "fatigue", "cough", "breath", "chest", "heart", "blood", "pressure",
    "medication", "drug", "ibuprofen", "aspirin", "antibiotic", "symptom",
    "swelling", "rash", "infection", "allergy", "anxiety", "depression",
    "sleep", "appetite", "weight", "vision", "hearing", "numbness",
}


def is_relevant(text: str, query_terms: set) -> bool:
    words = set(text.lower().split())
    return bool(words & MEDICAL_TERMS) or bool(words & query_terms)


# ── Serve frontend ─────────────────────────────────────────────────────
@app.route("/")
def serve_frontend():
    return send_from_directory("frontend/html", "index.html")

@app.route("/css/<path:path>")
def serve_css(path):
    return send_from_directory("frontend/css", path)

@app.route("/js/<path:path>")
def serve_js(path):
    return send_from_directory("frontend/js", path)


# ── /api/generate-tokens ──────────────────────────────────────────────
@app.route("/api/generate-tokens", methods=["POST"])
def generate_tokens():
    """Generate real LiveKit JWT tokens for patient and clinician."""
    try:
        data      = request.get_json() or {}
        room_name = data.get("room_name") or f"interview-{int(time.time())}"

        def make_token(identity, name):
            return (
                api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                .with_identity(identity)
                .with_name(name)
                .with_grants(api.VideoGrants(
                    room_join=True, room=room_name,
                    can_publish=True, can_subscribe=True,
                ))
                .to_jwt()
            )

        return jsonify({
            "success":         True,
            "room_name":       room_name,
            "patient_token":   make_token("patient",   "Patient"),
            "clinician_token": make_token("clinician", "Clinician"),
            "livekit_url":     LIVEKIT_URL,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/transcript ───────────────────────────────────────────────────
@app.route("/api/transcript", methods=["GET"])
def get_transcript():
    """Return current live transcript segments for a room."""
    try:
        room_name = request.args.get("room_name", "")
        segments  = read_live_transcript(room_name)
        parsed    = [
            {
                "timestamp": s["timestamp"],
                "speaker":   s["speaker"],
                "text":      s["text"],
                "raw":       f"[{s['timestamp']}] {s['speaker']}: {s['text']}",
            }
            for s in segments
        ]
        return jsonify({
            "success":     True,
            "transcripts": parsed[-100:],
            "count":       len(parsed),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/analysis ─────────────────────────────────────────────────────
@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    """
    Run four Groq Llama 3 analysis modules on the live transcript.
    Uses Groq instead of MedGemma so it works on any machine.

    Modules:
        1. Clinical summary
        2. Symptom extraction
        3. Referral recommendation
        4. Follow-up questions
    """
    try:
        room_name = request.args.get("room_name", "")
        segments  = read_live_transcript(room_name)

        if len(segments) < 3:
            return jsonify({
                "success":       True,
                "waiting":       True,
                "message":       "Not enough conversation yet. Analysis will appear once more speech is transcribed.",
                "segment_count": len(segments),
            })

        full_text      = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
        patient_text   = "\n".join(s["text"] for s in segments if s["speaker"] == "PATIENT")
        clinician_text = "\n".join(s["text"] for s in segments if s["speaker"] == "CLINICIAN")

        groq_client = get_groq()

        def ask(prompt: str) -> str:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role":    "system",
                        "content": (
                            "You are a clinical assistant helping analyse a patient-clinician interview. "
                            "Be concise, factual, and grounded only in the transcript provided. "
                            "Do not invent information. This is for educational purposes only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()

        summary = ask(
            f"Based on this clinical interview transcript, write a structured clinical summary. "
            f"Include: chief complaint, reported symptoms, duration, any medications mentioned, "
            f"and key clinician findings. Be concise.\n\nTranscript:\n{full_text}"
        )

        symptoms = ask(
            f"List all symptoms and health complaints reported by the PATIENT in this transcript. "
            f"Use bullet points (one per line starting with '-'). Include any duration or severity details mentioned.\n\n"
            f"Patient statements:\n{patient_text}"
        )

        referral = ask(
            f"Based on this clinical interview, should the patient be referred to a specialist? "
            f"State clearly: 'Referral recommended' or 'No referral needed', then give a brief 1-2 sentence "
            f"justification grounded in the transcript.\n\nFull transcript:\n{full_text}"
        )

        followup = ask(
            f"Based on what the patient reported and what the clinician asked, list 3-5 follow-up questions "
            f"the clinician should ask in a future appointment. Use bullet points starting with '-'.\n\n"
            f"Clinician questions so far:\n{clinician_text}\n\nPatient statements:\n{patient_text}"
        )

        return jsonify({
            "success":       True,
            "waiting":       False,
            "summary":       summary,
            "symptoms":      symptoms,
            "referral":      referral,
            "followup":      followup,
            "segment_count": len(segments),
        })

    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── /api/retrieve ─────────────────────────────────────────────────────
@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    """
    Semantic search over live transcript segments using all-MiniLM-L6-v2.

    Body:
        {
            "query":          "what symptoms does the patient have?",
            "k":              5,
            "speaker_filter": "all" | "patient" | "clinician",
            "room_name":      "interview-1234567890"
        }

    Returns top-K results with similarity scores + Precision@1/3/5.
    """
    try:
        body           = request.get_json() or {}
        query          = body.get("query", "").strip()
        k              = int(body.get("k", 5))
        speaker_filter = body.get("speaker_filter", "all").lower()
        room_name      = body.get("room_name", "")

        if not query:
            return jsonify({"success": False, "error": "query is required"}), 400

        all_segments = read_live_transcript(room_name)

        if not all_segments:
            return jsonify({
                "success":        True,
                "results":        [],
                "query":          query,
                "k":              k,
                "speaker_filter": speaker_filter,
                "precision_at_k": {},
                "message":        "No transcript segments available yet.",
            })

        # Apply speaker filter
        if speaker_filter == "patient":
            segments = [s for s in all_segments if s["speaker"] == "PATIENT"]
        elif speaker_filter == "clinician":
            segments = [s for s in all_segments if s["speaker"] == "CLINICIAN"]
        else:
            segments = all_segments

        if not segments:
            return jsonify({
                "success":        True,
                "results":        [],
                "query":          query,
                "k":              k,
                "speaker_filter": speaker_filter,
                "precision_at_k": {},
                "message":        f"No segments for filter: {speaker_filter}",
            })

        embedder = get_embedder()
        texts    = [s["text"] for s in segments]

        query_vec    = embedder.encode([query])
        segment_vecs = embedder.encode(texts)
        scores       = cosine_similarity(query_vec, segment_vecs)[0]

        ranked_indices = np.argsort(scores)[::-1]

        # Precision@K
        query_terms = set(query.lower().split())
        precision_at_k = {}
        for eval_k in [1, 3, 5]:
            top_k_idx = ranked_indices[:eval_k]
            relevant  = sum(
                1 for i in top_k_idx
                if is_relevant(segments[i]["text"], query_terms)
            )
            precision_at_k[f"P@{eval_k}"] = round(relevant / eval_k, 3)

        # Top-K results
        results = [
            {
                "rank":       int(rank + 1),
                "timestamp":  segments[i]["timestamp"],
                "speaker":    segments[i]["speaker"],
                "text":       segments[i]["text"],
                "similarity": round(float(scores[i]), 4),
            }
            for rank, i in enumerate(ranked_indices[:k])
        ]

        return jsonify({
            "success":        True,
            "results":        results,
            "query":          query,
            "k":              k,
            "speaker_filter": speaker_filter,
            "total_segments": len(segments),
            "precision_at_k": precision_at_k,
        })

    except Exception as e:
        print(f"Retrieve error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Live mode server starting on http://localhost:5000")
    print("Make sure app.py (offline mode) is running on port 5001")
    app.run(debug=True, port=5000)
