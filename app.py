"""
app.py
------
Flask API that wraps the clinical interview IR pipeline.

Endpoints:
    GET  /health              - check the API is running
    POST /upload              - upload an audio file from the frontend
    POST /pipeline            - run full offline pipeline on an audio file
    POST /index               - index a transcript into Supabase
    POST /retrieve            - retrieve segments for a query
    POST /analyze             - run all five MedGemma analysis modules
    POST /live/tokens         - generate real LiveKit JWT tokens
    POST /live/stop           - stop live session, index + analyze transcript
    GET  /live/transcript     - get current live transcript for a room
    GET  /                    - serve the frontend
"""

import os
import json
import time
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess
import requests

from speaker_separation.offline.pipeline import run_offline_pipeline
from ir.index import index_transcript, get_supabase_client, load_embedding_model
from ir.retrieve import retrieve
from ir.analyze import (
    summarize_interview,
    answer_symptom_question,
    analyze_interview_quality,
    recommend_referral,
    recommend_followup_questions,
    get_medgemma,
)
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER      = "audio"
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

print("Loading shared resources...")
supabase        = get_supabase_client()
embedding_model = load_embedding_model()
print("Ready!")


def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def live_transcript_path(room_name: str) -> Path:
    """
    The LiveKit agent (separate process) writes transcripts here.
    Flask reads from the same path — this is how the two processes share data.
    No shared memory needed: just a file both processes can see.
    """
    return Path("audio") / f"{room_name}_live_transcript.json"


N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/clinical-pipeline"

def notify_n8n(transcript_path, audio_path, segment_count, session_id):
    try:
        requests.post(N8N_WEBHOOK_URL, json={
            "transcript_path": str(transcript_path),
            "audio_path":      audio_path,
            "segments":        segment_count,
            "session_id":      session_id,
        }, timeout=5)
        print("n8n webhook triggered.")
    except Exception as e:
        print(f"Warning: Could not reach n8n: {e}")


# ── Serve frontend ──────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory("frontend/html", "index.html")

@app.route("/css/<path:filename>")
def css(filename):
    return send_from_directory("frontend/css", filename)

@app.route("/js/<path:filename>")
def js(filename):
    return send_from_directory("frontend/js", filename)


# ── Health ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Upload ──────────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Use wav, mp3, or m4a"}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(save_path)

    if not filename.lower().endswith(".wav"):
        wav_path = os.path.splitext(save_path)[0] + ".wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", save_path,
            "-ar", "16000", "-ac", "1", wav_path
        ], check=True, capture_output=True)
        os.remove(save_path)
        save_path = wav_path

    return jsonify({"audio_path": save_path})


# ── Offline Pipeline ────────────────────────────────────────────────────
@app.route("/pipeline", methods=["POST"])
def pipeline():
    data        = request.json
    audio_path  = data.get("audio_path")
    patient_spk = data.get("patient_speaker", None)

    if not audio_path:
        return jsonify({"error": "audio_path is required"}), 400
    if not Path(audio_path).exists():
        return jsonify({"error": f"Audio file not found: {audio_path}"}), 404

    session_id  = str(uuid.uuid4())
    output_path = Path(audio_path).stem + "_transcript.json"

    transcript = run_offline_pipeline(
        audio_path=audio_path,
        patient_speaker=patient_spk,
        output_path=output_path,
    )

    index_transcript(
        transcript,
        supabase=supabase,
        embedding_model=embedding_model,
        session_id=session_id,
    )

    notify_n8n(output_path, audio_path, len(transcript), session_id)

    return jsonify({
        "transcript":  transcript,
        "output_path": output_path,
        "session_id":  session_id,
        "segments":    len(transcript),
    })


# ── Index ───────────────────────────────────────────────────────────────
@app.route("/index", methods=["POST"])
def index_endpoint():
    data            = request.json
    transcript_path = data.get("transcript_path")
    session_id      = data.get("session_id")

    if not transcript_path:
        return jsonify({"error": "transcript_path is required"}), 400
    if not Path(transcript_path).exists():
        return jsonify({"error": f"Transcript not found: {transcript_path}"}), 404

    segments = json.loads(Path(transcript_path).read_text())
    index_transcript(
        segments,
        supabase=supabase,
        embedding_model=embedding_model,
        session_id=session_id,
    )
    return jsonify({"indexed": len(segments), "session_id": session_id})


# ── Retrieve ────────────────────────────────────────────────────────────
@app.route("/retrieve", methods=["POST"])
def retrieve_endpoint():
    data       = request.json
    query      = data.get("query")
    mode       = data.get("mode", "all")
    k          = data.get("k", 5)
    session_id = data.get("session_id")

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = retrieve(
        query, k=k, mode=mode,
        session_id=session_id,
        embedding_model=embedding_model,
    )
    return jsonify({"results": results, "query": query, "mode": mode, "k": k})


# ── Analyze ─────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    data            = request.json
    transcript_path = data.get("transcript_path")

    if not transcript_path:
        return jsonify({"error": "transcript_path is required"}), 400
    if not Path(transcript_path).exists():
        return jsonify({"error": f"Transcript not found: {transcript_path}"}), 404

    segments = json.loads(Path(transcript_path).read_text())
    get_medgemma()
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    summary  = summarize_interview(segments)
    qa       = answer_symptom_question(
                   "What symptoms does the patient have and how long have they had them?",
                   all_segments=segments, embedding_model=emb_model)
    quality  = analyze_interview_quality(segments)
    referral = recommend_referral(segments)
    followup = recommend_followup_questions(segments)

    return jsonify({
        "summary":    summary,
        "symptom_qa": qa,
        "quality":    quality,
        "referral":   referral,
        "followup":   followup,
    })


# ══════════════════════════════════════════════════════════════════════
# LIVE MODE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.route("/live/tokens", methods=["POST"])
def live_tokens():
    from livekit import api as livekit_api

    livekit_url    = os.getenv("LIVEKIT_URL")
    livekit_key    = os.getenv("LIVEKIT_API_KEY")
    livekit_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([livekit_url, livekit_key, livekit_secret]):
        return jsonify({"error": "LiveKit credentials missing from .env"}), 500

    data      = request.json or {}
    room_name = data.get("room_name") or f"interview-{int(time.time())}"

    try:
        patient_token = (
            livekit_api.AccessToken(livekit_key, livekit_secret)
            .with_identity("patient")
            .with_name("Patient")
            .with_grants(livekit_api.VideoGrants(
                room_join=True, room=room_name,
                can_publish=True, can_subscribe=True,
            ))
            .to_jwt()
        )

        clinician_token = (
            livekit_api.AccessToken(livekit_key, livekit_secret)
            .with_identity("clinician")
            .with_name("Clinician")
            .with_grants(livekit_api.VideoGrants(
                room_join=True, room=room_name,
                can_publish=True, can_subscribe=True,
            ))
            .to_jwt()
        )

        return jsonify({
            "room_name":       room_name,
            "livekit_url":     livekit_url,
            "patient_token":   patient_token,
            "clinician_token": clinician_token,
        })

    except Exception as e:
        return jsonify({"error": f"Token generation failed: {str(e)}"}), 500


@app.route("/live/transcript", methods=["GET"])
def live_transcript():
    """
    Reads directly from the file the LiveKit agent writes to.
    Works across processes — no shared memory needed.
    """
    room_name = request.args.get("room_name")
    if not room_name:
        return jsonify({"error": "room_name is required"}), 400

    fpath = live_transcript_path(room_name)

    if not fpath.exists():
        # Agent hasn't written anything yet — return empty, not an error
        return jsonify({"segments": [], "room_name": room_name})

    try:
        segments = json.loads(fpath.read_text())
        return jsonify({"segments": segments, "room_name": room_name})
    except Exception as e:
        return jsonify({"error": f"Could not read transcript file: {e}"}), 500


@app.route("/live/stop", methods=["POST"])
def live_stop():
    """
    Reads the transcript file written by the LiveKit agent,
    indexes it into Supabase, and runs full MedGemma analysis.
    """
    data      = request.json or {}
    room_name = data.get("room_name")

    if not room_name:
        return jsonify({"error": "room_name is required"}), 400

    fpath = live_transcript_path(room_name)

    if not fpath.exists():
        return jsonify({
            "error": (
                f"No transcript file found at {fpath}. "
                "Make sure the LiveKit transcriber agent is running: "
                "python speaker_separation/live/transcriber.py start"
            )
        }), 404

    try:
        segments = json.loads(fpath.read_text())
    except Exception as e:
        return jsonify({"error": f"Could not read transcript: {e}"}), 500

    if not segments:
        return jsonify({
            "error": "Transcript file exists but is empty. "
                     "No speech was captured — check the agent is connected to the room."
        }), 404

    session_id = str(uuid.uuid4())
    print(f"Indexing live transcript — {len(segments)} segments, session: {session_id}")
    index_transcript(
        segments,
        supabase=supabase,
        embedding_model=embedding_model,
        session_id=session_id,
    )

    print("Running MedGemma analysis on live transcript...")
    get_medgemma()
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    summary  = summarize_interview(segments)
    qa       = answer_symptom_question(
                   "What symptoms does the patient have and how long have they had them?",
                   all_segments=segments, embedding_model=emb_model)
    quality  = analyze_interview_quality(segments)
    referral = recommend_referral(segments)
    followup = recommend_followup_questions(segments)

    return jsonify({
        "transcript":      segments,
        "session_id":      session_id,
        "transcript_path": str(fpath),
        "summary":         summary,
        "symptom_qa":      qa,
        "quality":         quality,
        "referral":        referral,
        "followup":        followup,
    })


# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
