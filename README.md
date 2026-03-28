# Clinical Interview IR System
Intelligent Clinical Interview Analysis, Summarization & Retrieval System with Speaker Separation

CP423 — Information Retrieval & Search Engines | Winter 2026

> ⚠️ This system is for educational purposes only. It must not be used for real medical diagnosis or treatment decisions.

---

A system that processes spoken clinical interviews and enables structured summarization, symptom-based question answering, and automated interview analysis.

Two modes are supported:

- **Offline mode** — processes a pre-recorded audio file using Pyannote diarization and Whisper transcription
- **Live mode** — processes real-time audio streams using LiveKit speaker separation

Full offline pipeline flow:

```
User uploads audio → Flask API → Pyannote diarization → Whisper transcription
→ Alignment + role detection → Supabase indexing → MedGemma analysis → Frontend display
```

---

## Folder Structure

```
CP423-clinical-ir/
├── audio/                          ← interview audio files
├── speaker_separation/
│   ├── offline/
│   │   ├── __init__.py
│   │   ├── diarize.py              ← Pyannote speaker diarization
│   │   ├── transcribe.py           ← Whisper transcription via Groq
│   │   ├── align.py                ← timeline alignment + LLM role detection
│   │   └── pipeline.py             ← orchestrates offline pipeline
│   └── live/
│       ├── __init__.py
│       └── transcriber.py          ← LiveKit real-time transcription agent
├── ir/
│   ├── __init__.py
│   ├── index.py                    ← Supabase indexing with embeddings
│   ├── retrieve.py                 ← speaker-aware semantic retrieval
│   ├── analyze.py                  ← MedGemma clinical analysis modules
│   └── evaluate.py                 ← Precision@K and Recall@K evaluation
├── frontend/
│   ├── html/
│   │   └── index.html              ← main frontend page
│   ├── css/
│   │   └── style.css               ← styling
│   └── js/
│       └── main.js                 ← frontend logic and API calls
├── scripts/
│   ├── generate_tokens.py          ← CLI tool to generate LiveKit JWT tokens
│   ├── test_groq.py                ← verify Groq API key
│   ├── test_livekit.py             ← verify LiveKit connection
│   ├── test_pyannote.py            ← verify Pyannote model loads
│   ├── test_setup.py               ← full environment check
│   └── test_supabase.py            ← verify Supabase connection
├── app.py                          ← Flask API
├── .env                            ← API keys (never commit this)
├── .env.example                    ← template for API keys
└── README.md
```

---

## Requirements

- Python 3.11
- Apple Silicon Mac (M1/M2/M3) for MedGemma local inference via MLX
- Docker (for n8n)
- ffmpeg (for audio conversion)

Install ffmpeg:

```bash
brew install ffmpeg
```

Install Docker from [docker.com](https://docker.com) if you don't have it.

---

## API Keys Required

You need seven credentials before running the system. All services have a free tier.

### Hugging Face

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens → New token** (read access is enough)
3. Accept model terms at each of these pages:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) — accept the Health AI Developer Foundations terms

### Groq

1. Create a free account at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** and generate a new key
3. Used for Whisper transcription and Llama 3 role detection

### Supabase

1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to **Settings → API** and copy the **Project URL** and **anon public key**

### LiveKit (Live Mode only)

1. Create a free account at [livekit.io](https://livekit.io)
2. Create a new project and copy the **WebSocket URL**, **API Key**, and **API Secret**

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/nc-2074/CP423-Project-clinical-ir
cd CP423-Project-clinical-ir
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies with working versions

```bash
# Core ML packages
pip install torch==2.4.0 torchaudio==2.4.0
pip install numpy==1.24.3

# Hugging Face ecosystem
pip install huggingface_hub==0.26.0
pip install tokenizers==0.20.0
pip install transformers==4.48.0
pip install sentence-transformers==3.0.0

# Pyannote for diarization
pip install pyannote.audio==3.3.2

# MLX for Apple Silicon
pip install mlx-lm==0.23.0

# Groq API
pip install groq==1.0.0

# LiveKit for real-time transcription
pip install livekit livekit-api livekit-agents livekit-plugins-silero livekit-plugins-groq

# Other utilities
pip install matplotlib python-dotenv pydub soundfile
pip install supabase==2.28.0
pip install flask==3.0.0 flask-cors==4.0.0 requests==2.31.0
pip install pandas scikit-learn
```
### 4. Configure your `.env` file

```bash
cp env.example .env
nano .env
```
Open `.env` and fill in your keys:

```env
# Hugging Face — needed for Pyannote diarization model and MedGemma
HF_TOKEN=your_hugging_face_token_here

# Groq — free tier Whisper transcription + Llama 3 role detection
GROQ_API_KEY=your_groq_api_key_here

# Supabase — transcript storage and vector search
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_anon_public_key_here

# LiveKit — required only for live interview mode
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here
```

### 5. Set up the Supabase database

Go to your Supabase project → **SQL Editor** and run the following queries one at a time.

**Create the table:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE transcript_segments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT,
    speaker TEXT NOT NULL,
    role TEXT NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE transcript_segments
ADD CONSTRAINT unique_segment
UNIQUE (session_id, start_time, end_time, text);
```

**Create the retrieval function (all segments):**

```sql
CREATE OR REPLACE FUNCTION match_segments(
    query_embedding VECTOR(384),
    match_count INT,
    p_session_id TEXT
)
RETURNS TABLE (
    id UUID, speaker TEXT, role TEXT,
    start_time FLOAT, end_time FLOAT,
    text TEXT, similarity FLOAT
)
LANGUAGE SQL STABLE AS $$
    SELECT id, speaker, role, start_time, end_time, text,
           1 - (embedding <=> query_embedding) AS similarity
    FROM transcript_segments
    WHERE session_id = p_session_id
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

**Create the patient segments retrieval function:**

```sql
CREATE OR REPLACE FUNCTION match_patient_segments(
    query_embedding VECTOR(384),
    match_count INT,
    p_session_id TEXT
)
RETURNS TABLE (
    id UUID, speaker TEXT, role TEXT,
    start_time FLOAT, end_time FLOAT,
    text TEXT, similarity FLOAT
)
LANGUAGE SQL STABLE AS $$
    SELECT id, speaker, role, start_time, end_time, text,
           1 - (embedding <=> query_embedding) AS similarity
    FROM transcript_segments
    WHERE role = 'PATIENT'
    AND session_id = p_session_id
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

**Create the clinician segments retrieval function:**

```sql
CREATE OR REPLACE FUNCTION match_clinician_segments(
    query_embedding VECTOR(384),
    match_count INT,
    p_session_id TEXT
)
RETURNS TABLE (
    id UUID, speaker TEXT, role TEXT,
    start_time FLOAT, end_time FLOAT,
    text TEXT, similarity FLOAT
)
LANGUAGE SQL STABLE AS $$
    SELECT id, speaker, role, start_time, end_time, text,
           1 - (embedding <=> query_embedding) AS similarity
    FROM transcript_segments
    WHERE role = 'CLINICIAN'
    AND session_id = p_session_id
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

### 6. Set up n8n (automation workflow)

Start n8n using Docker:

```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

Or if you already have an n8n container:

```bash
docker start n8n
```

Go to [http://localhost:5678](http://localhost:5678), log in, and import the Clinical IR workflow JSON. Once imported, activate the workflow by toggling it on in the top right corner — the toggle should turn green.

---

## Verifying Your Setup

Run the full environment check before starting the server:

```bash
python scripts/test_setup.py
```

You can also test individual components:

```bash
# Test Groq API key (Whisper + Llama 3)
python scripts/test_groq.py

# Test Supabase connection and table setup
python scripts/test_supabase.py

# Test Pyannote model loads (downloads ~300MB on first run)
python scripts/test_pyannote.py

# Test LiveKit connection and token generation
python scripts/test_livekit.py
```

---

## Running the System

### Start the Flask API

```bash
source venv/bin/activate
python app.py
```

The API starts on port 5001. You should see:

```
Loading shared resources...
Embedding model loaded.
Ready!
 * Running on http://0.0.0.0:5001
```

### Open the frontend

Go to [http://localhost:5001](http://localhost:5001) in your browser.

---

## Offline Mode (Pre-recorded Interview)

1. Click **Offline Mode** in the frontend
2. Upload a `.wav`, `.mp3`, or `.m4a` audio file
3. Click **Process Interview**
4. Wait for the pipeline to complete — this typically takes 2–3 minutes
5. View results across three tabs:
   - **Transcript** — speaker-labeled segments with patient/clinician filter
   - **Analysis** — five MedGemma analysis modules (summary, symptom QA, interview quality, referral recommendation, follow-up questions)
   - **Retrieval** — semantic search over the transcript

### Run the offline pipeline manually

```bash
python -m speaker_separation.offline.pipeline audio/interview.wav
```

With optional flags:

```bash
# Force which speaker is the patient (skips LLM role detection)
python -m speaker_separation.offline.pipeline audio/interview.wav --patient SPEAKER_0

# Save output to a custom path
python -m speaker_separation.offline.pipeline audio/interview.wav --output results/transcript.json
```

---

## Live Mode (Real-time Interview)

Live mode requires the LiveKit transcriber agent running separately alongside the Flask API.

### Step 1: Start the Flask API

```bash
source venv/bin/activate
python app.py
```

### Step 2: Start the LiveKit transcriber agent

In a second terminal:

```bash
cd /Users/name/CP423-Project-clinical-ir
source venv/bin/activate
python speaker_separation/live/transcriber.py start
```

You should see:

```
Preloading models...
Models ready.
```

### Step 3: Generate tokens from the frontend

1. Click **Live Interview** in the frontend
2. Click **Generate Tokens**
3. Copy the patient token and clinician token

### Step 4: Join the LiveKit room

1. Go to [meet.livekit.io](https://meet.livekit.io)
2. Paste your LiveKit URL (shown in the frontend)
3. Paste the **patient token** for the patient's browser tab
4. Paste the **clinician token** for the clinician's browser tab
5. Both participants join and begin speaking — the transcript updates live in the frontend

### Step 5: End the interview

Click **End Interview & Analyze** in the frontend. This stops transcription, indexes the transcript into Supabase, and runs full MedGemma analysis.

### Generate tokens from the command line (for testing)

```bash
python scripts/generate_tokens.py

# With a custom room name
python scripts/generate_tokens.py --room my-custom-room
```

---

## Individual Module Usage

### Run semantic retrieval manually

```bash
# Search all segments
python -m ir.retrieve "what symptoms does the patient have?"

# Search patient segments only
python -m ir.retrieve "what symptoms does the patient have?" patient

# Search clinician segments only, top 3 results
python -m ir.retrieve "what medications were prescribed?" clinician 3
```

### Run MedGemma analysis manually

```bash
python -m ir.analyze path/to/transcript.json
```

### Run retrieval evaluation (Precision@K and Recall@K)

```bash
python -m ir.evaluate
```

Results are printed to the terminal and saved to `evaluation_results.json`.

### Index a transcript manually

```bash
python -m ir.index path/to/transcript.json
```

---

## API Endpoints

All endpoints are served by the Flask API at `http://localhost:5001`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check the API is running |
| POST | `/upload` | Upload an audio file from the frontend |
| POST | `/pipeline` | Run the full offline pipeline on an audio file |
| POST | `/index` | Index a transcript into Supabase |
| POST | `/retrieve` | Retrieve segments for a semantic query |
| POST | `/analyze` | Run all five MedGemma analysis modules |
| POST | `/live/tokens` | Generate LiveKit JWT tokens for patient and clinician |
| GET | `/live/transcript` | Get the current live transcript for a room |
| POST | `/live/stop` | End a live session, index transcript, and run analysis |
| GET | `/` | Serve the frontend |

### Example: retrieve endpoint

```bash
curl -X POST http://localhost:5001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "what symptoms does the patient have?", "mode": "patient", "k": 5}'
```

### Example: analyze endpoint

```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"transcript_path": "audio/interview_transcript.json"}'
```

---

## n8n Workflow

The included n8n workflow listens on a webhook at `/webhook-test/clinical-pipeline` and automatically triggers MedGemma analysis whenever the Flask pipeline completes.

The webhook receives:

```json
{
  "transcript_path": "audio/interview_transcript.json",
  "audio_path": "audio/interview.wav",
  "segments": 42,
  "session_id": "uuid-here"
}
```

To import the workflow: go to [http://localhost:5678](http://localhost:5678) → **Workflows → Import from file** → select the workflow JSON. Activate the workflow by enabling the toggle in the top right.

---

## Notes

- **Never commit your `.env` file** — it contains private API keys
- The Pyannote model downloads ~300MB on first run and is cached automatically
- The MedGemma model downloads ~2.5GB on first run and is cached automatically
- **MedGemma requires Apple Silicon (M1/M2/M3)** — on other hardware, replace it with Llama 3.3 via Groq in `ir/analyze.py` by swapping out the `medgemma_generate` function
- The Groq free tier has a daily transcription limit — sufficient for development and demos
- Audio files should be `.wav` at 16000Hz mono for best results — MP3 and M4A files are converted automatically by the upload endpoint
- Each uploaded interview gets a unique `session_id` so retrieval only searches within the current interview
- The embedding model (`all-MiniLM-L6-v2`, ~90MB) downloads on first run and is cached

---

## Ethics

This system is for educational purposes only. All outputs must be treated as preliminary and require independent verification by a qualified clinician. This system must not be used to provide real medical diagnoses or treatment recommendations.
