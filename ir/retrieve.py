"""
retrieve.py
-----------
Step 5 of offline pipeline: retrieve relevant transcript segments
from Supabase based on a query.

Supports three retrieval modes:
- "all"       -> search across all segments
- "patient"   -> search only patient segments
- "clinician" -> search only clinician segments

This enables queries like:
- "What symptoms did the patient describe?" (patient mode)
- "What did the clinician ask about medications?" (clinician mode)
- "When was food poisoning mentioned?" (all mode)
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_supabase_client() -> Client:
    """Create a Supabase client using keys from .env"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL or SUPABASE_KEY is missing from your .env file."
        )

    return create_client(url, key)


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer model for generating query embeddings."""
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")
    return model


def retrieve(
    query: str,
    k: int = 5,
    mode: str = "all",
    session_id: str = None,
    supabase: Client = None,
    embedding_model: SentenceTransformer = None,
) -> list[dict]:
    if supabase is None:
        supabase = get_supabase_client()

    if embedding_model is None:
        embedding_model = load_embedding_model()

    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()

    if mode == "patient":
        print(f"Retrieving top {k} PATIENT segments for: '{query}'")
        response = supabase.rpc("match_patient_segments", {
            "query_embedding": query_embedding,
            "match_count":     k,
            "p_session_id":    session_id,
        }).execute()

    elif mode == "clinician":
        print(f"Retrieving top {k} CLINICIAN segments for: '{query}'")
        response = supabase.rpc("match_clinician_segments", {
            "query_embedding": query_embedding,
            "match_count":     k,
            "p_session_id":    session_id,
        }).execute()

    else:
        print(f"Retrieving top {k} segments for: '{query}'")
        response = supabase.rpc("match_segments", {
            "query_embedding": query_embedding,
            "match_count":     k,
            "p_session_id":    session_id,
        }).execute()

    results = response.data
    print(f"  Found {len(results)} results.")
    return results


def print_results(results: list[dict]) -> None:
    """Pretty print retrieval results to the terminal."""
    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)
    for i, seg in enumerate(results, 1):
        print(f"\n[{i}] [{seg['role']:10s}] "
              f"{seg['start_time']:.1f}s – {seg['end_time']:.1f}s")
        print(f"     {seg['text']}")
        print(f"     Similarity: {seg['similarity']:.4f}")
    print("=" * 60)


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "what symptoms does the patient have?"
    mode  = sys.argv[2] if len(sys.argv) > 2 else "all"
    k     = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    results = retrieve(query, k=k, mode=mode)
    print_results(results)