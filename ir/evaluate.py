"""
evaluate.py
-----------
Evaluation of retrieval quality using Precision@K and Recall@K.

Precision@K: of the top K results returned, how many are actually relevant?
Recall@K:    of all relevant segments in the database, how many did we
             find in the top K results?

Evaluation is run three ways:
- Overall:          all segments
- Patient only:     only patient segments
- Clinician only:   only clinician segments
"""

import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from .retrieve import retrieve

load_dotenv()

# ── Ground truth ─────────────────────────────────────────────────────
# These are our manually defined relevant segments for each query.
# In a real system these would be annotated by a human expert.
# Format: { "query": [list of relevant segment texts] }

GROUND_TRUTH = {
    "what symptoms does the patient have?": [
        "I have a headache.",
        "No, just headache.",
        "For two days.",
        "Good morning, doctor. I have a headache."
    ],
    "what did the clinician diagnose?": [
        "Okay, take this medicine after meals.",
        "You'll be fine soon. Take care.",
        "Do you have a fever?",
        "How long have you had it?"
    ],
    "how long has the patient had symptoms?": [
        "For two days.",
        "I have a headache.",
        "Good morning, doctor. I have a headache."
    ]
}


def precision_at_k(retrieved: list[dict], relevant_texts: list[str], k: int) -> float:
    """
    Calculate Precision@K.

    Precision@K = number of relevant results in top K / K

    Parameters
    ----------
    retrieved : list[dict]
        The top K results returned by the retrieval system.
    relevant_texts : list[str]
        The ground truth relevant segment texts for this query.
    k : int
        The number of results to consider.

    Returns
    -------
    float
        Precision@K score between 0.0 and 1.0.
    """
    top_k = retrieved[:k]
    relevant_count = sum(
        1 for seg in top_k
        if any(rel.lower() in seg["text"].lower() or
               seg["text"].lower() in rel.lower()
               for rel in relevant_texts)
    )
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(retrieved: list[dict], relevant_texts: list[str], k: int) -> float:
    """
    Calculate Recall@K.

    Recall@K = number of relevant results in top K / total relevant segments

    Parameters
    ----------
    retrieved : list[dict]
        The top K results returned by the retrieval system.
    relevant_texts : list[str]
        The ground truth relevant segment texts for this query.
    k : int
        The number of results to consider.

    Returns
    -------
    float
        Recall@K score between 0.0 and 1.0.
    """
    top_k = retrieved[:k]
    relevant_found = sum(
        1 for seg in top_k
        if any(rel.lower() in seg["text"].lower() or
               seg["text"].lower() in rel.lower()
               for rel in relevant_texts)
    )
    total_relevant = len(relevant_texts)
    return relevant_found / total_relevant if total_relevant > 0 else 0.0


def evaluate(k_values: list[int] = [1, 3, 5]) -> dict:
    """
    Run full evaluation across all queries and K values.

    Parameters
    ----------
    k_values : list[int]
        The K values to evaluate at. Default is [1, 3, 5].

    Returns
    -------
    dict
        Evaluation results for all queries and modes.
    """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {}

    for query, relevant_texts in GROUND_TRUTH.items():
        print(f"\nEvaluating: '{query}'")
        results[query] = {}

        for mode in ["all", "patient", "clinician"]:
            retrieved = retrieve(
                query,
                k=max(k_values),
                mode=mode,
                embedding_model=embedding_model
            )

            mode_results = {}
            for k in k_values:
                p = precision_at_k(retrieved, relevant_texts, k)
                r = recall_at_k(retrieved, relevant_texts, k)
                mode_results[f"P@{k}"] = round(p, 4)
                mode_results[f"R@{k}"] = round(r, 4)

            results[query][mode] = mode_results

    return results


def print_evaluation_results(results: dict) -> None:
    """Pretty print evaluation results to the terminal."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for query, modes in results.items():
        print(f"\nQuery: '{query}'")
        print(f"{'Mode':<12} {'P@1':<8} {'R@1':<8} {'P@3':<8} {'R@3':<8} {'P@5':<8} {'R@5':<8}")
        print("-" * 60)

        for mode, metrics in modes.items():
            print(
                f"{mode:<12} "
                f"{metrics.get('P@1', 0):<8} "
                f"{metrics.get('R@1', 0):<8} "
                f"{metrics.get('P@3', 0):<8} "
                f"{metrics.get('R@3', 0):<8} "
                f"{metrics.get('P@5', 0):<8} "
                f"{metrics.get('R@5', 0):<8}"
            )


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    results = evaluate(k_values=[1, 3, 5])
    print_evaluation_results(results)

    # Save results to JSON
    Path("evaluation_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print("\nSaved to evaluation_results.json")