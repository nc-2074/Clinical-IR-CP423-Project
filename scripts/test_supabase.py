"""
scripts/test_supabase.py
------------------------
Verify Supabase connection and check that the transcript_segments
table exists and is queryable.

Usage:
    python scripts/test_supabase.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def test_supabase():
    print("=" * 50)
    print("TESTING SUPABASE CONNECTION")
    print("=" * 50)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or "your_" in url:
        print("❌ SUPABASE_URL not set in .env")
        return
    if not key or "your_" in key:
        print("❌ SUPABASE_KEY not set in .env")
        return

    print(f"✅ URL: {url}")
    print(f"✅ Key: {key[:10]}...{key[-4:]}")

    try:
        client = create_client(url, key)
        print("✅ Supabase client initialised")

        # Check transcript_segments table exists
        result = client.table("transcript_segments").select("id").limit(1).execute()
        print(f"✅ transcript_segments table accessible")
        print(f"   Rows returned in test query: {len(result.data)}")

        # Check pgvector functions exist
        try:
            client.rpc("match_segments", {
                "query_embedding": [0.0] * 384,
                "match_count": 1,
                "p_session_id": "test",
            }).execute()
            print("✅ match_segments RPC function exists")
        except Exception:
            print("⚠️  match_segments RPC not found — run the SQL setup from README.md")

    except Exception as e:
        print(f"❌ Supabase error: {e}")
        print("\nCommon fixes:")
        print("  - Check SUPABASE_URL and SUPABASE_KEY in .env")
        print("  - Run the SQL setup from README.md to create the table")


if __name__ == "__main__":
    test_supabase()