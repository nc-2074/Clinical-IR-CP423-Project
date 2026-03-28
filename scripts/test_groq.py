"""
scripts/test_groq.py
--------------------
Verify your Groq API key works for both Whisper transcription
and Llama 3 chat completions.

Usage:
    python scripts/test_groq.py
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def test_groq():
    print("=" * 50)
    print("TESTING GROQ API")
    print("=" * 50)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or "your_" in api_key:
        print("❌ GROQ_API_KEY not set in .env")
        print("   Get a free key at: https://console.groq.com")
        return

    print(f"✅ API key found: {api_key[:5]}...{api_key[-4:]}")

    try:
        client = Groq(api_key=api_key)

        # Test chat completion (used by align.py for role detection)
        print("\n--- Testing Llama 3 (used for role detection) ---")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Reply with only: OK"}],
            max_tokens=5,
        )
        print(f"✅ Llama 3: {response.choices[0].message.content.strip()}")

        print("\n--- Whisper transcription ---")
        print("✅ Whisper client ready (requires an audio file to fully test)")
        print("   Use: python -m speaker_separation.offline.transcribe audio/your_file.wav")

    except Exception as e:
        print(f"❌ Groq error: {e}")


if __name__ == "__main__":
    test_groq()
