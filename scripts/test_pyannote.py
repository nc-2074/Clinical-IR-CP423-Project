"""
scripts/test_pyannote.py
------------------------
Verify that the Pyannote diarization model loads correctly.
Downloads ~300MB on first run, then caches it.

Usage:
    python scripts/test_pyannote.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()


def test_pyannote():
    print("=" * 50)
    print("TESTING PYANNOTE DIARIZATION")
    print("=" * 50)

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token or "your_" in hf_token:
        print("❌ HF_TOKEN not set in .env")
        print("   1. Get a token at: https://huggingface.co/settings/tokens")
        print("   2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   3. Accept terms at: https://huggingface.co/pyannote/segmentation-3.0")
        return

    print(f"✅ HF token found: {hf_token[:5]}...{hf_token[-4:]}")
    print("\nLoading Pyannote pipeline (downloads ~300MB on first run)...")

    try:
        from pyannote.audio import Pipeline
        from huggingface_hub import login

        login(token=hf_token)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        print("✅ Pyannote pipeline loaded successfully!")
        print("\nYour environment is ready for offline speaker diarization.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nCommon fixes:")
        print("  - Make sure you accepted the model terms on Hugging Face")
        print("  - Check your HF_TOKEN has read access")


if __name__ == "__main__":
    test_pyannote()