"""
scripts/generate_tokens.py
--------------------------
Standalone CLI tool to generate LiveKit JWT tokens for testing.
Useful for testing outside of Flask — e.g. directly joining
meet.livekit.io to verify your LiveKit credentials work.

Usage:
    python scripts/generate_tokens.py
    python scripts/generate_tokens.py --room my-custom-room
"""

import os
import time
import argparse
from dotenv import load_dotenv
from livekit import api

load_dotenv()


def generate_tokens(room_name: str = None):
    url        = os.getenv("LIVEKIT_URL")
    api_key    = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([url, api_key, api_secret]):
        print("❌ Missing LiveKit credentials in .env")
        print("   Required: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET")
        return

    if room_name is None:
        room_name = f"interview-{int(time.time())}"

    patient_token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("patient")
        .with_name("Patient")
        .with_grants(api.VideoGrants(
            room_join=True, room=room_name,
            can_publish=True, can_subscribe=True,
        ))
        .to_jwt()
    )

    clinician_token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("clinician")
        .with_name("Clinician")
        .with_grants(api.VideoGrants(
            room_join=True, room=room_name,
            can_publish=True, can_subscribe=True,
        ))
        .to_jwt()
    )

    print("\n" + "=" * 60)
    print(f"ROOM:        {room_name}")
    print(f"LIVEKIT URL: {url}")
    print("=" * 60)
    print("\n👤 PATIENT TOKEN:")
    print(patient_token)
    print("\n👨‍⚕️ CLINICIAN TOKEN:")
    print(clinician_token)
    print("\n" + "=" * 60)
    print("Go to: https://meet.livekit.io")
    print("Paste the LiveKit URL, then paste each token for each participant.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LiveKit tokens")
    parser.add_argument("--room", default=None, help="Room name (auto-generated if not set)")
    args = parser.parse_args()
    generate_tokens(room_name=args.room)