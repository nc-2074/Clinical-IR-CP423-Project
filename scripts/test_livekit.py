"""
scripts/test_livekit.py
-----------------------
Test LiveKit connection and token generation.

Usage:
    python scripts/test_livekit.py
"""

import os
import asyncio
from dotenv import load_dotenv
from livekit import api

load_dotenv()


async def test_livekit():
    print("=" * 50)
    print("TESTING LIVEKIT CONNECTION")
    print("=" * 50)

    url        = os.getenv("LIVEKIT_URL")
    api_key    = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not url or "your_" in url:
        print("❌ LIVEKIT_URL not set in .env"); return
    if not api_key or "your_" in api_key:
        print("❌ LIVEKIT_API_KEY not set in .env"); return
    if not api_secret or "your_" in api_secret:
        print("❌ LIVEKIT_API_SECRET not set in .env"); return

    print(f"✅ URL:        {url}")
    print(f"✅ API Key:    {api_key[:5]}...{api_key[-4:]}")
    print(f"✅ API Secret: {api_secret[:5]}...{api_secret[-4:]}")

    try:
        livekit_api = api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret)
        rooms = await livekit_api.room.list_rooms(api.ListRoomsRequest())
        print(f"\n✅ Connected to LiveKit! Active rooms: {len(rooms.rooms)}")
        await livekit_api.aclose()
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")

    # Test token generation
    print("\n--- Token Generation ---")
    try:
        import time
        room = f"test-{int(time.time())}"
        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity("test-user")
            .with_name("Test User")
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
        print(f"✅ Token generated for room '{room}'")
        print(f"   {token[:40]}...")
    except Exception as e:
        print(f"❌ Token generation failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_livekit())