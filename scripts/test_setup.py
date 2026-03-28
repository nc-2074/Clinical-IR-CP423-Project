"""
scripts/test_setup.py
---------------------
Verify that all components are installed and configured correctly.
Run this first to ensure your environment is ready before starting the server.
 
Usage:
    python scripts/test_setup.py
"""
 
import sys
import platform
import os
import subprocess
 
 
def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)
 
 
def test_python():
    print_header("PYTHON ENVIRONMENT")
    print(f"Python version: {sys.version}")
    print(f"Platform:       {platform.platform()}")
    print(f"Virtual env:    {sys.prefix != sys.base_prefix}")
 
 
def test_packages():
    print_header("INSTALLED PACKAGES")
 
    packages = [
        ("pyannote.audio",      "pyannote.audio"),
        ("groq",                "groq"),
        ("livekit",             "livekit"),
        ("livekit-api",         "livekit-api"),
        ("supabase",            "supabase"),
        ("sentence_transformers","sentence-transformers"),
        ("sklearn",             "scikit-learn"),
        ("flask",               "flask"),
        ("dotenv",              "python-dotenv"),
        ("numpy",               "numpy"),
    ]
 
    for module_name, display_name in packages:
        try:
            mod     = __import__(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"✅ {display_name}: {version}")
        except ImportError:
            print(f"❌ {display_name}: NOT INSTALLED  →  pip install {display_name}")
        except Exception as e:
            print(f"⚠️  {display_name}: {e}")
 
    # livekit-agents needs a separate check
    try:
        import livekit.agents
        print(f"✅ livekit-agents: installed")
    except ImportError:
        print("❌ livekit-agents: NOT INSTALLED  →  pip install livekit-agents[silero]")
 
 
def test_ffmpeg():
    print_header("FFMPEG")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"✅ {result.stdout.split(chr(10))[0]}")
        else:
            print("❌ ffmpeg not working properly")
    except FileNotFoundError:
        print("❌ ffmpeg not found  →  brew install ffmpeg")
    except Exception as e:
        print(f"❌ ffmpeg error: {e}")
 
 
def test_env_file():
    print_header("ENVIRONMENT FILE (.env)")
    if not os.path.exists(".env"):
        print("❌ .env file missing  →  cp .env.example .env")
        return
 
    print("✅ .env file found")
    with open(".env") as f:
        content = f.read()
 
    keys = [
        ("HF_TOKEN",            "Hugging Face"),
        ("GROQ_API_KEY",        "Groq"),
        ("SUPABASE_URL",        "Supabase URL"),
        ("SUPABASE_KEY",        "Supabase Key"),
        ("LIVEKIT_URL",         "LiveKit URL"),
        ("LIVEKIT_API_KEY",     "LiveKit API Key"),
        ("LIVEKIT_API_SECRET",  "LiveKit API Secret"),
    ]
 
    for key, label in keys:
        if f"{key}=your_" in content or f"{key}=" not in content:
            print(f"⚠️  {label} ({key}): not set")
        else:
            print(f"✅ {label} ({key}): set")
 
 
def main():
    print("\n" + "=" * 60)
    print("   CLINICAL INTERVIEW IR SYSTEM — SETUP VERIFICATION")
    print("=" * 60)
 
    test_python()
    test_packages()
    test_ffmpeg()
    test_env_file()
 
    print_header("NEXT STEPS")
    print("1. Fix any ❌ items above")
    print("2. Fill in all keys in your .env file")
    print("3. Start the server:  python app.py")
    print("4. Open browser:      http://localhost:5001")
    print("5. For live mode, also run:")
    print("   python -m speaker_separation.live.transcriber")
 
 
if __name__ == "__main__":
    main()