"""
Setup script -- run this ONCE before starting the app.
Installs all dependencies.

Usage:
    python setup.py
"""
import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
print("\nDone! Now:")
print("  1. Copy .env.example to .env")
print("  2. Add your HuggingFace token to .env")
print("  3. Run: python -m streamlit run streamlit_app.py")
