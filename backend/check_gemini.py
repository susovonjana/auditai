"""
Diagnostic: list which Gemini models your API key can use.

Run from the backend folder with venv active:
    python check_gemini.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY", "")
if not key:
    raise SystemExit("GEMINI_API_KEY is not set in .env")

print(f"Key in use: {key[:8]}...{key[-4:]}")
print()

import google.generativeai as genai

genai.configure(api_key=key)

print("Models supporting generateContent:")
print("-" * 50)
found = 0
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        # Strip the 'models/' prefix for use in GEMINI_MODEL env var
        name = m.name.replace("models/", "")
        print(f"  {name}")
        found += 1

print("-" * 50)
print(f"Total: {found} model(s)")
print()
print("Put any of these names (without the 'models/' prefix) in your .env:")
print("    GEMINI_MODEL=...")
