"""Quick sanity check that all 3 API keys are present in .env.
Run: python check_env.py
"""
from dotenv import dotenv_values

v = dotenv_values(".env")
for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"]:
    val = v.get(k, "")
    if not val or val.endswith("..."):
        print(f"{k}: MISSING or still placeholder")
    else:
        print(f"{k}: set (len={len(val)}, starts={val[:6]}...)")
