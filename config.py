import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DB_PATH = Path(os.getenv("DB_PATH", "./data/research.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Context budget — leaves headroom below model limits
MAX_CONTEXT_TOKENS = 8000
MAX_SNIPPETS_PER_TURN = 8
SNIPPET_CHAR_LIMIT = 1200

# Search defaults
SEARCH_RESULTS_PER_QUERY = 5
MAX_PAGES_TO_FETCH = 6
FETCH_TIMEOUT_SECONDS = 8

# Conversation summarization kicks in beyond this
TURNS_BEFORE_SUMMARY = 6
