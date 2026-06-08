"""Quick sanity check before we build the agent loop.
Runs: storage round-trip, one Tavily search, one Groq call, one Gemini call.
"""
from storage import db
from tools import search as websearch
from llm import client as llm


def check_storage():
    print("[1/4] storage...")
    db.init_db()
    sid = db.create_session(title="smoke_test")
    db.add_message(sid, "user", "hello")
    msgs = db.get_messages(sid)
    assert len(msgs) == 1 and msgs[0]["content"] == "hello"
    print(f"      OK (session_id={sid[:8]}..., {len(msgs)} message)")


def check_search():
    print("[2/4] tavily search...")
    results = websearch.search("who won the 2024 nobel prize in physics", max_results=3)
    assert results, "no results returned"
    print(f"      OK ({len(results)} results)")
    print(f"      top: {results[0].title[:60]}  |  {results[0].url[:60]}")


def check_groq():
    print("[3/4] groq...")
    text, who = llm.complete(
        [{"role": "user", "content": "reply with exactly: pong"}],
        max_tokens=10, prefer="groq",
    )
    print(f"      OK (provider={who}, reply={text!r})")


def check_gemini():
    print("[4/4] gemini...")
    text, who = llm.complete(
        [{"role": "user", "content": "reply with exactly: pong"}],
        max_tokens=10, prefer="gemini",
    )
    if who != "gemini":
        raise RuntimeError(f"Gemini failed; silently fell back to {who}. "
                           f"Check GEMINI_MODEL value in .env.")
    print(f"      OK (provider={who}, reply={text!r})")


if __name__ == "__main__":
    check_storage()
    check_search()
    check_groq()
    check_gemini()
    print("\nAll checks passed.")
