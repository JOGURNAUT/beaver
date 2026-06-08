"""BUILD MESSAGES for the answer LLM call.

System prompt + (rolling summary + last N turns verbatim + numbered snippets + current Q)
all packed into one user message.

Also: summarize_history() for the rolling-summary fallback when conversation grows.
"""
from __future__ import annotations

from config import MAX_CONTEXT_TOKENS, TURNS_BEFORE_SUMMARY
from llm import client as llm
from agent.selector import Snippet, count_tokens


ANSWER_SYSTEM = """You are a deep research assistant. Answer the user's question
using ONLY the numbered snippets provided below as evidence. Each fact you state
must be followed by one or more citation markers like [1] or [2,3] that refer
to those numbered snippets.

Rules:
- If the snippets do not contain enough information to answer confidently, say
  so explicitly and suggest what additional research would help. Do not invent
  facts.
- If snippets disagree on a key fact (a number, date, name, or claim), state
  the disagreement explicitly and cite the conflicting snippets, e.g.
  "Source [2] reports X, while source [4] reports Y."
- Never write URLs in your answer. Only use the numeric [n] markers — the
  citation list is built from snippet metadata after you finish.
- Be concise. Aim for a direct answer, then optional supporting detail.
- Use regular punctuation (commas, periods, parentheses). Avoid em dashes
  and en dashes; substitute with commas or parentheses.
"""


def _format_snippets_block(snippets: list[Snippet]) -> str:
    if not snippets:
        return "(no snippets retrieved)"
    lines = []
    for i, s in enumerate(snippets, start=1):
        #include title + domain inline for source context- URLs intentionally omitted (mapped post-hoc)
        lines.append(f"[{i}] {s.title} — {s.domain}\n{s.text}")
    return "\n\n".join(lines)


def _format_history_block(recent_turns: list[dict], max_turns: int = 3) -> str:
    if not recent_turns:
        return ""
    tail = recent_turns[-max_turns:]   #only the last N turns verbatim, rest covered by rolling summary
    lines = []
    for t in tail:
        lines.append(f"User asked: {t['query']}")
        lines.append(f"You answered: {t['final_answer'][:400]}")
    return "Recent conversation:\n" + "\n".join(lines)


def build_messages(user_query: str, snippets: list[Snippet],
                   rolling_summary: str = "",
                   recent_turns: list[dict] | None = None) -> list[dict]:
    parts: list[str] = []
    if rolling_summary:
        parts.append(f"Earlier conversation summary:\n{rolling_summary}")
    history_block = _format_history_block(recent_turns or [])
    if history_block:
        parts.append(history_block)
    parts.append("Evidence snippets:\n" + _format_snippets_block(snippets))
    parts.append(f"Current question: {user_query}")

    user_msg = "\n\n".join(parts)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


#SUMMARIZER FALLBACK- triggered when history grows beyond TURNS_BEFORE_SUMMARY
_SUMMARIZER_SYSTEM = """Summarize the following research conversation into 4-6
bullet points capturing: the user's evolving topic, key facts established, and
any open questions. Do not include citations or URLs. Keep under 200 words."""


def summarize_history(turns: list[dict]) -> str:
    """Summarizes OLDER turns (everything except the most recent 2) into a
    rolling summary. Called only when len(turns) > TURNS_BEFORE_SUMMARY."""
    if len(turns) <= TURNS_BEFORE_SUMMARY:
        return ""
    older = turns[:-2]   #keep last 2 turns verbatim, summarize the rest
    convo = "\n\n".join(
        f"Q: {t['query']}\nA: {t['final_answer'][:500]}" for t in older
    )
    messages = [
        {"role": "system", "content": _SUMMARIZER_SYSTEM},
        {"role": "user", "content": convo},
    ]
    text, _ = llm.complete(messages, temperature=0.2, max_tokens=400)
    return text.strip()


def fits_in_budget(messages: list[dict], budget: int = MAX_CONTEXT_TOKENS) -> bool:
    total = sum(count_tokens(m["content"]) for m in messages)
    return total <= budget
