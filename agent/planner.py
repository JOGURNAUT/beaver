"""Stage 1: Plan.

Takes the user query (plus optional conversation context) and asks the LLM
to reformulate it into 3-5 targeted web search queries plus a one-line strategy.

Why an LLM here: the same factual question can be phrased many ways the web
isn't indexed under. A planner makes the search input less brittle than the
raw user text.

Output is JSON so we can parse deterministically; on parse failure we fall
back to using the raw user query as a single search query (graceful degrade,
the agent still works).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from llm import client as llm


@dataclass
class SearchPlan:
    queries: list[str]
    strategy: str = ""
    raw: str = field(default="", repr=False)


_PLANNER_SYSTEM = """You are a research planner. Given a user question and any
prior context, produce 3-5 distinct web search queries that together would
gather enough evidence to answer it well.

Rules:
- Queries should be diverse: rephrase, narrow, broaden, and pull synonyms.
- Prefer specific entities, dates, and numbers when relevant.
- Each query under ~10 words.
- If the user question is multi-part, cover each part with at least one query.

Reply with ONLY a JSON object, no prose, no markdown fences:
{"strategy": "<one sentence on how these queries cover the question>",
 "queries": ["q1", "q2", "q3"]}"""


def _build_planner_prompt(user_query: str, rolling_summary: str | None,
                          recent_turns: list[dict] | None) -> list[dict]:
    ctx = ""
    if rolling_summary:
        ctx += f"Prior conversation summary:\n{rolling_summary}\n\n"
    if recent_turns:
        last = recent_turns[-2:]  # only last 2 turns to keep planner cheap
        for t in last:
            ctx += f"Prior Q: {t['query']}\nPrior A (truncated): {t['final_answer'][:300]}\n\n"
    user = f"{ctx}Current question: {user_query}"
    return [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> dict:
    # try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to pull the first {...} block (LLMs sometimes wrap in prose despite instructions)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def plan(user_query: str, rolling_summary: str | None = None,
         recent_turns: list[dict] | None = None) -> SearchPlan:
    messages = _build_planner_prompt(user_query, rolling_summary, recent_turns)
    text, _provider = llm.complete(messages, temperature=0.3, max_tokens=400)
    parsed = _extract_json(text)

    queries = parsed.get("queries") or []
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not queries:
        # fallback: use the raw user query
        queries = [user_query]

    # cap at 5 — defense against runaway plans
    queries = queries[:5]

    return SearchPlan(
        queries=queries,
        strategy=str(parsed.get("strategy", "")).strip(),
        raw=text,
    )
