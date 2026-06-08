"""The agent loop- streaming generator that orchestrates the 5 stages.

run(session_id, user_query) yields events at every stage transition + per-token
chunks during the answer stream. Persists turn to SQLite at the end.

Event shapes:
  {"type": "plan",         "queries": [...], "strategy": "..."}
  {"type": "search_done",  "results": [...]}
  {"type": "fetch_done",   "pages": [...]}
  {"type": "select_done",  "snippets": [...]}
  {"type": "answer_token", "text": "..."}
  {"type": "answer_done",  "answer", "citations", "latency_ms", "provider"}
  {"type": "error",        "stage": "...", "message": "..."}

No chain-of-thought is ever streamed- only operational progress + final tokens.
"""
from __future__ import annotations
import time
from typing import Iterator
from urllib.parse import urlparse

from config import MAX_PAGES_TO_FETCH, TURNS_BEFORE_SUMMARY
from storage import db
from tools import search as websearch
from tools import fetch as webfetch
from llm import client as llm
from agent import planner as planner_mod
from agent import selector as selector_mod
from agent import context_builder as ctx_mod
from agent import citations as cite_mod


def _dedupe_results(results: list[websearch.SearchResult]) -> list[websearch.SearchResult]:
    #dedupe by URL across all plan queries
    seen: set[str] = set()
    out: list[websearch.SearchResult] = []
    for r in results:
        if not r.url or r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
    return out


def _pick_urls_to_fetch(results: list[websearch.SearchResult],
                        cap: int = MAX_PAGES_TO_FETCH) -> list[str]:
    """Pick up to `cap` URLs, prioritizing Tavily score and diversifying domains."""
    #sort by score desc, None last
    sorted_r = sorted(results, key=lambda r: (r.score is None, -(r.score or 0)))
    picked: list[str] = []
    per_domain: dict[str, int] = {}
    for r in sorted_r:
        if len(picked) >= cap:
            break
        d = urlparse(r.url).netloc.replace("www.", "")
        if per_domain.get(d, 0) >= 2:  #at most 2 URLs per domain at fetch stage
            continue
        picked.append(r.url)
        per_domain[d] = per_domain.get(d, 0) + 1
    return picked


def run(session_id: str, user_query: str) -> Iterator[dict]:
    """Main entry. Yields events as the agent works; persists at the end."""
    t0 = time.time()
    last_t = t0
    stage_latencies: dict[str, float] = {}   #per-stage seconds, persisted in save_turn

    #LOAD context- session + rolling summary + prior turns
    session = db.get_session(session_id)
    if not session:
        yield {"type": "error", "stage": "session", "message": f"unknown session_id {session_id}"}
        return
    rolling_summary = session.get("rolling_summary", "") or ""
    prior_turns = db.get_turns(session_id)

    #PLAN- LLM reformulates user query into 3-5 search queries
    try:
        plan_obj = planner_mod.plan(
            user_query,
            rolling_summary=rolling_summary or None,
            recent_turns=prior_turns or None,
        )
    except Exception as e:
        yield {"type": "error", "stage": "plan", "message": str(e)}
        plan_obj = planner_mod.SearchPlan(queries=[user_query], strategy="(planner failed, using raw query)")

    stage_latencies["plan"] = time.time() - last_t
    last_t = time.time()
    yield {"type": "plan", "queries": plan_obj.queries, "strategy": plan_obj.strategy}

    #SEARCH- one Tavily call per plan query, sequential
    all_results: list[websearch.SearchResult] = []
    for q in plan_obj.queries:
        try:
            res = websearch.search(q)
            all_results.extend(res)
        except Exception as e:
            yield {"type": "error", "stage": "search", "message": f"{q}: {e}"}

    all_results = _dedupe_results(all_results)
    stage_latencies["search"] = time.time() - last_t
    last_t = time.time()
    yield {
        "type": "search_done",
        "results": [
            {"title": r.title, "url": r.url,
             "domain": urlparse(r.url).netloc.replace("www.", ""),
             "snippet": r.snippet[:200]}
            for r in all_results
        ],
    }

    #FETCH- parallel via ThreadPool, 8s timeout per URL, trafilatura for main content
    urls = _pick_urls_to_fetch(all_results)
    pages: list[webfetch.FetchedPage] = []
    if urls:
        pages = webfetch.fetch_many(urls)
    stage_latencies["fetch"] = time.time() - last_t
    last_t = time.time()
    yield {
        "type": "fetch_done",
        "pages": [
            {"url": p.url, "ok": p.error is None, "chars": len(p.text or ""),
             "error": p.error, "title": p.title, "domain": p.domain}
            for p in pages
        ],
    }

    #SELECT- chunk + embed + budget-and-diversity-aware pick
    snippets = selector_mod.select(pages, user_query)
    stage_latencies["select"] = time.time() - last_t
    last_t = time.time()
    yield {
        "type": "select_done",
        "snippets": [
            {"idx": i + 1, "title": s.title, "domain": s.domain,
             "chars": len(s.text), "score": round(s.score, 3), "url": s.url}
            for i, s in enumerate(snippets)
        ],
    }

    #CONTEXT- summarize older history if conversation has grown beyond threshold
    if len(prior_turns) > TURNS_BEFORE_SUMMARY:
        try:
            new_summary = ctx_mod.summarize_history(prior_turns)
            if new_summary:
                rolling_summary = new_summary
                db.update_rolling_summary(session_id, rolling_summary)
        except Exception as e:
            yield {"type": "error", "stage": "summarize", "message": str(e)}

    messages = ctx_mod.build_messages(
        user_query=user_query,
        snippets=snippets,
        rolling_summary=rolling_summary,
        recent_turns=prior_turns,
    )

    #ANSWER- stream tokens, BOTH yield to UI AND accumulate for citation post-process
    chunks: list[str] = []
    provider_used = "unknown"
    try:
        for chunk in llm.stream(messages, temperature=0.2, max_tokens=900):
            chunks.append(chunk)
            yield {"type": "answer_token", "text": chunk}
    except Exception as e:
        yield {"type": "error", "stage": "answer", "message": str(e)}
        #fall through to persist what we got

    raw_answer = "".join(chunks)
    stage_latencies["answer"] = time.time() - last_t

    #POST-PROCESS CITATIONS- regex extract [n] markers, drop hallucinated
    cleaned_answer, citations = cite_mod.extract_citations(raw_answer, snippets)
    #strip em/en dashes that the model sometimes uses despite the system prompt
    cleaned_answer = cleaned_answer.replace("—", ",").replace("–", "-")
    latency_ms = int((time.time() - t0) * 1000)

    yield {
        "type": "answer_done",
        "answer": cleaned_answer,
        "citations": cite_mod.citations_to_dicts(citations),
        "latency_ms": latency_ms,
        "stage_latencies": stage_latencies,
        "provider": provider_used,
    }

    #PERSIST- chat messages + full audit trail to SQLite
    db.add_message(session_id, "user", user_query)
    db.add_message(session_id, "assistant", cleaned_answer)
    db.save_turn(
        session_id=session_id,
        query=user_query,
        plan=plan_obj.strategy,
        search_queries=plan_obj.queries,
        urls_opened=[
            {"url": p.url, "title": p.title, "domain": p.domain,
             "retrieved_at": p.retrieved_at, "ok": p.error is None}
            for p in pages
        ],
        snippets=[
            {"idx": i + 1, "title": s.title, "domain": s.domain,
             "url": s.url, "score": s.score, "text": s.text}
            for i, s in enumerate(snippets)
        ],
        final_answer=cleaned_answer,
        citations=cite_mod.citations_to_dicts(citations),
        latency_ms=latency_ms,
        stage_latencies=stage_latencies,
    )
