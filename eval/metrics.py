"""High-level per-question scoring.

score_turn() takes a question record (from dataset) and the agent's output
(snippets + answer + citations + latency) and returns a dict of metric values.

It calls the appropriate subset of judges based on the question's
expected_behavior tags so we don't waste judge calls on irrelevant checks.
"""
from __future__ import annotations
from typing import Any

from eval import judge


def score_turn(q: dict, agent_output: dict) -> dict[str, Any]:
    """q has fields: id, category, question, expected_behavior, gold_facts
    agent_output has fields: answer, snippets, citations, latency_ms"""

    answer = agent_output.get("answer", "") or ""
    snippets = agent_output.get("snippets", [])
    citations = agent_output.get("citations", [])
    latency_ms = agent_output.get("latency_ms", 0)
    expected = set(q.get("expected_behavior", []))

    result: dict[str, Any] = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "answer": answer,
        "n_snippets": len(snippets),
        "n_citations": len(citations),
        "n_unique_domains": len({c.get("domain", "") for c in citations if c.get("domain")}),
        "latency_ms": latency_ms,
    }

    # --- judges that apply to (almost) everyone ---
    if "must_answer" in expected or expected == {"must_flag_conflict"} or expected == {"must_refuse"}:
        # always run relevance; cheap and informative
        result["relevance"] = judge.judge_relevance(q["question"], answer)

    # only score faithfulness/citation when we actually expected a substantive answer
    if "must_answer" in expected or "must_flag_conflict" in expected or "must_cite_at_least_2_sources" in expected:
        result["faithfulness"] = judge.judge_faithfulness(q["question"], answer, snippets)
        result["citation_precision"] = judge.judge_citation_precision(answer, snippets)

    # --- conditional checks ---
    if "must_refuse" in expected:
        result["refusal"] = judge.judge_refusal(q["question"], answer)

    if "must_flag_conflict" in expected:
        result["conflict"] = judge.judge_conflict(q["question"], answer)

    if "must_cite_at_least_2_sources" in expected:
        result["multi_source"] = {
            "score": 1.0 if result["n_unique_domains"] >= 2 else 0.0,
            "n_unique_domains": result["n_unique_domains"],
        }

    # --- always-on gold-fact coverage (cheap signal) ---
    result["gold_facts"] = judge.gold_fact_coverage(answer, q.get("gold_facts", []))

    # --- multi-turn: score the follow-up if present ---
    if "follow_up_must_resolve_context" in expected and agent_output.get("follow_up_answer") is not None:
        fa = agent_output.get("follow_up_answer", "") or ""
        result["follow_up_answer"] = fa
        result["follow_up_latency_ms"] = agent_output.get("follow_up_latency_ms", 0)
        result["context_resolution"] = judge.judge_context_resolution(
            q["question"], answer, q["follow_up"], fa,
        )
        result["follow_up_gold_facts"] = judge.gold_fact_coverage(fa, q.get("follow_up_gold_facts", []))

    return result


def aggregate(results: list[dict]) -> dict[str, Any]:
    """Roll up per-question results into headline numbers grouped by category."""
    def avg(values):
        vs = [v for v in values if v is not None]
        return round(sum(vs) / len(vs), 3) if vs else None

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    headline: dict[str, Any] = {"n_questions": len(results)}
    headline["overall"] = {
        "avg_relevance": avg([r.get("relevance", {}).get("score") for r in results]),
        "avg_faithfulness": avg([r.get("faithfulness", {}).get("score") for r in results]),
        "avg_citation_precision": avg([r.get("citation_precision", {}).get("precision") for r in results]),
        "avg_refusal_score": avg([r.get("refusal", {}).get("score") for r in results if "refusal" in r]),
        "avg_conflict_score": avg([r.get("conflict", {}).get("score") for r in results if "conflict" in r]),
        "avg_multi_source": avg([r.get("multi_source", {}).get("score") for r in results if "multi_source" in r]),
        "avg_context_resolution": avg([r.get("context_resolution", {}).get("score") for r in results
                                       if "context_resolution" in r]),
        "avg_gold_fact_coverage": avg([r.get("gold_facts", {}).get("score") for r in results
                                       if r.get("gold_facts", {}).get("score") is not None]),
        "avg_latency_ms": avg([r.get("latency_ms") for r in results]),
        "avg_unique_domains_per_answer": avg([r.get("n_unique_domains") for r in results]),
    }

    headline["by_category"] = {}
    for cat, rs in by_cat.items():
        headline["by_category"][cat] = {
            "n": len(rs),
            "avg_relevance": avg([r.get("relevance", {}).get("score") for r in rs]),
            "avg_faithfulness": avg([r.get("faithfulness", {}).get("score") for r in rs]),
            "avg_citation_precision": avg([r.get("citation_precision", {}).get("precision") for r in rs]),
            "avg_refusal_score": avg([r.get("refusal", {}).get("score") for r in rs if "refusal" in r]),
            "avg_conflict_score": avg([r.get("conflict", {}).get("score") for r in rs if "conflict" in r]),
            "avg_latency_ms": avg([r.get("latency_ms") for r in rs]),
        }

    return headline
