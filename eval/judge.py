"""LLM-as-judge primitives.

We use Gemini as the judge (the answerer is Groq/Llama), so the judge is from
a different model family — reduces self-grading bias. Judge always runs with
temperature=0 for reproducibility.

Each function returns a parsed dict; on parse failure returns a minimal
fallback dict with score=0 and an `error` field, so the eval runner doesn't
crash on a single bad judgment.
"""
from __future__ import annotations
import json
import re

from llm import client as llm


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def _ask_judge(system: str, user: str, max_tokens: int = 600) -> dict:
    """Always uses Gemini, temperature 0. Returns parsed JSON or {error:...}.

    Gemini only, deliberately - and this is the important part.

    llm.complete() normally falls back to the other provider. For the judge
    that is not a degradation, it is a different experiment: the fallback is
    Groq, which is the model being evaluated, so a rate-limited Gemini turns
    cross-model judging into self-grading without changing anything visible in
    the output. A run of this harness did exactly that once Groq's daily token
    limit and Gemini's per-minute limit collided. Passing a single-element
    order makes a judge outage record as a missing score, which aggregate()
    now surfaces as incomplete coverage.

    Retries once with a much larger budget when the first call comes back
    unparseable.

    gemini-2.5-flash is a reasoning model: it spends part of max_tokens
    thinking before it emits any text at all. So a budget that is generous for
    the JSON itself can still return an empty string, or a response truncated
    to "{\n" - which is exactly what a run of this harness produced, silently,
    on the hardest three questions. Thinking cost scales with how tangled the
    input is, not with how long the answer needs to be, so sizing every caller
    for its worst case wastes tokens on the easy ones. Retrying only the
    failures is cheaper.
    """
    attempts = (max_tokens, max_tokens * 3)
    text = ""
    for budget in attempts:
        try:
            text, _ = llm.complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.0, max_tokens=budget,
                order=["gemini"],   # no fallback: see _JUDGE_PROVIDER note
            )
        except Exception as e:
            return {"error": f"judge_call_failed: {e}"}
        parsed = _extract_json(text)
        if parsed:
            return parsed
    return {"error": "judge_returned_unparseable",
            "raw": text[:500],
            "budgets_tried": list(attempts)}


# ---------------------------------------------------------------------------
# Faithfulness: are all claims in the answer supported by the snippets?
# ---------------------------------------------------------------------------

_FAITH_SYSTEM = """You are an evaluator. Given a question, an answer, and the
evidence snippets that the answerer was allowed to use, identify whether each
factual claim in the answer is supported by the snippets.

Reply ONLY with JSON:
{"total_claims": <int>,
 "supported_claims": <int>,
 "score": <0.0..1.0, equals supported/total>,
 "unsupported_claims": ["<short description of each unsupported claim>"]}

Rules:
- Count only factual claims (entities, numbers, dates, attributions). Ignore
  hedging language, transitions, and meta-statements like "I don't know".
- A claim is supported if its substance appears in at least one snippet.
- Be strict: if the answer adds detail not in the snippets, that's unsupported.
"""


def judge_faithfulness(question: str, answer: str, snippets: list[dict]) -> dict:
    snip_text = "\n\n".join(
        f"[{i+1}] {s.get('title','')} — {s.get('domain','')}\n{s.get('text','')}"
        for i, s in enumerate(snippets)
    ) or "(no snippets)"
    user = (
        f"Question: {question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Evidence snippets:\n{snip_text}"
    )
    out = _ask_judge(_FAITH_SYSTEM, user)
    if "score" not in out:
        # judge failed to produce a parsable score — record as None so
        # aggregate() can skip it instead of treating it as a real zero.
        out["score"] = None
    return out


# ---------------------------------------------------------------------------
# Citation precision: do cited snippets actually support the claims near them?
# ---------------------------------------------------------------------------

_CITE_SYSTEM = """You are an evaluator of citation accuracy.

You will see an answer that contains numeric markers like [1] or [2,3]. For
each citation event (an occurrence of one or more markers attached to a
specific claim), judge whether AT LEAST ONE of the cited snippets actually
supports the claim it is attached to.

Reply ONLY with JSON:
{"total_citation_events": <int>,
 "supported_events": <int>,
 "precision": <0.0..1.0>,
 "bad_events": [{"claim":"...","markers":[...],"reason":"..."}]}
"""


def judge_citation_precision(answer: str, snippets: list[dict]) -> dict:
    if not snippets:
        return {"total_citation_events": 0, "supported_events": 0, "precision": 1.0,
                "bad_events": [], "note": "no_snippets"}
    snip_text = "\n\n".join(
        f"[{i+1}] {s.get('title','')} — {s.get('domain','')}\n{s.get('text','')}"
        for i, s in enumerate(snippets)
    )
    user = f"Answer:\n{answer}\n\nSnippets:\n{snip_text}"
    out = _ask_judge(_CITE_SYSTEM, user)
    if "precision" not in out:
        out["precision"] = None
    return out


# ---------------------------------------------------------------------------
# Answer relevance: does it actually address the question?
# ---------------------------------------------------------------------------

_REL_SYSTEM = """Rate how directly the answer addresses the question on a scale
0.0 (not at all) to 1.0 (fully and directly). Penalize off-topic content,
padding, and answers that talk around the question.

Reply ONLY with JSON:
{"score": <0.0..1.0>, "reason": "<one sentence>"}
"""


def judge_relevance(question: str, answer: str) -> dict:
    user = f"Question: {question}\n\nAnswer:\n{answer}"
    out = _ask_judge(_REL_SYSTEM, user, max_tokens=400)
    if "score" not in out:
        out["score"] = None
    return out


# ---------------------------------------------------------------------------
# Refusal correctness: did the agent explicitly admit insufficient evidence?
# ---------------------------------------------------------------------------

_REFUSE_KEYWORDS = [
    "cannot determine", "cannot answer", "cannot confidently", "could not find",
    "couldn't find", "insufficient", "not enough information", "not enough to",
    "no reliable", "no available", "don't have enough", "do not have enough",
    "do not contain enough", "unable to confirm", "not confirmed",
    "not independently verified", "no information",
    "not found in the provided", "no evidence", "i don't know", "i do not know",
]


def judge_refusal(question: str, answer: str) -> dict:
    """LLM judge is the primary signal. Keywords are a confirmation: when LLM
    says refused AND keywords agree, that's a high-confidence refusal; when
    LLM says refused but keywords don't match, it still counts (the LLM saw
    the intent), just with a slightly lower confidence score."""
    a_lower = answer.lower()
    keyword_hit = any(k in a_lower for k in _REFUSE_KEYWORDS)

    sys = ("Decide whether the following answer explicitly states that it "
           "cannot answer the question due to missing, unavailable, or "
           "insufficient evidence. A clear admission of inability to answer "
           "counts as refused. "
           'Reply ONLY with JSON: {"refused": true|false, "reason": "..."}')
    user = f"Question: {question}\n\nAnswer:\n{answer}"
    out = _ask_judge(sys, user, max_tokens=400)
    if "refused" not in out:
        return {"score": None, "keyword_hit": keyword_hit, "llm_refused": None,
                "reason": "judge_failed"}
    llm_refused = bool(out["refused"])

    if llm_refused and keyword_hit:
        score = 1.0
    elif llm_refused:
        score = 0.75   # LLM saw the refusal even if our keyword list missed it
    elif keyword_hit:
        score = 0.25   # keywords suggest refusal but LLM disagreed; weak signal
    else:
        score = 0.0
    return {"score": score, "keyword_hit": keyword_hit, "llm_refused": llm_refused,
            "reason": out.get("reason", "")}


# ---------------------------------------------------------------------------
# Conflict handling: did the agent flag disagreement and cite both sides?
# ---------------------------------------------------------------------------

_CONFLICT_KEYWORDS = [
    "different sources", "sources disagree", "sources differ", "conflicting",
    "varies", "range of", "estimates range", "varies between",
    "some sources", "while other", "however", "on the other hand",
    "disagreement", "discrepan",
]


def judge_conflict(question: str, answer: str) -> dict:
    a_lower = answer.lower()
    keyword_hit = any(k in a_lower for k in _CONFLICT_KEYWORDS)

    sys = ("Decide whether the answer explicitly notes that different sources "
           "give different or conflicting information about the question, and "
           "cites more than one source in service of that. "
           'Reply ONLY with JSON: {"flagged": true|false, "reason": "..."}')
    user = f"Question: {question}\n\nAnswer:\n{answer}"
    out = _ask_judge(sys, user, max_tokens=400)
    if "flagged" not in out:
        return {"score": None, "keyword_hit": keyword_hit, "llm_flagged": None,
                "reason": "judge_failed"}
    llm_flagged = bool(out["flagged"])

    if llm_flagged and keyword_hit:
        score = 1.0
    elif llm_flagged:
        score = 0.75
    elif keyword_hit:
        score = 0.25
    else:
        score = 0.0
    return {"score": score, "keyword_hit": keyword_hit, "llm_flagged": llm_flagged,
            "reason": out.get("reason", "")}


# ---------------------------------------------------------------------------
# Gold-fact coverage (cheap deterministic check, complements judges)
# ---------------------------------------------------------------------------

def gold_fact_coverage(answer: str, gold_facts: list[str]) -> dict:
    """Soft check: how many of the listed gold facts appear (case-insensitive)
    anywhere in the answer. Useful as a quick sanity signal even when LLM-judge
    is wrong."""
    if not gold_facts:
        return {"score": None, "matched": [], "missing": []}
    a_lower = answer.lower()
    matched = [g for g in gold_facts if g.lower() in a_lower]
    missing = [g for g in gold_facts if g.lower() not in a_lower]
    return {"score": len(matched) / len(gold_facts), "matched": matched, "missing": missing}


#MULTI-TURN: did the follow-up correctly resolve pronouns/context from the main turn?
_CTX_SYSTEM = """You are evaluating a multi-turn conversation.

Given:
  - The MAIN question and the agent's MAIN answer
  - A FOLLOW-UP question that contains pronouns or implicit references (e.g.,
    "they", "it", "there") relying on context from the main turn
  - The agent's FOLLOW-UP answer

Decide whether the follow-up answer correctly resolved the context — i.e., it
answered the follow-up as if it understood what entity/topic the pronouns
referred to from the main turn. It does NOT have to be factually perfect; we
only check context resolution.

Reply ONLY with JSON:
{"resolved": true|false, "reason": "<one sentence>"}"""


def judge_context_resolution(main_question: str, main_answer: str,
                              follow_up_question: str, follow_up_answer: str) -> dict:
    user = (
        f"MAIN question: {main_question}\n"
        f"MAIN answer: {main_answer[:600]}\n\n"
        f"FOLLOW-UP question: {follow_up_question}\n"
        f"FOLLOW-UP answer: {follow_up_answer[:600]}"
    )
    out = _ask_judge(_CTX_SYSTEM, user, max_tokens=400)
    if "resolved" not in out:
        return {"score": None, "reason": "judge_failed"}
    return {"score": 1.0 if bool(out["resolved"]) else 0.0,
            "reason": out.get("reason", "")}
