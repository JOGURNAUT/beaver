"""Runs the agent over the dataset and scores every turn.

Usage:
    python eval/run_eval.py                        # all 15 questions
    python eval/run_eval.py --limit 3              # quick smoke
    python eval/run_eval.py --only c1,m1           # specific IDs
    python eval/run_eval.py --category multi_hop   # one category

Outputs:
    eval/results/<timestamp>/results.json          # full per-question data
    eval/results/<timestamp>/summary.json          # headline numbers
    eval/results/<timestamp>/report.md             # human-readable summary
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import db
from agent import loop as agent_loop
from eval import metrics


DATASET_PATH = ROOT / "eval" / "dataset" / "questions.json"
RESULTS_ROOT = ROOT / "eval" / "results"


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def _run_query_in_session(sid: str, query: str) -> dict:
    """Run one query in a given session, return the final outputs."""
    final = {"answer": "", "snippets": [], "citations": [], "latency_ms": 0}
    for ev in agent_loop.run(sid, query):
        t = ev.get("type")
        if t == "answer_done":
            final["answer"] = ev.get("answer", "")
            final["citations"] = ev.get("citations", [])
            final["latency_ms"] = ev.get("latency_ms", 0)
        elif t == "error":
            final.setdefault("errors", []).append(ev)
    return final


def run_one(q: dict) -> dict:
    """Run a question (and optional follow_up) in a session and collect outputs.
    For multi-turn questions, the follow_up runs in the SAME session so the
    planner sees prior turn context — that's what we're testing."""
    sid = db.create_session(title=f"eval_{q['id']}")

    #MAIN turn
    final = _run_query_in_session(sid, q["question"])

    #FOLLOW-UP turn if defined (multi-turn questions)
    if q.get("follow_up"):
        follow = _run_query_in_session(sid, q["follow_up"])
        final["follow_up_answer"] = follow.get("answer", "")
        final["follow_up_citations"] = follow.get("citations", [])
        final["follow_up_latency_ms"] = follow.get("latency_ms", 0)

    #pull snippets (with text) from the LAST saved turn for judge context
    turns = db.get_turns(sid)
    if turns:
        final["snippets"] = turns[-1].get("snippets", [])
        if q.get("follow_up") and len(turns) >= 2:
            final["follow_up_snippets"] = turns[-1].get("snippets", [])
            final["snippets"] = turns[-2].get("snippets", [])   #main turn's snippets
    return final


def write_report(outdir: Path, results: list[dict], summary: dict) -> None:
    md = ["# Evaluation Report", ""]
    md.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append(f"- Questions: {summary['n_questions']}")
    md.append("")

    md.append("## Overall")
    md.append("")
    for k, v in summary["overall"].items():
        md.append(f"- **{k}**: {v}")
    md.append("")

    md.append("## By category")
    md.append("")
    md.append("| Category | n | relevance | faithfulness | citation_prec | refusal | conflict | latency (ms) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for cat, m in summary["by_category"].items():
        md.append(f"| {cat} | {m['n']} | {m['avg_relevance']} | "
                  f"{m['avg_faithfulness']} | {m['avg_citation_precision']} | "
                  f"{m['avg_refusal_score']} | {m['avg_conflict_score']} | "
                  f"{m['avg_latency_ms']} |")
    md.append("")

    md.append("## Per-question highlights")
    md.append("")
    for r in results:
        md.append(f"### [{r['id']}] {r['category']} — {r['question']}")
        md.append("")
        md.append(f"- citations: {r['n_citations']} (unique domains: {r['n_unique_domains']})")
        md.append(f"- latency: {r['latency_ms']} ms")
        for key in ("relevance", "faithfulness", "citation_precision", "refusal", "conflict", "multi_source"):
            if key in r:
                v = r[key]
                score = v.get("score", v.get("precision"))
                md.append(f"- {key}: {score}")
        gf = r.get("gold_facts", {})
        if gf.get("score") is not None:
            md.append(f"- gold_facts: {gf['score']}  matched={gf['matched']}  missing={gf['missing']}")
        md.append("")
        md.append("**Answer:**")
        md.append("")
        md.append("> " + (r["answer"].replace("\n", "\n> ") if r["answer"] else "(empty)"))
        md.append("")

    (outdir / "report.md").write_text("\n".join(md), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="run only first N questions")
    p.add_argument("--only", type=str, default="", help="comma-separated question IDs")
    p.add_argument("--category", type=str, default="", help="filter by category")
    args = p.parse_args()

    db.init_db()
    qs = load_dataset()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        qs = [q for q in qs if q["id"] in wanted]
    if args.category:
        qs = [q for q in qs if q["category"] == args.category]
    if args.limit:
        qs = qs[: args.limit]

    if not qs:
        print("no questions selected")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = RESULTS_ROOT / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"writing results to {outdir}")

    results: list[dict] = []
    for i, q in enumerate(qs, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(qs)}] {q['id']}  {q['category']}  {q['question']!r}")
        try:
            agent_out = run_one(q)
            scored = metrics.score_turn(q, agent_out)
        except Exception as e:
            scored = {
                "id": q["id"], "category": q["category"],
                "question": q["question"], "answer": "",
                "error": str(e), "n_snippets": 0, "n_citations": 0,
                "n_unique_domains": 0, "latency_ms": 0,
            }
        results.append(scored)
        # incremental save so a partial run isn't lost
        (outdir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
        dt = time.time() - t0
        print(f"   done in {dt:.1f}s")

    summary = metrics.aggregate(results)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    write_report(outdir, results, summary)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary["overall"], indent=2))
    print(f"\nFull report: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
