"""CLI entrypoint for the research agent.

Usage:
  python cli.py                       # starts a new session
  python cli.py --session <id>        # resume an existing session
  python cli.py --list                # list sessions
"""
from __future__ import annotations
import argparse
import sys

from storage import db
from agent import loop as agent_loop


def _render_event(ev: dict) -> None:
    t = ev.get("type")
    if t == "plan":
        print(f"\n[PLAN] {ev.get('strategy', '')}")
        for i, q in enumerate(ev["queries"], 1):
            print(f"   {i}. {q}")
    elif t == "search_done":
        print(f"\n[SEARCH] {len(ev['results'])} unique results across queries")
        for r in ev["results"][:6]:
            print(f"   - {r['domain']:25}  {r['title'][:70]}")
    elif t == "fetch_done":
        ok = sum(1 for p in ev["pages"] if p["ok"])
        print(f"\n[FETCH] {ok}/{len(ev['pages'])} pages extracted")
        for p in ev["pages"]:
            mark = "ok " if p["ok"] else "FAIL"
            print(f"   {mark}  {p['chars']:>6} chars  {p['url'][:80]}")
    elif t == "select_done":
        print(f"\n[SELECT] {len(ev['snippets'])} snippets chosen")
        for s in ev["snippets"]:
            print(f"   [{s['idx']}] score={s['score']}  {s['domain']:20}  {s['title'][:50]}")
    elif t == "answer_token":
        sys.stdout.write(ev["text"])
        sys.stdout.flush()
    elif t == "answer_done":
        print(f"\n\n[CITATIONS] ({ev['latency_ms']} ms total)")
        for c in ev["citations"]:
            print(f"   [{c['marker']}] {c['title']} — {c['domain']}")
            print(f"        {c['url']}")
    elif t == "error":
        print(f"\n[ERROR @ {ev['stage']}] {ev['message']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", help="resume a session by id")
    p.add_argument("--list", action="store_true", help="list sessions and exit")
    args = p.parse_args()

    db.init_db()

    if args.list:
        for s in db.list_sessions():
            print(f"{s['id']}  {s['created_at']}  {s['title']}")
        return

    sid = args.session
    if not sid:
        sid = db.create_session(title="cli")
        print(f"started session {sid}")
    else:
        if not db.get_session(sid):
            print(f"session {sid} not found")
            return
        print(f"resumed session {sid}")

    print("\nType your question (or 'quit' to exit):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        for ev in agent_loop.run(sid, q):
            _render_event(ev)


if __name__ == "__main__":
    main()
