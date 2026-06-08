"""Streamlit UI- chat + live progress + session switcher + latency/confidence footer.

Run with: streamlit run ui/app.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

#allow running from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from storage import db
from agent import loop as agent_loop
from llm import client as llm


st.set_page_config(page_title="Deep Research Agent", layout="wide",
                   initial_sidebar_state="expanded")
db.init_db()


#MINIMAL CSS- only custom classes (no Streamlit internal selectors that could
#interfere with widget behavior). Theme colors come from .streamlit/config.toml.
st.markdown("""
<style>
.brand-title {
    color: #3B82F6;
    font-size: 20px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.3px;
}
.brand-tag {
    color: #64748b;
    font-size: 10px;
    margin-top: 4px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    font-weight: 600;
}
.section-label {
    color: #64748b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 16px 0 6px 0;
    font-weight: 700;
}
.sidebar-foot {
    margin-top: 22px;
    padding: 12px 14px;
    background: rgba(19, 34, 56, 0.5);
    border: 1px solid #1f2e44;
    border-radius: 8px;
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.55;
}
.sidebar-foot b {color: #cbd5e1;}

.chip-row {display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px 0; align-items: center;}
.chip {
    background: rgba(56, 189, 248, 0.10);
    color: #7dd3fc;
    padding: 4px 11px;
    border-radius: 14px;
    font-size: 12px;
    border: 1px solid rgba(56, 189, 248, 0.28);
    font-weight: 500;
}
.chip-label {
    color: #64748b;
    font-size: 10px;
    margin-right: 4px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
}

.latency-footer {
    color: #64748b;
    font-size: 11px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    margin-top: 12px;
    margin-bottom: 4px;
    padding: 10px 0 6px 0;
    border-top: 1px solid #1f2e44;
}
.latency-stage {margin-right: 14px; display: inline-block;}
.latency-stage b {color: #cbd5e1; font-weight: 600;}

.confidence-badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 600;
    margin: 10px 0 4px 0;
    letter-spacing: 0.3px;
}
.conf-strong {background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3);}
.conf-medium {background: rgba(234, 179, 8, 0.15); color: #fbbf24; border: 1px solid rgba(234, 179, 8, 0.3);}
.conf-weak {background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3);}
</style>
""", unsafe_allow_html=True)


#SESSION-STATE PLUMBING
def _ensure_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = None


def _new_session() -> None:
    sid = db.create_session(title="New chat")
    st.session_state.session_id = sid


def _load_session(sid: str) -> None:
    st.session_state.session_id = sid


def _auto_name_session(sid: str, first_query: str) -> None:
    """Generate a short title for a brand-new session from its first query."""
    try:
        text, _ = llm.complete(
            [{"role": "system", "content": "Generate a 4-5 word descriptive title for this research question. No quotes, no punctuation, no prefix. Just the title."},
             {"role": "user", "content": first_query}],
            temperature=0.3, max_tokens=20, prefer="groq",
        )
        title = text.strip().strip('"').strip("'")[:50] or first_query[:40]
    except Exception:
        title = first_query[:40]
    db.rename_session(sid, title)


def _smart_title(s: dict) -> str:
    raw = s.get("title") or ""
    if raw and raw != "New chat":
        return raw[:42]
    msgs = db.get_messages(s["id"])
    if msgs:
        return msgs[0]["content"][:42]
    return "Untitled"


def _confidence_class(n_citations: int, n_domains: int) -> tuple[str, str]:
    if n_citations == 0:
        return "conf-weak", f"Limited evidence . 0 sources"
    if n_citations >= 3 and n_domains >= 2:
        return "conf-strong", f"Strong evidence . {n_citations} citations . {n_domains} domains"
    if n_citations >= 2:
        return "conf-medium", f"Moderate evidence . {n_citations} citations . {n_domains} domains"
    return "conf-weak", f"Limited evidence . {n_citations} citation . {n_domains} domain"


_ensure_session()

#ensure a session exists on first load so chat_input has a target
if st.session_state.session_id is None:
    _new_session()


#SIDEBAR
with st.sidebar:
    st.markdown("""
    <p class="brand-title">Deep Research</p>
    <p class="brand-tag">Plan . Search . Fetch . Select . Answer</p>
    """, unsafe_allow_html=True)

    if st.button("+ New session", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    sessions = db.list_sessions()
    if sessions:
        st.markdown('<div class="section-label">Sessions</div>', unsafe_allow_html=True)
        labels = {s["id"]: _smart_title(s) for s in sessions}
        ids = list(labels.keys())
        current = st.session_state.session_id
        default_idx = ids.index(current) if current in ids else 0
        picked = st.selectbox(
            label="Sessions",
            label_visibility="collapsed",
            options=ids,
            index=default_idx,
            format_func=lambda i: labels[i],
            key="session_picker",
        )
        if picked != st.session_state.session_id:
            _load_session(picked)
            st.rerun()

    turns_count = len(db.get_turns(st.session_state.session_id))
    st.markdown(f"""
    <div class="sidebar-foot">
    <b>Turns:</b> {turns_count}<br>
    <b>Model:</b> Groq Llama 3.3 70B<br>
    <b>Search:</b> Tavily
    </div>
    """, unsafe_allow_html=True)


#MAIN PANEL
sid: str = st.session_state.session_id

#CHAT INPUT- form so text_input value is batched atomically with the submit click
with st.form(key="query_form", clear_on_submit=True):
    typed = st.text_input(
        "Query",
        placeholder="Ask a research question...",
        label_visibility="collapsed",
        key="query_text",
    )
    submitted = st.form_submit_button("Send", type="primary")
user_input = typed.strip() if submitted and typed and typed.strip() else None

#REPLAY history (always renders, regardless of user_input)
messages = db.get_messages(sid)
turns = db.get_turns(sid)
turn_iter = iter(turns)
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            t = next(turn_iter, None)
            if t:
                if t.get("search_queries"):
                    chips = "".join(f'<span class="chip">{q}</span>' for q in t["search_queries"][:5])
                    st.markdown(f'<div class="chip-row"><span class="chip-label">Searched</span>{chips}</div>',
                                unsafe_allow_html=True)
                cites = t.get("citations", [])
                n_cites = len(cites)
                n_domains = len({c.get("domain", "") for c in cites if c.get("domain")})
                cls, label = _confidence_class(n_cites, n_domains)
                st.markdown(f'<span class="confidence-badge {cls}">{label}</span>',
                            unsafe_allow_html=True)
                if cites:
                    with st.expander(f"Sources ({len(cites)})"):
                        for c in cites:
                            st.markdown(
                                f"**[{c['marker']}] {c['title']}** - *{c['domain']}*  \n"
                                f"<{c['url']}>"
                            )
                stage_lat = t.get("stage_latencies", {}) or {}
                lat_ms = t.get("latency_ms", 0)
                if stage_lat:
                    parts = " . ".join(
                        f'<span class="latency-stage"><b>{name.title()}</b> {dt:.1f}s</span>'
                        for name, dt in stage_lat.items()
                    )
                    if lat_ms:
                        parts += f' . <span class="latency-stage"><b>Total</b> {lat_ms/1000:.1f}s</span>'
                    st.markdown(f'<div class="latency-footer">{parts}</div>',
                                unsafe_allow_html=True)
                elif lat_ms:
                    st.markdown(
                        f'<div class="latency-footer"><span class="latency-stage">'
                        f'<b>Total</b> {lat_ms/1000:.1f}s</span></div>',
                        unsafe_allow_html=True,
                    )


#PROCESS user input (after history is rendered)
if user_input:
    is_first_turn = len(turns) == 0

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        progress = st.status("Researching...", expanded=True)
        chips_box = st.empty()
        answer_box = st.empty()
        badge_box = st.empty()
        sources_box = st.container()
        footer_box = st.empty()

        buffer: list[str] = []
        latest_citations: list[dict] = []

        t_start = time.time()
        stage_times: dict[str, float] = {}
        last_t = t_start

        for ev in agent_loop.run(sid, user_input):
            t = ev.get("type")
            now = time.time()

            if t == "plan":
                stage_times["plan"] = now - last_t
                last_t = now
                progress.write(f"**Plan** - {ev.get('strategy','')}")
                chips = "".join(f'<span class="chip">{q}</span>' for q in ev.get("queries", [])[:5])
                chips_box.markdown(
                    f'<div class="chip-row"><span class="chip-label">Searched</span>{chips}</div>',
                    unsafe_allow_html=True,
                )

            elif t == "search_done":
                stage_times["search"] = now - last_t
                last_t = now
                progress.write(f"**Search** - {len(ev['results'])} unique results")

            elif t == "fetch_done":
                stage_times["fetch"] = now - last_t
                last_t = now
                ok = sum(1 for p in ev["pages"] if p["ok"])
                progress.write(f"**Fetch** - {ok}/{len(ev['pages'])} pages extracted")
                for p in ev["pages"]:
                    mark = "ok" if p["ok"] else "FAIL"
                    progress.write(f"  . {mark} ({p['chars']} chars) - {p['url'][:80]}")

            elif t == "select_done":
                stage_times["select"] = now - last_t
                last_t = now
                progress.write(f"**Select** - {len(ev['snippets'])} snippets")
                for s in ev["snippets"]:
                    progress.write(
                        f"  . [{s['idx']}] score={s['score']} `{s['domain']}` - {s['title'][:60]}"
                    )

            elif t == "answer_token":
                buffer.append(ev["text"])
                answer_box.markdown("".join(buffer))

            elif t == "answer_done":
                stage_times["answer"] = now - last_t
                latest_citations = ev["citations"]
                progress.update(
                    label=f"Done in {ev['latency_ms']/1000:.1f}s",
                    state="complete", expanded=False,
                )

            elif t == "error":
                progress.write(f":red[**Error @ {ev['stage']}**] - {ev['message']}")

        n_cites = len(latest_citations)
        n_domains = len({c.get("domain", "") for c in latest_citations if c.get("domain")})
        cls, label = _confidence_class(n_cites, n_domains)
        badge_box.markdown(f'<span class="confidence-badge {cls}">{label}</span>',
                           unsafe_allow_html=True)

        if latest_citations:
            with sources_box.expander(f"Sources ({len(latest_citations)})", expanded=True):
                for c in latest_citations:
                    st.markdown(
                        f"**[{c['marker']}] {c['title']}** - *{c['domain']}*  \n"
                        f"<{c['url']}>"
                    )

        stage_html = " . ".join(
            f'<span class="latency-stage"><b>{name.title()}</b> {dt:.1f}s</span>'
            for name, dt in stage_times.items()
        )
        footer_box.markdown(f'<div class="latency-footer">{stage_html}</div>',
                            unsafe_allow_html=True)

    if is_first_turn:
        _auto_name_session(sid, user_input)
        st.rerun()
