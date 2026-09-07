"""Gradio UI for the Deep Research Agent.

Alternative to the Streamlit UI- more reliable streaming + chat interface.
Both UIs consume the same agent generator, so agent/loop.py stays UI-agnostic.

Gradio is an optional dependency, deliberately not in requirements.txt- the
deployed container only ever serves the Streamlit UI.

    pip install -r requirements-gradio.txt
    python ui/gradio_app.py        # http://localhost:7860
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

#allow running from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from storage import db
from agent import loop as agent_loop


db.init_db()


def _format_stage_footer(stage_times: dict[str, float], total_ms: int) -> str:
    parts = " . ".join(f"**{name.title()}** {dt:.1f}s" for name, dt in stage_times.items())
    if total_ms:
        parts += f" . **Total** {total_ms/1000:.1f}s"
    return f"\n\n---\n_{parts}_"


def _format_sources(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = ["\n\n---\n**Sources**\n"]
    for c in citations:
        lines.append(f"[{c['marker']}] **{c['title']}** - *{c['domain']}*  \n   {c['url']}\n")
    return "\n".join(lines)


def _format_chips(queries: list[str]) -> str:
    if not queries:
        return ""
    chips = " ".join(f"`{q}`" for q in queries)
    return f"_Searched: {chips}_\n\n"


def _format_confidence(n_cites: int, n_domains: int) -> str:
    if n_cites == 0:
        return "🔴 **Limited evidence** . 0 sources"
    if n_cites >= 3 and n_domains >= 2:
        return f"🟢 **Strong evidence** . {n_cites} citations . {n_domains} domains"
    if n_cites >= 2:
        return f"🟡 **Moderate evidence** . {n_cites} citations . {n_domains} domains"
    return f"🔴 **Limited evidence** . {n_cites} citation . {n_domains} domain"


def chat_fn(message: str, history: list, session_state: dict):
    """Streams agent events as the assistant's answer builds up.

    Yields partial markdown that Gradio renders progressively.
    """
    #ensure we have a session
    sid = session_state.get("sid")
    if not sid:
        sid = db.create_session(title="Gradio chat")
        session_state["sid"] = sid

    chips_block = ""
    answer = ""
    badge = ""
    sources = ""
    footer = ""

    stage_times: dict[str, float] = {}
    last_t = time.time()
    t_start = last_t

    for ev in agent_loop.run(sid, message):
        t = ev.get("type")
        now = time.time()

        if t == "plan":
            stage_times["plan"] = now - last_t
            last_t = now
            chips_block = _format_chips(ev.get("queries", []))
            yield chips_block + "_Researching..._"

        elif t == "search_done":
            stage_times["search"] = now - last_t
            last_t = now
            yield chips_block + f"_Searched {len(ev['results'])} results, fetching pages..._"

        elif t == "fetch_done":
            stage_times["fetch"] = now - last_t
            last_t = now
            ok = sum(1 for p in ev["pages"] if p["ok"])
            yield chips_block + f"_Fetched {ok}/{len(ev['pages'])} pages, selecting evidence..._"

        elif t == "select_done":
            stage_times["select"] = now - last_t
            last_t = now
            yield chips_block + f"_Selected {len(ev['snippets'])} snippets, generating answer..._"

        elif t == "answer_token":
            answer += ev["text"]
            yield chips_block + answer

        elif t == "answer_done":
            stage_times["answer"] = now - last_t
            citations = ev.get("citations", [])
            n_cites = len(citations)
            n_domains = len({c.get("domain", "") for c in citations if c.get("domain")})
            badge = "\n\n" + _format_confidence(n_cites, n_domains)
            sources = _format_sources(citations)
            footer = _format_stage_footer(stage_times, ev.get("latency_ms", 0))
            yield chips_block + answer + badge + sources + footer

        elif t == "error":
            yield chips_block + answer + f"\n\n🔴 _Error at {ev['stage']}: {ev['message']}_"


def new_session(session_state: dict):
    sid = db.create_session(title="Gradio chat")
    session_state["sid"] = sid
    return [], session_state, "_New session started._"


_CUSTOM_CSS = """
/* tighter, more modern layout */
.gradio-container {max-width: 1000px !important; margin: 0 auto;}
footer {display: none !important;}
.show-api {display: none !important;}

/* hide the Textbox label box at top of input */
.gradio-container .label-wrap {display: none !important;}
.gradio-container [data-testid="textbox"] label {display: none !important;}

/* nicer header */
#title-wrap {padding: 14px 0 6px 0; border-bottom: 1px solid #1f2e44; margin-bottom: 18px;}
#title-wrap h1 {
    background: linear-gradient(135deg, #3B82F6 0%, #38BDF8 50%, #F97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    font-size: 26px;
    font-weight: 700;
}
#title-wrap .tagline {
    color: #64748b;
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 4px;
}

/* chat input */
textarea, input[type="text"] {
    border-radius: 12px !important;
    font-size: 14px;
}

/* example chips */
.examples {margin-top: 8px;}
"""

with gr.Blocks(title="Deep Research Agent", css=_CUSTOM_CSS) as demo:
    gr.HTML("""
    <div id="title-wrap">
        <h1>Deep Research Agent</h1>
        <div class="tagline">Plan . Search . Fetch . Select . Answer</div>
    </div>
    """)

    session_state = gr.State({})

    with gr.Row():
        new_btn = gr.Button("+ New session", variant="primary", scale=1)
        status = gr.Markdown("_Ready._")

    chatbot = gr.Chatbot(height=560, show_label=False)

    msg_input = gr.Textbox(
        placeholder="Ask a research question...",
        show_label=False,
        autofocus=True,
        container=False,
    )

    examples = gr.Examples(
        examples=[
            "What is the population of Tokyo?",
            "Who founded Anthropic and what were their roles at OpenAI?",
            "What will OpenAI's revenue be in 2028?",
            "What is the official elevation of Mount Everest?",
        ],
        inputs=msg_input,
    )

    def on_submit(message, chat_history, state):
        #Gradio 6 Chatbot requires messages format: list of {"role", "content"}
        if not message or not message.strip():
            return "", chat_history, state
        chat_history = (chat_history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},
        ]
        for partial in chat_fn(message, chat_history[:-2], state):
            chat_history[-1] = {"role": "assistant", "content": partial}
            yield "", chat_history, state

    msg_input.submit(on_submit, inputs=[msg_input, chatbot, session_state],
                     outputs=[msg_input, chatbot, session_state])

    new_btn.click(new_session, inputs=session_state,
                  outputs=[chatbot, session_state, status])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True,
                theme=gr.themes.Soft(primary_hue="blue"))
