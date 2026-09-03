"""Unified LLM client with Groq primary, Gemini fallback.

Both providers wrapped behind the same two methods:
  - complete(messages, **kw) -> str         (blocking, full text)
  - stream(messages, **kw)   -> Iterator[str]  (token chunks)

Why this shape: the agent loop should not know which provider answered.
Lets us demo provider-fallback in interview without touching call sites.
"""
from __future__ import annotations
import logging
import time
from typing import Iterator

from groq import Groq
from google import genai
from google.genai import types as gtypes

from config import GROQ_API_KEY, GEMINI_API_KEY, GROQ_MODEL, GEMINI_MODEL


log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


_groq: Groq | None = None
_gemini: genai.Client | None = None


def _groq_client() -> Groq:
    global _groq
    if _groq is None:
        if not GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY missing")
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def _gemini_client() -> genai.Client:
    global _gemini
    if _gemini is None:
        if not GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY missing")
        _gemini = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini


def _msgs_to_gemini(messages: list[dict]) -> tuple[str, list[gtypes.Content]]:
    """Gemini wants system as a separate arg and user/assistant as contents."""
    system = ""
    contents: list[gtypes.Content] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system += m["content"] + "\n"
            continue
        gemini_role = "user" if role == "user" else "model"
        contents.append(gtypes.Content(role=gemini_role,
                                       parts=[gtypes.Part.from_text(text=m["content"])]))
    return system.strip(), contents


def complete(messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 1024, prefer: str = "groq") -> tuple[str, str]:
    """Returns (text, provider_used)."""
    order = ["groq", "gemini"] if prefer == "groq" else ["gemini", "groq"]
    last_err = None
    for attempt, provider in enumerate(order):
        try:
            if provider == "groq":
                resp = _groq_client().chat.completions.create(
                    model=GROQ_MODEL, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                text, model = resp.choices[0].message.content or "", GROQ_MODEL
            else:
                system, contents = _msgs_to_gemini(messages)
                resp = _gemini_client().models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system or None,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                text, model = (resp.text or ""), GEMINI_MODEL

            if attempt > 0:
                log.warning("LLM fallback: %s answered after %s failed",
                            provider, order[0])
            if not text.strip():
                # Reasoning models spend part of max_tokens on thinking before
                # emitting text, so a budget that is too small comes back empty
                # with no error at all.
                log.warning("LLM empty response from %s (model=%s, "
                            "max_tokens=%d)", provider, model, max_tokens)
            return text, provider
        except Exception as e:
            log.warning("LLM provider %s failed: %s: %s",
                        provider, type(e).__name__, e)
            last_err = e
            time.sleep(0.5)
            continue
    log.error("LLM all providers failed; last error: %s", last_err)
    raise LLMError(f"All providers failed; last error: {last_err}")


def stream(messages: list[dict], temperature: float = 0.2,
           max_tokens: int = 1024, prefer: str = "groq") -> Iterator[str]:
    """Yields text chunks. Falls back to the other provider if the primary
    raises OR returns zero chunks (some free-tier APIs silently drop the
    streaming response on overload)."""
    order = ["groq", "gemini"] if prefer == "groq" else ["gemini", "groq"]
    last_err = None
    for attempt, provider in enumerate(order):
        try:
            yielded_any = False
            if provider == "groq":
                resp = _groq_client().chat.completions.create(
                    model=GROQ_MODEL, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, stream=True,
                )
                for chunk in resp:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yielded_any = True
                        yield delta
            else:
                system, contents = _msgs_to_gemini(messages)
                resp_iter = _gemini_client().models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system or None,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                for chunk in resp_iter:
                    if chunk.text:
                        yielded_any = True
                        yield chunk.text
            if yielded_any:
                if attempt > 0:
                    log.warning("LLM stream fallback: %s answered after %s failed",
                                provider, order[0])
                return
            #zero chunks AND no exception- treat as silent failure, try next provider
            log.warning("LLM stream from %s returned 0 chunks (no error raised)",
                        provider)
            last_err = RuntimeError(f"{provider} stream returned 0 chunks")
        except Exception as e:
            log.warning("LLM stream provider %s failed: %s: %s",
                        provider, type(e).__name__, e)
            last_err = e
            continue
    log.error("LLM stream all providers failed; last error: %s", last_err)
    raise LLMError(f"All providers failed; last error: {last_err}")
