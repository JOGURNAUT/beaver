"""Tests for the SELECT stage's pure parts.

The embedder and the tokenizer are both stubbed out, so nothing here downloads a
model or touches the network. That is deliberate: these are the functions whose
boundary conditions are easy to get wrong, and they should be checkable in a
second.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from agent import selector
from tools.fetch import FetchedPage


# ---------------------------------------------------------------- chunking

def test_short_paragraphs_are_dropped():
    # Anything under 100 chars is treated as boilerplate that survived
    # extraction (nav text, captions, cookie notices).
    text = "Too short.\n\nAlso short."
    assert selector._chunk_text(text) == []


def test_paragraph_within_the_limit_is_kept_whole():
    para = "a" * 400
    assert selector._chunk_text(para, limit=1200) == [para]


def test_long_paragraph_is_split_on_sentence_boundaries():
    sentence = "This sentence is long enough to matter for the packing test. "
    para = (sentence * 6).strip()          # ~360 chars, over the limit below
    chunks = selector._chunk_text(para, limit=120)

    assert len(chunks) > 1
    # every chunk respects the limit, and nothing is lost
    assert all(len(c) <= 120 for c in chunks)
    assert "".join(chunks).replace(" ", "") == para.replace(" ", "")


def test_sentences_are_packed_rather_than_emitted_one_at_a_time():
    # Two short sentences that both fit inside the limit should share a chunk.
    para = ("First sentence here padded out to clear the hundred character "
            "minimum for a paragraph. Second sentence follows it.")
    chunks = selector._chunk_text(para, limit=1200)

    assert len(chunks) == 1


# ---------------------------------------------------------------- recency

def test_missing_publish_date_is_neutral():
    assert selector._recency_factor(None) == 0.5


def test_unparseable_publish_date_is_neutral():
    assert selector._recency_factor("not-a-date") == 0.5


def test_today_scores_full_recency():
    today = datetime.now().strftime("%Y-%m-%d")
    assert selector._recency_factor(today) == pytest.approx(1.0, abs=0.01)


def test_recency_decays_linearly_over_two_years():
    one_year = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    assert selector._recency_factor(one_year) == pytest.approx(0.5, abs=0.01)


def test_very_old_content_hits_the_floor_rather_than_zero():
    ancient = (datetime.now() - timedelta(days=3000)).strftime("%Y-%m-%d")
    assert selector._recency_factor(ancient) == 0.3


def test_a_future_date_is_clamped_to_full_recency():
    future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    assert selector._recency_factor(future) == 1.0


# ---------------------------------------------------------------- cosine

def test_identical_vectors_score_one():
    v = np.array([1.0, 2.0, 3.0])
    assert selector._cosine(v, np.array([v]))[0] == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    a = np.array([1.0, 0.0])
    b = np.array([[0.0, 1.0]])
    assert selector._cosine(a, b)[0] == pytest.approx(0.0)


def test_a_zero_vector_does_not_divide_by_zero():
    # The +1e-9 in the norm exists for this case; an empty embedding should
    # score, not raise.
    a = np.array([0.0, 0.0])
    assert selector._cosine(a, np.array([[1.0, 1.0]]))[0] == pytest.approx(0.0)


# ---------------------------------------------------------------- selection

@pytest.fixture
def stub_scoring(monkeypatch):
    """Score snippets by position: the first page's chunks rank highest.

    Removes the embedding model and the tokenizer from these tests entirely, so
    what is being checked is the budget-and-diversity walk, not the maths.
    """
    class _FakeModel:
        def encode(self, texts, **kw):
            # query -> [1, 0]; snippets -> descending similarity to it
            if len(texts) == 1:
                return np.array([[1.0, 0.0]])
            n = len(texts)
            return np.array([[1.0 - i / (n + 1), i / (n + 1)] for i in range(n)])

    monkeypatch.setattr(selector, "_embedder", lambda: _FakeModel())
    monkeypatch.setattr(selector, "count_tokens", lambda t: len(t) // 4)


def _page(domain: str, n_chunks: int = 1, chunk_chars: int = 200) -> FetchedPage:
    body = "\n\n".join(["x" * chunk_chars for _ in range(n_chunks)])
    return FetchedPage(url=f"https://{domain}/a", title=domain, text=body,
                       domain=domain, retrieved_at="2026-01-01")


def test_no_usable_pages_yields_no_snippets(stub_scoring):
    assert selector.select([], "q") == []


def test_pages_that_failed_to_fetch_are_ignored(stub_scoring):
    broken = FetchedPage(url="https://x.com/a", title="", text="", domain="x.com",
                         retrieved_at="2026-01-01", error="timeout")
    assert selector.select([broken], "q") == []


def test_domain_cap_limits_how_much_one_source_can_contribute(stub_scoring):
    # Five chunks from one page, all scoring above anything else, but the cap
    # says two. This is what makes multi-source diversity possible at all.
    pages = [_page("dominant.com", n_chunks=5), _page("other.com", n_chunks=2)]
    chosen = selector.select(pages, "q", max_snippets=6, max_per_domain=2)

    per_domain = {}
    for s in chosen:
        per_domain[s.domain] = per_domain.get(s.domain, 0) + 1
    assert per_domain["dominant.com"] == 2


def test_snippet_quota_is_respected(stub_scoring):
    pages = [_page(f"d{i}.com", n_chunks=2) for i in range(5)]
    chosen = selector.select(pages, "q", max_snippets=3, max_per_domain=2)

    assert len(chosen) == 3


def test_an_oversized_snippet_is_skipped_but_later_ones_still_fit(stub_scoring):
    """The budget walk uses `continue`, not `break`, and this is why.

    The highest-scoring chunk is too large for the remaining budget. Breaking
    there would throw away every cheaper chunk below it and return almost
    nothing; continuing keeps filling the budget with what does fit.
    """
    huge = _page("huge.com", n_chunks=1, chunk_chars=4000)     # ~1000 tokens
    small = _page("small.com", n_chunks=1, chunk_chars=200)    # ~50 tokens

    chosen = selector.select([huge, small], "q", max_snippets=5, token_budget=100)

    assert [s.domain for s in chosen] == ["small.com"]
