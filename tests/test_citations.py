"""The citation guard is the one place the no-hallucinated-URL guarantee lives.

The model only ever emits [n] markers; URLs are attached afterwards from snippet
metadata. That means the guarantee is only as good as the marker handling, and
every case below is a way it could leak or lose a citation.
"""
from agent.citations import extract_citations
from agent.selector import Snippet


def _snip(url: str, title: str = "t", domain: str = "example.com") -> Snippet:
    return Snippet(text="body", url=url, title=title, domain=domain, score=1.0)


def _snips(n: int) -> list[Snippet]:
    return [_snip(f"https://site{i}.com/a", f"title {i}", f"site{i}.com")
            for i in range(1, n + 1)]


def test_valid_marker_is_kept_and_mapped():
    answer = "The sky is blue [1]."
    cleaned, cites = extract_citations(answer, _snips(2))

    assert cleaned == "The sky is blue [1]."
    assert [c.marker for c in cites] == [1]
    assert cites[0].url == "https://site1.com/a"


def test_out_of_range_marker_is_stripped_entirely():
    # [9] against two snippets is a hallucinated reference. It must not survive
    # into the answer text, because a reader would see a citation that maps to
    # nothing.
    cleaned, cites = extract_citations("Unsupported claim [9].", _snips(2))

    assert "[9]" not in cleaned
    assert cleaned == "Unsupported claim ."
    assert cites == []


def test_partly_hallucinated_marker_keeps_only_the_real_indices():
    cleaned, cites = extract_citations("Mixed [1,9,2].", _snips(2))

    assert cleaned == "Mixed [1,2]."
    assert [c.marker for c in cites] == [1, 2]


def test_marker_with_whitespace_is_parsed():
    cleaned, cites = extract_citations("Spaced [1, 2].", _snips(2))

    assert cleaned == "Spaced [1,2]."
    assert [c.marker for c in cites] == [1, 2]


def test_no_snippets_means_every_marker_is_hallucinated():
    # An answer produced with zero evidence cannot cite anything, even if the
    # model wrote markers anyway.
    cleaned, cites = extract_citations("Claim [1] and [2].", [])

    assert "[" not in cleaned
    assert cites == []


def test_same_snippet_cited_twice_yields_one_citation():
    _, cites = extract_citations("First [1]. Again [1].", _snips(1))

    assert len(cites) == 1


def test_citations_are_in_first_mention_order():
    _, cites = extract_citations("Later [3]. Earlier [1].", _snips(3))

    assert [c.marker for c in cites] == [3, 1]


def test_two_snippets_from_one_page_collapse_in_the_source_list():
    # One page chunked into several snippets should appear once in the sources
    # panel. The markers in the text stay as the model wrote them; only the
    # displayed list is deduped.
    same_page = [
        _snip("https://one.com/x", "chunk a", "one.com"),
        _snip("https://one.com/x", "chunk b", "one.com"),
    ]
    cleaned, cites = extract_citations("Both [1] and [2].", same_page)

    assert cleaned == "Both [1] and [2]."
    assert len(cites) == 1
    assert cites[0].url == "https://one.com/x"


def test_unmarked_answer_is_returned_unchanged():
    cleaned, cites = extract_citations("No citations here.", _snips(2))

    assert cleaned == "No citations here."
    assert cites == []
