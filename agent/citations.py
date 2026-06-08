"""POST PROCESS answer to extract citations.
LLM only writes [n] markers- we map those to snippet metadata we gave it.
Out-of-range markers (hallucinations) get stripped here- LLM never types a URL.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from agent.selector import Snippet


@dataclass
class Citation:
    marker: int   #1-based, matches what the LLM wrote: [1]
    url: str
    title: str
    domain: str


_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  #matches [1] / [2,3] / [1, 2, 4]


def extract_citations(answer: str, snippets: list[Snippet]) -> tuple[str, list[Citation]]:
    """Returns (cleaned_answer, citations_used).
    cleaned_answer = answer with hallucinated markers stripped.
    citations_used = deduped, in-order list with URL/title/domain from our snippets.
    """
    used_indices: list[int] = []  #preserves first-mention order
    seen: set[int] = set()        #same snippet cited twice = one entry

    def _normalize_marker(m: re.Match) -> str:
        nums = [int(x.strip()) for x in m.group(1).split(",")]  #split on commas, strip whitespace, to int
        kept = [n for n in nums if 1 <= n <= len(snippets)]     #drop anything out of range (hallucinations)
        for n in kept:
            if n not in seen:   #dedupe via set, append to ordered list
                seen.add(n)
                used_indices.append(n)
        if not kept:
            return ""  #whole marker was hallucinated, strip from text
        return "[" + ",".join(str(n) for n in kept) + "]"  #[1,5,99] -> [1,5]

    cleaned = _MARKER_RE.sub(_normalize_marker, answer)
    cleaned = re.sub(r"  +", " ", cleaned).strip()  #tidy double spaces left by stripped markers

    #BUILD Citation list from kept indices (preserves first-mention order)
    citations: list[Citation] = []
    for n in used_indices:
        s = snippets[n - 1]
        citations.append(Citation(marker=n, url=s.url, title=s.title, domain=s.domain))

    #DEDUPE by URL- same page chunked into multiple snippets shouldnt show twice in Sources panel
    #(marker numbers in the answer text stay as the LLM wrote them- we only dedupe the displayed list)
    seen_urls: set[str] = set()
    deduped: list[Citation] = []
    for c in citations:
        if c.url in seen_urls:
            continue
        seen_urls.add(c.url)
        deduped.append(c)

    return cleaned, deduped


def citations_to_dicts(cs: list[Citation]) -> list[dict]:
    #used by eval + db.save_turn since dataclasses don't json-serialize directly
    return [{"marker": c.marker, "url": c.url, "title": c.title, "domain": c.domain}
            for c in cs]
