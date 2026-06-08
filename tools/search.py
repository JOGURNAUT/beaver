from dataclasses import dataclass
from tavily import TavilyClient

from config import TAVILY_API_KEY, SEARCH_RESULTS_PER_QUERY


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
        }


_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY missing in .env")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def search(query: str, max_results: int = SEARCH_RESULTS_PER_QUERY) -> list[SearchResult]:
    client = _get_client()
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",  # 'advanced' costs more credits, basic is fine for snippets
    )
    out: list[SearchResult] = []
    for r in resp.get("results", []):
        out.append(SearchResult(
            title=r.get("title", "") or r.get("url", ""),
            url=r.get("url", ""),
            snippet=r.get("content", "") or "",
            score=r.get("score"),
        ))
    return out
