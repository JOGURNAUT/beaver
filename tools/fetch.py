import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import trafilatura

from config import FETCH_TIMEOUT_SECONDS


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    domain: str
    retrieved_at: str
    publish_date: str | None = None   #ISO date YYYY-MM-DD if extractable, else None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "retrieved_at": self.retrieved_at,
            "publish_date": self.publish_date,
            "error": self.error,
        }


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/0.1)"
}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def fetch_one(url: str, timeout: int = FETCH_TIMEOUT_SECONDS) -> FetchedPage:
    now = datetime.now(timezone.utc).isoformat()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_HEADERS) as c:
            r = c.get(url)
            r.raise_for_status()
            html = r.text
    except Exception as e:
        return FetchedPage(url=url, title="", text="", domain=_domain(url),
                           retrieved_at=now, error=str(e))

    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "") or ""
    publish_date = (meta.date if meta and meta.date else None)  #ISO YYYY-MM-DD or None
    if not title:
        #very rough fallback
        if "<title>" in html.lower():
            try:
                title = html.split("<title>", 1)[1].split("</title>", 1)[0].strip()[:200]
            except Exception:
                title = url

    return FetchedPage(
        url=url, title=title or url, text=text, domain=_domain(url),
        retrieved_at=now, publish_date=publish_date,
        error=None if text else "no_extractable_text",
    )


def fetch_many(urls: list[str], timeout: int = FETCH_TIMEOUT_SECONDS,
               max_workers: int = 6) -> list[FetchedPage]:
    if not urls:
        return []
    out: list[FetchedPage | None] = [None] * len(urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, u, timeout): i for i, u in enumerate(urls)}
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                out[i] = FetchedPage(url=urls[i], title="", text="", domain=_domain(urls[i]),
                                     retrieved_at=datetime.now(timezone.utc).isoformat(),
                                     error=str(e))
    return [p for p in out if p is not None]
