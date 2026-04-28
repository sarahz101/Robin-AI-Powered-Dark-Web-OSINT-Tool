"""
search.py
Dark web search engine integration.

Phase 3 additions:
  - Per-engine HTML parsers for the 5 most reliable engines (Ahmia, Tor66,
    OnionLand, Excavator, Find Tor). All other engines fall back to the
    generic anchor-scraping parser that was used for everything before.
  - Keyword pre-scoring: after aggregating results across all engines,
    each result is scored by how many query terms appear in its title and
    URL. Results are sorted descending by score before being handed to the
    LLM filter, so the most relevant candidates always appear first.
"""

import re
import random
import logging
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from constants import USER_AGENTS

import warnings
warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------

SEARCH_ENGINES = [
    {"name": "Ahmia",            "url": "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/?q={query}"},
    {"name": "OnionLand",        "url": "http://3bbad7fauom4d6sgppalyqddsqbf5u5p56b5k5uk2zxsy3d6ey2jobad.onion/search?q={query}"},
    {"name": "Torgle",           "url": "http://iy3544gmoeclh5de6gez2256v6pjh4omhpqdh2wpeeppjtvqmjhkfwad.onion/torgle/?query={query}"},
    {"name": "Amnesia",          "url": "http://amnesia7u5odx5xbwtpnqk3edybgud5bmiagu75bnqx2crntw5kry7ad.onion/search?query={query}"},
    {"name": "Kaizer",           "url": "http://kaizerwfvp5gxu6cppibp7jhcqptavq3iqef66wbxenh6a2fklibdvid.onion/search?q={query}"},
    {"name": "Anima",            "url": "http://anima4ffe27xmakwnseih3ic2y7y3l6e7fucwk4oerdn4odf7k74tbid.onion/search?q={query}"},
    {"name": "Tornado",          "url": "http://tornadoxn3viscgz647shlysdy7ea5zqzwda7hierekeuokh5eh5b3qd.onion/search?q={query}"},
    {"name": "TorNet",           "url": "http://tornetupfu7gcgidt33ftnungxzyfq2pygui5qdoyss34xbgx2qruzid.onion/search?q={query}"},
    {"name": "Torland",          "url": "http://torlbmqwtudkorme6prgfpmsnile7ug2zm4u3ejpcncxuhpu4k2j4kyd.onion/index.php?a=search&q={query}"},
    {"name": "Find Tor",         "url": "http://findtorroveq5wdnipkaojfpqulxnkhblymc7aramjzajcvpptd4rjqd.onion/search?q={query}"},
    {"name": "Excavator",        "url": "http://2fd6cemt4gmccflhm6imvdfvli3nf7zn6rfrwpsy7uhxrgbypvwf5fad.onion/search?query={query}"},
    {"name": "Onionway",         "url": "http://oniwayzz74cv2puhsgx4dpjwieww4wdphsydqvf5q7eyz4myjvyw26ad.onion/search.php?s={query}"},
    {"name": "Tor66",            "url": "http://tor66sewebgixwhcqfnp5inzp5x5uohhdy3kvtnyfxc2e5mxiuh34iid.onion/search?q={query}"},
    {"name": "OSS",              "url": "http://3fzh7yuupdfyjhwt3ugzqqof6ulbcl27ecev33knxe3u7goi3vfn2qqd.onion/oss/index.php?search={query}"},
    {"name": "Torgol",           "url": "http://torgolnpeouim56dykfob6jh5r2ps2j73enc42s2um4ufob3ny4fcdyd.onion/?q={query}"},
    {"name": "The Deep Searches","url": "http://searchgf7gdtauh7bhnbyed4ivxqmuoat3nm6zfrg3ymkq6mtnpye3ad.onion/search?q={query}"},
]

DEFAULT_SEARCH_ENGINES = [e["url"] for e in SEARCH_ENGINES]

# Onion URL pattern reused across parsers
_ONION_URL_RE = re.compile(r'https?://[a-z2-7]{16,56}\.onion[^\s"\'<>]*')


# ---------------------------------------------------------------------------
# Tor session factory
# ---------------------------------------------------------------------------

def get_tor_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3, read=3, connect=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.proxies = {
        "http":  "socks5h://127.0.0.1:9050",
        "https": "socks5h://127.0.0.1:9050",
    }
    return session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_useful_result(link: str, title: str) -> bool:
    """Basic quality gate applied by every parser."""
    if not link or not title:
        return False
    if len(title) <= 3:
        return False
    if "search" in link.lower():
        return False
    return True


def _extract_onion_href(href: str) -> str | None:
    """Return the first .onion URL found in an href string, or None."""
    match = _ONION_URL_RE.search(href)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# Per-engine parsers
# Each function receives a BeautifulSoup object and returns
# a list of {"title": str, "link": str} dicts.
# ---------------------------------------------------------------------------

def _parse_ahmia(soup: BeautifulSoup) -> list:
    """
    Ahmia result structure:
      <li class="result">
        <h4><a href="https://...onion/...">Title text</a></h4>
        <p class="lead">Description</p>
        ...
      </li>
    """
    results = []
    for li in soup.select("li.result"):
        a = li.select_one("h4 a") or li.select_one("a")
        if not a:
            continue
        href = a.get("href", "")
        title = a.get_text(strip=True)
        link = _extract_onion_href(href) or _extract_onion_href(str(a))
        if link and _is_useful_result(link, title):
            results.append({"title": title, "link": link})
    return results


def _parse_tor66(soup: BeautifulSoup) -> list:
    """
    Tor66 result structure:
      <div class="result-block"> or <div class="r">
        <a href="http://...onion/...">Title</a>
        <span class="url">...</span>
      </div>
    Each result <a> has the onion URL directly in href.
    """
    results = []
    # Try structured result containers first, fall back to all anchors.
    containers = soup.select("div.result-block, div.r, div.result")
    targets = containers if containers else [soup]
    for container in targets:
        for a in container.find_all("a", href=True):
            href = a["href"]
            link = _extract_onion_href(href)
            title = a.get_text(strip=True)
            if link and _is_useful_result(link, title):
                results.append({"title": title, "link": link})
    return results


def _parse_onionland(soup: BeautifulSoup) -> list:
    """
    OnionLand result structure:
      <div class="g">
        <div class="r"><a href="http://...onion/...">Title</a></div>
        <div class="s"><span class="st">Snippet</span></div>
      </div>
    """
    results = []
    for div in soup.select("div.g"):
        a = div.select_one("div.r a") or div.select_one("a")
        if not a:
            continue
        href = a.get("href", "")
        title = a.get_text(strip=True)
        link = _extract_onion_href(href)
        if link and _is_useful_result(link, title):
            results.append({"title": title, "link": link})
    return results


def _parse_excavator(soup: BeautifulSoup) -> list:
    """
    Excavator result structure:
      <div class="search-result">
        <a class="result-title" href="http://...onion/...">Title</a>
        <p class="result-snippet">...</p>
      </div>
    Falls back to any anchor with an onion href if classes differ.
    """
    results = []
    for div in soup.select("div.search-result, div.result"):
        a = div.select_one("a.result-title") or div.select_one("a[href]")
        if not a:
            continue
        href = a.get("href", "")
        title = a.get_text(strip=True)
        link = _extract_onion_href(href)
        if link and _is_useful_result(link, title):
            results.append({"title": title, "link": link})
    return results


def _parse_findtor(soup: BeautifulSoup) -> list:
    """
    Find Tor result structure:
      <div class="site">
        <a href="http://...onion">Title</a>
        <p>Description</p>
      </div>
    """
    results = []
    for div in soup.select("div.site, div.result, article"):
        a = div.select_one("a[href]")
        if not a:
            continue
        href = a.get("href", "")
        title = a.get_text(strip=True)
        link = _extract_onion_href(href)
        if link and _is_useful_result(link, title):
            results.append({"title": title, "link": link})
    return results


def _parse_generic(soup: BeautifulSoup) -> list:
    """
    Generic fallback: scan all <a> tags for onion URLs.
    Used for every engine that doesn't have a dedicated parser.
    """
    results = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)
        link = _extract_onion_href(href)
        if link and _is_useful_result(link, title):
            results.append({"title": title, "link": link})
    return results


# Map engine name → parser function.
# Engines absent from this dict automatically use _parse_generic.
_ENGINE_PARSERS = {
    "Ahmia":    _parse_ahmia,
    "Tor66":    _parse_tor66,
    "OnionLand": _parse_onionland,
    "Excavator": _parse_excavator,
    "Find Tor": _parse_findtor,
}


# ---------------------------------------------------------------------------
# Pre-scoring
# ---------------------------------------------------------------------------

def _score_result(result: dict, query_terms: list[str]) -> int:
    """
    Score a result by how many distinct query terms appear in its title + URL.
    Returns an integer count (higher = more relevant).
    """
    haystack = (result.get("title", "") + " " + result.get("link", "")).lower()
    return sum(1 for term in query_terms if term in haystack)


def score_and_sort(results: list, query: str) -> list:
    """
    Attach a relevance score to each result and return the list sorted
    highest-score-first.  Results with identical scores preserve their
    original relative order (stable sort).

    This runs before the LLM filter step so the model sees the strongest
    candidates at the top of every batch.
    """
    terms = [t.lower() for t in re.split(r"\W+", query) if len(t) > 2]
    if not terms:
        return results

    scored = [
        {**r, "_score": _score_result(r, terms)}
        for r in results
    ]
    scored.sort(key=lambda r: r["_score"], reverse=True)

    # Strip the internal score key before returning.
    for r in scored:
        r.pop("_score", None)
    return scored


# ---------------------------------------------------------------------------
# Fetch + parse for a single engine
# ---------------------------------------------------------------------------

def fetch_search_results(endpoint: str, query: str, engine_name: str = "") -> list:
    """
    Fetch results from one search engine endpoint and parse them.

    Uses the dedicated parser for the engine if one exists, otherwise
    falls back to the generic anchor-scraping parser.
    """
    url = endpoint.format(query=query)
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    session = get_tor_session()
    parser_fn = _ENGINE_PARSERS.get(engine_name, _parse_generic)

    try:
        response = session.get(url, headers=headers, timeout=40)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        results = parser_fn(soup)
        _logger.debug("[%s] %d results parsed.", engine_name or endpoint, len(results))
        return results
    except Exception as exc:
        _logger.debug("[%s] fetch failed: %s", engine_name or endpoint, exc)
        return []


# ---------------------------------------------------------------------------
# Aggregate across all engines
# ---------------------------------------------------------------------------

def get_search_results(refined_query: str, max_workers: int = 5) -> list:
    """
    Query all registered search engines concurrently, deduplicate the
    combined results, apply keyword pre-scoring, and return sorted by
    relevance (highest first).
    """
    # Build (endpoint, engine_name) pairs so fetch_search_results knows
    # which parser to use.
    engine_map = {e["url"]: e["name"] for e in SEARCH_ENGINES}

    raw_results: list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_search_results, endpoint, refined_query, engine_map[endpoint]
            ): endpoint
            for endpoint in DEFAULT_SEARCH_ENGINES
        }
        for future in as_completed(futures):
            raw_results.extend(future.result())

    # Deduplicate by normalised link (strip trailing slash).
    seen_links: set = set()
    unique_results: list = []
    for res in raw_results:
        clean_link = res.get("link", "").rstrip("/")
        if clean_link and clean_link not in seen_links:
            seen_links.add(clean_link)
            unique_results.append(res)

    # Pre-score and sort before handing off to the LLM filter.
    sorted_results = score_and_sort(unique_results, refined_query)
    _logger.info(
        "get_search_results: %d unique results (from %d raw) sorted by keyword score.",
        len(sorted_results), len(raw_results),
    )
    return sorted_results