"""
crawler.py
Deep Crawl engine for Robin — replaces loader.py + html_txt.py + automated_loader.py.

Two-tier crawl strategy
-----------------------
Tier 1 (preferred): Selenium + Tor Browser binary
  Full JS rendering, screenshot, human-like interaction, CAPTCHA detection.
  Requires: Tor Browser installed + geckodriver in PATH.
  Configured via env var TORBROWSER_BINARY (path to Browser/firefox binary).

Tier 2 (fallback): requests + SOCKS5 Tor proxy
  Uses Robin's existing scrape.py infrastructure — no extra deps needed.
  Activated automatically when Tier 1 prerequisites are absent.

Output
------
Crawled content is returned as a plain-text string (title + visible text)
that plugs directly into Robin's existing filter_results → generate_summary
pipeline as if it were scraped content.

Optionally, raw HTML is saved to investigations/crawled/<url_hash>/ for audit.
"""

import os
import re
import time
import random
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from scrape import get_tor_session

import warnings
warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CRAWL_OUTPUT_DIR   = Path("investigations") / "crawled"
MAX_CONTENT_CHARS  = 50_000   # chars kept from raw HTML before text extraction
MAX_RETURN_CHARS   = 8_000    # chars returned as plain text per page
PAGE_LOAD_TIMEOUT  = 60       # seconds (both tiers); first-run circuits + slow
                              # .onion sites routinely take 30-50s, so 45 was
                              # tight and timed out on borderline cases.
POST_LOAD_SLEEP    = 3        # seconds after JS load (Tier 1 only)
RETRIES            = 2
RETRY_DELAY        = 5        # seconds between retries
MAX_WORKERS        = 5        # concurrent crawl threads

# Tor Browser binary — override with TORBROWSER_BINARY env var
_TORBROWSER_BINARY = os.getenv("TORBROWSER_BINARY", "")

# Tor SOCKS ports to probe (in order)
_TOR_SOCKS_PORTS   = (9050, 9150)
_TOR_SOCKS_HOST    = "127.0.0.1"

# CAPTCHA / block detection keywords
_BLOCK_KEYWORDS    = [
    "captcha", "are you human", "please enable javascript",
    "access denied", "verify you are human", "cloudflare",
    "ddos protection", "ray id",
]


# ---------------------------------------------------------------------------
# Tier detection
# ---------------------------------------------------------------------------

def _selenium_available() -> bool:
    """Return True if selenium package is importable."""
    try:
        import selenium  # noqa: F401
        return True
    except ImportError:
        return False


def _geckodriver_in_path() -> bool:
    """Return True if geckodriver binary is findable on PATH."""
    import shutil
    return shutil.which("geckodriver") is not None


def _torbrowser_binary() -> str:
    """Return the Tor Browser Firefox binary path if it exists, else empty string."""
    path = _TORBROWSER_BINARY.strip()
    if path and Path(path).exists():
        return path
    # Common Linux default
    default = Path.home() / ".local/share/torbrowser/tbb/x86_64/tor-browser/Browser/firefox"
    if default.exists():
        return str(default)
    return ""


def probe_tier() -> str:
    """
    Probe which crawl tier is available. Returns 'selenium' or 'requests'.

    Selenium tier requires only:
      - the `selenium` Python package
      - `geckodriver` somewhere on PATH

    The Tor Browser binary is *preferred* (its anti-fingerprinting hardening
    is stronger than stock Firefox), but is NOT required. When it's missing,
    `_crawl_selenium` falls back to whichever Firefox geckodriver discovers
    on the system and routes its traffic through the local Tor SOCKS5 proxy
    (the same `network.proxy.socks` preferences are set unconditionally).
    For OSINT scraping the two paths are functionally equivalent — both go
    out via Tor, both render JS, both can solve light CAPTCHA.

    This relaxed check is what lets the bundled Docker image use selenium
    out of the box, since shipping the full Tor Browser bundle inside the
    image adds ~150 MB and license/redistribution friction.
    """
    if _selenium_available() and _geckodriver_in_path():
        return "selenium"
    return "requests"


def selenium_subtier() -> str:
    """
    Return a human-readable label for which selenium variant will run:
      - 'tor-browser'   : Tor Browser binary detected, used as-is
      - 'firefox+socks' : stock Firefox routed through 127.0.0.1:9050
      - 'unavailable'   : selenium tier won't even start
    Used by the UI so it can display the truth instead of always claiming
    "Tor Browser".
    """
    if not (_selenium_available() and _geckodriver_in_path()):
        return "unavailable"
    return "tor-browser" if _torbrowser_binary() else "firefox+socks"


def _find_socks_port() -> Optional[int]:
    """Return the first reachable Tor SOCKS port, or None."""
    import socket
    for port in _TOR_SOCKS_PORTS:
        try:
            with socket.create_connection((_TOR_SOCKS_HOST, port), timeout=2):
                return port
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# HTML → plain text (shared by both tiers)
# ---------------------------------------------------------------------------

def _html_to_text(html: str, title_hint: str = "") -> str:
    """Extract clean visible text from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "head", "meta", "link"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    text = text[:MAX_CONTENT_CHARS]
    prefix = f"{title_hint} — " if title_hint else ""
    return (prefix + text)[:MAX_RETURN_CHARS]


def _is_blocked(html: str) -> bool:
    lower = html.lower()
    return any(kw in lower for kw in _BLOCK_KEYWORDS)


def _save_html(url_hash: str, html: str, tier: str) -> None:
    """Optionally persist raw HTML for audit purposes."""
    try:
        out = CRAWL_OUTPUT_DIR / url_hash
        out.mkdir(parents=True, exist_ok=True)
        (out / "rendered.html").write_text(html, encoding="utf-8", errors="replace")
        (out / "tier.txt").write_text(tier)
    except Exception as exc:
        _logger.debug("Could not save HTML for %s: %s", url_hash, exc)


# ---------------------------------------------------------------------------
# Tier 1: Selenium + Tor Browser
# ---------------------------------------------------------------------------

def _crawl_selenium(url: str, url_hash: str) -> str:
    """
    Full JS-rendered crawl using Selenium + Tor Browser.
    Returns plain text, or raises on failure.
    """
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import TimeoutException, WebDriverException

    binary = _torbrowser_binary()
    socks_port = _find_socks_port()
    if not socks_port:
        raise RuntimeError("No Tor SOCKS port reachable — is Tor running?")

    opts = Options()
    opts.add_argument("--headless")
    if binary:
        opts.binary_location = binary
        _logger.info("[DeepCrawl/selenium] Tor Browser binary in use: %s", binary)
    else:
        _logger.info(
            "[DeepCrawl/selenium] Tor Browser not detected — using stock "
            "Firefox routed through 127.0.0.1:%d (Tor SOCKS).", socks_port,
        )

    # Route all traffic through Tor SOCKS
    opts.set_preference("network.proxy.type", 1)
    opts.set_preference("network.proxy.socks", _TOR_SOCKS_HOST)
    opts.set_preference("network.proxy.socks_port", int(socks_port))
    opts.set_preference("network.proxy.socks_version", 5)
    opts.set_preference("network.proxy.socks_remote_dns", True)

    # Privacy / performance tweaks
    opts.set_preference("dom.webnotifications.enabled", False)
    opts.set_preference("browser.cache.disk.enable", False)
    opts.set_preference("browser.privatebrowsing.autostart", True)
    opts.set_preference("toolkit.telemetry.enabled", False)
    opts.set_preference("datareporting.healthreport.uploadEnabled", False)

    driver = None
    try:
        service = FirefoxService()
        driver = webdriver.Firefox(service=service, options=opts)
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

        driver.get(url)

        # Minimal human-like interaction
        time.sleep(random.uniform(0.5, 1.5))
        try:
            ActionChains(driver).move_by_offset(
                random.randint(10, 150), random.randint(10, 150)
            ).perform()
        except Exception:
            pass

        # Scroll to trigger lazy-load content
        try:
            height = driver.execute_script(
                "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"
            )
            for y in range(0, min(height, 3000), random.randint(300, 700)):
                driver.execute_script("window.scrollTo(0, arguments[0]);", y)
                time.sleep(random.uniform(0.1, 0.4))
        except Exception:
            pass

        time.sleep(POST_LOAD_SLEEP + random.uniform(0, 1.5))

        html = driver.page_source or ""

        if _is_blocked(html):
            raise RuntimeError("CAPTCHA/block page detected.")

        _save_html(url_hash, html, tier="selenium")

        title = ""
        try:
            title = driver.title
        except Exception:
            pass

        return _html_to_text(html, title_hint=title)

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tier 2: requests + SOCKS5 (fallback)
# ---------------------------------------------------------------------------

def _crawl_requests(url: str, url_hash: str, title_hint: str = "") -> str:
    """
    Plain HTTP crawl through the existing Tor session from scrape.py.
    Returns plain text, or raises on failure.
    """
    from constants import USER_AGENTS
    session = get_tor_session()
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    resp = session.get(url, headers=headers, timeout=(10, PAGE_LOAD_TIMEOUT), stream=True)
    resp.raise_for_status()

    # Read up to MAX_CONTENT_CHARS worth of bytes
    chunks, total = [], 0
    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        total += len(chunk)
        chunks.append(chunk)
        if total >= MAX_CONTENT_CHARS * 3:   # rough byte budget
            break

    html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")

    if _is_blocked(html):
        raise RuntimeError("CAPTCHA/block page detected.")

    _save_html(url_hash, html, tier="requests")

    # Extract title from HTML
    title = title_hint
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        if t and t.get_text(strip=True):
            title = t.get_text(strip=True)
    except Exception:
        pass

    return _html_to_text(html, title_hint=title)


# ---------------------------------------------------------------------------
# Single-URL crawl dispatcher (with retry + tier fallback)
# ---------------------------------------------------------------------------

def crawl_url(url: str, title_hint: str = "", tier: Optional[str] = None) -> dict:
    """
    Crawl a single .onion (or clearweb) URL and return a result dict:
      {
        url:     str,
        text:    str,   # clean plain text for LLM consumption
        tier:    str,   # 'selenium' or 'requests'
        success: bool,
        error:   str or None,
      }

    Args:
        url:        Target URL.
        title_hint: Optional page title (used if we can't extract one).
        tier:       Force 'selenium' or 'requests'. Auto-detected if None.
    """
    if tier is None:
        tier = probe_tier()

    url_hash = hashlib.sha256(url.encode()).hexdigest()
    result   = {"url": url, "text": "", "tier": tier, "success": False, "error": None}

    for attempt in range(1, RETRIES + 1):
        try:
            if tier == "selenium":
                result["text"] = _crawl_selenium(url, url_hash)
            else:
                result["text"] = _crawl_requests(url, url_hash, title_hint=title_hint)

            result["success"] = True
            result["tier"]    = tier
            _logger.info("[DeepCrawl] ✓ %s via %s (%d chars)", url, tier, len(result["text"]))
            return result

        except Exception as exc:
            _logger.warning(
                "[DeepCrawl] attempt %d/%d failed for %s (%s): %s",
                attempt, RETRIES, url, tier, exc,
            )
            result["error"] = str(exc)

            # If Selenium failed, fall back to requests on next attempt
            if tier == "selenium":
                _logger.info("[DeepCrawl] Falling back to requests tier for %s", url)
                tier = "requests"
                result["tier"] = tier

            if attempt < RETRIES:
                time.sleep(RETRY_DELAY + random.uniform(0, 2))

    return result


# ---------------------------------------------------------------------------
# Batch crawl (used by UI for "crawl all sources")
# ---------------------------------------------------------------------------

def crawl_sources(
    sources: list,
    max_workers: int = MAX_WORKERS,
    tier: Optional[str] = None,
    progress_callback=None,
) -> dict:
    """
    Concurrently deep-crawl a list of source dicts ({"link":..., "title":...}).
    Returns a dict mapping URL → plain text (mirrors scrape.scrape_multiple's
    return format so it can drop straight into generate_summary).

    Args:
        sources:           List of {"link": url, "title": title_hint} dicts.
        max_workers:       Thread pool size.
        tier:              Force crawl tier or auto-detect.
        progress_callback: Optional callable(completed_count, total) for UI updates.

    Returns:
        {url: text_content, ...}  — only successful crawls are included.
    """
    if tier is None:
        tier = probe_tier()

    results    = {}
    total      = len(sources)
    completed  = {"n": 0}
    lock       = threading.Lock()

    def _task(src):
        url   = src.get("link", "").strip()
        title = src.get("title", "")
        if not url:
            return url, None
        res = crawl_url(url, title_hint=title, tier=tier)
        with lock:
            completed["n"] += 1
            if progress_callback:
                try:
                    progress_callback(completed["n"], total)
                except Exception:
                    pass
        return url, res["text"] if res["success"] else None

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 10))) as pool:
        futures = {pool.submit(_task, src): src for src in sources}
        for future in as_completed(futures):
            try:
                url, text = future.result()
                if url and text:
                    results[url] = text
            except Exception as exc:
                _logger.debug("[DeepCrawl] Worker exception: %s", exc)

    _logger.info(
        "[DeepCrawl] Batch complete: %d/%d succeeded (tier=%s).",
        len(results), total, tier,
    )
    return results