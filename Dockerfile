# ─────────────────────────────────────────────────────────────────────────────
# Robin — AI-Powered Dark Web OSINT Tool
#
# This Dockerfile produces a fully self-contained image. After build, the
# container ships with:
#   • Tor daemon          — SOCKS5 proxy on 127.0.0.1:9050 (started by entrypoint)
#   • Firefox-ESR         — JS-rendering browser for the Selenium deep-crawl tier
#   • geckodriver         — WebDriver bridge between Selenium and Firefox
#   • All Python deps     — installed from requirements.txt
#   • Common system fonts — needed for non-blank rendering of CJK / emoji pages
#
# Zero host-side setup is required besides Docker itself: the Selenium
# "Tier 1" deep-crawl works out of the box, routed through the embedded
# Tor daemon. Tor Browser proper is NOT bundled (≈150 MB extra and
# license-redistribution friction); Robin's crawler.py automatically uses
# stock Firefox + the local Tor SOCKS proxy when the Tor Browser binary
# isn't found, which is functionally equivalent for OSINT scraping.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Pin geckodriver — keep aligned with whichever firefox-esr major version
# the upstream Debian base ships. 0.36.0 supports Firefox >= 102 (covers
# the 115 / 128 ESR lines currently in Debian 12 bookworm-slim).
ARG GECKODRIVER_VERSION=0.36.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOZ_HEADLESS=1

# ── System packages ───────────────────────────────────────────────────────
# Grouped by purpose so the layer is readable. Everything in one RUN to
# keep the final image to a single APT layer with no stale caches.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      # Tor daemon (provides 127.0.0.1:9050 SOCKS5 + 127.0.0.1:9051 control)
      tor \
      # Firefox ESR + its runtime dependencies for headless operation
      firefox-esr \
      libgtk-3-0 libdbus-glib-1-2 libxt6 libasound2 libx11-xcb1 libxcb-shm0 \
      libnss3 libgbm1 libgl1 \
      # Fonts — Firefox renders blanks for missing scripts (CJK, emoji, etc.)
      fonts-liberation fonts-noto-core fonts-noto-cjk fonts-noto-color-emoji \
      # Build & TLS toolchain — required to compile Python wheels (cryptography, etc.)
      build-essential libssl-dev libffi-dev \
      # Runtime niceties
      curl ca-certificates xdg-utils && \
    # Install geckodriver from the upstream Mozilla release.
    curl -fsSL "https://github.com/mozilla/geckodriver/releases/download/v${GECKODRIVER_VERSION}/geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz" \
      | tar -xz -C /usr/local/bin && \
    chmod +x /usr/local/bin/geckodriver && \
    # Trim the apt cache.
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────
# Copied separately so changes to app code don't bust the pip cache layer.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────
COPY . .
RUN chmod +x /app/entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]
