#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Robin container entrypoint.
#
# 1. Start the embedded Tor daemon, capture its log to /tmp/tor.log.
# 2. Wait until the SOCKS port is listening AND a circuit is fully built
#    (Tor reports "Bootstrapped 100%"). Skipping the bootstrap wait is the
#    classic source of "Connection timeout" failures on the very first
#    deep-crawl after `docker run` — the SOCKS port opens long before
#    traffic can actually be routed.
# 3. Print a runtime-check banner so bad images / missing components are
#    obvious before Streamlit even comes up.
# 4. Hand control to Streamlit.
# ─────────────────────────────────────────────────────────────────────────────
set -e

TOR_LOG=/tmp/tor.log
: > "$TOR_LOG"

echo "Starting Tor..."
# Tee Tor's stdout/stderr to a file so we can grep for the bootstrap line.
tor 2>&1 | tee "$TOR_LOG" &

echo "Waiting for Tor SOCKS socket (127.0.0.1:9050) to open..."
timeout 60 bash -c '
until python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect((\"127.0.0.1\", 9050)); s.close()" 2>/dev/null; do
  echo "  …socket not ready yet"
  sleep 2
done
'
if [ $? -ne 0 ]; then
  echo "ERROR: Tor failed to start or is not listening on port 9050."
  exit 1
fi

echo "Waiting for Tor circuit to be usable..."
# Two ready signals — first to win lets us proceed:
#   1. "Bootstrapped 100%"   — Tor's official "done" message
#   2. "consensus contains exit nodes" — earlier signal that means circuits
#      to clearweb can already be built (typically appears 30-90s before
#      100% on a fresh container).
# Cap the wait at 240s — first-time bootstrap on a slow connection can
# crawl up to ~3 min; longer than that means something's wrong.
READY_RE='(Bootstrapped 100|current consensus contains exit nodes)'
if timeout 240 bash -c "until grep -qE '${READY_RE}' \"${TOR_LOG}\"; do sleep 1; done"; then
    ready_line=$(grep -E "${READY_RE}" "$TOR_LOG" | tail -1)
    echo "  ${ready_line}"
else
    echo "WARNING: Tor did not become routable within 240s. Robin will"
    echo "         still start, but the first deep-crawl may time out"
    echo "         until the circuit is fully built. Inspect '$TOR_LOG'."
fi

# ── Pre-warm the circuit ────────────────────────────────────────────────
# "Routable" doesn't mean "fast" — the very first request through a fresh
# circuit can take 20-40s while exit-node descriptors load. That's enough
# to blow Selenium's 45s page-load timeout on the user's first crawl.
# Force one cheap HTTPS request through SOCKS5 here so the circuit is
# warm by the time Streamlit hands off to crawler.py. Failures are
# logged but non-fatal — Robin's per-crawl retry/fallback still catches.
echo "Pre-warming Tor circuit (one HTTPS request via SOCKS5)..."
if curl --socks5-hostname 127.0.0.1:9050 -fsS --max-time 45 \
        -o /dev/null https://check.torproject.org/ 2>/dev/null; then
    echo "  Tor exit reachable — first selenium crawl will hit a warm circuit."
else
    echo "  WARN: pre-warm request failed; first crawl may need a retry."
fi

# ── Runtime check ────────────────────────────────────────────────────────
# Surface the actual state of every component the deep-crawl pipeline
# relies on. Missing components don't abort the container (Robin gracefully
# falls back to the requests tier), but seeing them in `docker logs` makes
# misbuilds easy to spot.
echo
echo "─── Robin runtime check ─────────────────────────────────"
firefox_v="$(firefox-esr --version 2>/dev/null    || echo MISSING)"
gecko_v="$(geckodriver --version 2>/dev/null | head -1 || echo MISSING)"
python_v="$(python3 --version 2>/dev/null         || echo MISSING)"
echo "  Tor SOCKS    : 127.0.0.1:9050 ✓"
echo "  ${python_v}"
echo "  Firefox      : ${firefox_v}"
echo "  geckodriver  : ${gecko_v}"
case "${firefox_v}${gecko_v}" in
  *MISSING*)
    echo "  Deep-crawl tier : Tier 2 (requests) — Selenium tier disabled"
    ;;
  *)
    echo "  Deep-crawl tier : Tier 1 (Firefox + Tor SOCKS) ready"
    ;;
esac
echo "─────────────────────────────────────────────────────────"
echo

echo "Starting Robin: AI-Powered Dark Web OSINT Tool..."
exec streamlit run ui.py --server.port=8501 --server.address=0.0.0.0
