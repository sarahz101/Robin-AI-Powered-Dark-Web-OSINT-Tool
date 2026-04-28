"""
tor_utils.py
Utility helpers for interacting with the Tor daemon control port.

Tor's control protocol (https://spec.torproject.org/control-spec) is simple
plain-text, so we communicate over a raw socket with no extra dependencies.

Typical setup:
  ControlPort 9051
  CookieAuthentication 0   (no password)
  -- or --
  HashedControlPassword <hash>   (with password)

Pass the password via TOR_CONTROL_PASSWORD env var or the function argument.
Leave it as None / empty string for unauthenticated control ports.
"""

import os
import socket
import time
import logging

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 9051
# Grace period (seconds) to wait after NEWNYM before the new circuit is ready.
NEWNYM_WAIT = 2

_logger = logging.getLogger(__name__)


def _send_command(sock: socket.socket, command: str) -> str:
    """Send a single control-protocol command and return the full response."""
    sock.sendall((command + "\r\n").encode())
    response = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk
        # Tor responses end with a line starting with a 3-digit code + space.
        decoded = response.decode(errors="replace")
        lines = decoded.strip().splitlines()
        if lines and len(lines[-1]) >= 4 and lines[-1][3] == " ":
            break
    return response.decode(errors="replace")


def refresh_tor_circuit(
    control_port: int = CONTROL_PORT,
    password: str | None = None,
) -> dict:
    """
    Send a NEWNYM signal to the Tor control port to request a fresh circuit.

    Args:
        control_port: Tor control port (default 9051).
        password:     Control port password. Falls back to the
                      TOR_CONTROL_PASSWORD environment variable, then tries
                      an unauthenticated connection.

    Returns:
        dict with keys:
          status  : "ok" | "error"
          message : Human-readable result string.
    """
    password = password or os.getenv("TOR_CONTROL_PASSWORD", "")

    try:
        with socket.create_connection((CONTROL_HOST, control_port), timeout=5) as sock:
            sock.settimeout(5)

            # Authenticate
            if password:
                auth_cmd = f'AUTHENTICATE "{password}"'
            else:
                auth_cmd = "AUTHENTICATE"

            auth_resp = _send_command(sock, auth_cmd)
            if not auth_resp.startswith("250"):
                return {
                    "status": "error",
                    "message": f"Authentication failed: {auth_resp.strip()}",
                }

            # Request new identity / circuit
            newnym_resp = _send_command(sock, "SIGNAL NEWNYM")
            if not newnym_resp.startswith("250"):
                return {
                    "status": "error",
                    "message": f"NEWNYM failed: {newnym_resp.strip()}",
                }

            # Brief pause so Tor has time to build the new circuit before
            # the next request goes out.
            time.sleep(NEWNYM_WAIT)

            _logger.info("Tor circuit refreshed successfully.")
            return {
                "status": "ok",
                "message": "New Tor circuit established. Your exit node has changed.",
            }

    except ConnectionRefusedError:
        return {
            "status": "error",
            "message": (
                f"Control port {control_port} refused connection. "
                "Check that ControlPort is enabled in your torrc "
                "(add `ControlPort 9051`) and restart Tor."
            ),
        }
    except socket.timeout:
        return {
            "status": "error",
            "message": f"Timed out connecting to Tor control port {control_port}.",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def get_tor_exit_ip(session) -> str | None:
    """
    Return the current Tor exit IP by fetching https://check.torproject.org/api/ip
    through the provided Tor-proxied requests session.
    Returns None on failure.
    """
    try:
        resp = session.get("https://check.torproject.org/api/ip", timeout=15)
        resp.raise_for_status()
        return resp.json().get("IP")
    except Exception:
        return None