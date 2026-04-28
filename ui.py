import base64
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import webbrowser
import streamlit as st
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from scrape import scrape_multiple
from search import get_search_results
from llm_utils import BufferedStreamingHandler, get_model_choices
from llm import get_llm, refine_query, filter_results, generate_summary, PRESET_PROMPTS
from export import generate_pdf
import investigations as inv_db
import seeds as seed_db
import presets as preset_db
from crawler import crawl_sources, crawl_url, probe_tier, selenium_subtier, _torbrowser_binary
from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OLLAMA_BASE_URL,
    LLAMA_CPP_BASE_URL,
)
from health import check_llm_health, check_search_engines, check_tor_proxy


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def _validate_config() -> list:
    warnings = []
    def _check(name, value, prefix):
        if not value:
            return
        v = str(value).strip()
        if v.startswith("your_") or v.startswith(prefix + "_"):
            warnings.append(f"**{name}** looks like a placeholder — double-check your `.env`.")
        if len(v) < 20 and "KEY" in name:
            warnings.append(f"**{name}** seems too short to be a valid API key.")
    _check("OPENAI_API_KEY",     OPENAI_API_KEY,     "sk")
    _check("ANTHROPIC_API_KEY",  ANTHROPIC_API_KEY,  "sk-ant")
    _check("GOOGLE_API_KEY",     GOOGLE_API_KEY,     "AIza")
    _check("OPENROUTER_API_KEY", OPENROUTER_API_KEY, "sk-or")
    return warnings


def _running_headless() -> bool:
    """
    True when this Streamlit process can't possibly launch a GUI app —
    typically inside Docker (the recommended deployment) or any other
    server context where no X display / browser / xdg-utils is available.

    Detection strategy (any one is enough):
      - /.dockerenv exists (Docker default) or /run/.containerenv (Podman)
      - the cgroup file mentions docker/containerd/podman/lxc/kubepods
      - on Linux/BSD, no DISPLAY / WAYLAND_DISPLAY env var is set
    """
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text() if Path("/proc/1/cgroup").exists() else ""
        if any(t in cgroup for t in ("docker", "containerd", "podman", "lxc", "kubepods")):
            return True
    except Exception:
        pass
    if platform.system() not in ("Darwin", "Windows"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return True
    return False


def _open_path_in_system_app(path: str) -> tuple[bool, str]:
    """
    Hand `path` to the OS so the user's default app (text editor for .txt,
    browser for .html, etc.) opens it. Server-side: only meaningful when
    Streamlit runs on the same machine the user is sitting at.

    Tries a long chain of fallbacks so we don't fail just because one
    helper isn't installed:
      1. `xdg-open` and friends (the canonical Linux openers)
      2. The configured Tor Browser binary (already detected for crawler)
      3. System browsers (firefox, chromium, chrome, brave) — works for
         text and HTML files alike via file:// loading
      4. Common text editors for .txt files (gedit, kate, mousepad, code…)
         honoring $EDITOR / $VISUAL when set
      5. Python's `webbrowser.open()` as a final shim
      6. Clear error pointing at the Download button when nothing works
         (e.g. running inside Docker without xdg-utils or DISPLAY).

    Returns (ok, message).
    """
    system = platform.system()
    is_url = path.startswith(("http://", "https://", "file://"))
    abs_path = path if is_url else os.path.abspath(path)
    is_text  = (not is_url) and abs_path.lower().endswith(
        (".txt", ".md", ".log", ".json", ".csv")
    )

    candidates: list[list[str]] = []

    if system == "Darwin":
        # `-t` forces the system text editor for plain-text files; for
        # URLs/HTML the bare `open` form picks the registered handler.
        if is_text and not is_url:
            candidates.append(["open", "-t", abs_path])
        candidates.append(["open", abs_path])
    elif system == "Windows":
        try:
            os.startfile(abs_path)  # type: ignore[attr-defined]
            return True, f"Launched system handler for `{abs_path}`."
        except Exception as exc:
            return False, f"Could not open `{abs_path}`: {exc}"
    else:
        # Linux / BSD canonical openers.
        for cmd in (
            ["xdg-open",   abs_path],
            ["gio", "open", abs_path],
            ["gnome-open", abs_path],
            ["kde-open5",  abs_path],
            ["kde-open",   abs_path],
            ["wslview",    abs_path],
        ):
            candidates.append(cmd)

        # User-preferred editor for text files.
        if is_text:
            for env_key in ("VISUAL", "EDITOR"):
                editor = os.environ.get(env_key, "").strip()
                if editor:
                    candidates.append([editor, abs_path])

        # Common Linux GUI text editors — only applied to text files so
        # we don't try to render HTML in gedit.
        if is_text:
            for editor in (
                "code", "codium", "gedit", "kate", "mousepad",
                "geany", "leafpad", "pluma", "xed",
            ):
                candidates.append([editor, abs_path])

        # System browsers — work for both text (rendered as plain text)
        # and HTML. Tor Browser is preferred when present, since on a
        # tor-using machine it's likely the user's `.onion`-aware browser.
        target = abs_path if is_url else f"file://{abs_path}"
        torbrowser = _torbrowser_binary()
        if torbrowser:
            candidates.append([torbrowser, target])
        for browser in (
            "firefox", "firefox-esr", "chromium", "chromium-browser",
            "google-chrome", "chrome", "brave-browser", "vivaldi",
        ):
            candidates.append([browser, target])

    tried: list[str] = []
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        tried.append(cmd[0])
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, f"Opened `{abs_path}` with `{os.path.basename(cmd[0])}`."
        except Exception:
            continue

    # Final shim: webbrowser module. Often picks BROWSER env var.
    try:
        target = abs_path if is_url else f"file://{abs_path}"
        if webbrowser.open(target):
            return True, f"Opened `{abs_path}` in your default browser."
    except Exception:
        pass

    if _running_headless():
        return False, (
            f"This Robin instance is running headlessly (Docker / no DISPLAY) — "
            f"there's no GUI here to open `{abs_path}` with. "
            "**Use the Download button above** to save the file to your machine, "
            "then open it locally."
        )

    suffix = f" (tried: {', '.join(tried)})" if tried else ""
    return False, (
        f"No system opener succeeded for `{abs_path}`{suffix}. "
        "Install one of: xdg-utils, firefox, chromium, gedit. "
        "Otherwise use the Download button above."
    )


def _open_url_in_tor_browser(url: str) -> tuple[bool, str]:
    """
    Launch Tor Browser pointed at `url`. Uses the same binary discovery
    crawler.py uses for Selenium (env var TORBROWSER_BINARY, then the
    common Linux default path).
    """
    binary = _torbrowser_binary()
    if not binary:
        return False, (
            "Tor Browser binary not found. Set the `TORBROWSER_BINARY` "
            "env var to the path of `Browser/firefox` inside your Tor "
            "Browser install, then restart the app."
        )
    try:
        subprocess.Popen(
            [binary, url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True, f"Launched Tor Browser → {url[:60]}{'…' if len(url) > 60 else ''}"
    except Exception as exc:
        return False, f"Could not launch Tor Browser: {exc}"


def _crawled_html_path(url: str) -> Path | None:
    """Return the path to the saved rendered.html for `url`, or None if absent."""
    h = hashlib.sha256(url.encode()).hexdigest()
    candidate = Path("investigations") / "crawled" / h / "rendered.html"
    return candidate if candidate.exists() else None


def _render_pipeline_error(stage: str, err: Exception) -> None:
    message = str(err).strip() or err.__class__.__name__
    lower_msg = message.lower()
    hints = [
        "- Confirm the relevant API key is set in your `.env` before launching.",
        "- Keys copied from dashboards often include hidden spaces — re-copy if auth keeps failing.",
        "- Restart the app after updating environment variables.",
    ]
    if any(t in lower_msg for t in ("anthropic", "x-api-key", "invalid api key", "authentication")):
        hints.insert(0, "- Claude/Anthropic models require a valid `ANTHROPIC_API_KEY`.")
    elif "openrouter" in lower_msg or "user not found" in lower_msg or "code: 401" in lower_msg:
        hints.insert(0, "- OpenRouter 401 usually means an invalid/expired key or extra whitespace.")
    elif "openai" in lower_msg or "gpt" in lower_msg:
        hints.insert(0, "- OpenAI models require `OPENAI_API_KEY` with access to the chosen model.")
    elif "google" in lower_msg or "gemini" in lower_msg:
        hints.insert(0, "- Google Gemini needs `GOOGLE_API_KEY` or Application Default Credentials.")
    st.error("❌ Failed to {}.\n\nError: {}\n\n{}".format(stage, message, "\n".join(hints)))
    st.stop()


# ---------------------------------------------------------------------------
# Cached backend calls
# ---------------------------------------------------------------------------

@st.cache_data(ttl=200, show_spinner=False)
def cached_search_results(refined_query: str, threads: int):
    # Percent-encode the query so special characters (& ? = # /) don't break
    # the engine URL templates that interpolate it as `?q={query}`.
    return get_search_results(quote(refined_query, safe=""), max_workers=threads)


@st.cache_data(ttl=200, show_spinner=False)
def cached_scrape_multiple(filtered: list, threads: int, max_content_chars: int):
    return scrape_multiple(filtered, max_workers=threads, max_return_chars=max_content_chars)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Robin: AI-Powered Dark Web OSINT Tool",
    page_icon="🕵️‍♂️",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .aStyle { font-size:18px; font-weight:bold; padding:5px 0; text-align:left; }
    .deep-crawl-box {
        border: 1px solid #444; border-radius: 8px;
        padding: 12px 16px; margin-top: 10px; background: #111;
    }
</style>""", unsafe_allow_html=True)

if "startup_warnings" not in st.session_state:
    st.session_state.startup_warnings = _validate_config()
if st.session_state.startup_warnings:
    with st.expander("⚠️ Configuration warnings", expanded=False):
        for w in st.session_state.startup_warnings:
            st.warning(w)


# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------

st.sidebar.title("Robin")
st.sidebar.text("AI-Powered Dark Web OSINT Tool")
st.sidebar.markdown("Made by [Apurv Singh Gautam](https://www.linkedin.com/in/apurvsinghgautam/)")
st.sidebar.subheader("Settings")


def _env_is_set(v) -> bool:
    return bool(v and str(v).strip() and "your_" not in str(v))


model_options = get_model_choices()
if not model_options:
    st.sidebar.error("⛔ No LLM models available. Set at least one API key in `.env` and restart.")
    st.stop()

# Preferred default models, in priority order. The first one that's actually
# present in `model_options` (i.e. has its API key configured) wins.
_PREFERRED_DEFAULTS = (
    "claude-sonnet-4-5",
    "gpt-5.1",
    "gpt-4.1",
    "gemini-2.5-pro",
    "claude-sonnet-4.5-openrouter",
)
default_model_index = next(
    (model_options.index(m) for m in _PREFERRED_DEFAULTS if m in model_options),
    0,
)
model = st.sidebar.selectbox("Select LLM Model", model_options, index=default_model_index, key="model_select")

threads           = st.sidebar.slider("Scraping Threads", 1, 16, 4, key="thread_slider")
max_results       = st.sidebar.slider("Max Results to Filter", 10, 100, 50, key="max_results_slider",
                        help="Cap raw results sent to the LLM filter step.")
max_scrape        = st.sidebar.slider("Max Pages to Scrape", 3, 20, 10, key="max_scrape_slider",
                        help="Cap filtered results that get scraped.")
max_content_chars = st.sidebar.slider("Content Size per Page", 2_000, 10_000, 2_000, step=1_000,
                        key="max_content_chars_slider",
                        help="Max chars kept per scraped page.")

st.sidebar.divider()
st.sidebar.subheader("Provider Configuration")
for name, value, is_cloud in [
    ("OpenAI",     OPENAI_API_KEY,     True),
    ("Anthropic",  ANTHROPIC_API_KEY,  True),
    ("Google",     GOOGLE_API_KEY,     True),
    ("OpenRouter", OPENROUTER_API_KEY, True),
    ("Ollama",     OLLAMA_BASE_URL,    False),
    ("llama.cpp",  LLAMA_CPP_BASE_URL, False),
]:
    if _env_is_set(value):
        st.sidebar.markdown(f"&ensp;✅ **{name}** — configured")
    elif is_cloud:
        st.sidebar.markdown(f"&ensp;⚠️ **{name}** — API key not set")
    else:
        st.sidebar.markdown(f"&ensp;🔵 **{name}** — not configured *(optional)*")

# Built-in domains. These four MUST keep their existing keys + behavior, so
# they're hard-coded here exactly as before.
_BUILTIN_PRESETS = {
    "🔍 Dark Web Threat Intel":           "threat_intel",
    "🦠 Ransomware / Malware Focus":       "ransomware_malware",
    "👤 Personal / Identity Investigation": "personal_identity",
    "🏢 Corporate Espionage / Data Leaks":  "corporate_espionage",
}
_BUILTIN_PLACEHOLDERS = {
    "threat_intel":       "e.g. Pay extra attention to cryptocurrency wallet addresses.",
    "ransomware_malware": "e.g. Highlight double-extortion tactics or RaaS affiliates.",
    "personal_identity":  "e.g. Flag passport numbers and note country of origin.",
    "corporate_espionage":"e.g. Prioritize source code repos, API keys, Slack dumps.",
}

with st.sidebar.expander("⚙️ Prompt Settings"):
    custom_presets = preset_db.list_presets()

    # Build the dropdown: built-ins first (preserves their existing labels),
    # then any user-defined domains prefixed with ✨.
    label_to_key: dict[str, str] = dict(_BUILTIN_PRESETS)
    for cp in custom_presets:
        label_to_key[f"✨ {cp['name']}"] = cp["key"]

    selected_preset_label = st.selectbox(
        "Research Domain", list(label_to_key.keys()), key="preset_select",
    )
    selected_preset = label_to_key[selected_preset_label]
    is_custom = preset_db.is_custom_key(selected_preset)

    if is_custom:
        cp = preset_db.get_preset_by_key(selected_preset)
        if cp is None:
            # Stale selectbox state — shouldn't happen, but recover gracefully.
            st.warning("This custom domain no longer exists. Pick another.")
            selected_preset = "threat_intel"
            cp = None
            current_system_prompt = PRESET_PROMPTS["threat_intel"]
        else:
            current_system_prompt = cp["system_prompt"]
            st.caption(
                f"Created {cp['created_at'][:10]}"
                + (f" · updated {cp['updated_at'][:10]}"
                   if cp['updated_at'] != cp['created_at'] else "")
            )

        # Editable system prompt for custom domains.
        edited = st.text_area(
            "System Prompt (editable)",
            value=current_system_prompt,
            height=240,
            key=f"system_prompt_edit_{cp['id'] if cp else 'none'}",
        )
        edited_desc = st.text_input(
            "Description (optional)",
            value=cp.get("description", "") if cp else "",
            key=f"desc_edit_{cp['id'] if cp else 'none'}",
        )

        col_save, col_del = st.columns(2)
        with col_save:
            if cp and st.button("💾 Save", key=f"save_preset_{cp['id']}",
                                use_container_width=True):
                try:
                    preset_db.update_preset(
                        cp["id"],
                        system_prompt=edited,
                        description=edited_desc,
                    )
                    st.success("Saved.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with col_del:
            if cp and st.button("🗑️ Delete", key=f"del_preset_{cp['id']}",
                                use_container_width=True, type="secondary"):
                preset_db.delete_preset(cp["id"])
                st.success(f"Deleted '{cp['name']}'.")
                st.rerun()

        custom_placeholder = (
            "e.g. Cross-reference any leaked email addresses against known "
            "breach corpora and flag overlapping infrastructure."
        )
    else:
        # Built-in domain — keep the legacy read-only display verbatim.
        current_system_prompt = PRESET_PROMPTS[selected_preset]
        st.text_area(
            "System Prompt", value=current_system_prompt.strip(),
            height=200, disabled=True, key="system_prompt_display",
        )
        custom_placeholder = _BUILTIN_PLACEHOLDERS[selected_preset]

    custom_instructions = st.text_area(
        "Custom Instructions (optional)",
        placeholder=custom_placeholder,
        height=100,
        key="custom_instructions",
    )

    # ── Create new custom domain ─────────────────────────────────────────
    st.divider()
    st.caption("➕ Create a new custom research domain")
    with st.form("new_preset_form", clear_on_submit=True):
        new_preset_name = st.text_input(
            "Domain name",
            placeholder="e.g. Crypto Tracing",
        )
        new_preset_desc = st.text_input(
            "Description (optional)",
            placeholder="One-line summary of what this domain focuses on",
        )
        new_preset_prompt = st.text_area(
            "System prompt",
            height=200,
            placeholder=(
                "Describe the analyst persona, the analysis rules, and the "
                "output format. {query} will be substituted with the user's "
                "query at runtime."
            ),
        )
        if st.form_submit_button("➕ Create domain"):
            try:
                preset_db.create_preset(
                    new_preset_name, new_preset_prompt, new_preset_desc,
                )
                st.success(f"Created '{new_preset_name.strip()}'.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

# Resolve the active system prompt once, after the expander block, so the
# rest of the script can pass it straight into generate_summary regardless
# of whether the user picked a built-in or custom domain.
_active_custom = (
    preset_db.get_preset_by_key(selected_preset)
    if preset_db.is_custom_key(selected_preset) else None
)
selected_system_prompt = (
    _active_custom["system_prompt"] if _active_custom
    else PRESET_PROMPTS.get(selected_preset, PRESET_PROMPTS["threat_intel"])
)


# ---------------------------------------------------------------------------
# Sidebar — Health Checks
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Health Checks")

if st.sidebar.button("🔌 Check LLM Connection", use_container_width=True):
    with st.sidebar, st.spinner(f"Testing {model}..."):
        result = check_llm_health(model)
    if result["status"] == "up":
        st.sidebar.success(f"✅ **{result['provider']}** — Connected ({result['latency_ms']}ms)")
    else:
        st.sidebar.error(f"❌ **{result['provider']}** — Failed\n\n{result['error']}")

if st.sidebar.button("🔍 Check Search Engines", use_container_width=True):
    with st.sidebar, st.spinner("Checking Tor proxy..."):
        tor_result = check_tor_proxy()
    if tor_result["status"] == "down":
        st.sidebar.error(f"❌ **Tor Proxy** — Not reachable\n\n{tor_result['error']}")
    else:
        st.sidebar.success(f"✅ **Tor Proxy** — Connected ({tor_result['latency_ms']}ms)")
        with st.sidebar, st.spinner("Pinging search engines..."):
            engine_results = check_search_engines()
        up_count = sum(1 for r in engine_results if r["status"] == "up")
        total    = len(engine_results)
        lbl = (f"✅ **All {total} engines reachable**" if up_count == total else
               f"⚠️ **{up_count}/{total} engines reachable**" if up_count else
               f"❌ **0/{total} engines reachable**")
        (st.sidebar.success if up_count == total else
         st.sidebar.warning if up_count else st.sidebar.error)(lbl)
        for r in engine_results:
            icon   = "🟢" if r["status"] == "up" else "🔴"
            detail = f"{r['latency_ms']}ms" if r["status"] == "up" else r["error"]
            st.sidebar.markdown(f"&ensp;{icon} **{r['name']}** — {detail}")

# ---------------------------------------------------------------------------
# Sidebar — Open Robin UI in Tor Browser
# ---------------------------------------------------------------------------
#
# Tor Browser refuses localhost traffic by default — every request goes
# through Tor, and 127.0.0.1 isn't reachable from a Tor circuit. To use
# Tor Browser as the Robin UI itself (so .onion links opened from Robin
# reuse the same browser window), the user has to add localhost to
# `network.proxy.no_proxies_on` in about:config. This expander gives
# them the instructions and a one-click launch.

st.sidebar.divider()
with st.sidebar.expander("🦊 Use Tor Browser as Robin UI", expanded=False):
    st.caption(
        "Tor Browser blocks localhost by default, so `http://localhost:8501` "
        "won't load until you whitelist it once."
    )
    st.markdown(
        "**One-time Tor Browser setup**\n\n"
        "1. Open Tor Browser → address bar → `about:config` → *Accept Risk*.\n"
        "2. Search for `network.proxy.no_proxies_on`.\n"
        "3. Set the value to `localhost, 127.0.0.1`.\n"
        "4. Reload the Robin URL in Tor Browser.\n\n"
        "After that, `.onion` links you click in Robin open in a new tab "
        "in the same Tor Browser window."
    )
    robin_url = st.text_input(
        "Robin URL",
        value=os.environ.get("ROBIN_URL", "http://localhost:8501"),
        help="Where Streamlit is serving Robin — defaults to localhost:8501.",
        key="robin_url_input",
    )
    if _running_headless():
        st.caption(
            "ℹ️ Robin is running headlessly (Docker / no DISPLAY) — the launch "
            "button is disabled here because the container has no GUI / Tor "
            "Browser binary. Open Tor Browser **on your own machine** and "
            "point it at the Robin URL after applying steps 1–4."
        )
    elif st.button(
        "🦊 Open Robin in Tor Browser",
        use_container_width=True,
        key="open_robin_in_tor_btn",
        help="Launches Tor Browser pointed at the Robin URL above. "
             "Requires the about:config tweak from step 3.",
    ):
        ok, msg = _open_url_in_tor_browser(robin_url)
        (st.success if ok else st.error)(msg)


# ---------------------------------------------------------------------------
# Sidebar — Seed Manager
# ---------------------------------------------------------------------------

st.sidebar.divider()
with st.sidebar.expander("🌱 Seed Manager", expanded=False):
    st.caption("Add .onion URLs and inspect previously deep-crawled content.")

    with st.form("add_seed_form", clear_on_submit=True):
        new_url  = st.text_input("URL", placeholder="http://example.onion")
        new_name = st.text_input("Label (optional)")
        if st.form_submit_button("➕ Add Seed"):
            if new_url.strip():
                try:
                    seed_db.add_seed(new_url.strip(), new_name.strip())
                    st.success("Seed added.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a URL.")

    all_seeds = seed_db.get_all_seeds()
    if all_seeds:
        # Filter dropdown — same UX as Past Investigations.
        seed_filter = st.selectbox(
            "Filter seeds",
            ["all", "crawled", "uncrawled", "with content"],
            key="seed_filter",
        )
        if seed_filter == "crawled":
            visible = [s for s in all_seeds if s["crawled"]]
        elif seed_filter == "uncrawled":
            visible = [s for s in all_seeds if not s["crawled"]]
        elif seed_filter == "with content":
            visible = [s for s in all_seeds if (s.get("content") or "")]
        else:
            visible = all_seeds

        st.caption(f"**{len(visible)} / {len(all_seeds)} seeds**")

        if visible:
            def _seed_label(s):
                icon  = "✅" if s["crawled"] else "⏳"
                doc   = "📄" if (s.get("content") or "") else "—"
                short = s["url"][:34] + ("…" if len(s["url"]) > 34 else "")
                return f"{icon}{doc} {short}"

            seed_labels = [_seed_label(s) for s in visible]
            # Keying by filter ensures the picker resets when filter changes,
            # so it can't reference an item that's been filtered out.
            seed_key = f"seed_select_{seed_filter}"
            chosen = st.selectbox(
                "Open seed", ["(none)"] + seed_labels, key=seed_key,
            )
            if chosen != "(none)":
                try:
                    s = visible[seed_labels.index(chosen)]
                except ValueError:
                    s = None
                if s:
                    status_word = "crawled" if s["crawled"] else "pending"
                    code = s.get("status_code")
                    code_part = f" (HTTP {code})" if code else ""
                    crawled_at = s.get("crawled_at")
                    crawled_line = (
                        f"  \n**Last crawled:** {crawled_at[:16]}"
                        if crawled_at else ""
                    )
                    st.markdown(f"**Label:** {s.get('name') or '—'}")
                    st.markdown(
                        f"**URL:** `{s['url']}`  \n"
                        f"**Status:** {status_word}{code_part}  \n"
                        f"**Added:** {s['added_at'][:16]}"
                        f"{crawled_line}"
                    )

                    # ── Crawl-this-seed action ─────────────────────────────
                    # Works for both uncrawled seeds and previously-crawled
                    # ones (re-crawl). Persists the new content into the
                    # seeds DB so the rest of the UI sees it.
                    crawl_label = (
                        "🕷️ Re-crawl this seed now" if s["crawled"]
                        else "🕷️ Crawl this seed now"
                    )
                    if st.button(
                        crawl_label,
                        key=f"crawl_seed_{s['id']}",
                        use_container_width=True,
                    ):
                        with st.spinner(f"Deep crawling {s['url'][:40]}…"):
                            res = crawl_url(
                                s["url"],
                                title_hint=s.get("name", ""),
                                tier=probe_tier(),
                            )
                        if res["success"]:
                            try:
                                seed_db.mark_crawled(
                                    s["id"], status_code=200,
                                    content=res["text"],
                                )
                                st.success(
                                    f"✅ Crawled — {len(res['text']):,} chars saved "
                                    f"via {res['tier']}."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Crawl ok but save failed: {exc}")
                        else:
                            st.error(f"❌ Crawl failed: {res['error']}")

                    # ── Open URL: Tor Browser preferred, then plain link ───
                    # Tor Browser is the only browser that resolves .onion
                    # without extra config, so we surface it as the primary
                    # action when the binary is detected. The plain
                    # target="_blank" link stays as a universal fallback.
                    #
                    # In headless / Docker mode there's no GUI at all on
                    # the server side, so we hide all "Open" buttons and
                    # surface a single explanatory caption — the Download
                    # buttons further down work everywhere.
                    headless = _running_headless()
                    tor_binary_present = bool(_torbrowser_binary()) and not headless

                    if headless:
                        st.caption(
                            "🛈 Robin is running headlessly (Docker / no GUI), "
                            "so server-side **Open** actions are disabled. "
                            "Use the **Download** buttons below to save files "
                            "and open them locally."
                        )
                    elif tor_binary_present:
                        if st.button(
                            "🦊 Open URL in Tor Browser",
                            key=f"open_tor_{s['id']}",
                            use_container_width=True,
                            help="Launches Tor Browser pointed at this URL.",
                        ):
                            ok, msg = _open_url_in_tor_browser(s["url"])
                            (st.success if ok else st.error)(msg)
                    else:
                        st.caption(
                            "🦊 Tor Browser not detected — set `TORBROWSER_BINARY` "
                            "in your `.env` to enable a one-click launch."
                        )
                    st.markdown(
                        f'🌐 <a href="{s["url"]}" target="_blank" '
                        f'rel="noopener noreferrer">Open URL in current browser ↗</a> '
                        '<sub>(only works if your browser handles .onion)</sub>',
                        unsafe_allow_html=True,
                    )

                    content = s.get("content") or ""
                    if content:
                        st.caption(f"📄 {len(content):,} chars saved")
                        with st.expander("View extracted content", expanded=False):
                            st.text(content[:5000] + ("…" if len(content) > 5000 else ""))
                    else:
                        st.caption("No content saved yet — click crawl above to populate.")

                    # ── Download / open extracted content ──────────────────
                    if content:
                        # Universal fallback — always works regardless of
                        # whether the server has xdg-open installed.
                        st.download_button(
                            "💾 Download content as .txt",
                            data=content,
                            file_name=f"robin_seed_{s['id']}.txt",
                            mime="text/plain",
                            key=f"dl_seed_{s['id']}",
                            use_container_width=True,
                        )
                        # Server-side launch only when there's plausibly a GUI.
                        if not headless and st.button(
                            "📝 Open content in system app",
                            key=f"editor_open_{s['id']}",
                            help="Writes the extracted text to a temp .txt and asks "
                                 "the OS handler to open it. Falls back to your default "
                                 "browser if no opener is installed.",
                            use_container_width=True,
                        ):
                            try:
                                fd, tmp = tempfile.mkstemp(
                                    prefix=f"robin_seed_{s['id']}_",
                                    suffix=".txt",
                                )
                                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                                    fh.write(f"# {s['url']}\n")
                                    fh.write(f"# Crawled: {s.get('crawled_at') or '—'}\n")
                                    fh.write(f"# {len(content):,} chars\n\n")
                                    fh.write(content)
                                ok, msg = _open_path_in_system_app(tmp)
                                if ok:
                                    st.success(f"📝 {msg}")
                                else:
                                    st.error(f"❌ {msg}\n\nFile saved at `{tmp}`.")
                            except Exception as exc:
                                st.error(f"❌ Could not prepare temp file: {exc}")

                    # If the saved rendered.html still exists, offer to:
                    #   - download it (universal: works no matter how/where
                    #     Streamlit is hosted, just like the .txt download)
                    #   - open it via the system handler (`xdg-open file.html`
                    #     style — only when a GUI is reachable)
                    #   - open it in Tor Browser via file:// (when detected
                    #     AND running with a GUI)
                    html_path = _crawled_html_path(s["url"])
                    if html_path is not None:
                        try:
                            html_bytes = html_path.read_bytes()
                        except Exception as exc:
                            html_bytes = None
                            st.caption(f"Saved HTML could not be read: {exc}")
                        if html_bytes is not None:
                            st.download_button(
                                "💾 Download saved HTML",
                                data=html_bytes,
                                file_name=f"robin_seed_{s['id']}.html",
                                mime="text/html",
                                key=f"dl_html_{s['id']}",
                                use_container_width=True,
                                help=f"Downloads {html_path.name} so you can "
                                     "open it directly with your browser.",
                            )
                        if not headless and st.button(
                            "🌐 Open saved HTML (xdg-open / system)",
                            key=f"html_open_{s['id']}",
                            use_container_width=True,
                            help=f"Equivalent to running `xdg-open {html_path}` "
                                 "on the host. Falls back to firefox/chromium and "
                                 "your default browser if xdg-utils is missing.",
                        ):
                            ok, msg = _open_path_in_system_app(str(html_path))
                            (st.success if ok else st.error)(msg)
                        if tor_binary_present and st.button(
                            "🦊 Open saved HTML in Tor Browser",
                            key=f"html_tor_{s['id']}",
                            use_container_width=True,
                            help=f"Opens {html_path} in Tor Browser via file://",
                        ):
                            ok, msg = _open_url_in_tor_browser(
                                f"file://{html_path.resolve()}"
                            )
                            (st.success if ok else st.error)(msg)

                    if st.button("🗑️ Delete seed", key=f"del_seed_{s['id']}"):
                        seed_db.delete_seed(s["id"])
                        st.rerun()
        else:
            st.caption("No seeds match this filter.")
    else:
        st.caption("No seeds yet.")


# ---------------------------------------------------------------------------
# Sidebar — Past Investigations
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("📂 Past Investigations")
all_tags      = inv_db.get_all_tags()
filter_status = st.sidebar.selectbox("Filter by status", ["all"] + list(inv_db.VALID_STATUSES), key="filter_status")
filter_tag    = st.sidebar.selectbox("Filter by tag", ["all"] + all_tags, key="filter_tag") if all_tags else "all"

saved = inv_db.load_all(
    status_filter=None if filter_status == "all" else filter_status,
    tag_filter=None if filter_tag == "all" else filter_tag,
)

if saved:
    inv_labels = [
        f"{inv['timestamp'][:16]} — {inv['query'][:38]} [{inv['status']}]"
        for inv in saved
    ]
    # Including the active filters in the widget key forces Streamlit to
    # reset the picker when the filter set changes — otherwise a previously
    # selected label can persist into a filter set that no longer contains
    # it, which causes the load selectbox to render blank/freeze on scroll.
    inv_select_key = f"inv_select_{filter_status}_{filter_tag}"
    selected_label = st.sidebar.selectbox(
        "Load investigation", ["(none)"] + inv_labels, key=inv_select_key,
    )
    if selected_label != "(none)":
        try:
            selected_inv = saved[inv_labels.index(selected_label)]
        except ValueError:
            selected_inv = None
        if selected_inv and st.sidebar.button(
            "📂 Load", use_container_width=True, key="load_inv_btn",
        ):
            st.session_state["loaded_investigation"] = selected_inv
            # Clear any leftover fresh-run state so the loaded view
            # doesn't accidentally re-summarize stale pipeline data.
            for k in ("refined", "results", "filtered", "scraped",
                      "streamed_summary", "last_inv_id"):
                st.session_state.pop(k, None)
            st.rerun()
else:
    st.sidebar.caption("No saved investigations yet.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

_, logo_col, _ = st.columns(3)
with logo_col:
    st.image(".github/assets/robin_logo.png", width=200)

with st.form("search_form", clear_on_submit=True):
    col_input, col_button = st.columns([10, 1])
    query      = col_input.text_input("Enter Dark Web Search Query",
                                       placeholder="Enter Dark Web Search Query",
                                       label_visibility="collapsed", key="query_input")
    run_button = col_button.form_submit_button("Run")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_findings(summary_text: str, inv_dict: dict, target=None):
    """
    Render findings with Markdown + PDF download buttons.

    `target` is the placeholder/container to render into. If omitted, the
    module-level `findings_placeholder` is used (which only exists once
    the script reaches the pipeline-placeholder block, so callers in the
    loaded-investigation view must always pass an explicit target).
    """
    container = target.container() if target is not None else findings_placeholder.container()
    with container:
        st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
        st.markdown(summary_text)
        col_md, col_pdf = st.columns(2)
        now = datetime.now().strftime("%Y-%m-%d")
        with col_md:
            b64 = base64.b64encode(summary_text.encode()).decode()
            st.markdown(
                f'<div class="aStyle">📥 <a href="data:file/markdown;base64,{b64}" '
                f'download="summary_{now}.md">Download Markdown</a></div>',
                unsafe_allow_html=True,
            )
        with col_pdf:
            pdf_bytes = generate_pdf(inv_dict)
            st.download_button(
                "📄 Download PDF Report", data=pdf_bytes,
                file_name=f"robin_investigation_{now}.pdf",
                mime="application/pdf",
                key=f"pdf_dl_{now}_{abs(hash(summary_text)) % 99999}",
            )


def _run_summary_stage(
    llm, query_text, scraped, preset, custom_instr, summary_slot,
    system_prompt_override: str | None = None,
):
    """
    Stream the LLM summary into summary_slot, return full accumulated text.

    `system_prompt_override`, if given, is used verbatim as the system prompt
    — that's how user-defined custom research domains flow through.
    Built-in domains continue to resolve via `preset` against PRESET_PROMPTS.
    """
    streamed = {"text": ""}
    def ui_emit(chunk: str):
        streamed["text"] += chunk
        summary_slot.markdown(streamed["text"])
    stream_handler = BufferedStreamingHandler(ui_callback=ui_emit)
    llm.callbacks  = [stream_handler]
    generate_summary(
        llm, query_text, scraped,
        preset=preset, custom_instructions=custom_instr,
        system_prompt_override=system_prompt_override,
    )
    return streamed["text"]


def _deep_crawl_sources_section(
    sources: list,
    scraped_key: str,
    query_text: str,
    section_label: str = "sources",
):
    """
    Render the deep crawl UI block below a sources list.
    - Shows the crawl tier detected (Selenium vs requests).
    - Lets the user crawl individual sources or all at once.
    - On completion, re-runs the LLM summary with deep-crawled content
      merged into the existing scraped data and re-renders findings.

    Args:
        sources:      List of {"link":..., "title":...} dicts.
        scraped_key:  session_state key holding the existing scraped dict
                      (so we can merge deep content into it).
        query_text:   Query used for the summary stage.
        section_label: Display label for messages.
    """
    tier = probe_tier()
    if tier == "selenium":
        sub = selenium_subtier()
        if sub == "tor-browser":
            tier_label = "🦊 **Tier 1 — Tor Browser** (full JS, hardened anti-fingerprint)"
        else:
            tier_label = "🦊 **Tier 1 — Firefox + Tor SOCKS** (full JS via 127.0.0.1:9050)"
    else:
        tier_label = "🌐 **Tier 2 — requests + SOCKS** (lightweight fallback)"

    st.markdown(f"**Deep Crawl available** &nbsp;|&nbsp; {tier_label}")
    if tier == "requests":
        st.caption(
            "Selenium prerequisites missing — Robin needs the `selenium` Python "
            "package and `geckodriver` on PATH. The bundled Docker image ships "
            "with both; if you're running locally, install geckodriver "
            "(`apt install firefox-esr` + the upstream geckodriver release) to "
            "enable the heavier Tier 1 path."
        )

    # Drain the post-crawl success message stashed by a prior rerun so the
    # user sees the confirmation in the same place the deep-crawl section
    # already lives (the toast is fleeting and may be missed).
    post_msg = st.session_state.pop("_post_crawl_msg", None)
    if post_msg:
        st.success(post_msg)

    # Per-source crawl buttons
    with st.expander(f"🔗 Crawl individual {section_label}", expanded=False):
        for i, src in enumerate(sources):
            url   = src.get("link", "")
            title = src.get("title", "Untitled")
            if not url:
                continue
            col_lbl, col_btn = st.columns([8, 2])
            with col_lbl:
                st.markdown(f"**{i+1}.** {title[:60]}{'…' if len(title)>60 else ''}")
                st.caption(url[:70] + ("…" if len(url) > 70 else ""))
            with col_btn:
                if st.button("🕷️ Crawl", key=f"crawl_single_{scraped_key}_{i}"):
                    with st.spinner(f"Deep crawling {url[:40]}…"):
                        result = crawl_url(url, title_hint=title, tier=tier)
                    if result["success"]:
                        existing = st.session_state.get(scraped_key, {})
                        existing[url] = result["text"]
                        st.session_state[scraped_key] = existing
                        # Persist the crawl into the seeds DB so it survives reloads.
                        saved_to_db = False
                        try:
                            seed_db.add_seed(url, title)
                            sid = seed_db.get_seed_by_url(url)
                            if sid:
                                seed_db.mark_crawled(
                                    sid["id"], status_code=200, content=result["text"],
                                )
                                saved_to_db = True
                        except Exception:
                            pass
                        if saved_to_db:
                            # st.toast survives the rerun below, so the user
                            # sees the confirmation AND the refreshed sidebar.
                            st.toast(
                                f"✅ Crawled {url[:40]} — saved to Seed Manager",
                                icon="🕷️",
                            )
                            # Force a rerun so the sidebar re-queries the
                            # seeds DB on this same interaction. Without
                            # this, the sidebar stays stuck showing the
                            # pre-crawl snapshot until the user clicks
                            # something else (Streamlit renders the sidebar
                            # before this main-pane handler runs).
                            st.rerun()
                        else:
                            st.warning(
                                f"⚠️ Crawled `{url[:40]}` — {len(result['text']):,} chars "
                                f"via {result['tier']}, but could not save to Seed "
                                "Manager (DB write failed). HTML still archived."
                            )
                    else:
                        st.error(f"❌ Failed: {result['error']}")

    # Crawl all sources at once
    if st.button(
        f"🕷️ Deep Crawl all {len(sources)} {section_label}",
        key=f"crawl_all_{scraped_key}",
        use_container_width=True,
    ):
        progress_bar  = st.progress(0.0, text="Starting deep crawl…")
        progress_text = st.empty()
        completed_ref = {"n": 0}

        def _on_progress(done, total):
            completed_ref["n"] = done
            pct = done / total if total else 0
            progress_bar.progress(pct, text=f"Deep crawling… {done}/{total}")
            progress_text.caption(f"{done} of {total} pages crawled")

        with st.spinner(f"Deep crawling {len(sources)} pages via {tier}…"):
            crawled = crawl_sources(
                sources,
                max_workers=min(threads, 5),
                tier=tier,
                progress_callback=_on_progress,
            )

        progress_bar.empty()
        progress_text.empty()

        if crawled:
            existing = st.session_state.get(scraped_key, {})
            existing.update(crawled)
            st.session_state[scraped_key] = existing

            # Persist every successfully crawled URL into the seeds DB.
            saved_count = 0
            for url, text in crawled.items():
                src_title = next((s.get("title","") for s in sources if s.get("link")==url), "")
                try:
                    seed_db.add_seed(url, src_title)
                    sid = seed_db.get_seed_by_url(url)
                    if sid:
                        seed_db.mark_crawled(sid["id"], status_code=200, content=text)
                        saved_count += 1
                except Exception:
                    pass

            # Toast persists across the rerun below so the user sees BOTH
            # the success notice AND the freshly-refreshed Seed Manager
            # in the sidebar (which otherwise paints before this handler
            # runs and shows the pre-crawl snapshot).
            total_chars = sum(len(v) for v in crawled.values())
            st.toast(
                f"✅ {len(crawled)}/{len(sources)} pages crawled · "
                f"{saved_count} saved to Seed Manager · "
                f"{total_chars:,} chars",
                icon="🕷️",
            )
            # Stash the same message in session_state so we can also show
            # it on the next render (toast is fleeting, ~4 s). The fresh
            # render reads + clears this so it only appears once.
            st.session_state["_post_crawl_msg"] = (
                f"✅ Deep crawled {len(crawled)}/{len(sources)} pages "
                f"({total_chars:,} total chars). "
                f"{saved_count} saved to **Seed Manager** (sidebar). "
                "HTML archived to `investigations/crawled/<hash>/`. "
                "Use **Re-summarize** below to regenerate findings with this richer content."
            )
            st.rerun()
        else:
            st.warning("No pages could be deep crawled. Check Tor connectivity.")


# ---------------------------------------------------------------------------
# Loaded investigation view
# ---------------------------------------------------------------------------

if "loaded_investigation" in st.session_state and not run_button:
    inv    = st.session_state["loaded_investigation"]
    inv_id = inv.get("id")

    st.info(f"📂 **{inv['query']}** — {inv['timestamp'][:16]}")

    with st.expander("📋 Notes & Management", expanded=False):
        st.markdown(f"**Refined Query:** `{inv.get('refined_query','')}`")
        st.markdown(f"**Model:** `{inv.get('model','')}` &nbsp;&nbsp; **Domain:** {inv.get('preset','')}")
        st.markdown(f"**Sources:** {len(inv['sources'])}")
        col_s, col_t = st.columns(2)
        with col_s:
            new_status = st.selectbox("Status", inv_db.VALID_STATUSES,
                                       index=list(inv_db.VALID_STATUSES).index(inv.get("status","active")),
                                       key="edit_status")
        with col_t:
            new_tags = st.text_input("Tags (comma-separated)", value=inv.get("tags",""), key="edit_tags")
        if st.button("💾 Save changes", key="save_meta_btn"):
            inv_db.update_status(inv_id, new_status)
            inv_db.update_tags(inv_id, new_tags)
            st.success("Updated.")
            st.session_state["loaded_investigation"] = inv_db.load_one(inv_id)
            st.rerun()
        if st.button("🗑️ Delete investigation", key="delete_inv_btn", type="secondary"):
            inv_db.delete_investigation(inv_id)
            del st.session_state["loaded_investigation"]
            st.rerun()

    with st.expander(f"🔗 Sources ({len(inv['sources'])} results)", expanded=False):
        for i, item in enumerate(inv["sources"], 1):
            st.markdown(f"{i}. [{item.get('title','Untitled')}]({item.get('link','')})")

    st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
    st.markdown(inv["summary"])

    col_md, col_pdf = st.columns(2)
    with col_md:
        b64 = base64.b64encode(inv["summary"].encode()).decode()
        st.markdown(
            f'<div class="aStyle">📥 <a href="data:file/markdown;base64,{b64}" '
            f'download="summary_{inv["timestamp"][:10]}.md">Download Markdown</a></div>',
            unsafe_allow_html=True,
        )
    with col_pdf:
        pdf_bytes = generate_pdf(inv)
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                            file_name=f"robin_investigation_{inv['timestamp'][:10]}.pdf",
                            mime="application/pdf", key="pdf_dl_loaded")

    # Deep crawl for loaded investigation sources
    if inv["sources"]:
        st.divider()
        st.markdown("### 🕷️ Deep Crawl")
        st.caption(
            "Fetch richer content directly from the sources referenced in this investigation, "
            "then regenerate the findings with that deeper context."
        )
        _deep_crawl_sources_section(
            sources=inv["sources"],
            scraped_key="loaded_inv_scraped",
            query_text=inv.get("refined_query", inv.get("query", "")),
            section_label="investigation sources",
        )

        # ── Re-summarize THIS investigation ──────────────────────────────────
        st.divider()
        st.markdown("### 🔁 Re-summarize this investigation")
        st.caption(
            "Regenerate the findings using the loaded investigation's sources "
            "and the currently selected model/preset. Deep-crawled content (if any) "
            "is used automatically; otherwise sources are re-scraped on the fly."
        )
        if st.button(
            "🔁 Re-summarize this investigation",
            use_container_width=True,
            key=f"resummarize_loaded_{inv_id}",
        ):
            with st.status("✍️ Re-summarizing loaded investigation…", expanded=True) as rs_status:
                st.write("🔄 Loading LLM…")
                try:
                    llm = get_llm(model)
                except Exception as e:
                    rs_status.update(label="❌ Failed", state="error")
                    _render_pipeline_error("load the selected LLM", e)

                # Prefer deep-crawled content; fall back to a fresh scrape of
                # the loaded investigation's sources so this button always works.
                content_dict = st.session_state.get("loaded_inv_scraped") or {}
                if not content_dict:
                    st.write("📜 Scraping loaded investigation sources…")
                    content_dict = cached_scrape_multiple(
                        inv["sources"], threads, max_content_chars,
                    )
                    st.write(f"&nbsp;&nbsp;&nbsp;→ {len(content_dict)} pages scraped")
                else:
                    st.write(
                        f"📦 Using {len(content_dict)} deep-crawled pages "
                        f"({sum(len(v) for v in content_dict.values()):,} chars)."
                    )

                rs_query = inv.get("refined_query") or inv.get("query", "")
                st.write("✍️ Streaming new summary…")

                # Stream into a temporary slot so the user can watch the
                # tokens arrive. Once streaming finishes we replace this
                # block with the full _render_findings() output (Markdown +
                # download buttons) — using a SECOND placeholder defined
                # below as the explicit render target.
                streaming_slot = st.empty()
                with streaming_slot.container():
                    st.subheader(":red[🔎 Findings (re-summarized)]", anchor=None, divider="gray")
                    summary_slot = st.empty()

                new_summary = _run_summary_stage(
                    llm, rs_query, content_dict,
                    selected_preset, custom_instructions, summary_slot,
                    system_prompt_override=selected_system_prompt,
                )
                rs_status.update(label="✅ Re-summarization complete", state="complete", expanded=False)

            # Clear the streaming placeholder so the final findings render
            # below replaces it cleanly (no duplicate headings).
            streaming_slot.empty()

            # Save as a NEW investigation linked to the same query so the
            # original record stays intact for comparison.
            new_inv_id = inv_db.save_investigation(
                query=inv.get("query", ""),
                refined_query=rs_query,
                model=model,
                preset_label=selected_preset_label,
                sources=inv["sources"],
                summary=new_summary,
            )
            st.success(
                f"✅ Saved as new investigation #{new_inv_id}. "
                "The original investigation is unchanged."
            )

            inv_dict_for_pdf = {
                "query":         inv.get("query", ""),
                "refined_query": rs_query,
                "model":         model,
                "preset":        selected_preset_label,
                "status":        "active",
                "tags":          "",
                "timestamp":     datetime.now().isoformat(),
                "sources":       inv["sources"],
                "summary":       new_summary,
            }
            # Pass an explicit local placeholder — the module-level
            # `findings_placeholder` doesn't exist yet at this point in
            # script execution (it's defined below the loaded-inv block).
            _render_findings(new_summary, inv_dict_for_pdf, target=st.empty())

    if st.button("✖ Clear"):
        del st.session_state["loaded_investigation"]
        st.session_state.pop("loaded_inv_scraped", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Pipeline placeholders
# ---------------------------------------------------------------------------

status_slot          = st.empty()
notes_placeholder    = st.empty()
sources_placeholder  = st.empty()
findings_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

if run_button and query:
    st.session_state.pop("loaded_investigation", None)
    for k in [
        "refined", "results", "filtered", "scraped", "streamed_summary",
        "last_inv_id", "last_query", "last_model", "last_preset_label",
        "last_max_content_chars", "last_timestamp",
    ]:
        st.session_state.pop(k, None)

    with st.status("🔄 Running investigation pipeline…", expanded=True) as pipeline_status:

        st.write("🔄 Loading LLM…")
        try:
            llm = get_llm(model)
        except Exception as e:
            pipeline_status.update(label="❌ Pipeline failed", state="error", expanded=True)
            _render_pipeline_error("load the selected LLM", e)

        st.write("✏️ Refining query…")
        try:
            st.session_state.refined = refine_query(llm, query)
            st.write(f"&nbsp;&nbsp;&nbsp;→ `{st.session_state.refined}`")
        except Exception as e:
            pipeline_status.update(label="❌ Pipeline failed", state="error", expanded=True)
            _render_pipeline_error("refine the query", e)

        st.write("🔍 Searching dark web (results pre-scored by keyword relevance)…")
        st.session_state.results = cached_search_results(st.session_state.refined, threads)
        if len(st.session_state.results) > max_results:
            st.session_state.results = st.session_state.results[:max_results]
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.results)} unique results")

        num_batches = (len(st.session_state.results) + 24) // 25
        st.write(f"🗂️ Filtering results ({num_batches} batch{'es' if num_batches > 1 else ''})…")
        st.session_state.filtered = filter_results(
            llm, st.session_state.refined, st.session_state.results
        )
        if len(st.session_state.filtered) > max_scrape:
            st.session_state.filtered = st.session_state.filtered[:max_scrape]
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.filtered)} selected")

        st.write("📜 Scraping content…")
        st.session_state.scraped = cached_scrape_multiple(
            st.session_state.filtered, threads, max_content_chars
        )
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.scraped)} pages scraped")

        st.write("✍️ Generating summary…")
        with findings_placeholder.container():
            st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
            summary_slot = st.empty()

        st.session_state.streamed_summary = _run_summary_stage(
            llm, query, st.session_state.scraped,
            selected_preset, custom_instructions, summary_slot,
            system_prompt_override=selected_system_prompt,
        )

        pipeline_status.update(label="✅ Pipeline complete", state="complete", expanded=False)

    # Save to SQLite + capture run-time context so the rendering block
    # below can faithfully re-render on subsequent reruns (e.g. after a
    # deep-crawl button click). Without this, clicking deep-crawl would
    # cause Streamlit to re-execute the script with `run_button=False`
    # and the entire results view would disappear.
    inv_id = inv_db.save_investigation(
        query=query,
        refined_query=st.session_state.refined,
        model=model,
        preset_label=selected_preset_label,
        sources=st.session_state.filtered,
        summary=st.session_state.streamed_summary,
    )
    st.session_state["last_inv_id"]            = inv_id
    st.session_state["last_query"]             = query
    st.session_state["last_model"]             = model
    st.session_state["last_preset_label"]      = selected_preset_label
    st.session_state["last_max_content_chars"] = max_content_chars
    st.session_state["last_timestamp"]         = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Render fresh-run results — every rerun
#
# This block fires whenever the session has data from a completed pipeline
# AND no past investigation is loaded. Decoupling the rendering from the
# pipeline-execution `if run_button and query:` block above is what makes
# the Deep Crawl / Re-summarize / save-status-tags buttons keep working
# after the form has been submitted (a button click triggers a rerun
# during which `run_button` is False).
# ---------------------------------------------------------------------------

if (
    st.session_state.get("streamed_summary")
    and st.session_state.get("filtered") is not None
    and "loaded_investigation" not in st.session_state
):
    fresh_query        = st.session_state.get("last_query", "")
    fresh_model        = st.session_state.get("last_model", model)
    fresh_preset_label = st.session_state.get("last_preset_label", selected_preset_label)
    fresh_chars        = st.session_state.get("last_max_content_chars", max_content_chars)
    fresh_timestamp    = st.session_state.get("last_timestamp", datetime.now().isoformat())
    fresh_inv_id       = st.session_state.get("last_inv_id")

    # Notes
    with notes_placeholder.container():
        with st.expander("📋 Notes", expanded=False):
            st.markdown(f"**Refined Query:** `{st.session_state.refined}`")
            st.markdown(
                f"**Model:** `{fresh_model}` &nbsp;&nbsp; "
                f"**Domain:** {fresh_preset_label}"
            )
            st.markdown(
                f"**Results found:** {len(st.session_state.get('results', []))} &nbsp;&nbsp; "
                f"**Filtered to:** {len(st.session_state.filtered)} &nbsp;&nbsp; "
                f"**Scraped:** {len(st.session_state.get('scraped', {}))} &nbsp;&nbsp; "
                f"**Content size:** {fresh_chars:,} chars/page"
            )
            st.markdown("---")
            col_s, col_t = st.columns(2)
            with col_s:
                quick_status = st.selectbox(
                    "Set status", inv_db.VALID_STATUSES, key="quick_status",
                )
            with col_t:
                quick_tags = st.text_input(
                    "Add tags (comma-separated)", key="quick_tags",
                )
            if st.button("💾 Save status & tags", key="quick_save_btn"):
                if fresh_inv_id is not None:
                    inv_db.update_status(fresh_inv_id, quick_status)
                    inv_db.update_tags(fresh_inv_id, quick_tags)
                    st.success("Saved.")
                else:
                    st.error("No investigation id in session — cannot save.")

    # Sources only — Deep Crawl is intentionally NOT offered for fresh runs.
    # The intended workflow is: run → review/save → load from Past
    # Investigations → deep-crawl from there. Loading the saved record gives
    # you a stable target with metadata (status, tags) to attach the deeper
    # crawl content to, instead of mutating an in-memory cache that vanishes
    # on the next form submit.
    with sources_placeholder.container():
        with st.expander(
            f"🔗 Sources ({len(st.session_state.filtered)} results)", expanded=False
        ):
            for i, item in enumerate(st.session_state.filtered, 1):
                st.markdown(f"{i}. [{item.get('title','Untitled')}]({item.get('link','')})")
        st.caption(
            "🕷️ Deep Crawl is available from the **Past Investigations** panel "
            "after this run is loaded — open it from the sidebar to deep-crawl "
            "and re-summarize with richer content."
        )

    # Findings
    inv_dict_for_pdf = {
        "query":         fresh_query,
        "refined_query": st.session_state.refined,
        "model":         fresh_model,
        "preset":        fresh_preset_label,
        "status":        "active",
        "tags":          "",
        "timestamp":     fresh_timestamp,
        "sources":       st.session_state.filtered,
        "summary":       st.session_state.streamed_summary,
    }
    _render_findings(st.session_state.streamed_summary, inv_dict_for_pdf)


# ---------------------------------------------------------------------------
# Re-summarize — reuses cached scraped data (updated by deep crawl)
#
# Only fires for a fresh-run context. When a past investigation is loaded,
# the loaded view has its own Re-summarize button that operates on that
# investigation's sources — without this guard, the bottom button would
# operate on stale fresh-run data even while a different investigation is
# being viewed.
# ---------------------------------------------------------------------------

if (
    st.session_state.get("scraped")
    and not run_button
    and "loaded_investigation" not in st.session_state
):
    st.divider()
    st.caption(
        "🔁 **Re-summarize** — change model/preset above, or after a deep crawl, "
        "to regenerate findings with updated content."
    )
    if st.button("🔁 Re-summarize with current settings", use_container_width=True, key="resummarize_btn"):

        with st.status("✍️ Re-generating summary…", expanded=True) as rs_status:
            st.write("🔄 Loading LLM…")
            try:
                llm = get_llm(model)
            except Exception as e:
                rs_status.update(label="❌ Failed", state="error")
                _render_pipeline_error("load the selected LLM", e)

            original_query = st.session_state.get("refined", "")
            st.write("✍️ Streaming new summary…")

            with findings_placeholder.container():
                st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
                summary_slot = st.empty()

            st.session_state.streamed_summary = _run_summary_stage(
                llm, original_query, st.session_state.scraped,
                selected_preset, custom_instructions, summary_slot,
                system_prompt_override=selected_system_prompt,
            )
            rs_status.update(label="✅ Re-summarization complete", state="complete", expanded=False)

        # Prefer the original user-typed query for the new investigation's
        # `query` field; fall back to the refined query if the run pre-dates
        # this enhancement.
        save_query = st.session_state.get("last_query") or original_query
        new_inv_id = inv_db.save_investigation(
            query=save_query,
            refined_query=st.session_state.get("refined", original_query),
            model=model,
            preset_label=selected_preset_label,
            sources=st.session_state.get("filtered", []),
            summary=st.session_state.streamed_summary,
        )
        st.session_state["last_inv_id"]        = new_inv_id
        st.session_state["last_model"]         = model
        st.session_state["last_preset_label"]  = selected_preset_label
        st.session_state["last_timestamp"]     = datetime.now().isoformat()

        inv_dict_for_pdf = {
            "query":         save_query,
            "refined_query": st.session_state.get("refined", original_query),
            "model":         model,
            "preset":        selected_preset_label,
            "status":        "active",
            "tags":          "",
            "timestamp":     datetime.now().isoformat(),
            "sources":       st.session_state.get("filtered", []),
            "summary":       st.session_state.streamed_summary,
        }
        _render_findings(st.session_state.streamed_summary, inv_dict_for_pdf)