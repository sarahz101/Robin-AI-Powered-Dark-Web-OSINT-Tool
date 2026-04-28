import re
import time
import logging
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_utils import (
    BufferedStreamingHandler,
    _common_llm_params,
    resolve_model_config,
    get_model_choices,
)
from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENROUTER_API_KEY,
)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
# Number of attempts before giving up on a single LLM chain call.
LLM_MAX_RETRIES = 3
# Base delay (seconds) for exponential back-off: 2s → 4s → 8s
LLM_RETRY_BASE_DELAY = 2

# Exceptions that are safe to retry (transient failures).
# Provider SDKs (anthropic, google.api_core) are imported defensively: if the
# package is missing the entry is simply omitted, so retry coverage scales
# with whichever providers are installed.
_retryable: list = [
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,   # 5xx from OpenAI-compatible providers
]

try:
    import anthropic
    _retryable.extend([
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    ])
except ImportError:
    pass

try:
    from google.api_core import exceptions as _google_exc
    _retryable.extend([
        _google_exc.ResourceExhausted,    # 429
        _google_exc.ServiceUnavailable,   # 503
        _google_exc.DeadlineExceeded,     # 504
        _google_exc.InternalServerError,  # 500
    ])
except ImportError:
    pass

_RETRYABLE_EXCEPTIONS = tuple(_retryable)

# Batch size for the LLM filter step — keeps each prompt well within context limits.
FILTER_BATCH_SIZE = 25


def get_llm(model_choice):
    config = resolve_model_config(model_choice)

    if config is None:
        supported_models = get_model_choices()
        raise ValueError(
            f"Unsupported LLM model: '{model_choice}'. "
            f"Supported models (case-insensitive match) are: {', '.join(supported_models)}"
        )

    llm_class = config["class"]
    model_specific_params = config["constructor_params"]

    # Fresh handler per instance — prevents shared-singleton streaming bugs.
    fresh_handler = BufferedStreamingHandler(buffer_limit=60)
    all_params = {
        **_common_llm_params,
        **model_specific_params,
        "callbacks": [fresh_handler],
    }

    _ensure_credentials(model_choice, llm_class, model_specific_params)

    return llm_class(**all_params)


def _ensure_credentials(model_choice: str, llm_class, model_params: dict) -> None:
    """Raise a clear error if the user selects a hosted model without a key."""

    def _require(key_value, env_var, provider_name):
        if key_value:
            return
        raise ValueError(
            f"{provider_name} model '{model_choice}' selected but `{env_var}` is not set.\n"
            "Add it to your .env file or export it before running the app."
        )

    class_name = getattr(llm_class, "__name__", str(llm_class))

    if "ChatAnthropic" in class_name:
        _require(ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY", "Anthropic")
    elif "ChatGoogleGenerativeAI" in class_name:
        _require(GOOGLE_API_KEY, "GOOGLE_API_KEY", "Google Gemini")
    elif "ChatOpenAI" in class_name:
        base_url = (model_params or {}).get("base_url", "").lower()
        if "openrouter" in base_url:
            _require(OPENROUTER_API_KEY, "OPENROUTER_API_KEY", "OpenRouter")
        else:
            _require(OPENAI_API_KEY, "OPENAI_API_KEY", "OpenAI")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _invoke_with_retry(chain, inputs: dict, stage: str = "LLM call"):
    """
    Invoke a LangChain chain with exponential back-off retry on transient errors.

    Retries on: rate limits, timeouts, connection errors, and 5xx responses.
    Propagates immediately on any other exception (e.g. auth failures).

    Args:
        chain:  A LangChain Runnable (prompt | llm | parser).
        inputs: Variables passed to chain.invoke().
        stage:  Human-readable label used in log/warning messages.

    Returns:
        The chain's string output.

    Raises:
        The last retryable exception if all attempts are exhausted.
        Any non-retryable exception on first occurrence.
    """
    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return chain.invoke(inputs)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt >= LLM_MAX_RETRIES:
                # Out of attempts — raise immediately rather than sleeping.
                break
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logging.warning(
                "[%s] attempt %d/%d failed (%s: %s) — retrying in %.0fs",
                stage, attempt, LLM_MAX_RETRIES,
                type(exc).__name__, str(exc)[:120], delay,
            )
            time.sleep(delay)
        except Exception:
            raise   # Auth errors, bad requests, etc. — don't retry
    raise last_exc


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def refine_query(llm, user_input):
    system_prompt = """
    You are a Cybercrime Threat Intelligence Expert. Your task is to refine the provided user query that needs to be sent to darkweb search engines. 
    
    Rules:
    1. Analyze the user query and think about how it can be improved to use as search engine query
    2. Refine the user query by adding or removing words so that it returns the best result from dark web search engines
    3. Don't use any logical operators (AND, OR, etc.)
    4. Keep the final refined query limited to 5 words or less
    5. Output just the user query and nothing else

    INPUT:
    """
    prompt_template = ChatPromptTemplate(
        [("system", system_prompt), ("user", "{query}")]
    )
    chain = prompt_template | llm | StrOutputParser()
    return _invoke_with_retry(chain, {"query": user_input}, stage="refine_query")


def _filter_batch(llm, query: str, batch: list, batch_offset: int) -> list:
    """
    Ask the LLM to select the most relevant indices from a single batch.

    The batch is a slice of the original results list. batch_offset is the
    number of items before this slice, used to convert local (1-based within
    the batch) indices back to global (1-based across all results) indices.

    Returns a list of global 1-based indices.
    """
    system_prompt = """
    You are a Cybercrime Threat Intelligence Expert. You are given a dark web search query and a list of search results in the form of index, link and title. 
    Your task is to select the Top 10 relevant results that best match the search query for further investigation.
    Rule:
    1. Output ONLY the indices (comma-separated list), no more than 10, that best match the input query.

    Search Query: {query}
    Search Results:
    """
    final_str = _generate_final_string(batch)
    prompt_template = ChatPromptTemplate(
        [("system", system_prompt), ("user", "{results}")]
    )
    chain = prompt_template | llm | StrOutputParser()

    try:
        raw = _invoke_with_retry(chain, {"query": query, "results": final_str}, stage="filter_batch")
    except openai.RateLimitError:
        # Last-resort fallback: truncated titles only to minimise token use.
        final_str = _generate_final_string(batch, truncate=True)
        raw = _invoke_with_retry(
            chain, {"query": query, "results": final_str}, stage="filter_batch_truncated"
        )

    # Parse local indices (1-based within this batch) and convert to global.
    parsed = []
    for match in re.findall(r"\d+", raw):
        try:
            local_idx = int(match)
            if 1 <= local_idx <= len(batch):
                global_idx = batch_offset + local_idx  # 1-based global
                parsed.append(global_idx)
        except ValueError:
            continue

    return parsed


def filter_results(llm, query: str, results: list) -> list:
    """
    Filter search results down to the top 20 most relevant entries.

    Results are processed in batches of FILTER_BATCH_SIZE (default 25) so that
    each LLM prompt stays well within context limits. Each batch independently
    selects up to 10 candidates; the combined candidates are de-duplicated and
    capped at 20.
    """
    if not results:
        return []

    all_selected_indices = []

    for batch_start in range(0, len(results), FILTER_BATCH_SIZE):
        batch = results[batch_start: batch_start + FILTER_BATCH_SIZE]
        selected = _filter_batch(llm, query, batch, batch_offset=batch_start)
        all_selected_indices.extend(selected)

    # De-duplicate while preserving order.
    seen: set = set()
    unique_indices = [i for i in all_selected_indices if not (i in seen or seen.add(i))]

    if not unique_indices:
        num_batches = (len(results) + FILTER_BATCH_SIZE - 1) // FILTER_BATCH_SIZE
        logging.warning(
            "filter_results: LLM returned no usable indices across %d batch(es). "
            "Defaulting to top %d results.",
            num_batches,
            min(len(results), 20),
        )
        unique_indices = list(range(1, min(len(results), 20) + 1))

    # Cap at 20 and convert to 0-based for list indexing.
    return [results[i - 1] for i in unique_indices[:20]]


def _generate_final_string(results, truncate=False):
    """Generate a formatted string from search results for LLM processing."""
    final_str = []
    for i, res in enumerate(results):
        truncated_link = re.sub(r"(?<=\.onion).*", "", res["link"])
        title = re.sub(r"[^0-9a-zA-Z\-\.]", " ", res["title"])
        if truncated_link == "" and title == "":
            continue

        if truncate:
            max_title_length = 30
            title = (title[:max_title_length] + "..." if len(title) > max_title_length else title)
            truncated_link = ""

        final_str.append(f"{i+1}. {truncated_link} - {title}")

    return "\n".join(s for s in final_str)


PRESET_PROMPTS = {
    "threat_intel": """
    You are an Cybercrime Threat Intelligence Expert tasked with generating context-based technical investigative insights from dark web osint search engine results.

    Rules:
    1. Analyze the Darkweb OSINT data provided using links and their raw text.
    2. Output the Source Links referenced for the analysis.
    3. Provide a detailed, contextual, evidence-based technical analysis of the data.
    4. Provide intellgience artifacts along with their context visible in the data.
    5. The artifacts can include indicators like name, email, phone, cryptocurrency addresses, domains, darkweb markets, forum names, threat actor information, malware names, TTPs, etc.
    6. Generate 3-5 key insights based on the data.
    7. Each insight should be specific, actionable, context-based, and data-driven.
    8. Include suggested next steps and queries for investigating more on the topic.
    9. Be objective and analytical in your assessment.
    10. Ignore not safe for work texts from the analysis

    Output Format:
    1. Input Query: {query}
    2. Source Links Referenced for Analysis - this heading will include all source links used for the analysis
    3. Investigation Artifacts - this heading will include all technical artifacts identified including name, email, phone, cryptocurrency addresses, domains, darkweb markets, forum names, threat actor information, malware names, etc.
    4. Key Insights
    5. Next Steps - this includes next investigative steps including search queries to search more on a specific artifacts for example or any other topic.

    Format your response in a structured way with clear section headings.

    INPUT:
    """,
    "ransomware_malware": """
    You are a Malware and Ransomware Intelligence Expert tasked with analyzing dark web data for malware-related threats.

    Rules:
    1. Analyze the Darkweb OSINT data provided using links and their raw text.
    2. Output the Source Links referenced for the analysis.
    3. Focus specifically on ransomware groups, malware families, exploit kits, and attack infrastructure.
    4. Identify malware indicators: file hashes, C2 domains/IPs, staging URLs, payload names, and obfuscation techniques.
    5. Map TTPs to MITRE ATT&CK where possible.
    6. Identify victim organizations, sectors, or geographies mentioned.
    7. Generate 3-5 key insights focused on threat actor behavior and malware evolution.
    8. Include suggested next steps for containment, detection, and further hunting.
    9. Be objective and analytical. Ignore not safe for work texts.

    Output Format:
    1. Input Query: {query}
    2. Source Links Referenced for Analysis
    3. Malware / Ransomware Indicators (hashes, C2s, payload names, TTPs)
    4. Threat Actor Profile (group name, aliases, known victims, sector targeting)
    5. Key Insights
    6. Next Steps (hunting queries, detection rules, further investigation)

    Format your response in a structured way with clear section headings.

    INPUT:
    """,
    "personal_identity": """
    You are a Personal Threat Intelligence Expert tasked with analyzing dark web data for identity and personal information exposure.

    Rules:
    1. Analyze the Darkweb OSINT data provided using links and their raw text.
    2. Output the Source Links referenced for the analysis.
    3. Focus on personally identifiable information (PII): names, emails, phone numbers, addresses, SSNs, passport data, financial account details.
    4. Identify breach sources, data brokers, and marketplaces selling personal data.
    5. Assess exposure severity: what data is available and how actionable is it for a threat actor.
    6. Generate 3-5 key insights on the individual's exposure risk.
    7. Include recommended protective actions and further investigation queries.
    8. Be objective. Ignore not safe for work texts. Handle all personal data with discretion.

    Output Format:
    1. Input Query: {query}
    2. Source Links Referenced for Analysis
    3. Exposed PII Artifacts (type, value, source context)
    4. Breach / Marketplace Sources Identified
    5. Exposure Risk Assessment
    6. Key Insights
    7. Next Steps (protective actions, further queries)

    Format your response in a structured way with clear section headings.

    INPUT:
    """,
    "corporate_espionage": """
    You are a Corporate Intelligence Expert tasked with analyzing dark web data for corporate data leaks and espionage activity.

    Rules:
    1. Analyze the Darkweb OSINT data provided using links and their raw text.
    2. Output the Source Links referenced for the analysis.
    3. Focus on leaked corporate data: credentials, source code, internal documents, financial records, employee data, customer databases.
    4. Identify threat actors, insider threat indicators, and data broker activity targeting the organization.
    5. Assess business impact: what competitive or operational damage could result from the exposure.
    6. Generate 3-5 key insights on the corporate risk posture.
    7. Include recommended incident response steps and further investigation queries.
    8. Be objective and analytical. Ignore not safe for work texts.

    Output Format:
    1. Input Query: {query}
    2. Source Links Referenced for Analysis
    3. Leaked Corporate Artifacts (credentials, documents, source code, databases)
    4. Threat Actor / Broker Activity
    5. Business Impact Assessment
    6. Key Insights
    7. Next Steps (IR actions, legal considerations, further queries)

    Format your response in a structured way with clear section headings.

    INPUT:
    """,
}


def generate_summary(
    llm,
    query,
    content,
    preset: str = "threat_intel",
    custom_instructions: str = "",
    system_prompt_override: str | None = None,
):
    """
    Generate the final investigation summary.

    Resolution order for the system prompt:
      1. `system_prompt_override` (if provided) — used verbatim. This is how
         the UI passes user-defined custom-domain prompts through.
      2. `PRESET_PROMPTS[preset]` — one of the four built-in domains.
      3. `PRESET_PROMPTS["threat_intel"]` — final fallback for unknown keys.
    """
    if system_prompt_override and system_prompt_override.strip():
        system_prompt = system_prompt_override
    else:
        system_prompt = PRESET_PROMPTS.get(preset, PRESET_PROMPTS["threat_intel"])

    if custom_instructions and custom_instructions.strip():
        system_prompt = system_prompt.rstrip() + f"\n\nAdditionally focus on: {custom_instructions.strip()}"
    prompt_template = ChatPromptTemplate(
        [("system", system_prompt), ("user", "{content}")]
    )
    chain = prompt_template | llm | StrOutputParser()
    return _invoke_with_retry(chain, {"query": query, "content": content}, stage="generate_summary")