import requests
from urllib.parse import urljoin
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Callable, Optional, List
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks.base import BaseCallbackHandler
from config import (
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    LLAMA_CPP_BASE_URL,
)


class BufferedStreamingHandler(BaseCallbackHandler):
    def __init__(self, buffer_limit: int = 60, ui_callback: Optional[Callable[[str], None]] = None):
        self.buffer = ""
        self.buffer_limit = buffer_limit
        self.ui_callback = ui_callback

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.buffer += token
        if "\n" in token or len(self.buffer) >= self.buffer_limit:
            print(self.buffer, end="", flush=True)
            if self.ui_callback:
                self.ui_callback(self.buffer)
            self.buffer = ""

    def on_llm_end(self, response, **kwargs) -> None:
        if self.buffer:
            print(self.buffer, end="", flush=True)
            if self.ui_callback:
                self.ui_callback(self.buffer)
            self.buffer = ""


# --- Configuration Data ---

# Common parameters shared by all LLMs.
# NOTE: 'callbacks' is intentionally excluded here. A fresh BufferedStreamingHandler
# is created per get_llm() call in llm.py to prevent the shared-singleton bug where
# multiple pipeline stages (refine, filter, summarize) would reference the same
# handler object and potentially leak ui_callback state between calls.
_common_llm_params = {
    "temperature": 0,
    "streaming": True,
}

# Map input model choices (lowercased) to their configuration.
# Each config includes the class and any model-specific constructor parameters.
_llm_config_map = {
    'gpt-4.1': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-4.1'}
    },
    'gpt-5.2': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5.2'}
    },
    'gpt-5.1': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5.1'}
    },
    'gpt-5-mini': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-mini'}
    },
    'gpt-5-nano': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-nano'}
    },
    'claude-sonnet-4-5': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-5'}
    },
    'claude-sonnet-4-0': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-0'}
    },
    'gemini-2.5-flash': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash', 'google_api_key': GOOGLE_API_KEY}
    },
    'gemini-2.5-flash-lite': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash-lite', 'google_api_key': GOOGLE_API_KEY}
    },
    'gemini-2.5-pro': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-pro', 'google_api_key': GOOGLE_API_KEY}
    },
    'qwen3-80b-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'qwen/qwen3-next-80b-a3b-instruct:free',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY,
        }
    },
    'nemotron-nano-9b-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'nvidia/nemotron-nano-9b-v2:free',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY,
        }
    },
    'gpt-oss-120b-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'openai/gpt-oss-120b:free',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY,
        }
    },
    'gpt-5.1-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'openai/gpt-5.1',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY,
        }
    },
    'gpt-5-mini-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'openai/gpt-5-mini',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY,
        }
    },
    'claude-sonnet-4.5-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'anthropic/claude-sonnet-4.5',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY,
        }
    },
    'grok-4.1-fast-openrouter': {
        'class': ChatOpenAI,
        'constructor_params': {
            'model_name': 'x-ai/grok-4.1-fast',
            'base_url': OPENROUTER_BASE_URL,
            'api_key': OPENROUTER_API_KEY,
        }
    },
}


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


def _get_ollama_base_url() -> Optional[str]:
    if not OLLAMA_BASE_URL:
        return None
    return OLLAMA_BASE_URL.rstrip("/") + "/"


def fetch_ollama_models() -> List[str]:
    """
    Retrieve the list of locally available Ollama models by querying the Ollama HTTP API.
    Returns an empty list if the API isn't reachable or the base URL is not defined.
    """
    base_url = _get_ollama_base_url()
    if not base_url:
        return []

    try:
        resp = requests.get(urljoin(base_url, "api/tags"), timeout=3)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        available = []
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                available.append(name)
        return available
    except (requests.RequestException, ValueError):
        return []


def fetch_llama_cpp_models() -> List[str]:
    """
    Retrieve available models from an OpenAI-compatible llama.cpp server.
    """
    if not LLAMA_CPP_BASE_URL:
        return []

    base = LLAMA_CPP_BASE_URL.rstrip("/")
    try:
        resp = requests.get(f"{base}/v1/models", timeout=3)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [m["id"] for m in data if "id" in m]
    except (requests.RequestException, ValueError, KeyError):
        return []


def _is_set(v: Optional[str]) -> bool:
    return bool(v and str(v).strip() and "your_" not in str(v))


def get_model_choices() -> List[str]:
    """
    Combine configured cloud models with locally available Ollama/llama.cpp models.
    Cloud models are shown only if the required API keys are present.
    """
    gated_base_models: List[str] = []

    openai_ok = _is_set(OPENAI_API_KEY)
    anthropic_ok = _is_set(ANTHROPIC_API_KEY)
    google_ok = _is_set(GOOGLE_API_KEY)
    openrouter_ok = _is_set(OPENROUTER_API_KEY) and _is_set(OPENROUTER_BASE_URL)

    for k, cfg in _llm_config_map.items():
        cls = cfg.get("class")
        ctor = cfg.get("constructor_params", {}) or {}

        if cls is ChatOpenAI and (ctor.get("base_url") == OPENROUTER_BASE_URL or "openrouter" in k):
            if openrouter_ok:
                gated_base_models.append(k)
            continue

        if cls is ChatOpenAI:
            if openai_ok:
                gated_base_models.append(k)
            continue

        if cls is ChatAnthropic:
            if anthropic_ok:
                gated_base_models.append(k)
            continue

        if cls is ChatGoogleGenerativeAI:
            if google_ok:
                gated_base_models.append(k)
            continue

        gated_base_models.append(k)

    dynamic_models = []
    dynamic_models += fetch_ollama_models()
    dynamic_models += fetch_llama_cpp_models()

    normalized = {_normalize_model_name(m): m for m in gated_base_models}
    for dm in dynamic_models:
        key = _normalize_model_name(dm)
        if key not in normalized:
            normalized[key] = dm

    ordered_dynamic = sorted(
        [name for key, name in normalized.items() if name not in gated_base_models],
        key=_normalize_model_name,
    )
    return gated_base_models + ordered_dynamic


def resolve_model_config(model_choice: str):
    """
    Resolve a model choice (case-insensitive) to the corresponding configuration.
    Supports both predefined remote models and locally installed Ollama/llama.cpp models.
    """
    model_choice_lower = _normalize_model_name(model_choice)
    config = _llm_config_map.get(model_choice_lower)
    if config:
        return config

    for llama_model in fetch_llama_cpp_models():
        if _normalize_model_name(llama_model) == model_choice_lower:
            return {
                "class": ChatOpenAI,
                "constructor_params": {
                    "model_name": llama_model,
                    "base_url": LLAMA_CPP_BASE_URL,
                    "api_key": OPENAI_API_KEY or "sk-local",
                },
            }

    for ollama_model in fetch_ollama_models():
        if _normalize_model_name(ollama_model) == model_choice_lower:
            return {
                "class": ChatOllama,
                "constructor_params": {"model": ollama_model, "base_url": OLLAMA_BASE_URL},
            }

    return None