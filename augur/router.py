"""
Augur — Multi-Provider Model Router

Routes specialist LLM calls to the correct provider SDK based on the model string.
Supports Anthropic (Claude), OpenAI (GPT), Google (Gemini), and OpenRouter
(DeepSeek, Llama, Mistral, etc. via OpenAI-compatible API).
"""
from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(model: str) -> Provider:
    """Detect the provider for a model string using prefix matching."""
    if model.startswith("claude-"):
        return Provider.ANTHROPIC
    if model.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-")):
        return Provider.OPENAI
    if model.startswith("gemini-"):
        return Provider.GOOGLE
    if "/" in model:
        return Provider.OPENROUTER
    raise ValueError(f"Cannot detect provider for model: {model}")


# ---------------------------------------------------------------------------
# API key mapping
# ---------------------------------------------------------------------------

_API_KEY_MAP: Dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GOOGLE: "GOOGLE_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
}


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def available_providers() -> Dict[Provider, bool]:
    """Check which providers have API keys set in the environment."""
    return {p: _API_KEY_MAP[p] in os.environ for p in Provider}


# ---------------------------------------------------------------------------
# Model resolution from TOML config
# ---------------------------------------------------------------------------

def resolve_model(model_config: dict) -> Tuple[str, Provider]:
    """Resolve a model from a TOML ``[model]`` config section.

    Tries ``primary`` first, then ``fallback``.  Returns ``(model_name, provider)``
    for the first model whose provider API key is available.
    """
    avail = available_providers()
    missing_keys: list[str] = []

    for field in ("primary", "fallback"):
        model_name = model_config.get(field)
        if model_name is None:
            continue
        provider = detect_provider(model_name)
        if avail.get(provider):
            return model_name, provider
        missing_keys.append(_API_KEY_MAP[provider])

    if not missing_keys:
        raise RuntimeError(
            "Model config must contain at least a 'primary' or 'fallback' key."
        )
    raise RuntimeError(
        f"No API key available. Set one of: {', '.join(missing_keys)}"
    )


# ---------------------------------------------------------------------------
# Dispatch to provider SDKs
# ---------------------------------------------------------------------------

async def send_message(
    model: str,
    provider: Provider,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    """Send a message to the correct provider SDK and return the text response."""
    key = _API_KEY_MAP[provider]

    if provider == Provider.ANTHROPIC:
        return await _send_anthropic(model, key, system, user, temperature, max_tokens, timeout)

    if provider in (Provider.OPENAI, Provider.OPENROUTER):
        return await _send_openai(model, provider, key, system, user, temperature, max_tokens, timeout)

    if provider == Provider.GOOGLE:
        return await _send_google(model, key, system, user, temperature, max_tokens, timeout)

    raise ValueError(f"Unsupported provider: {provider}")  # pragma: no cover


async def _send_anthropic(
    model: str, key: str, system: str, user: str,
    temperature: float, max_tokens: int, timeout: int,
) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic SDK not installed. pip install anthropic")

    client = anthropic.AsyncAnthropic(api_key=os.environ[key])
    response = await asyncio.wait_for(
        client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ),
        timeout=timeout,
    )
    return response.content[0].text


async def _send_openai(
    model: str, provider: Provider, key: str, system: str, user: str,
    temperature: float, max_tokens: int, timeout: int,
) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai SDK not installed. pip install openai")

    kwargs: dict = {"api_key": os.environ[key]}
    if provider == Provider.OPENROUTER:
        kwargs["base_url"] = "https://openrouter.ai/api/v1"

    client = openai.AsyncOpenAI(**kwargs)
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ),
        timeout=timeout,
    )
    return response.choices[0].message.content


async def _send_google(
    model: str, key: str, system: str, user: str,
    temperature: float, max_tokens: int, timeout: int,
) -> str:
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
    except ImportError:
        raise RuntimeError("google-genai SDK not installed. pip install google-genai")

    client = genai.Client(api_key=os.environ[key])
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=model,
            contents=user,
            config=GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        ),
        timeout=timeout,
    )
    return response.text
