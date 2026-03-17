"""Tests for the Augur multi-provider model router."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augur.router import (
    Provider,
    _API_KEY_MAP,
    available_providers,
    detect_provider,
    resolve_model,
    send_message,
)


# ---------------------------------------------------------------------------
# detect_provider
# ---------------------------------------------------------------------------

class TestDetectProvider:

    def test_claude_sonnet(self):
        assert detect_provider("claude-sonnet-4-6") == Provider.ANTHROPIC

    def test_claude_haiku(self):
        assert detect_provider("claude-haiku-4-5-20251001") == Provider.ANTHROPIC

    def test_gpt_4o(self):
        assert detect_provider("gpt-4o") == Provider.OPENAI

    def test_gpt_4o_mini(self):
        assert detect_provider("gpt-4o-mini") == Provider.OPENAI

    def test_o3_mini(self):
        assert detect_provider("o3-mini") == Provider.OPENAI

    def test_o4_mini(self):
        assert detect_provider("o4-mini") == Provider.OPENAI

    def test_gemini_25_flash(self):
        assert detect_provider("gemini-2.5-flash") == Provider.GOOGLE

    def test_gemini_20_flash(self):
        assert detect_provider("gemini-2.0-flash") == Provider.GOOGLE

    def test_deepseek_openrouter(self):
        assert detect_provider("deepseek/deepseek-r1-0528") == Provider.OPENROUTER

    def test_llama_openrouter(self):
        assert detect_provider("meta-llama/llama-3-70b") == Provider.OPENROUTER

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            detect_provider("unknown-model")


# ---------------------------------------------------------------------------
# available_providers
# ---------------------------------------------------------------------------

class TestAvailableProviders:

    def test_only_anthropic_key_set(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        result = available_providers()
        assert result[Provider.ANTHROPIC] is True
        assert result[Provider.OPENAI] is False
        assert result[Provider.GOOGLE] is False
        assert result[Provider.OPENROUTER] is False

    def test_all_keys_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
        monkeypatch.setenv("GOOGLE_API_KEY", "sk-g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-r")

        result = available_providers()
        assert all(result.values())

    def test_no_keys_set(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = available_providers()
        assert not any(result.values())


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:

    def test_primary_available(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        config = {"primary": "claude-sonnet-4-6", "fallback": "gpt-4o"}
        model, provider = resolve_model(config)
        assert model == "claude-sonnet-4-6"
        assert provider == Provider.ANTHROPIC

    def test_primary_unavailable_fallback_available(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = {"primary": "claude-sonnet-4-6", "fallback": "gpt-4o"}
        model, provider = resolve_model(config)
        assert model == "gpt-4o"
        assert provider == Provider.OPENAI

    def test_neither_available_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {"primary": "claude-sonnet-4-6", "fallback": "gpt-4o"}
        with pytest.raises(RuntimeError, match="No API key available"):
            resolve_model(config)

    def test_missing_primary_tries_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = {"fallback": "gpt-4o"}
        model, provider = resolve_model(config)
        assert model == "gpt-4o"
        assert provider == Provider.OPENAI

    def test_missing_both_keys_raises(self):
        config = {}
        with pytest.raises(RuntimeError, match="must contain"):
            resolve_model(config)


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------

class TestSendMessage:

    @pytest.mark.asyncio
    async def test_anthropic_dispatch(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        mock_content = MagicMock()
        mock_content.text = "Anthropic response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_create = AsyncMock(return_value=mock_response)
        mock_messages = MagicMock()
        mock_messages.create = mock_create
        mock_client = MagicMock()
        mock_client.messages = mock_messages

        mock_module = MagicMock()
        mock_module.AsyncAnthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            result = await send_message(
                model="claude-sonnet-4-6",
                provider=Provider.ANTHROPIC,
                system="You are a forecaster.",
                user="Will it rain?",
                temperature=0.3,
                max_tokens=1024,
                timeout=30,
            )

        assert result == "Anthropic response"
        mock_module.AsyncAnthropic.assert_called_once_with(api_key="sk-test")
        mock_create.assert_called_once_with(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            temperature=0.3,
            system="You are a forecaster.",
            messages=[{"role": "user", "content": "Will it rain?"}],
        )

    @pytest.mark.asyncio
    async def test_openai_dispatch(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_message = MagicMock()
        mock_message.content = "OpenAI response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_module = MagicMock()
        mock_module.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_module}):
            result = await send_message(
                model="gpt-4o",
                provider=Provider.OPENAI,
                system="You are a forecaster.",
                user="Will it rain?",
                temperature=0.5,
                max_tokens=2048,
                timeout=60,
            )

        assert result == "OpenAI response"
        mock_module.AsyncOpenAI.assert_called_once_with(api_key="sk-test")
        mock_create.assert_called_once_with(
            model="gpt-4o",
            max_tokens=2048,
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a forecaster."},
                {"role": "user", "content": "Will it rain?"},
            ],
        )

    @pytest.mark.asyncio
    async def test_openrouter_dispatch_sets_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        mock_message = MagicMock()
        mock_message.content = "OpenRouter response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_response)
        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        mock_module = MagicMock()
        mock_module.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_module}):
            result = await send_message(
                model="deepseek/deepseek-r1-0528",
                provider=Provider.OPENROUTER,
                system="You are a forecaster.",
                user="Will it rain?",
                temperature=0.3,
                max_tokens=1024,
                timeout=30,
            )

        assert result == "OpenRouter response"
        mock_module.AsyncOpenAI.assert_called_once_with(
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )

    @pytest.mark.asyncio
    async def test_google_dispatch(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.text = "Google response"

        mock_generate = AsyncMock(return_value=mock_response)
        mock_models = MagicMock()
        mock_models.generate_content = mock_generate
        mock_aio = MagicMock()
        mock_aio.models = mock_models
        mock_client = MagicMock()
        mock_client.aio = mock_aio

        mock_config_cls = MagicMock()
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = mock_config_cls

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        # Wire the google parent module so `from google import genai` works
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            result = await send_message(
                model="gemini-2.5-flash",
                provider=Provider.GOOGLE,
                system="You are a forecaster.",
                user="Will it rain?",
                temperature=0.4,
                max_tokens=512,
                timeout=30,
            )

        assert result == "Google response"
        mock_genai.Client.assert_called_once_with(api_key="sk-test")
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs.kwargs["contents"] == "Will it rain?"

    @pytest.mark.asyncio
    async def test_timeout_propagates(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        async def slow_create(**kwargs):
            await asyncio.sleep(10)

        mock_messages = MagicMock()
        mock_messages.create = slow_create
        mock_client = MagicMock()
        mock_client.messages = mock_messages

        mock_module = MagicMock()
        mock_module.AsyncAnthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            with pytest.raises(asyncio.TimeoutError):
                await send_message(
                    model="claude-sonnet-4-6",
                    provider=Provider.ANTHROPIC,
                    system="sys",
                    user="usr",
                    temperature=0.0,
                    max_tokens=100,
                    timeout=0.01,
                )
