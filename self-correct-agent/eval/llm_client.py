"""Small OpenAI-compatible client with deterministic request caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMConfigurationError(RuntimeError):
    pass


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # 不覆盖已导出的环境变量，方便命令行临时替换模型配置。
            os.environ.setdefault(key, value)


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
        cache_dir: Path | None = None,
    ) -> None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL")
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir or Path(__file__).resolve().parent / "cache"

        if not self.api_key:
            raise LLMConfigurationError("Set OPENAI_API_KEY (or DEEPSEEK_API_KEY) before running evaluation.")
        if not self.model:
            raise LLMConfigurationError("Set OPENAI_MODEL (or DEEPSEEK_MODEL) before running evaluation.")

    def complete(self, prompt: str) -> str:
        # 缓存键包含模型、端点、提示词和生成参数，避免不同实验条件错误复用结果。
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        cache_key = hashlib.sha256(
            json.dumps({"base_url": self.base_url, **payload}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["content"]

        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach LLM endpoint {self.base_url}: {error.reason}") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected LLM response: {body!r}") from error

        # 先缓存再返回，使中断后的 --resume 不会再次调用已完成请求。
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
        return content


class LocalLlamaServerClient(OpenAICompatibleClient):
    """OpenAI-compatible client for a local llama.cpp `llama-server` instance."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
        cache_dir: Path | None = None,
    ) -> None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        host = os.getenv("LOCAL_LLM_HOST", "127.0.0.1")
        port = os.getenv("LOCAL_LLM_PORT", "8080")
        # llama.cpp 复用 OpenAI Chat Completions 协议，因此可继承通用客户端。
        server_url = base_url or os.getenv("LOCAL_LLM_BASE_URL") or f"http://{host}:{port}/v1"
        super().__init__(
            api_key=os.getenv("LOCAL_LLM_API_KEY") or "local",
            base_url=server_url,
            model=model or os.getenv("LOCAL_LLM_MODEL") or "local-qwen3-8b",
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
        )


class AnthropicCompatibleClient:
    """Client for Anthropic Messages-compatible endpoints, including DeepSeek."""

    def __init__(
        self,
        *,
        auth_token: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        thinking: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        self.auth_token = auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.model = model or os.getenv("ANTHROPIC_MODEL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking or os.getenv("ANTHROPIC_THINKING") or "disabled"
        self.cache_dir = cache_dir or Path(__file__).resolve().parent / "cache"

        if not self.auth_token:
            raise LLMConfigurationError("Set ANTHROPIC_AUTH_TOKEN before running evaluation.")
        if not self.model:
            raise LLMConfigurationError("Set ANTHROPIC_MODEL before running evaluation.")
        if self.thinking not in {"enabled", "disabled"}:
            raise LLMConfigurationError("ANTHROPIC_THINKING must be 'enabled' or 'disabled'.")

    def complete(self, prompt: str) -> str:
        last_body: dict[str, Any] | None = None
        # Some reasoning models emit a long hidden-thinking block before text. Retry
        # only that incomplete protocol response with a larger output allowance.
        for max_tokens in (self.max_tokens, self.max_tokens * 2):
            # 部分推理模型会先耗尽隐藏思考 token；仅对这种无文本响应加大预算重试。
            payload: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "thinking": {"type": self.thinking},
                "messages": [{"role": "user", "content": prompt}],
            }
            cache_key = hashlib.sha256(
                json.dumps({"base_url": self.base_url, **payload}, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_path = self.cache_dir / f"anthropic-{cache_key}.json"
            if cache_path.exists():
                return json.loads(cache_path.read_text(encoding="utf-8"))["content"]

            request = Request(
                f"{self.base_url}/v1/messages",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "x-api-key": self.auth_token,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
            except URLError as error:
                raise RuntimeError(f"Could not reach LLM endpoint {self.base_url}: {error.reason}") from error

            try:
                content = "".join(
                    block["text"] for block in body["content"] if block.get("type") == "text"
                )
            except (KeyError, TypeError) as error:
                raise RuntimeError(f"Unexpected LLM response: {body!r}") from error
            if content:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
                return content
            last_body = body

        raise RuntimeError(f"LLM response contains no text content after retry: {last_body!r}")
