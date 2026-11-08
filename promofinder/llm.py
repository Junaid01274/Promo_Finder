"""One small interface over whichever model is available.

Most providers now speak the OpenAI chat-completions format, so one adapter
covers Gemini, Groq, OpenRouter, a local Ollama, OpenAI itself, or any
in-house gateway (pass --base-url and --api-key-env). Anthropic gets its own
adapter through the official SDK.

Every response is cached on disk by a hash of (provider, model, prompt), so a
re-run costs nothing and gives the same output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

PRESETS = {
    # name: (base_url, default_model, env var holding the key)
    # Prefer "-latest" aliases: a pinned default (gemini-2.5-flash) was retired for
    # new keys while this was being built. Pass --model to pin one for a run.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai",
               "gemini-flash-latest", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",
                   "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "llama3.1", ""),
    "openai": ("https://api.openai.com/v1", "gpt-4.1-mini", "OPENAI_API_KEY"),
    "anthropic": ("", "claude-opus-5-5", "ANTHROPIC_API_KEY"),
}


_LOCK = threading.Lock()


class LLMError(RuntimeError):
    pass


def load_dotenv() -> None:
    """Read KEY=value lines from a .env next to the project, if there is one.
    Keeps API keys out of shell history and chat. Never overrides a real env var."""
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


@dataclass
class LLM:
    provider: str
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    cache_dir: Path = Path(".cache/llm")
    max_tokens: int = 4000
    timeout: int = 240
    calls: int = 0
    cache_hits: int = 0
    chars_in: int = 0
    chars_out: int = 0

    def __post_init__(self):
        load_dotenv()
        if self.provider not in PRESETS and not self.base_url:
            raise LLMError(f"unknown provider {self.provider!r}; pass --base-url for a custom one")
        base, model, env = PRESETS.get(self.provider, ("", "", ""))
        self.base_url = self.base_url or base
        self.model = self.model or model
        self.api_key_env = self.api_key_env or env
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def _key(self) -> str:
        if not self.api_key_env:
            return ""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise LLMError(f"set {self.api_key_env} to use provider {self.provider!r}")
        return key

    def complete(self, system: str, user: str) -> str:
        digest = hashlib.sha256(
            json.dumps([self.provider, self.model, system, user]).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            with _LOCK:
                self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        text = self._call_anthropic(system, user) if self.provider == "anthropic" \
            else self._call_openai_compatible(system, user)
        with _LOCK:                              # episodes are judged in parallel
            self.calls += 1
            self.chars_in += len(system) + len(user)
            self.chars_out += len(text)
        path.write_text(json.dumps({"model": self.label, "text": text}), encoding="utf-8")
        return text

    def _call_openai_compatible(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        last = ""
        for attempt in range(4):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                # A timeout used to crash the whole run. Now it retries, then the
                # caller falls back to the rule picks for that episode.
                last = f"{type(e).__name__}"
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if r.status_code == 429 and "per day" in r.text.lower():
                # Daily free-tier cap: retrying won't help, so fail fast and let the
                # remaining episodes fall back to the rule picks.
                raise LLMError(f"{self.label}: daily quota used up ({r.status_code})")
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(min(60, 5 * 2 ** attempt))   # free tiers rate-limit hard
                continue
            if r.status_code != 200:
                raise LLMError(f"{self.label} HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                raise LLMError(f"{self.label}: unexpected response shape {str(data)[:200]}")
        raise LLMError(f"{self.label}: gave up after 4 tries ({last})")

    def _call_anthropic(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("pip install anthropic to use provider 'anthropic'") from e
        client = anthropic.Anthropic(api_key=self._key())
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens * 4,      # leave room for adaptive thinking
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if msg.stop_reason == "refusal":
            raise LLMError(f"{self.label} refused the request")
        return "".join(b.text for b in msg.content if b.type == "text")


def parse_json(text: str):
    """Models wrap JSON in prose or ``` fences more often than they should."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i: j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model output")
