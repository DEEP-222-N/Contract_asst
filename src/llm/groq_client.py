"""Thin wrapper around the Groq SDK with on-disk response caching."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "llm"


class GroqLLM:
    """A tiny chat wrapper that caches by (model, system, user, temperature)."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        self.temperature = temperature
        self._client = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _client_lazy(self):
        if self._client is None:
            from groq import Groq  # lazy import so eval runs without the dep installed
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to .env or export it."
                )
            self._client = Groq(api_key=key)
        return self._client

    def _cache_path(self, system: str, user: str) -> Path:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(f"\0{self.temperature}\0".encode())
        h.update(system.encode("utf-8"))
        h.update(b"\0")
        h.update(user.encode("utf-8"))
        return CACHE_DIR / f"{h.hexdigest()[:20]}.json"

    def chat(
        self,
        system: str,
        user: str,
        *,
        force: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        cp = self._cache_path(system, user)
        if not force and cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))["content"]

        client = self._client_lazy()
        t0 = time.time()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "system": system,
            "user": user,
            "content": content,
            "latency_s": time.time() - t0,
        }
        cp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return content
