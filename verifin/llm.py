"""Minimal OpenAI-compatible chat client (works with NVIDIA NIM at build.nvidia.com,
vLLM, Ollama, etc.). Zero dependencies: stdlib urllib only."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional


class ChatClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: int = 120):
        self.base_url = (base_url or os.getenv("VERIFIN_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("VERIFIN_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
        self.model = model or os.getenv("VERIFIN_MODEL", "meta/llama-3.1-70b-instruct")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> Optional["ChatClient"]:
        c = cls()
        return c if c.api_key else None

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]
