"""Unified LLM client with provider switching (OpenAI / Groq).

Reads LLM_PROVIDER from .env to decide the backend.
Groq uses the OpenAI-compatible API, so a single openai.OpenAI client
handles both — just swap base_url and api_key.
"""
import os
from typing import List, Dict, Optional, Any

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False


# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq":   "qwen-qwq-32b",
}

PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq":   "https://api.groq.com/openai/v1",
}


class LLMClient:
    """Provider-agnostic LLM client.

    Usage:
        client = LLMClient()                       # reads .env
        reply  = client.chat("Summarise this...")   # single string shortcut
        reply  = client.chat_messages([...])        # full messages list
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower().strip()
        raw_model = model or os.environ.get("LLM_MODEL", "") or DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.model = raw_model.strip()

        # Resolve API key
        if api_key:
            self._api_key = api_key.strip()
        elif self.provider == "openai":
            self._api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        elif self.provider == "groq":
            self._api_key = os.environ.get("GROQ_API_KEY", "").strip()
        else:
            self._api_key = (os.environ.get("OPENAI_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")).strip()

        self.client: Optional[Any] = None
        self._available = False

        if not OPENAI_SDK_AVAILABLE:
            print("[LLMClient] openai SDK not installed. LLM calls will use local fallback.")
            return

        if not self._api_key:
            print(f"[LLMClient] No API key for provider '{self.provider}'. LLM calls will use local fallback.")
            return

        base_url = PROVIDER_URLS.get(self.provider, PROVIDER_URLS["openai"])
        try:
            self.client = OpenAI(api_key=self._api_key, base_url=base_url)
            self._available = True
            print(f"[LLMClient] Initialized — provider={self.provider}, model={self.model}")
        except Exception as e:
            print(f"[LLMClient] Failed to initialize ({e}). Using local fallback.")

    @property
    def is_available(self) -> bool:
        return self._available

    def chat(self, prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 256) -> str:
        """Simple single-prompt chat. Returns the assistant reply text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_messages(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Send a full messages list and return the assistant content string."""
        if not self._available:
            return ""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if hasattr(resp, "usage") and resp.usage:
                print(f"  [{self.provider.upper()} API HIT] ID: {resp.id} | Model: {resp.model} | Total Tokens: {resp.usage.total_tokens}", flush=True)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to direct HTTP POST via requests (avoids Windows WDAC DLL load blocks on jiter.pyd)
            try:
                import requests
                base_url = PROVIDER_URLS.get(self.provider, PROVIDER_URLS["openai"])
                url = f"{base_url.rstrip('/')}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                r = requests.post(url, headers=headers, json=payload, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    usage = data.get("usage", {})
                    call_id = data.get("id", "req-http")
                    print(f"  [{self.provider.upper()} HTTP HIT] ID: {call_id} | Model: {self.model} | Total Tokens: {usage.get('total_tokens')}", flush=True)
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[LLMClient HTTP Error] {r.status_code}: {r.text[:150]}")
            except Exception as http_err:
                pass
            print(f"[LLMClient] API call failed: {e}")
            return ""

    def __repr__(self):
        status = "active" if self._available else "fallback"
        return f"LLMClient(provider={self.provider}, model={self.model}, status={status})"

