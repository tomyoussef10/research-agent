"""Thin wrappers over the LLM APIs (Google Gemini by default, Anthropic Claude optional).

Everything else in the agent talks to one interface, create(system, messages, tools),
using Anthropic-style message dicts. Each wrapper converts to its own API, so tests can
swap in a fake LLM and the provider can be changed with one setting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class LLMReply:
    blocks: list[dict]          # [{"type": "text", "text": ...} | {"type": "tool_use", "id", "name", "input"}]
    stop_reason: str = ""

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.blocks if b["type"] == "tool_use"]

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.blocks if b["type"] == "text").strip()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def cost(self, price_in: float, price_out: float) -> float:
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 1500):
        import anthropic

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        # The SDK retries rate limits and 5xx errors itself; 2 retries keeps runs predictable.
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=60)
        self.model = model
        self.max_tokens = max_tokens
        self.usage = Usage()

    def create(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)

        self.usage.calls += 1
        self.usage.input_tokens += resp.usage.input_tokens
        self.usage.output_tokens += resp.usage.output_tokens

        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return LLMReply(blocks=blocks, stop_reason=resp.stop_reason or "")


def _gemini_schema(schema: dict) -> dict:
    """Gemini wants upper-case type names (OBJECT, STRING, ...) in function schemas."""
    out = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif key == "properties":
            out[key] = {name: _gemini_schema(prop) for name, prop in value.items()}
        else:
            out[key] = value
    return out


class GeminiLLM:
    """Google Gemini through the google-genai SDK (free tier works for this project).

    The SDK's Content objects for model turns are kept and sent back unchanged, which
    preserves Gemini's thought signatures across tool calls.
    """

    RETRY_CODES = {429, 500, 503}

    def __init__(self, api_key: str, model: str, max_tokens: int = 8192, max_retries: int = 5):
        # Gemini 2.5 "thinking" tokens count toward max_tokens, so leave plenty of room.
        from google import genai

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.usage = Usage()
        self._n = 0

    def _contents(self, messages: list[dict]) -> list:
        from google.genai import types

        names: dict[str, str] = {}  # tool_use id -> tool name
        contents = []
        for msg in messages:
            content = msg["content"]
            if msg["role"] == "assistant":
                for b in content:
                    if b["type"] == "tool_use":
                        names[b["id"]] = b["name"]
                raw = next((b["_raw"] for b in content if "_raw" in b), None)
                if raw is not None:
                    contents.append(raw)
                else:  # e.g. history written by hand or by tests
                    contents.append(types.Content(role="model", parts=[types.Part(text=b.get("text", "")) for b in content if b["type"] == "text"] or [types.Part(text="")]))
            elif isinstance(content, str):
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            else:  # list of tool_result blocks
                parts = []
                for r in content:
                    key = "error" if r.get("is_error") else "result"
                    parts.append(types.Part.from_function_response(
                        name=names.get(r["tool_use_id"], "tool"), response={key: r["content"]}))
                contents.append(types.Content(role="user", parts=parts))
        return contents

    def create(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        from google.genai import errors, types

        config = {"system_instruction": system, "max_output_tokens": self.max_tokens}
        if tools:
            config["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters=_gemini_schema(t["input_schema"]))
                for t in tools])]

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=self._contents(messages),
                    config=types.GenerateContentConfig(**config))
                break
            except errors.APIError as e:
                if getattr(e, "code", None) in self.RETRY_CODES and attempt < self.max_retries:
                    wait = min(60, 5 * 2 ** attempt)  # free tier has per-minute limits
                    print(f"  [gemini busy or rate-limited ({e.code}); waiting {wait} s]")
                    time.sleep(wait)
                    continue
                raise

        self.usage.calls += 1
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            self.usage.input_tokens += meta.prompt_token_count or 0
            self.usage.output_tokens += meta.candidates_token_count or 0

        candidate = resp.candidates[0] if resp.candidates else None
        raw = candidate.content if candidate and candidate.content else None
        blocks: list[dict] = []
        for part in (raw.parts or []) if raw else []:
            if part.function_call:
                self._n += 1
                blocks.append({"type": "tool_use", "id": f"call_{self._n}",
                               "name": part.function_call.name, "input": dict(part.function_call.args or {})})
            elif part.text and not getattr(part, "thought", False):
                blocks.append({"type": "text", "text": part.text})
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        if raw is not None:
            blocks[0]["_raw"] = raw
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return LLMReply(blocks=blocks, stop_reason=stop)


def make_llm(cfg) -> "GeminiLLM | AnthropicLLM":
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg.anthropic_api_key, cfg.model)
    if cfg.provider == "gemini":
        return GeminiLLM(cfg.gemini_api_key, cfg.model)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{cfg.provider}' (use 'gemini' or 'anthropic')")
