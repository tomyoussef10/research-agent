"""Thin wrapper over the Anthropic Messages API.

Everything else in the agent talks to this interface, so tests can swap in a fake LLM
and the model provider can be changed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
