"""Test doubles: a scripted LLM and simple fake tools. No network or API keys needed."""

from __future__ import annotations

from agent.llm import LLMReply, Usage
from agent.tools.base import Source, Tool, ToolError, ToolResult


class ScriptedLLM:
    """Replays tool calls in order, then says it is done. The synthesizer call gets `final`."""

    def __init__(self, tool_calls: list[tuple[str, dict]], final: str = "final answer"):
        self.tool_calls = list(tool_calls)
        self.final = final
        self.usage = Usage()
        self.requests: list[dict] = []
        self._n = 0

    def create(self, system, messages, tools=None):
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        self.usage.calls += 1
        if tools is None:  # synthesizer
            return LLMReply(blocks=[{"type": "text", "text": self.final}], stop_reason="end_turn")
        if self.tool_calls:
            name, args = self.tool_calls.pop(0)
            self._n += 1
            return LLMReply(blocks=[{"type": "tool_use", "id": f"t{self._n}", "name": name, "input": args}],
                            stop_reason="tool_use")
        return LLMReply(blocks=[{"type": "text", "text": "done"}], stop_reason="end_turn")


class FakeSearch(Tool):
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def __init__(self, name: str, fail_times: int = 0, retryable: bool = True, empty: bool = False):
        self.name = name
        self.retryable_errors = retryable
        self.fail_times = fail_times
        self.empty = empty
        self.calls = 0

    def run(self, args):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ToolError("simulated outage", retryable=self.retryable_errors)
        if self.empty:
            return ToolResult(content="nothing", empty=True)
        return ToolResult(content=f"{self.name} result for {args['query']}",
                          sources=[Source(title=f"{self.name} doc", location=f"https://example.org/{self.name}")])
