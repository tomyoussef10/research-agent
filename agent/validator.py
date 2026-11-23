"""Runs tool calls safely: checks arguments, blocks repeats, retries, falls back, logs.

The orchestrator never calls a tool directly; it always goes through Validator.run().
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from .step_log import StepLog
from .tools.base import Tool, ToolError, ToolResult, check_args

# If a tool fails outright, which tool to try instead (with the same "query").
DEFAULT_FALLBACKS = {"web_search": "search_papers", "search_papers": "web_search"}

MAX_BAD_ARGS_PER_TOOL = 2


@dataclass
class Outcome:
    """What goes back to the LLM as the tool_result."""
    content: str
    is_error: bool


class Validator:
    def __init__(
        self,
        tools: dict[str, Tool],
        log: StepLog,
        retries: int = 2,
        backoff_seconds: float = 1.0,
        fallbacks: dict[str, str] | None = None,
        fail_tools: frozenset[str] = frozenset(),
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.tools = tools
        self.log = log
        self.retries = retries
        self.backoff = backoff_seconds
        self.fallbacks = DEFAULT_FALLBACKS if fallbacks is None else fallbacks
        self.fail_tools = fail_tools
        self.sleep = sleep
        self._seen: dict[str, str] = {}       # call key -> result content (repeat guard + cache)
        self._bad_args: dict[str, int] = {}   # tool name -> count of malformed calls

    # ---- public -------------------------------------------------------------

    def run(self, name: str, args: dict) -> Outcome:
        tool = self.tools.get(name)
        if tool is None:
            return self._reject(name, args, f"unknown tool '{name}'. Available: {', '.join(self.tools)}")

        if self._bad_args.get(name, 0) >= MAX_BAD_ARGS_PER_TOOL:
            return self._reject(name, args, f"'{name}' was called with bad arguments too many times; use another tool or answer.")

        problem = check_args(tool.input_schema, args)
        if problem:
            self._bad_args[name] = self._bad_args.get(name, 0) + 1
            return self._reject(name, args, f"invalid arguments for '{name}': {problem}. Fix them and try once more.")

        key = self._key(name, args)
        if key in self._seen:
            return self._reject(name, args, "you already made this exact call; its result is above. Use it or try a different query.")

        outcome = self._run_with_retries(tool, args)
        if outcome is None:  # hard failure -> try fallback
            outcome = self._fallback(name, args)
        self._seen[key] = outcome.content
        return outcome

    # ---- internals ----------------------------------------------------------

    @staticmethod
    def _key(name: str, args: dict) -> str:
        return name + ":" + json.dumps(args, sort_keys=True).lower()

    def _reject(self, name: str, args: dict, message: str) -> Outcome:
        self.log.add(name, args, "rejected", message)
        return Outcome(content=message, is_error=True)

    def _call(self, tool: Tool, args: dict) -> ToolResult:
        if tool.name in self.fail_tools:
            raise ToolError(f"injected failure for testing ({tool.name})", retryable=tool.retryable_errors)
        return tool.run(args)

    def _run_with_retries(self, tool: Tool, args: dict) -> Outcome | None:
        """Returns an Outcome, or None if the tool failed in a way a fallback might fix."""
        attempts, start, last_error = 0, time.time(), ""
        while True:
            attempts += 1
            try:
                result = self._call(tool, args)
            except ToolError as e:
                last_error = str(e)
                if e.retryable and attempts <= self.retries:
                    self.sleep(self.backoff * 2 ** (attempts - 1))
                    continue
                self.log.add(tool.name, args, "error", last_error, attempts=attempts, seconds=time.time() - start)
                if tool.name in self.fallbacks:
                    return None
                # No fallback (e.g. run_python): hand the error to the LLM so it can fix its code.
                return Outcome(content=f"Tool error: {last_error}", is_error=True)

            seconds = time.time() - start
            if result.empty:
                self.log.add(tool.name, args, "empty", result.content, attempts=attempts, seconds=seconds)
                return Outcome(content=result.content, is_error=True)

            self.log.add(tool.name, args, "ok", result.content, result.sources, attempts=attempts, seconds=seconds)
            return Outcome(content=result.content, is_error=False)

    def _fallback(self, failed: str, args: dict) -> Outcome:
        alt_name = self.fallbacks.get(failed)
        alt = self.tools.get(alt_name) if alt_name else None
        if alt is None or "query" not in args:
            return Outcome(content=f"'{failed}' is unavailable and there is no fallback. Answer with what you have and say what is missing.", is_error=True)

        note = f"'{failed}' was unavailable, so '{alt_name}' was used instead."
        start = time.time()
        try:
            result = self._call(alt, {"query": args["query"]})
        except ToolError as e:
            self.log.add(alt_name, {"query": args["query"]}, "error", str(e), note=note, seconds=time.time() - start)
            return Outcome(content=f"'{failed}' and its fallback '{alt_name}' both failed. Answer with what you have and say what is missing.", is_error=True)

        status = "empty" if result.empty else "fallback"
        self.log.add(alt_name, {"query": args["query"]}, status, result.content, result.sources, note=note, seconds=time.time() - start)
        return Outcome(content=f"NOTE: {note}\n\n{result.content}", is_error=result.empty)
