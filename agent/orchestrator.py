"""The agent loop: the LLM picks a tool, the validator runs it, repeat until done."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .step_log import Step, StepLog
from .synthesizer import synthesize
from .validator import Validator

ORCHESTRATOR_PROMPT = """You are a research assistant that answers multi-step technical questions.
You do not answer from memory. You gather evidence with tools, one call at a time:

- search_papers: facts, numbers, or methods from our local AI/ML paper collection.
- web_search: recent or external facts that are not in those papers.
- run_python: any arithmetic, ratio, percentage, or conversion. Never do math in your head.

Work step by step. Before each tool call, think about what you still need. Use the results
you already have. If a tool fails or returns nothing, try a different query or another tool.
You may make at most {max_calls} tool calls in total.

When you have enough evidence, or cannot get more, stop calling tools and reply with a
short note saying you are done. A separate step writes the final answer from the tool log."""


@dataclass
class AgentResult:
    question: str
    answer: str
    log: StepLog
    hit_step_limit: bool


class Orchestrator:
    def __init__(self, llm, validator: Validator, log: StepLog, max_tool_calls: int = 6,
                 on_step: Callable[[Step], None] | None = None):
        self.llm = llm
        self.validator = validator
        self.log = log
        self.max_tool_calls = max_tool_calls
        self.on_step = on_step or (lambda step: None)

    def run(self, question: str) -> AgentResult:
        specs = [t.spec() for t in self.validator.tools.values()]
        system = ORCHESTRATOR_PROMPT.format(max_calls=self.max_tool_calls)
        messages: list[dict] = [{"role": "user", "content": question}]
        calls, hit_limit = 0, False

        while True:
            reply = self.llm.create(system=system, messages=messages, tools=specs)
            messages.append({"role": "assistant", "content": reply.blocks})
            if not reply.tool_calls and not reply.text and not self.log.steps and len(messages) == 2:
                # A blank first reply (rare with Gemini): nudge once to use a tool.
                messages.append({"role": "user", "content": "Please use the tools to gather evidence first."})
                continue
            if not reply.tool_calls:
                break

            results = []
            for call in reply.tool_calls:
                if calls >= self.max_tool_calls:
                    hit_limit = True
                    results.append({"type": "tool_result", "tool_use_id": call["id"],
                                    "content": "Tool-call limit reached. Stop and say you are done.",
                                    "is_error": True})
                    continue
                calls += 1
                before = len(self.log.steps)
                outcome = self.validator.run(call["name"], call["input"])
                for step in self.log.steps[before:]:
                    self.on_step(step)
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": outcome.content, "is_error": outcome.is_error})
            messages.append({"role": "user", "content": results})

            if hit_limit:
                break

        answer = synthesize(self.llm, question, self.log)
        return AgentResult(question=question, answer=answer, log=self.log, hit_step_limit=hit_limit)
