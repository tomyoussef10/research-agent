"""The step log: every tool call, its input, outcome, and sources. Saved as a JSON trace."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .tools.base import Source


@dataclass
class Step:
    n: int
    tool: str
    args: dict
    status: str  # "ok", "empty", "error", "rejected", "fallback"
    output: str
    sources: list[dict] = field(default_factory=list)
    note: str = ""
    attempts: int = 1
    seconds: float = 0.0


class StepLog:
    def __init__(self):
        self.steps: list[Step] = []
        self.started = time.time()

    def add(self, tool: str, args: dict, status: str, output: str,
            sources: list[Source] | None = None, note: str = "", attempts: int = 1,
            seconds: float = 0.0) -> Step:
        step = Step(
            n=len(self.steps) + 1, tool=tool, args=args, status=status, output=output,
            sources=[s.to_dict() for s in (sources or [])], note=note, attempts=attempts,
            seconds=round(seconds, 2),
        )
        self.steps.append(step)
        return step

    def useful_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status in ("ok", "fallback")]

    def numbered_sources(self) -> list[dict]:
        """Unique sources across useful steps, in first-seen order. Index i = citation [i+1]."""
        seen, out = set(), []
        for s in self.useful_steps():
            for src in s.sources:
                if src["location"] not in seen:
                    seen.add(src["location"])
                    out.append(src)
        return out

    def tools_used(self) -> list[str]:
        return [s.tool for s in self.useful_steps()]

    def save(self, trace_dir: Path, question: str, answer: str, usage: dict) -> Path:
        trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = trace_dir / f"trace-{stamp}.json"
        path.write_text(json.dumps({
            "question": question,
            "answer": answer,
            "steps": [asdict(s) for s in self.steps],
            "tools_used": self.tools_used(),
            "seconds_total": round(time.time() - self.started, 2),
            "usage": usage,
        }, indent=2))
        return path
