"""Writes the final answer using ONLY the step log, with numbered citations."""

from __future__ import annotations

from .step_log import StepLog

SYNTH_PROMPT = """You write the final answer to a technical question using ONLY the evidence below.
Rules:
- Use only facts that appear in the evidence. Do not add facts from memory.
- After each fact taken from a source, cite it with its number, like [1] or [2].
- Numbers computed by run_python may be stated as computed; mention the inputs.
- If the evidence does not answer part of the question, say exactly what is missing.
- If a tool was unavailable (see notes), say so in one sentence.
- Be concise: a direct answer first, then at most a short paragraph of support."""


def _format_evidence(log: StepLog) -> str:
    sources = log.numbered_sources()
    index = {src["location"]: i + 1 for i, src in enumerate(sources)}
    parts = []
    for step in log.steps:
        if step.status in ("ok", "fallback"):
            cites = ", ".join(f"[{index[s['location']]}]" for s in step.sources)
            header = f"Step {step.n}: {step.tool}({step.args})" + (f" -> sources {cites}" if cites else "")
            if step.note:
                header += f"\nNote: {step.note}"
            parts.append(f"{header}\n{step.output}")
        elif step.status == "error" and step.note:
            parts.append(f"Step {step.n}: {step.tool} failed. Note: {step.note}")
    if not parts:
        return "(No tool returned usable evidence.)"
    src_lines = "\n".join(f"[{i + 1}] {s['title']} - {s['location']}" for i, s in enumerate(sources))
    return "\n\n".join(parts) + ("\n\nSOURCES\n" + src_lines if src_lines else "")


def synthesize(llm, question: str, log: StepLog) -> str:
    evidence = _format_evidence(log)
    reply = llm.create(
        system=SYNTH_PROMPT,
        messages=[{"role": "user", "content": f"QUESTION\n{question}\n\nEVIDENCE\n{evidence}"}],
    )
    answer = reply.text
    sources = log.numbered_sources()
    if sources:
        answer += "\n\nSources:\n" + "\n".join(
            f"[{i + 1}] {s['title']} - {s['location']}" for i, s in enumerate(sources)
        )
    return answer
