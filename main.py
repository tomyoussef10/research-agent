"""Command-line interface.

    python main.py "How many times more parameters does GPT-3 have than BERT-large?"
    python main.py            # interactive mode
"""

from __future__ import annotations

import argparse
import sys

from agent import Config, Step, ask


def print_step(step: Step) -> None:
    icon = {"ok": "+", "fallback": "~", "empty": "o", "error": "x", "rejected": "!"}.get(step.status, "?")
    args = ", ".join(f"{k}={v!r}"[:90] for k, v in step.args.items())
    print(f"  [{icon}] step {step.n}: {step.tool}({args}) -> {step.status}"
          + (f" after {step.attempts} attempts" if step.attempts > 1 else ""))
    if step.note:
        print(f"      note: {step.note}")
    if step.status in ("error", "rejected"):
        print(f"      {step.output.splitlines()[0][:150]}")


def answer_one(question: str, cfg: Config, quiet: bool) -> None:
    print(f"\nQ: {question}\n")
    result, usage = ask(question, cfg, on_step=None if quiet else print_step)
    print("\n" + "=" * 70)
    print(result.answer)
    print("=" * 70)
    if result.hit_step_limit:
        print("(stopped at the tool-call limit)")
    print(f"tools used: {', '.join(result.log.tools_used()) or 'none'} | "
          f"tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out | "
          f"est. cost ${usage.get('cost_usd', 0):.4f}")
    print(f"trace saved to {usage.get('trace')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-tool research assistant agent")
    parser.add_argument("question", nargs="*", help="question to answer (omit for interactive mode)")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide the step-by-step tool log")
    args = parser.parse_args()
    cfg = Config()

    if args.question:
        answer_one(" ".join(args.question), cfg, args.quiet)
        return

    print("Research assistant. Type a question, or 'quit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            try:
                answer_one(q, cfg, args.quiet)
            except Exception as e:  # keep the session alive on API errors
                print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
