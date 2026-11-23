"""Run the agent on the test set and score it against the M2 success criteria.

    python eval/run_eval.py                 # all 30 questions
    python eval/run_eval.py --limit 5       # quick check
    python eval/run_eval.py --failures      # failure-injection suite (10 runs)

Writes eval/results-<time>.csv and prints a summary. Answers with "expected": null
must be graded by hand: fill in the 'correct' column in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Config, ask  # noqa: E402

TARGETS = {"correct": 0.75, "tool_choice": 0.85, "recovery": 0.80, "median_seconds": 20, "avg_cost": 0.05}

_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:e[+-]?\d+)?", re.I)
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "trillion": 1e12, "t": 1e12}


def numbers_in(text: str) -> list[float]:
    """All numbers in the text, also scaled by a following 'million'/'B' etc. (110M -> 1.1e8)."""
    text = text.split("\n\nSources:")[0]  # ignore the source list (arXiv ids etc.)
    out = []
    for m in _NUM.finditer(text):
        try:
            value = float(m.group().replace(",", ""))
        except ValueError:
            continue
        out.append(value)
        after = re.match(r"\s*(thousand|million|billion|trillion|[kmbt])\b", text[m.end():], re.I)
        if after:
            out.append(value * _SCALE[after.group(1).lower()])
    return out


def is_correct(answer: str, expected) -> bool | None:
    if expected is None:
        return None
    return any(abs(n - expected) <= max(abs(expected) * 0.02, 1e-9) for n in numbers_in(answer))


def run(items: list[dict], cfg: Config, label: str) -> list[dict]:
    rows = []
    for i, item in enumerate(items, 1):
        print(f"[{label} {i}/{len(items)}] {item['id']}: {item['question'][:70]}")
        start = time.time()
        try:
            result, usage = ask(item["question"], cfg)
            answer, used = result.answer, sorted(set(result.log.tools_used()))
            fallback = any(s.status == "fallback" for s in result.log.steps)
            crashed = ""
        except Exception as e:  # count crashes as failures, keep going
            answer, used, usage, fallback, crashed = "", [], {}, False, str(e)
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item.get("expected"),
            "correct": is_correct(answer, item.get("expected")),
            "expected_tools": " ".join(sorted(item["tools"])),
            "tools_used": " ".join(used),
            "tool_choice_ok": set(used) == set(item["tools"]),
            "fallback_used": fallback,
            "recovered": bool(answer) and not crashed,
            "seconds": round(time.time() - start, 1),
            "cost_usd": usage.get("cost_usd", 0),
            "trace": usage.get("trace", ""),
            "error": crashed,
            "answer": answer.replace("\n", " ")[:600],
        })
    return rows


def summarize(rows: list[dict], failures: bool) -> None:
    def pct(xs):
        return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "n/a"

    print("\n" + "=" * 60)
    if failures:
        rec = [r["recovered"] for r in rows]
        print(f"Recovered from injected failure: {pct(rec)} (target >= {TARGETS['recovery']:.0%})")
        print(f"Used a fallback tool:            {pct([r['fallback_used'] for r in rows])}")
    else:
        graded = [r["correct"] for r in rows if r["correct"] is not None]
        print(f"Correct (auto-graded {len(graded)}/{len(rows)}): {pct(graded)} (target >= {TARGETS['correct']:.0%})")
        print(f"Correct tool choice:             {pct([r['tool_choice_ok'] for r in rows])} (target >= {TARGETS['tool_choice']:.0%})")
    secs = [r["seconds"] for r in rows]
    costs = [r["cost_usd"] for r in rows]
    print(f"Median time per question:        {statistics.median(secs):.1f} s (target <= {TARGETS['median_seconds']} s)")
    print(f"Average cost per question:       ${statistics.mean(costs):.4f} (target <= ${TARGETS['avg_cost']})")
    ungraded = [r["id"] for r in rows if r["correct"] is None and not failures]
    if ungraded:
        print(f"Grade by hand in the CSV: {', '.join(ungraded)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--failures", action="store_true", help="run the failure-injection suite")
    args = parser.parse_args()

    items = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    if args.failures:
        # 10 runs: 5 questions with web search down, 5 with the paper database down.
        web_qs = [q for q in items if "web_search" in q["tools"]][:5]
        paper_qs = [q for q in items if "search_papers" in q["tools"]][:5]
        rows = run(web_qs, Config(fail_tools=frozenset({"web_search"})), "web down")
        rows += run(paper_qs, Config(fail_tools=frozenset({"search_papers"})), "papers down")
    else:
        rows = run(items, Config(), "eval")

    out = ROOT / "eval" / f"results-{'failures-' if args.failures else ''}{datetime.now():%Y%m%d-%H%M%S}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows, args.failures)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
