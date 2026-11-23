# Multi-Tool Research Assistant Agent

PRV201 Frontiers of Technology, Track A (Build), M3 prototype.

The agent answers multi-step technical questions by choosing among three tools:

| Tool | Used for | Built on |
| --- | --- | --- |
| `search_papers` | Facts and numbers from our local collection of 10 AI/ML papers | ChromaDB + all-MiniLM-L6-v2 embeddings |
| `web_search` | Recent or external facts not in the papers | Tavily Search API |
| `run_python` | Arithmetic, ratios, conversions | Python subprocess, 10 s timeout, sockets disabled |

An LLM orchestrator (Google Gemini 3.5 Flash on the free tier; Claude also supported) picks one tool at a time. A validator
checks every call, retries failures, and falls back to another tool. A synthesizer then
writes the answer **only** from the logged tool results, with numbered citations. Every
run is saved as a JSON trace in `traces/`.

## Setup (about 10 minutes)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then paste your two API keys (both free)
python scripts/ingest_papers.py                        # downloads 10 arXiv papers, builds the index
```

Keys: Gemini at aistudio.google.com/apikey and Tavily at app.tavily.com. Both free tiers are enough.
To use Claude instead, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env` (paid).
The first ingest also downloads the embedding model (about 90 MB).

## Run it

```bash
python main.py "How many times more parameters does GPT-3 have than BERT-large?"
python main.py          # interactive mode
```

You will see each tool call as it happens, then the answer, token use, cost, and the trace file:

```
  [+] step 1: search_papers(query='GPT-3 largest model parameters') -> ok
  [x] step 2: web_search(query='...') -> error after 3 attempts
      note: ...
  [~] step 3: search_papers(query='...') -> fallback
```

`+` ok, `~` fallback tool used, `o` empty result, `x` error, `!` call rejected by the validator.

## Test and evaluate

```bash
pytest                                   # 28 offline tests, no API keys needed
python eval/run_eval.py                  # 30-question test set vs. the M2 targets
python eval/run_eval.py --failures       # 10 runs with a tool forced to fail
AGENT_FAIL_TOOLS=web_search python main.py "..."   # try a failure by hand
```

`run_eval.py` auto-grades numeric answers (within 2%), checks tool choice against the
labels in `eval/questions.jsonl`, and writes a CSV. Questions with `"expected": null` are
graded by hand in that CSV.

## Layout

```
agent/
  orchestrator.py   the loop: LLM picks a tool -> validator runs it -> repeat (max 6 calls)
  validator.py      arg checks, repeat guard, retries with backoff, fallback tool, logging
  synthesizer.py    final answer from the step log only, with citations
  step_log.py       records every call; saves the JSON trace
  llm.py            Gemini and Anthropic API wrappers + token/cost tracking
  config.py         settings from .env
  tools/            web_search.py, paper_db.py, python_exec.py, base.py
scripts/            ingest_papers.py, papers.txt (the paper list)
eval/               questions.jsonl (30 labeled questions), run_eval.py
tests/              offline unit tests with a scripted fake LLM
traces/             one JSON file per run (the executed traces)
```

## Known limits

- `run_python` is a soft sandbox (separate process, timeout, no sockets). It is fine for
  model-written math but is not a security boundary; use Docker for untrusted users.
- The paper database only knows the papers in `scripts/papers.txt`.
- Gemini's free tier has per-minute request limits; the agent waits and retries on HTTP 429,
  so a full evaluation can take 20-30 minutes.
- Cost in traces is $0 on the free tier. For a paid model, set PRICE_IN/OUT_PER_MTOK in `.env`.
