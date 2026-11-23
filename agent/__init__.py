"""Multi-tool research assistant agent (PRV201 Track A)."""

from __future__ import annotations

from .config import Config
from .orchestrator import AgentResult, Orchestrator
from .step_log import Step, StepLog
from .tools import ChromaStore, PaperSearchTool, PythonExecTool, Tool, WebSearchTool
from .validator import Validator


def build_tools(cfg: Config) -> dict[str, Tool]:
    """Create the three tools. The paper tool is skipped (with a warning) if ChromaDB isn't set up."""
    tools: dict[str, Tool] = {}
    try:
        tools["search_papers"] = PaperSearchTool(ChromaStore(cfg.chroma_dir), min_score=cfg.min_paper_score)
    except ImportError as e:
        print(f"[warning] paper database disabled ({e}); install requirements.txt")
    tools["web_search"] = WebSearchTool(cfg.tavily_api_key)
    tools["run_python"] = PythonExecTool(timeout=cfg.python_timeout)
    return tools


def ask(question: str, cfg: Config | None = None, llm=None, tools: dict[str, Tool] | None = None,
        on_step=None, save_trace: bool = True) -> tuple[AgentResult, dict]:
    """Answer one question. Returns the result and a usage summary (tokens, cost, trace path)."""
    from .llm import make_llm

    cfg = cfg or Config()
    llm = llm or make_llm(cfg)
    tools = tools if tools is not None else build_tools(cfg)

    log = StepLog()
    validator = Validator(tools, log, retries=cfg.retries, backoff_seconds=cfg.backoff_seconds,
                          fail_tools=cfg.fail_tools)
    result = Orchestrator(llm, validator, log, cfg.max_tool_calls, on_step=on_step).run(question)

    usage = {}
    if hasattr(llm, "usage"):
        u = llm.usage
        usage = {"llm_calls": u.calls, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cost_usd": round(u.cost(cfg.price_in_per_mtok, cfg.price_out_per_mtok), 4)}
    if save_trace:
        usage["trace"] = str(log.save(cfg.trace_dir, question, result.answer, usage))
    return result, usage
