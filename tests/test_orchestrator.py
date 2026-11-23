from agent import Config, ask
from agent.tools import PythonExecTool

from .fakes import FakeSearch, ScriptedLLM


def tools(**web_kw):
    return {"search_papers": FakeSearch("search_papers"), "web_search": FakeSearch("web_search", **web_kw),
            "run_python": PythonExecTool()}


def test_multi_step_run(tmp_path):
    llm = ScriptedLLM([("search_papers", {"query": "GPT-3 parameters"}),
                       ("search_papers", {"query": "BERT-large parameters"}),
                       ("run_python", {"code": "print(round(175e9 / 340e6, 1))"})],
                      final="GPT-3 has about 514.7x the parameters of BERT-large [1].")
    cfg = Config()
    cfg.trace_dir = tmp_path
    result, usage = ask("ratio?", cfg, llm=llm, tools=tools())

    assert result.log.tools_used() == ["search_papers", "search_papers", "run_python"]
    assert result.log.steps[-1].output == "514.7"
    assert "Sources:" in result.answer and not result.hit_step_limit
    assert (tmp_path / usage["trace"].split("/")[-1]).exists()

    # The tool results were passed back to the LLM before its next decision.
    last_turn = llm.requests[3]["messages"][-1]["content"][0]
    assert last_turn["type"] == "tool_result" and last_turn["content"] == "514.7"


def test_synthesizer_sees_only_the_log():
    llm = ScriptedLLM([("web_search", {"query": "x"})])
    ask("q", Config(), llm=llm, tools=tools(), save_trace=False)
    synth = llm.requests[-1]
    assert synth["tools"] is None
    assert "web_search result for x" in synth["messages"][0]["content"]


def test_step_limit_stops_the_loop():
    calls = [("web_search", {"query": f"q{i}"}) for i in range(10)]
    llm = ScriptedLLM(calls)
    cfg = Config()
    cfg.max_tool_calls = 3
    result, _ = ask("q", cfg, llm=llm, tools=tools(), save_trace=False)
    assert result.hit_step_limit and len(result.log.steps) == 3


def test_web_outage_recovers_via_papers():
    llm = ScriptedLLM([("web_search", {"query": "x"})])
    cfg = Config()
    cfg.backoff_seconds = 0
    result, _ = ask("q", cfg, llm=llm, tools=tools(fail_times=99), save_trace=False)
    assert [s.status for s in result.log.steps] == ["error", "fallback"]
    assert "unavailable" in llm.requests[-1]["messages"][0]["content"]
