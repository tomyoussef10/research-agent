from agent.step_log import StepLog
from agent.validator import Validator

from .fakes import FakeSearch


def make(tools, **kw):
    log = StepLog()
    return Validator({t.name: t for t in tools}, log, sleep=lambda s: None, **kw), log


def test_success_is_logged_with_sources():
    v, log = make([FakeSearch("web_search")])
    out = v.run("web_search", {"query": "q"})
    assert not out.is_error
    assert log.steps[0].status == "ok" and log.steps[0].sources


def test_retries_then_succeeds():
    web = FakeSearch("web_search", fail_times=2)
    v, log = make([web], retries=2)
    assert not v.run("web_search", {"query": "q"}).is_error
    assert web.calls == 3 and log.steps[0].attempts == 3


def test_falls_back_when_retries_run_out():
    web, papers = FakeSearch("web_search", fail_times=99), FakeSearch("search_papers")
    v, log = make([web, papers], retries=2)
    out = v.run("web_search", {"query": "q"})
    assert not out.is_error and "search_papers" in out.content
    assert [s.status for s in log.steps] == ["error", "fallback"]
    assert log.tools_used() == ["search_papers"]


def test_both_down_reports_missing():
    v, _ = make([FakeSearch("web_search", fail_times=99), FakeSearch("search_papers", fail_times=99)])
    out = v.run("web_search", {"query": "q"})
    assert out.is_error and "both failed" in out.content


def test_injected_failure_uses_fallback():
    v, log = make([FakeSearch("web_search"), FakeSearch("search_papers")], fail_tools=frozenset({"web_search"}))
    v.run("web_search", {"query": "q"})
    assert log.steps[-1].status == "fallback"


def test_empty_result_goes_back_as_error_without_retry():
    papers = FakeSearch("search_papers", empty=True)
    v, log = make([papers])
    assert v.run("search_papers", {"query": "q"}).is_error
    assert papers.calls == 1 and log.steps[0].status == "empty"


def test_bad_args_rejected_then_tool_blocked():
    v, log = make([FakeSearch("web_search")])
    assert "invalid arguments" in v.run("web_search", {"q": "x"}).content
    v.run("web_search", {})
    assert "too many times" in v.run("web_search", {"query": "fine"}).content
    assert all(s.status == "rejected" for s in log.steps)


def test_duplicate_call_blocked():
    web = FakeSearch("web_search")
    v, _ = make([web])
    v.run("web_search", {"query": "Same"})
    out = v.run("web_search", {"query": "same"})
    assert out.is_error and "already made this exact call" in out.content and web.calls == 1


def test_unknown_tool():
    v, _ = make([FakeSearch("web_search")])
    assert "unknown tool" in v.run("calculator", {"x": 1}).content
