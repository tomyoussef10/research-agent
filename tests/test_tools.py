import pytest

from agent.tools import PaperSearchTool, PythonExecTool, ToolError, check_args
from agent.tools.paper_db import Passage


# ---- run_python (runs for real) --------------------------------------------

def test_python_prints_result():
    assert PythonExecTool().run({"code": "print(175e9 / 340e6)"}).content.startswith("514.7")


def test_python_error_is_reported():
    with pytest.raises(ToolError, match="ZeroDivisionError"):
        PythonExecTool().run({"code": "print(1/0)"})


def test_python_timeout():
    with pytest.raises(ToolError, match="longer than 1 s"):
        PythonExecTool(timeout=1).run({"code": "while True: pass"})


def test_python_no_output_is_empty():
    assert PythonExecTool().run({"code": "x = 2 + 2"}).empty


def test_python_network_blocked():
    with pytest.raises(ToolError, match="network access is disabled"):
        PythonExecTool().run({"code": "import socket\nsocket.create_connection(('example.org', 80))"})


# ---- search_papers (fake store) --------------------------------------------

class FakeStore:
    def __init__(self, passages):
        self.passages = passages

    def query(self, text, k):
        return self.passages[:k]


def test_papers_filters_low_scores():
    tool = PaperSearchTool(FakeStore([Passage("BERT-base has 110M parameters", "BERT", 4, 0.62),
                                      Passage("unrelated", "ResNet", 1, 0.10)]), min_score=0.35)
    result = tool.run({"query": "BERT size"})
    assert "110M" in result.content and "unrelated" not in result.content
    assert [s.location for s in result.sources] == ["BERT, p. 4"]


def test_papers_nothing_relevant_is_empty():
    tool = PaperSearchTool(FakeStore([Passage("x", "A", 1, 0.1)]), min_score=0.35)
    assert tool.run({"query": "q"}).empty


# ---- argument checking ------------------------------------------------------

SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


@pytest.mark.parametrize("args,problem", [
    ({}, "missing required"),
    ({"query": 5}, "must be of type string"),
    ({"query": "  "}, "must not be empty"),
    ({"query": "a", "extra": 1}, "unknown argument"),
    ("not a dict", "JSON object"),
])
def test_check_args_rejects(args, problem):
    assert problem in check_args(SCHEMA, args)


def test_check_args_accepts():
    assert check_args(SCHEMA, {"query": "transformer layers"}) is None
