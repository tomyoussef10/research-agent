"""Run short Python snippets in a separate process with a timeout.

This is a *soft* sandbox: a child process, an empty temp folder, a time limit, and
sockets disabled. It is good enough for math and small scripts written by the model,
but it is not a security boundary. For untrusted users, run it inside Docker instead.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .base import Tool, ToolError, ToolResult

MAX_OUTPUT = 4000

# Prepended to every snippet: block network access from inside the snippet.
_PRELUDE = """\
import socket as _s
def _blocked(*a, **k):
    raise OSError("network access is disabled in the sandbox")
_s.socket = _blocked
_s.create_connection = _blocked
del _s
"""


class PythonExecTool(Tool):
    name = "run_python"
    description = (
        "Run a short Python 3 program and return what it prints. Use for arithmetic, "
        "percentages, ratios, unit conversions, or small calculations on numbers you already "
        "found. Always print() the final result. Standard library only, no network, "
        "10-second time limit."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code that prints its result."},
        },
        "required": ["code"],
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def run(self, args: dict[str, Any]) -> ToolResult:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(_PRELUDE + args["code"])
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
                )
            except subprocess.TimeoutExpired as e:
                raise ToolError(f"code ran longer than {self.timeout} s and was stopped") from e

        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()
            # Show the last lines of the traceback, which hold the actual error.
            raise ToolError("code raised an error:\n" + "\n".join(err[-6:]))

        out = proc.stdout.strip()
        if not out:
            return ToolResult(content="The code ran but printed nothing. Use print() for the result.", empty=True)
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n...[output truncated]"
        return ToolResult(content=out)
