"""Web search through the Tavily Search API (plain HTTP, no SDK needed)."""

from __future__ import annotations

from typing import Any

import requests

from .base import Source, Tool, ToolError, ToolResult

TAVILY_URL = "https://api.tavily.com/search"


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web. Use for recent facts, current versions or releases, documentation, "
        "or anything NOT likely to be in our local collection of research papers. "
        "Returns the top results with their URL and page text."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A short, specific search query."},
        },
        "required": ["query"],
    }
    retryable_errors = True

    def __init__(self, api_key: str, max_results: int = 4, timeout: float = 15.0):
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout

    def run(self, args: dict[str, Any]) -> ToolResult:
        if not self.api_key:
            raise ToolError("TAVILY_API_KEY is not set", retryable=False)
        try:
            resp = requests.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": args["query"], "max_results": self.max_results},
                timeout=self.timeout,
            )
        except requests.Timeout as e:
            raise ToolError(f"web search timed out: {e}", retryable=True) from e
        except requests.RequestException as e:
            raise ToolError(f"web search request failed: {e}", retryable=True) from e

        if resp.status_code == 429 or resp.status_code >= 500:
            raise ToolError(f"web search returned HTTP {resp.status_code}", retryable=True)
        if resp.status_code != 200:
            raise ToolError(f"web search returned HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            results = resp.json().get("results", [])
        except ValueError as e:
            raise ToolError("web search returned malformed JSON", retryable=True) from e

        results = [r for r in results if r.get("url") and r.get("content")]
        if not results:
            return ToolResult(content="No web results found.", empty=True)

        lines, sources = [], []
        for i, r in enumerate(results, 1):
            title = r.get("title") or r["url"]
            lines.append(f"[{i}] {title}\nURL: {r['url']}\n{r['content'][:1200]}")
            sources.append(Source(title=title, location=r["url"]))
        return ToolResult(content="\n\n".join(lines), sources=sources)
