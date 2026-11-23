"""Settings read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(ROOT / ".env")


@dataclass
class Config:
    # "gemini" (default, free tier) or "anthropic"
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower())
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL") or (
        "claude-sonnet-5-5" if os.getenv("LLM_PROVIDER", "gemini").lower() == "anthropic" else "gemini-2.5-flash"))

    max_tool_calls: int = int(os.getenv("AGENT_MAX_TOOL_CALLS", "6"))
    retries: int = int(os.getenv("AGENT_RETRIES", "2"))
    backoff_seconds: float = float(os.getenv("AGENT_BACKOFF_SECONDS", "1.0"))
    python_timeout: int = int(os.getenv("AGENT_PYTHON_TIMEOUT", "10"))
    min_paper_score: float = float(os.getenv("AGENT_MIN_PAPER_SCORE", "0.35"))

    chroma_dir: Path = ROOT / "data" / "chroma"
    pdf_dir: Path = ROOT / "data" / "papers"
    trace_dir: Path = ROOT / "traces"

    # Price per million tokens, used only for the cost estimate in traces.
    # Gemini's free tier costs $0. For a paid model, set these to its current prices.
    price_in_per_mtok: float = float(os.getenv("PRICE_IN_PER_MTOK", "0"))
    price_out_per_mtok: float = float(os.getenv("PRICE_OUT_PER_MTOK", "0"))

    # Comma-separated tool names to force-fail (failure-injection testing).
    fail_tools: frozenset[str] = field(
        default_factory=lambda: frozenset(
            t.strip() for t in os.getenv("AGENT_FAIL_TOOLS", "").split(",") if t.strip()
        )
    )
