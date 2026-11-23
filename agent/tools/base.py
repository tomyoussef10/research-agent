"""Shared types for tools: the Tool interface, results, errors, and argument checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """A tool failed. `retryable` tells the validator whether trying again could help."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Source:
    title: str
    location: str  # URL, or "paper title, p. N"

    def to_dict(self) -> dict:
        return {"title": self.title, "location": self.location}


@dataclass
class ToolResult:
    content: str
    sources: list[Source] = field(default_factory=list)
    empty: bool = False  # ran fine but found nothing useful


class Tool:
    """Base class. Subclasses set name/description/input_schema and implement run()."""

    name: str = ""
    description: str = ""
    input_schema: dict = {}
    retryable_errors: bool = False

    def run(self, args: dict[str, Any]) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError

    def spec(self) -> dict:
        """Tool definition in the format the Anthropic Messages API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def check_args(schema: dict, args: Any) -> str | None:
    """Minimal JSON-schema check. Returns an error message, or None if the args are fine."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in props:
            return f"unknown argument '{key}'"
        expected = _TYPES.get(props[key].get("type", ""))
        if expected and (not isinstance(value, expected) or isinstance(value, bool) and expected is not bool):
            return f"argument '{key}' must be of type {props[key]['type']}"
        if props[key].get("type") == "string" and not value.strip():
            return f"argument '{key}' must not be empty"
    return None
