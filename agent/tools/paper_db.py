"""Semantic search over our local paper collection (ChromaDB + sentence-transformers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .base import Source, Tool, ToolError, ToolResult

COLLECTION = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Passage:
    text: str
    title: str
    page: int
    score: float  # cosine similarity, higher = more relevant


class PassageStore(Protocol):
    def query(self, text: str, k: int) -> list[Passage]: ...


class ChromaStore:
    """Wraps a persistent ChromaDB collection built by scripts/ingest_papers.py."""

    def __init__(self, path: Path):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        client = chromadb.PersistentClient(path=str(path))
        self.collection = client.get_or_create_collection(
            COLLECTION,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
            metadata={"hnsw:space": "cosine"},
        )

    def query(self, text: str, k: int) -> list[Passage]:
        if self.collection.count() == 0:
            raise ToolError("paper database is empty; run scripts/ingest_papers.py first")
        res = self.collection.query(query_texts=[text], n_results=k)
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(Passage(text=doc, title=meta.get("title", "?"), page=int(meta.get("page", 0)), score=1 - dist))
        return out


class PaperSearchTool(Tool):
    name = "search_papers"
    description = (
        "Search our local collection of AI/ML research papers (e.g. Transformer, BERT, GPT-3, "
        "ResNet, LoRA, Adam, Chinchilla, ReAct, Toolformer, RAG). Use for facts, numbers, or "
        "methods stated in those papers. Returns the most relevant passages with paper and page."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
        },
        "required": ["query"],
    }

    def __init__(self, store: PassageStore, k: int = 4, min_score: float = 0.35):
        self.store = store
        self.k = k
        self.min_score = min_score

    def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            passages = self.store.query(args["query"], self.k)
        except ToolError:
            raise
        except Exception as e:  # chromadb / embedding errors
            raise ToolError(f"paper database error: {e}") from e

        good = [p for p in passages if p.score >= self.min_score]
        if not good:
            return ToolResult(
                content=f"No passages scored above the relevance threshold ({self.min_score}).",
                empty=True,
            )

        lines, sources = [], []
        for i, p in enumerate(good, 1):
            lines.append(f"[{i}] {p.title}, p. {p.page} (score {p.score:.2f})\n{p.text[:1200]}")
            sources.append(Source(title=p.title, location=f"{p.title}, p. {p.page}"))
        return ToolResult(content="\n\n".join(lines), sources=sources)
