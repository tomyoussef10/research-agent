"""Download the papers in scripts/papers.txt from arXiv and index them in ChromaDB.

    python scripts/ingest_papers.py            # download + index everything
    python scripts/ingest_papers.py --reset    # wipe the index first

To add your own PDFs, drop them in data/papers/ and add a line "<file stem> | <title>"
to papers.txt (the stem is the file name without .pdf).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from agent.config import Config  # noqa: E402
from agent.tools.paper_db import COLLECTION, EMBED_MODEL  # noqa: E402

CHUNK_CHARS = 1000
OVERLAP = 200


def read_list(path: Path) -> list[tuple[str, str]]:
    items = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            pid, title = (part.strip() for part in line.split("|", 1))
            items.append((pid, title))
    return items


def download(pid: str, pdf_dir: Path) -> Path:
    path = pdf_dir / f"{pid}.pdf"
    if path.exists():
        return path
    url = f"https://arxiv.org/pdf/{pid}"
    print(f"  downloading {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    time.sleep(3)  # be polite to arXiv
    return path


def chunk(text: str) -> list[str]:
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + CHUNK_CHARS])
        start += CHUNK_CHARS - OVERLAP
    return [c for c in out if len(c) > 100]


def main() -> None:
    import chromadb
    import fitz  # PyMuPDF
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    if args.reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(
        COLLECTION,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
        metadata={"hnsw:space": "cosine"},
    )

    for pid, title in read_list(ROOT / "scripts" / "papers.txt"):
        print(f"{title}")
        try:
            pdf = download(pid, cfg.pdf_dir)
        except requests.RequestException as e:
            print(f"  skipped: {e}")
            continue
        ids, docs, metas = [], [], []
        with fitz.open(pdf) as doc:
            for page_no, page in enumerate(doc, 1):
                for i, piece in enumerate(chunk(page.get_text())):
                    ids.append(f"{pid}-p{page_no}-c{i}")
                    docs.append(piece)
                    metas.append({"title": title, "page": page_no, "paper_id": pid})
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"  indexed {len(ids)} chunks")

    print(f"\nDone. {col.count()} chunks in the paper database at {cfg.chroma_dir}")


if __name__ == "__main__":
    main()
