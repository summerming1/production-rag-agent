from __future__ import annotations

import json
from pathlib import Path

from .types import Document


def load_jsonl_corpus(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if not text:
                raise ValueError(f"empty text at line {line_no}")
            doc_id = str(row.get("id") or f"doc-{line_no}")
            source = str(row.get("source") or row.get("metadata", {}).get("source") or "unknown")
            metadata = dict(row.get("metadata") or {})
            docs.append(Document(doc_id, text, source, metadata))
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("document ids must be unique")
    return docs
