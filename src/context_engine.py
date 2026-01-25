from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

from retrieval import RetrievalArtifacts, search


@dataclass
class ContextEngineResult:
    query: str
    hits: pd.DataFrame
    context_text: str
    meta: Dict[str, Any]


def build_context_text(hits: pd.DataFrame, max_items: int = 8) -> str:
    """Build a compact context block for the LLM."""
    if hits is None or hits.empty:
        return "No matching items were found."

    cols = [c for c in ["item_id", "title", "category", "brand", "model", "price", "tags_norm"] if c in hits.columns]
    view = hits[cols].head(max_items).copy()

    lines = []
    for i, row in view.iterrows():
        tags = row.get("tags_norm", "")
        lines.append(
            f"- [{row.get('item_id')}] {row.get('title')} | {row.get('category')} | "
            f"{row.get('brand')} {row.get('model')} | ${row.get('price'):.2f} | tags: {tags}"
        )
    return "Candidate items:\n" + "\n".join(lines)


def run_context_engine(
    query: str,
    art: RetrievalArtifacts,
    top_k: int = 10,
    max_price: Optional[float] = None,
) -> ContextEngineResult:
    hits = search(query, art, top_k=top_k, max_price=max_price)

    context_text = build_context_text(hits, max_items=min(8, top_k))

    meta = {
        "n_hits": 0 if hits is None else len(hits),
        "top_k": top_k,
        "max_price": max_price,
    }

    return ContextEngineResult(
        query=query,
        hits=hits,
        context_text=context_text,
        meta=meta,
    )
