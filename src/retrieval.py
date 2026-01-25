# src/retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as e:
    raise ImportError(
        "faiss no está instalado. En Windows suele ser 'faiss-cpu'. "
        "Instala dependencias con: pip install faiss-cpu"
    ) from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise ImportError(
        "sentence-transformers no está instalado. Instala con: pip install sentence-transformers"
    ) from e


@dataclass
class RetrievalArtifacts:
    embeddings: np.ndarray          # shape: (N, d), float32, normalized
    index: "faiss.Index"            # FAISS index
    item_360: pd.DataFrame          # dataframe original
    texts: list[str]                # textos indexados


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure required columns exist (create empty if missing)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out


def build_search_text(df: pd.DataFrame) -> pd.Series:
    """
    Build a single text field per row for semantic retrieval.

    Expected columns (missing ones are filled with empty string):
      - title, category, brand, model, tags_norm
    """
    df = _ensure_columns(df, ["title", "category", "brand", "model", "tags_norm"])

    def tags_to_str(x) -> str:
        # None / NaN
        if x is None:
            return ""
        # pandas NA scalar
        try:
            if pd.isna(x):
                return ""
        except Exception:
            # pd.isna sobre listas/arrays puede devolver array de bool -> ignoramos aquí
            pass

        # numpy array / list / tuple
        if isinstance(x, (list, tuple, np.ndarray)):
            try:
                return " ".join(map(str, list(x)))
            except Exception:
                return str(x)

        # dict (por si tags viene como dict)
        if isinstance(x, dict):
            try:
                return " ".join([f"{k}:{v}" for k, v in x.items()])
            except Exception:
                return str(x)

        # cualquier otro tipo
        return str(x)

    return (
        df["title"].fillna("").astype(str)
        + " | category: " + df["category"].fillna("").astype(str)
        + " | brand: " + df["brand"].fillna("").astype(str)
        + " | model: " + df["model"].fillna("").astype(str)
        + " | tags: " + df["tags_norm"].apply(tags_to_str)
    )


def build_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    """Encode texts into normalized float32 embeddings."""
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # clave para cosine con IP
    )
    emb = np.asarray(emb, dtype="float32")
    return emb


def fit_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """Fit a FAISS IndexFlatIP (cosine if embeddings are normalized)."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def build_retrieval_artifacts(
    item_360: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> RetrievalArtifacts:
    """
    Build embeddings + FAISS index over item_360.
    """
    # Construye textos (Series) y pásalos a lista
    texts_series = build_search_text(item_360)
    texts = texts_series.astype(str).tolist()

    emb = build_embeddings(texts, model_name=model_name)
    index = fit_faiss_index(emb)

    return RetrievalArtifacts(
        embeddings=emb,
        index=index,
        item_360=item_360.reset_index(drop=True).copy(),
        texts=texts,
    )


def search(
    query: str,
    artifacts: RetrievalArtifacts,
    top_k: int = 10,
    max_price: Optional[float] = None,
) -> pd.DataFrame:
    """
    Semantic search over the built FAISS index.

    Returns a DataFrame with: item_360 columns + score
    """
    if not query or not query.strip():
        return pd.DataFrame()

    # Encode query
    q_emb = build_embeddings([query], model_name="all-MiniLM-L6-v2")  # (1, d)
    scores, idxs = artifacts.index.search(q_emb, top_k)

    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    # Filtra -1 (a veces FAISS devuelve -1 si no hay)
    pairs = [(i, s) for i, s in zip(idxs, scores) if i is not None and i >= 0]
    if not pairs:
        return pd.DataFrame()

    rows = artifacts.item_360.iloc[[i for i, _ in pairs]].copy()
    rows["score"] = [s for _, s in pairs]

    if max_price is not None and "price" in rows.columns:
        rows = rows[pd.to_numeric(rows["price"], errors="coerce") <= float(max_price)]

    return rows.sort_values("score", ascending=False).reset_index(drop=True)
