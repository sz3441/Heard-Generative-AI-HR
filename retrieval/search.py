import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def tokenize(text):
    """Lowercase, split into words, and drop common words like 'the' and 'and'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def load_index(index_dir):
    """Load passages, vectors, the embedding model and a BM25 index from one folder."""
    index_dir = Path(index_dir)
    with open(index_dir / "manifest.json") as f:
        info = json.load(f)

    passages = pd.read_parquet(index_dir / "passages.parquet")

    return {
        "passages": passages,
        "embeddings": np.load(index_dir / "embeddings.npy"),
        "model": SentenceTransformer(info["model"], device="cpu"),
        "bm25": BM25Okapi([tokenize(t) for t in passages["text"]]),
        "info": info,
    }


def semantic_search(index, query, k=5, field=None, filters=None):
    """Return the top-k passages ranked by semantic (embedding) similarity."""
    q = index["model"].encode(query, normalize_embeddings=True)

    results = index["passages"].copy()
    results["score"] = index["embeddings"] @ q
    if field:
        results = results[results["field"] == field]
    if filters:
        for col, value in filters.items():
            results = results[results[col] == value]
    return results.sort_values("score", ascending=False).head(k)


def bm25_search(index, query, k=5, field=None, filters=None):
    """Return the top-k passages ranked by BM25 keyword score."""
    results = index["passages"].copy()
    results["score"] = index["bm25"].get_scores(tokenize(query))
    if field:
        results = results[results["field"] == field]
    if filters:
        for col, value in filters.items():
            results = results[results[col] == value]
    return results.sort_values("score", ascending=False).head(k)



