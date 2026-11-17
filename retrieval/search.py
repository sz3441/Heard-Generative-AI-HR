import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer



def tokenize(text):
    """Lowercase, split into words, and drop common words like 'the' and 'and'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]

stemmer = PorterStemmer()
QUERY_WORDS = {
    "complain", "complaint", "like", "dislike", "problem", "issue", "concern",
    "employee", "staff", "worker", "say", "mention", "think", "feel",
    "review", "feedback",
}  # Words that describe the request, not its content
QUERY_STEMS = {stemmer.stem(w) for w in QUERY_WORDS}

def tokenize_stem(text, is_query=False):
    """tokenize(), reduce words to stems, then drop request words (queries only)."""
    stems = [stemmer.stem(w) for w in tokenize(text)]
    if is_query:
        stems = [s for s in stems if s not in QUERY_STEMS]
    return stems

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
        "bm25_stem": BM25Okapi([tokenize_stem(t) for t in passages["text"]]),
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


def bm25_stem_search(index, query, k=5, field=None, filters=None):
    """BM25 with stemming and request words removed from the query."""
    results = index["passages"].copy()
    results["score"] = index["bm25_stem"].get_scores(tokenize_stem(query, is_query=True))
    if field:
        results = results[results["field"] == field]
    if filters:
        for col, value in filters.items():
            results = results[results[col] == value]
    return results.sort_values("score", ascending=False).head(k)

def hybrid_search(index, query, k=5, field=None, filters=None, candidates=50, rrf_k=60):
    """Combine semantic and stemmed-BM25 rankings with Reciprocal Rank Fusion (RRF)."""
    rankings = [
        semantic_search(index, query, k=candidates, field=field, filters=filters),
        bm25_stem_search(index, query, k=candidates, field=field, filters=filters),
    ]

    fused = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking["passage_id"], start=1):
            fused[pid] = fused.get(pid, 0) + 1 / (rrf_k + rank)

    results = index["passages"][index["passages"]["passage_id"].isin(fused)].copy()
    results["score"] = results["passage_id"].map(fused)
    return results.sort_values("score", ascending=False).head(k)


