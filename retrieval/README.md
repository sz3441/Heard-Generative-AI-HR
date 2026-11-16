# Retrieval (Part A)

Owner: Zirou Zhang

## Files
- `search.py`: `load_index`, `semantic_search`, `bm25_search`
- `evaluate.py`: `recall_at_k`, `reciprocal_rank`, `evaluate`
- `build_index.py`: index building (to be completed)

## Index
`indexes/brightdata_eval/` (Bright Data, 431 reviews → 1,058 passages)
- `passages.parquet`: one row per passage (`passage_id`, `text`, `field`, `firm`, `year`, ...)
- `embeddings.npy`: 1,058 × 384 vectors, same row order as `passages.parquet`
- `manifest.json`: source file, model (`all-MiniLM-L6-v2`), chunking settings

## Usage
```python
from retrieval.search import load_index, semantic_search, bm25_search

index = load_index("indexes/brightdata_eval")   # load once
hits = semantic_search(index, "complaints about salary", k=5, field="cons")
hits[["passage_id", "text", "score"]]
```
Cite passages by `passage_id` (format `source:review_id:field:chunk`).

## Dependencies
`pandas`, `numpy`, `pyarrow`, `sentence-transformers`, `rank-bm25`, `scikit-learn`
