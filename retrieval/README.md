# Retrieval (Part A)

Owner: Zirou Zhang

## Files
- `build_index.py`: `load_reviews`, `make_passages`, `save_index`, `build_index`
- `search.py`: `load_index`, `semantic_search`, `bm25_search`, `bm25_stem_search`, `hybrid_search`
- `evaluate.py`: `recall_at_k`, `reciprocal_rank`, `evaluate`

## Indexes
| Folder | Data | Use |
|---|---|---|
| `indexes/brightdata_eval/` | Bright Data, 431 reviews → 1,058 passages | Evaluation (A3/A4/A6) |
| `indexes/kaggle_demo/` | Kaggle test set, Deloitte only, 4,229 reviews → 8,508 passages | Demo only (do not tune models on it) |

Each folder has:
- `passages.parquet`: one row per passage (`passage_id`, `review_id`, `field`, `chunk`, `text`, `firm`, `year`, `employment_status`, `job_title`, `overall_rating`, `sentiment_label`)
- `embeddings.npy`: one 384-dim vector per passage, same row order as `passages.parquet`
- `manifest.json`: source file, firm, model (`all-MiniLM-L6-v2`), chunking settings

Passage IDs: `source:review_id:field:chunk`, e.g. `brightdata:59978917:cons:0`, `kaggle_186553:pros:0`. Cite passages by this ID.

## Usage
```python
from retrieval.search import load_index, semantic_search, hybrid_search

index = load_index("indexes/brightdata_eval")   # load once at start-up
hits = semantic_search(index, "complaints about salary", k=5, field="cons")
hits = semantic_search(index, "bad management", k=5, field="cons", filters={"year": 2023})
hits[["passage_id", "text", "score"]]
```
All search functions take the same arguments and return the same table, so switching methods only changes the function name. The final method (semantic or hybrid) will be chosen in A4.

Rebuild an index:
```python
from retrieval.build_index import build_index

build_index("data/processed/brightdata_recent_clean_reviews.csv", "indexes/brightdata_eval", id_prefix="brightdata:")
build_index("data/processed/kaggle_reviews_test.csv", "indexes/kaggle_demo", firm="Deloitte")
```

## Dependencies
`pandas`, `numpy`, `pyarrow`, `sentence-transformers`, `rank-bm25`, `scikit-learn`, `nltk`
