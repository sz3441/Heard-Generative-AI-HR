# B4 — Cited Answer Generation with Refusal

This module answers HR questions from A's retrieved employee-feedback passages. It runs locally, validates every citation against the current retrieval result, and refuses when privacy suppression is active or evidence is insufficient.

## Model decision

- Generator: `Qwen/Qwen2.5-0.5B-Instruct` (the B2 base model), loaded on CPU in float32.
- Do **not** load B2's `lora_final_adapter`: it was trained only for aspect sentiment JSON classification, not grounded answer generation.
- No paid API is used. The Hugging Face weights must be cached in the Docker image or downloaded during image build.
- The generator is loaded lazily and cached once per application process. Suppressed and insufficient-evidence requests refuse before loading Qwen.

## Integration

```python
from retrieval.search import load_index
from part_b.B4_Cited_Answer_Generation.cited_answer import generate_cited_answer

index = load_index("indexes/brightdata_eval")
result = generate_cited_answer(
    index,
    "Why are employees unhappy about promotions?",
    search_method="hybrid",  # or "semantic"
    field="cons",
)
```

The return value is:

```python
{"answer": "...", "citations": ["brightdata:...:cons:0"], "refused": False}
```

Privacy suppression returns A's `safe_search(...)["message"]` with no citations. Empty/malformed evidence, fewer than `min_evidence` usable passages (default 2), an empty answer, or no valid citation produces a generic refusal. The exact `group_size` is never returned.

The validator accepts exact IDs from the current retrieval result. To tolerate the
small generator occasionally returning a quotation instead of its ID, it may also
map a quotation back to a passage only when the quotation is at least 20 characters,
appears verbatim in the retrieved text, and identifies exactly one retrieved passage.
All other missing, invented, shortened, or ambiguous citations are rejected.

## Dependencies

Generation: `torch`, `transformers`, `accelerate`.

Retrieval remains owned by A and additionally requires `pandas`, `numpy`,
`pyarrow>=26.0.1`, `sentence-transformers`, `rank-bm25`, `scikit-learn`, and
`nltk`. PyArrow 19 could not read the committed Bright Data parquet file; the
project was verified with PyArrow 26.0.1. Install runtime dependencies with:

```bash
pip install -r part_b/B4_Cited_Answer_Generation/requirements.txt
```

For development and tests:

```bash
pip install -r part_b/B4_Cited_Answer_Generation/requirements-dev.txt
python -m pytest -p no:debugging -q \
  part_b/B4_Cited_Answer_Generation/test_cited_answer.py
```

Run commands from the repository root so both `retrieval` and `part_b` can be
imported. The Qwen and MiniLM weights should be downloaded during the Docker
image build for offline CPU-only runtime.

## Verification

- 10 unit tests cover suppression, insufficient evidence, search-method routing,
  invalid citations, citation objects, singular citation keys, case restoration,
  and safe quotation-to-ID mapping.
- End-to-end verification used `indexes/brightdata_eval` (431 reviews, 1,058
  passages) with both semantic and hybrid retrieval.
- `indexes/kaggle_demo` was not used to tune prompts, thresholds, or decoding.

Do not use `indexes/kaggle_demo` to select thresholds, prompts, decoding settings, or search method. Choose semantic versus hybrid using A4 evaluation data; use Kaggle only for the final demo.
