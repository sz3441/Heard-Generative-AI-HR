# Part B — Local Models and Grounded Generation

Part B contains the data preparation, LoRA sentiment model, and cited-answer
generation modules used by Heard.

## Structure

| Folder | Purpose |
|---|---|
| `B1_Preparing_Data/` | Prepare instruction data for aspect-based sentiment training. |
| `B2_LoRA_Fine_Tuning/` | Fine-tune and run the Qwen sentiment classifier with LoRA. |
| `B4_Cited_Answer_Generation/` | Generate evidence-grounded answers from A's retrieval results, validate passage citations, and refuse unsafe or unsupported requests. |

## B4 quick check

Run from the repository root:

```bash
pip install -r part_b/B4_Cited_Answer_Generation/requirements-dev.txt
python -m pytest -p no:debugging -q \
  part_b/B4_Cited_Answer_Generation/test_cited_answer.py
```

See `B4_Cited_Answer_Generation/README.md` for integration examples and privacy
behavior. B4 depends on A's `retrieval` package and reads indexes from the
repository-level `indexes/` directory.
