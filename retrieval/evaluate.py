import pandas as pd


def recall_at_k(retrieved_ids, gold_ids, k):
    """Share of gold passages that appear in the top-k results."""
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(gold_ids)) / len(gold_ids)


def reciprocal_rank(retrieved_ids, gold_ids):
    """1 / rank of the first gold passage; 0 if none is retrieved."""
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in gold_ids:
            return 1 / rank
    return 0.0


def evaluate(search_fn, index, questions, k=10):
    """Run one search method on every question; return one row of scores per question."""
    rows = []
    for q in questions.itertuples():
        field = q.field if isinstance(q.field, str) else None
        filters = {"year": int(q.year)} if pd.notna(q.year) else None
        hits = search_fn(index, q.question, k=k, field=field, filters=filters)
        retrieved = hits["passage_id"].tolist()
        gold = q.gold_ids.split(";") if isinstance(q.gold_ids, str) else []

        row = {
            "question_id": q.question_id,
            "query_type": q.query_type,
            "top1_score": hits["score"].iloc[0] if len(hits) else None,
        }
        if gold:  # unanswerable questions have no gold passages: skip recall and MRR
            row["recall@5"] = recall_at_k(retrieved, gold, 5)
            row["recall@10"] = recall_at_k(retrieved, gold, 10)
            row["rr"] = reciprocal_rank(retrieved, gold)
        rows.append(row)
    return pd.DataFrame(rows)

