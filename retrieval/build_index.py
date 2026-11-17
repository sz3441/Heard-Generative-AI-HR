import pandas as pd

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_reviews(csv_path):
    """Read a cleaned review file and add year and employment_status."""
    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["date_review"], format="ISO8601", utc=True)
    df["year"] = df["date"].dt.year

    status = df["current"].fillna("").str.lower()
    df["employment_status"] = "unknown"
    df.loc[status.str.startswith("current"), "employment_status"] = "current"
    df.loc[status.str.startswith("former"), "employment_status"] = "former"

    return df


FIELDS = {"pros": "pros", "cons": "cons", "advice": "advice_to_management"}
META_COLS = ["firm", "year", "employment_status", "job_title", "overall_rating", "sentiment_label"]

def split_text(text, max_words=150, overlap=30):
    """Split text longer than max_words into overlapping chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks

def make_passages(df, id_prefix=""):
    """Split each review into pros/cons/advice passages, then chunk long passages."""
    rows = []
    for _, review in df.iterrows():
        for field, col in FIELDS.items():
            if col not in df.columns:          # Kaggle has no advice column
                continue
            text = review[col]
            if pd.isna(text) or len(str(text).split()) < 3:
                continue
            for i, chunk in enumerate(split_text(str(text).strip())):
                row = {
                    "passage_id": f"{id_prefix}{review['review_id']}:{field}:{i}",
                    "review_id": review["review_id"],
                    "field": field,
                    "chunk": i,
                    "text": chunk,
                }
                for c in META_COLS:
                    row[c] = review[c]
                rows.append(row)
    return pd.DataFrame(rows)

def save_index(passages, out_dir, source_file, firm=None):
    """Embed passages and save passages.parquet, embeddings.npy and manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = model.encode(
        passages["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    passages.to_parquet(out_dir / "passages.parquet", index=False)
    np.save(out_dir / "embeddings.npy", embeddings)

    info = {
        "corpus": out_dir.name,
        "source_file": Path(source_file).name,
        "firm": firm,
        "model": MODEL_NAME,
        "n_reviews": int(passages["review_id"].nunique()),
        "n_passages": len(passages),
        "dims": int(embeddings.shape[1]),
        "max_words": 150,
        "overlap_words": 30,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def build_index(csv_path, out_dir, id_prefix="", firm=None):
    """Load reviews, optionally keep one company, make passages and save the index."""
    df = load_reviews(csv_path)
    if firm:
        df = df[df["firm"] == firm]
    passages = make_passages(df, id_prefix=id_prefix)
    return save_index(passages, out_dir, csv_path, firm=firm)






