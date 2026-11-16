import pandas as pd


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