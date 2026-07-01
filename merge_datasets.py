"""
merge_datasets.py
-----------------
Merges two Kaggle movie datasets into one clean unified file:

  Source A - TMDB 5000 Movie Dataset
    data/tmdb_5000_movies.csv
    data/tmdb_5000_credits.csv

  Source B - The Movies Dataset (rounakbanik, ~45,000 titles)
    data/movies_metadata.csv
    data/credits.csv
    data/keywords.csv

Output -> data/merged_raw.csv  (~44,000 unique films after dedup)

Run ONCE before build_model.py:
    python merge_datasets.py
    python build_model.py
"""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path("data")

KEEP_COLS = [
    "id", "title", "overview", "genres", "keywords",
    "cast", "crew", "vote_average", "vote_count",
    "release_date", "popularity",
]


def check_files(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("\n[ERROR] Missing files:")
        for m in missing:
            print(f"  x  {m}")
        print("\nDownload from Kaggle and place in the data/ folder, then re-run.")
        sys.exit(1)


def safe_parse(val):
    """Parse stringified Python objects to clean JSON strings."""
    if pd.isna(val) or str(val).strip() in ("", "[]", "{}"):
        return "[]"
    try:
        parsed = ast.literal_eval(str(val))
        return json.dumps(parsed)
    except (ValueError, SyntaxError):
        return "[]"


def load_tmdb_5000():
    print("Loading TMDB 5000 ...")
    check_files(DATA / "tmdb_5000_movies.csv", DATA / "tmdb_5000_credits.csv")
    movies  = pd.read_csv(DATA / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA / "tmdb_5000_credits.csv")
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    df = df.rename(columns={"title_x": "title"})
    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(safe_parse)
    df = df[KEEP_COLS].copy()
    df["source"] = "tmdb_5000"
    print(f"  -> {len(df):,} rows")
    return df


def load_full_dataset():
    print("Loading The Movies Dataset (~45,000 rows) ...")
    check_files(DATA / "movies_metadata.csv", DATA / "credits.csv", DATA / "keywords.csv")
    meta     = pd.read_csv(DATA / "movies_metadata.csv", low_memory=False)
    credits  = pd.read_csv(DATA / "credits.csv")
    keywords = pd.read_csv(DATA / "keywords.csv")
    meta = meta[pd.to_numeric(meta["id"], errors="coerce").notna()].copy()
    meta["id"]     = meta["id"].astype(int)
    credits["id"]  = credits["id"].astype(int)
    keywords["id"] = keywords["id"].astype(int)
    df = meta.merge(credits, on="id", how="inner").merge(keywords, on="id", how="inner")
    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(safe_parse)
    df = df[KEEP_COLS].copy()
    df["source"] = "full_dataset"
    print(f"  -> {len(df):,} rows")
    return df


def merge_and_dedup(a, b):
    print("Merging and deduplicating on (title, release_year) ...")
    combined = pd.concat([a, b], ignore_index=True)
    combined["_year"] = pd.to_datetime(combined["release_date"], errors="coerce").dt.year
    combined["_key"]  = combined["title"].str.lower().str.strip()
    combined["_pri"]  = (combined["source"] == "tmdb_5000").astype(int)
    combined = combined.sort_values("_pri", ascending=False)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["_key", "_year"], keep="first")
    after = len(combined)
    combined = combined.drop(columns=["_key", "_pri"])
    combined = combined.rename(columns={"_year": "release_year"})
    print(f"  -> {before:,} combined -> {after:,} unique  ({before - after:,} duplicates removed)")
    return combined.reset_index(drop=True)


def quality_filter(df):
    """Remove rows that would produce empty or useless recommendation vectors."""
    before = len(df)
    df["overview"] = df["overview"].fillna("").str.strip()
    df = df[df["title"].notna()]
    df = df[df["overview"].str.len() >= 20]
    df = df[df["genres"].apply(lambda g: g != "[]")]
    after = len(df)
    print(f"  -> Quality filter: {before:,} -> {after:,}  ({before - after:,} removed)")
    return df.reset_index(drop=True)


def main():
    print("=" * 50)
    print("  Movie Recommender - Dataset Merge")
    print("=" * 50)
    tmdb   = load_tmdb_5000()
    full   = load_full_dataset()
    merged = merge_and_dedup(tmdb, full)
    print("Applying quality filter ...")
    merged = quality_filter(merged)
    out = DATA / "merged_raw.csv"
    merged.to_csv(out, index=False)
    print()
    print("=" * 50)
    print(f"  Saved  ->  {out}")
    print(f"  Total movies : {len(merged):,}")
    if len(merged) >= 10_000:
        print("  Resume bullet TRUE: 10,000+ titles confirmed")
    else:
        print("  WARNING: still under 10,000 - check CSV files")
    print("=" * 50)
    print("\nNext step:  python build_model.py\n")


if __name__ == "__main__":
    main()