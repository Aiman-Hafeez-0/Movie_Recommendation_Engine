"""
build_model.py
--------------
Offline build step. Reads data/merged_raw.csv and produces:

  movie_list.pkl   — cleaned DataFrame with Bayesian weighted ratings
  vectors.pkl      — sparse TF-IDF matrix (one row per movie)
  tfidf.pkl        — fitted TfidfVectorizer
  genre_top.pkl    — {genre: [top-12 movie dicts]} for Discover tab
  director_map.pkl — {director: [movie dicts]} for filmography section

Run:
    python build_model.py
Re-run whenever the data changes.
"""

import ast
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

DATA        = Path("data")
MERGED_CSV  = DATA / "merged_raw.csv"
MOVIES_CSV  = DATA / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA / "tmdb_5000_credits.csv"

# Feature weights — repeated tokens in the tags string act as TF-IDF emphasis
WEIGHTS = {
    "overview":  1,
    "genres":    4,
    "keywords":  3,
    "cast":      2,
    "crew":      4,   # director signal matters most for "feel alike"
}

ps = PorterStemmer()


# ── parsers ───────────────────────────────────────────────────────────────────

def parse_names(val, top=None):
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return []
    try:
        items = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        try:
            items = ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return []
    names = [i.get("name", "") for i in items if isinstance(i, dict)]
    return names[:top] if top else names


def get_director(val):
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return []
    try:
        items = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        try:
            items = ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return []
    for i in items:
        if isinstance(i, dict) and i.get("job") == "Director":
            return [i.get("name", "")]
    return []


def no_spaces(tokens):
    return [t.replace(" ", "") for t in tokens if t]


def stem_text(text):
    return " ".join(ps.stem(w) for w in text.split())


# ── load ──────────────────────────────────────────────────────────────────────

def load_data():
    if MERGED_CSV.exists():
        print(f"Using merged dataset: {MERGED_CSV}")
        df = pd.read_csv(MERGED_CSV, low_memory=False)
        df["overview"] = df["overview"].fillna("").str.strip()
        return df.dropna(subset=["title"]).reset_index(drop=True)
    if not MOVIES_CSV.exists() or not CREDITS_CSV.exists():
        print("[ERROR] Run merge_datasets.py first.")
        sys.exit(1)
    print("Falling back to TMDB 5000 only.")
    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    df = movies.merge(credits, left_on="id", right_on="movie_id")
    df = df.rename(columns={"title_x": "title"})
    df["overview"] = df["overview"].fillna("").str.strip()
    return df.dropna(subset=["title"]).reset_index(drop=True)


# ── feature engineering ───────────────────────────────────────────────────────

def engineer(df):
    print("Parsing metadata columns ...")
    df["genres_list"]   = df["genres"].apply(lambda v: parse_names(v))
    df["keywords_list"] = df["keywords"].apply(lambda v: parse_names(v, top=10))
    df["cast_list"]     = df["cast"].apply(lambda v: parse_names(v, top=5))
    df["crew_list"]     = df["crew"].apply(get_director)

    df["genres_t"]   = df["genres_list"].apply(no_spaces)
    df["keywords_t"] = df["keywords_list"].apply(no_spaces)
    df["cast_t"]     = df["cast_list"].apply(no_spaces)
    df["crew_t"]     = df["crew_list"].apply(no_spaces)

    print("Building weighted tag strings ...")
    def build_tags(row):
        parts = (
            row["overview"].split()    * WEIGHTS["overview"] +
            row["genres_t"]            * WEIGHTS["genres"]   +
            row["keywords_t"]          * WEIGHTS["keywords"] +
            row["cast_t"]              * WEIGHTS["cast"]     +
            row["crew_t"]              * WEIGHTS["crew"]
        )
        return " ".join(parts).lower()

    df["tags"] = df.apply(build_tags, axis=1)

    print("Stemming ...")
    df["tags"] = df["tags"].apply(stem_text)

    df["genres_display"]   = df["genres_list"].apply(lambda g: ", ".join(g[:5]))
    df["cast_display"]     = df["cast_list"].apply(lambda c: ", ".join(c[:3]))
    df["director_display"] = df["crew_list"].apply(lambda c: c[0] if c else "Unknown")
    df["year"]             = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["vote_average"]     = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0)
    df["vote_count"]       = pd.to_numeric(df.get("vote_count",   0), errors="coerce").fillna(0)
    df["popularity"]       = pd.to_numeric(df.get("popularity",   0), errors="coerce").fillna(0)
    return df


def compute_bayesian_rating(df):
    """
    IMDB Bayesian Weighted Rating formula:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    v = vote_count, m = minimum-votes threshold, R = movie rating, C = global mean
    This balances a movie's own rating against the global mean, preventing
    low-vote movies from unfairly dominating top lists.
    """
    print("Computing Bayesian weighted ratings ...")
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.70)
    v = df["vote_count"]
    R = df["vote_average"]
    df["bayesian_wr"] = (v / (v + m)) * R + (m / (v + m)) * C

    # Normalize 0-1 for hybrid scoring blend in the app
    lo, hi = df["bayesian_wr"].min(), df["bayesian_wr"].max()
    df["bayesian_wr_norm"] = (df["bayesian_wr"] - lo) / (hi - lo)
    print(f"  -> global mean rating: {C:.2f}, vote threshold: {m:.0f}")
    return df


# ── vectorize ─────────────────────────────────────────────────────────────────

def vectorize(df):
    print("Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(
        max_features=15_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    vectors = tfidf.fit_transform(df["tags"])
    print(f"  -> {vectors.shape}  ({vectors.nnz:,} non-zero entries, sparse)")
    return tfidf, vectors


# ── secondary lookup artifacts ────────────────────────────────────────────────

def build_genre_top(movie_list):
    """Top 12 movies per genre by Bayesian score — powers the Discover tab."""
    print("Building genre top-lists ...")
    all_genres = sorted({
        g.strip()
        for gs in movie_list["genres_display"].fillna("")
        for g in gs.split(",")
        if g.strip()
    })
    genre_top = {}
    for genre in all_genres:
        mask = movie_list["genres_display"].str.contains(genre, case=False, na=False)
        top  = (
            movie_list[mask]
            .sort_values("bayesian_wr", ascending=False)
            .head(12)
            .to_dict("records")
        )
        genre_top[genre] = top
    print(f"  -> {len(genre_top)} genres indexed")
    return genre_top


def build_director_map(movie_list):
    """Map director name -> list of their movies — powers the filmography section."""
    print("Building director filmography map ...")
    director_map = {}
    for _, row in movie_list.iterrows():
        d = str(row.get("director_display", "")).strip()
        if d and d not in ("Unknown", "", "nan"):
            director_map.setdefault(d, []).append(row.to_dict())
    # Sort each director's list by Bayesian rating descending
    for d in director_map:
        director_map[d].sort(key=lambda x: x.get("bayesian_wr", 0), reverse=True)
    print(f"  -> {len(director_map)} directors indexed")
    return director_map


# ── main ──────────────────────────────────────────────────────────────────────

SAVE_COLS = [
    "id", "title", "overview", "genres_display", "cast_display",
    "director_display", "vote_average", "vote_count", "year",
    "popularity", "bayesian_wr", "bayesian_wr_norm", "tags",
]


def main():
    print("=" * 52)
    print("  Movie Recommender — Model Build")
    print("=" * 52)

    df = load_data()
    print(f"  -> {len(df):,} movies loaded")

    df = engineer(df)
    df = compute_bayesian_rating(df)

    df = df[df["tags"].str.split().str.len() >= 5].reset_index(drop=True)
    print(f"  -> {len(df):,} movies after quality filter")

    tfidf, vectors = vectorize(df)

    movie_list = df[SAVE_COLS].copy()

    genre_top    = build_genre_top(movie_list)
    director_map = build_director_map(movie_list)

    print("\nSaving artifacts ...")
    artifacts = {
        "movie_list.pkl":   movie_list,
        "vectors.pkl":      vectors,
        "tfidf.pkl":        tfidf,
        "genre_top.pkl":    genre_top,
        "director_map.pkl": director_map,
    }
    import os
    for fname, obj in artifacts.items():
        with open(fname, "wb") as f:
            pickle.dump(obj, f)
        size = os.path.getsize(fname) / 1_048_576
        print(f"  {fname:22s}  {size:.1f} MB")

    print()
    print("=" * 52)
    print(f"  Build complete — {len(df):,} movies ready.")
    print("=" * 52)
    print("\nNext:  streamlit run app.py\n")


if __name__ == "__main__":
    main()