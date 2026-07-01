# 🎬 Movie Recommendation Engine

A content-based movie recommender that suggests similar titles using a weighted
TF-IDF representation of each movie's overview, genres, cast, director, and
keywords, scored with cosine similarity. Built with Streamlit.

## Demo

> Add a screenshot or GIF here, and your live Streamlit Cloud link once deployed.

## How it works

1. **Offline build step (`build_model.py`)** — merges the TMDB movies + credits
   datasets, parses the stringified JSON columns (genres, cast, crew,
   keywords), and builds a per-movie "tags" string. Each signal is weighted
   by repetition (director and genre count more than a single cast member or
   a word from the overview) before TF-IDF vectorization.
2. **Vectorization** — `TfidfVectorizer` produces a sparse matrix instead of a
   dense one, and instead of precomputing a full N×N similarity matrix
   (which becomes hundreds of MB at 10,000+ titles), similarity is computed
   on the fly per query: `cosine_similarity(vectors[idx], vectors)`. This
   keeps memory flat as the dataset grows.
3. **Live app (`app.py`)** — loads the pickled artifacts once via
   `@st.cache_resource`, lets the user search/select a movie, applies
   genre/rating/year filters, and (optionally) enriches results with live
   poster/rating data from the TMDB API.

## Features

- Weighted hybrid similarity (genre, cast, director, keywords, overview)
- Search-as-you-type movie picker instead of a long static dropdown
- Sidebar filters: genre, minimum rating, release year range
- "Why this was recommended" tags (shared genre / director)
- Graceful degradation when no TMDB API key is set (no crashes, just no posters)
- Cached data loading and API calls for snappy reruns
- Trailer search links per result
- Scales to 10,000+ titles without precomputing a giant similarity matrix

## Project structure

```
movie-recommender/
├── data/
│   ├── tmdb_5000_movies.csv      # not included — download from Kaggle
│   └── tmdb_5000_credits.csv     # not included — download from Kaggle
├── app.py                        # Streamlit app
├── build_model.py                # offline data prep + vectorization
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
git clone <your-repo-url>
cd movie-recommender
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
from Kaggle and place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` into `data/`.

(Optional) Get a free TMDB API key at themoviedb.org → Settings → API, then:

```bash
cp .env.example .env
# edit .env and paste your key
```

## Build the model

```bash
python build_model.py
```

This generates `movie_list.pkl`, `vectors.pkl`, and `tfidf.pkl` — the
precomputed artifacts the app reads. Re-run this whenever the underlying
data or feature weights change.

## Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploying

1. Push the repo to GitHub (the `.gitignore` already excludes the large
   `.pkl` files and raw CSVs — either commit smaller pickles directly, use
   Git LFS, or have your deploy step run `build_model.py` on first boot).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set `app.py` as the entry point.
3. Add `TMDB_API_KEY` under the app's **Secrets** if you want posters/ratings.

## Possible extensions

- Swap content-based filtering for a hybrid model (add collaborative
  filtering once you have user rating data)
- Approximate nearest neighbors (`sklearn.neighbors.NearestNeighbors`,
  cosine metric) once the catalog grows well past 10,000 titles
- Expand the dataset by merging in MovieLens 25M or "The Movies Dataset"
  (45,000 titles) on title/year
- Add a "more like this person's taste" mode using a short user-rated list

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `scipy` · `streamlit` · `nltk` · `requests`
