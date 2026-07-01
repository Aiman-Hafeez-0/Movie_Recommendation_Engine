"""
app.py — CineMatch Movie Recommendation Engine
------------------------------------------------
Run:  streamlit run app.py

Tabs:
  Discover  — Trending top-movies by genre (Bayesian-ranked)
  Recommend — Hybrid content + quality recommendations for any title
"""

import os
import pickle
import random
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
ALPHA        = 0.65   # weight of content similarity vs Bayesian quality score

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Reset & Base ---------- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

[data-testid="stAppViewContainer"] {
    background: #090e1a;
    font-family: 'Inter', sans-serif;
    color: #d4dce8;
}
[data-testid="stSidebar"] {
    background: #0d1422;
    border-right: 1px solid #1e2c42;
}
[data-testid="stSidebar"] > div { padding-top: 1rem; }
section[data-testid="stSidebar"] * { color: #d4dce8; }
h1, h2, h3 { font-family: 'Inter', sans-serif; color: #fff; }

/* ---------- Film-Strip Header ---------- */
.filmstrip-wrap {
    margin: -1rem -1rem 0;
    background: #0d1422;
    border-bottom: 1px solid #1e2c42;
}
.sprocket-rail {
    height: 22px;
    background-color: #f5a623;
    background-image: radial-gradient(circle, #090e1a 7px, transparent 7px);
    background-size: 30px 22px;
    background-repeat: repeat-x;
    background-position: 8px center;
}
.header-center {
    padding: 1.6rem 2.5rem 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
    line-height: 1;
}
.logo span { color: #f5a623; }
.header-meta { text-align: right; }
.header-meta p { font-size: 0.8rem; color: #5a7a99; line-height: 1.7; }
.catalog-pill {
    display: inline-block;
    background: #f5a623;
    color: #090e1a;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ---------- Tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1e2c42;
    gap: 0.5rem;
    padding: 0 0.5rem;
    margin-top: 0.5rem;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    color: #5a7a99;
    border: none;
    background: transparent;
    padding: 0.7rem 1.2rem;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    transition: color 0.2s, border-color 0.2s;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #f5a623;
    border-bottom: 2px solid #f5a623;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: #f5a623; }

/* ---------- Sidebar widgets ---------- */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stTextInput"] input {
    background: #1c2537 !important;
    border: 1px solid #2a3548 !important;
    color: #d4dce8 !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #f5a623 !important;
    box-shadow: 0 0 0 2px rgba(245,166,35,0.15) !important;
}

/* ---------- Buttons ---------- */
[data-testid="stButton"] button {
    background: #f5a623;
    color: #090e1a;
    font-weight: 700;
    border: none;
    border-radius: 7px;
    padding: 0.55rem 1.8rem;
    font-size: 0.92rem;
    transition: background 0.18s, transform 0.1s;
    font-family: 'Inter', sans-serif;
}
[data-testid="stButton"] button:hover {
    background: #ffc34d;
    transform: translateY(-1px);
}
[data-testid="stButton"] button:active { transform: translateY(0); }

/* ---------- Section heading ---------- */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    padding: 1.4rem 0 0.8rem;
    border-bottom: 1px solid #1e2c42;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-heading .hl { color: #f5a623; }
.section-heading .badge {
    font-size: 0.7rem;
    font-weight: 600;
    background: rgba(245,166,35,0.12);
    color: #f5a623;
    padding: 2px 8px;
    border-radius: 10px;
    margin-left: 0.3rem;
}

/* ---------- Movie Cards ---------- */
.mc {
    background: #1c2537;
    border: 1px solid #2a3548;
    border-radius: 10px;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.mc:hover {
    transform: translateY(-5px);
    border-color: #f5a623;
    box-shadow: 0 8px 30px rgba(245,166,35,0.15);
}
.mc-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}
.mc-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: linear-gradient(160deg, #1c2537, #111827);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.5rem;
    color: #2a3548;
}
.mc-body { padding: 0.85rem 0.9rem 0.9rem; flex: 1; display: flex; flex-direction: column; gap: 0.35rem; }
.mc-title { font-size: 0.9rem; font-weight: 700; color: #fff; line-height: 1.3; }
.mc-year  { color: #f5a623; font-size: 0.78rem; font-weight: 600; }
.mc-dir   { font-size: 0.74rem; color: #7a8fa8; }
.mc-score-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.2rem 0; }
.mc-score-label { font-size: 0.7rem; color: #7a8fa8; white-space: nowrap; }
.mc-bar-track { flex: 1; height: 4px; background: #2a3548; border-radius: 2px; }
.mc-bar-fill  { height: 4px; border-radius: 2px; background: linear-gradient(90deg, #f5a623, #ffd280); }
.mc-rating { font-size: 0.72rem; color: #f5a623; font-weight: 600; white-space: nowrap; }
.mc-genres { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 0.2rem; }
.mc-pill {
    font-size: 0.64rem;
    padding: 2px 7px;
    border-radius: 10px;
    background: rgba(245,166,35,0.08);
    color: #c9a84c;
    border: 1px solid rgba(245,166,35,0.15);
}
.mc-why {
    font-size: 0.7rem;
    color: #4a6278;
    font-style: italic;
    margin-top: auto;
    padding-top: 0.4rem;
    border-top: 1px solid #222e42;
}
.mc-trailer {
    display: block;
    text-align: center;
    background: rgba(245,166,35,0.06);
    color: #f5a623 !important;
    border: 1px solid rgba(245,166,35,0.2);
    border-radius: 5px;
    padding: 4px 8px;
    margin-top: 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: background 0.15s;
}
.mc-trailer:hover { background: #f5a623; color: #090e1a !important; }

/* ---------- Discover Genre Pills ---------- */
.genre-pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
.gpill {
    padding: 5px 14px;
    border-radius: 20px;
    background: #1c2537;
    border: 1px solid #2a3548;
    font-size: 0.8rem;
    color: #8aa0b8;
    cursor: pointer;
    transition: all 0.15s;
}
.gpill.active { background: #f5a623; color: #090e1a; border-color: #f5a623; font-weight: 700; }

/* ---------- Recently Viewed ---------- */
.rv-item {
    padding: 0.45rem 0.6rem;
    background: #141d2b;
    border: 1px solid #1e2c42;
    border-left: 3px solid #f5a623;
    border-radius: 5px;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    color: #aab8c8;
}
.rv-item .rv-yr { color: #5a7a99; font-size: 0.7rem; margin-left: 4px; }

/* ---------- Stats table ---------- */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1a2234;
    font-size: 0.78rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key  { color: #7a8fa8; }
.stat-val  { color: #f5a623; font-weight: 700; }

/* ---------- Info box ---------- */
.info-box {
    background: rgba(245,166,35,0.06);
    border: 1px solid rgba(245,166,35,0.2);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: #9aaa7a;
    margin-bottom: 1rem;
}
.info-box b { color: #f5a623; }

/* ---------- Director section ---------- */
.dir-header {
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    padding: 1.2rem 0 0.7rem;
    border-bottom: 1px solid #1e2c42;
    margin-bottom: 0.9rem;
}
.dir-header span { color: #f5a623; }

/* ---------- No-results ---------- */
.no-res { text-align: center; padding: 3rem; color: #2a3d55; font-size: 1rem; }

/* ---------- Spinner override ---------- */
[data-testid="stSpinner"] { color: #f5a623 !important; }

/* ---------- Divider ---------- */
hr { border-color: #1e2c42 !important; }
</style>
""", unsafe_allow_html=True)


# ── Artifact loading (cached per session) ─────────────────────────────────────

@st.cache_resource(show_spinner="Loading CineMatch catalog …")
def load_artifacts():
    required = ["movie_list.pkl", "vectors.pkl"]
    missing  = [f for f in required if not Path(f).exists()]
    if missing:
        st.error(f"Run `python build_model.py` first. Missing: {', '.join(missing)}")
        st.stop()

    with open("movie_list.pkl",  "rb") as f: movies    = pickle.load(f)
    with open("vectors.pkl",     "rb") as f: vectors   = pickle.load(f)

    genre_top    = {}
    director_map = {}
    if Path("genre_top.pkl").exists():
        with open("genre_top.pkl",    "rb") as f: genre_top    = pickle.load(f)
    if Path("director_map.pkl").exists():
        with open("director_map.pkl", "rb") as f: director_map = pickle.load(f)

    return movies, vectors, genre_top, director_map


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(movie_id: int):
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=4,
        )
        r.raise_for_status()
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w400{path}" if path else None
    except requests.RequestException:
        return None


movies, vectors, genre_top, director_map = load_artifacts()

# Pre-compute some catalog stats
all_genres = sorted({
    g.strip()
    for gs in movies["genres_display"].fillna("")
    for g in gs.split(",") if g.strip()
})
valid_years  = movies["year"].dropna().astype(int)
year_min, year_max = int(valid_years.min()), int(valid_years.max())


# ── Session state ─────────────────────────────────────────────────────────────

if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []   # list of dicts {title, year}


def push_recently_viewed(title, year):
    rv = st.session_state.recently_viewed
    rv = [x for x in rv if x["title"] != title]   # dedupe
    rv.insert(0, {"title": title, "year": year})
    st.session_state.recently_viewed = rv[:8]


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_year(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def why_recommended(base_row, cand_row):
    hints = []
    base_g = {g.strip() for g in str(base_row.get("genres_display","")).split(",") if g.strip()}
    cand_g = {g.strip() for g in str(cand_row.get("genres_display","")).split(",") if g.strip()}
    shared = base_g & cand_g
    if shared:
        hints.append("Genre: " + ", ".join(sorted(shared)))
    bd = str(base_row.get("director_display","")).strip()
    cd = str(cand_row.get("director_display","")).strip()
    if bd and bd not in ("Unknown","nan") and bd == cd:
        hints.append(f"Director: {bd}")
    return " · ".join(hints) if hints else "Similar themes & story"


def movie_card_html(movie: dict, sim_score: float = None, why: str = None, show_score_bar=True) -> str:
    title    = movie.get("title", "")
    year     = safe_year(movie.get("year"))
    yr_str   = str(year) if year else "N/A"
    rating   = float(movie.get("vote_average") or 0)
    director = str(movie.get("director_display","")).strip()
    genres   = str(movie.get("genres_display","")).strip()
    overview = str(movie.get("overview","")).strip()
    mid      = int(movie.get("id", 0))

    poster_url = fetch_poster(mid) if TMDB_API_KEY else None
    poster_html = (
        f'<img class="mc-poster" src="{poster_url}" alt="{title}" loading="lazy">'
        if poster_url else
        '<div class="mc-placeholder">🎬</div>'
    )

    genre_pills = "".join(
        f'<span class="mc-pill">{g.strip()}</span>'
        for g in genres.split(",") if g.strip()
    )

    score_bar = ""
    if show_score_bar and sim_score is not None:
        pct = round(sim_score * 100)
        score_bar = f"""
        <div class="mc-score-row">
          <span class="mc-score-label">Match</span>
          <div class="mc-bar-track"><div class="mc-bar-fill" style="width:{pct}%"></div></div>
          <span class="mc-rating">{pct}%</span>
        </div>"""

    rating_bar_pct = round(rating * 10)
    rating_bar = f"""
        <div class="mc-score-row">
          <span class="mc-score-label">Rating</span>
          <div class="mc-bar-track"><div class="mc-bar-fill" style="width:{rating_bar_pct}%"></div></div>
          <span class="mc-rating">⭐ {rating:.1f}</span>
        </div>"""

    snippet  = (overview[:130] + "…") if len(overview) > 130 else overview
    dir_line = f'<div class="mc-dir">Dir. {director}</div>' if director and director not in ("Unknown","nan") else ""
    why_line = f'<div class="mc-why">↳ {why}</div>' if why else ""

    trailer_q   = f"{title} {yr_str} trailer".replace(" ", "+")
    trailer_url = f"https://www.youtube.com/results?search_query={trailer_q}"

    return f"""
<div class="mc">
  {poster_html}
  <div class="mc-body">
    <div class="mc-title">{title}</div>
    <div class="mc-year">{yr_str}</div>
    {dir_line}
    {score_bar}
    {rating_bar}
    <div class="mc-genres">{genre_pills}</div>
    <div style="font-size:0.72rem;color:#5a7a99;margin-top:0.3rem;line-height:1.5">{snippet}</div>
    {why_line}
    <a class="mc-trailer" href="{trailer_url}" target="_blank">▶ Trailer on YouTube</a>
  </div>
</div>"""


def render_card_grid(movie_list_dicts, cols=4, sim_scores=None, base_row=None):
    COLS = cols
    groups = [movie_list_dicts[i:i+COLS] for i in range(0, len(movie_list_dicts), COLS)]
    for grp in groups:
        columns = st.columns(COLS)
        for col, movie in zip(columns, grp):
            idx     = movie_list_dicts.index(movie)
            sim     = sim_scores[idx] if sim_scores else None
            why_str = why_recommended(base_row, movie) if base_row is not None and sim is not None else None
            with col:
                st.html(movie_card_html(movie, sim_score=sim, why=why_str))


# ── Recommendation engine ─────────────────────────────────────────────────────

def recommend_hybrid(title: str, n: int = 10) -> pd.DataFrame:
    """
    Hybrid ranking:
      final_score = ALPHA * cosine_similarity + (1 - ALPHA) * bayesian_wr_norm
    Blends content closeness with a quality signal so you don't surface
    obscure but textually-similar films ahead of beloved ones.
    """
    idx_matches = movies.index[movies["title"].str.lower() == title.lower()]
    if len(idx_matches) == 0:
        return pd.DataFrame()
    idx  = idx_matches[0]
    sims = cosine_similarity(vectors[idx], vectors).flatten()

    result          = movies.copy()
    result["_sim"]  = sims
    result          = result.drop(index=idx)

    if "bayesian_wr_norm" in result.columns:
        result["_score"] = ALPHA * result["_sim"] + (1 - ALPHA) * result["bayesian_wr_norm"]
    else:
        result["_score"] = result["_sim"]

    return result.sort_values("_score", ascending=False).head(n).reset_index(drop=True)


# ── Film-strip header ─────────────────────────────────────────────────────────

decade_counts = (
    movies["year"].dropna().astype(int)
    .apply(lambda y: f"{(y // 10) * 10}s")
    .value_counts()
    .sort_index()
)

st.markdown(f"""
<div class="filmstrip-wrap">
  <div class="sprocket-rail"></div>
  <div class="header-center">
    <div>
      <div class="logo">Cine<span>Match</span></div>
      <div style="font-size:0.8rem;color:#5a7a99;margin-top:4px">
        Content-based recommendations &nbsp;·&nbsp; TF-IDF + Cosine Similarity + Bayesian Ranking
      </div>
    </div>
    <div class="header-meta">
      <p>{len(movies):,} films indexed</p>
      <p>{len(all_genres)} genres &nbsp;·&nbsp; {len(director_map):,} directors</p>
      <span class="catalog-pill">42 k+ Catalog</span>
    </div>
  </div>
  <div class="sprocket-rail"></div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")
    genre_filter = st.multiselect("Genre", all_genres, placeholder="Any genre")
    min_rating   = st.slider("Min rating ⭐", 0.0, 10.0, 0.0, 0.5)
    year_range   = st.slider("Release year", year_min, year_max, (1980, year_max))
    n_results    = st.slider("Results to show", 3, 16, 8)

    st.markdown("---")
    st.markdown("### 📊 Catalog stats")
    stats = [
        ("Total films",    f"{len(movies):,}"),
        ("Genres",         str(len(all_genres))),
        ("Directors",      f"{len(director_map):,}"),
        ("Earliest year",  str(year_min)),
        ("Latest year",    str(year_max)),
    ]
    for k, v in stats:
        st.html(f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>')

    st.markdown("---")
    st.markdown("### Decade breakdown")
    for decade, count in decade_counts.items():
        pct = int(count / len(movies) * 100)
        st.html(f'<div class="stat-row"><span class="stat-key">{decade}</span>'
            f'<span class="stat-val">{count:,}</span></div>')

    if st.session_state.recently_viewed:
        st.markdown("---")
        st.markdown("### 🕐 Recently viewed")
        for rv in st.session_state.recently_viewed:
            yr = f"({rv['year']})" if rv.get("year") else ""
            st.html(f'<div class="rv-item">{rv["title"]}<span class="rv-yr">{yr}</span></div>')

    st.markdown("---")
    st.markdown("### ⚙️ Recommendation method")
    st.caption(
        f"**Hybrid scoring** — content similarity weighted {int(ALPHA*100)}% "
        f"and Bayesian quality {int((1-ALPHA)*100)}%. "
        "Vectors: weighted TF-IDF (genre 4×, director 4×, keywords 3×, cast 2×, overview 1×). "
        "Similarity: cosine on a 42 k × 15 k sparse matrix, computed per query."
    )


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_discover, tab_recommend = st.tabs(["🏠  Discover", "🔍  Find Similar"])


# ════════════════════════════════════════════════════════
#  TAB 1 — DISCOVER
# ════════════════════════════════════════════════════════

with tab_discover:
    st.html('<div class="section-heading">Trending by Genre'
        '<span class="badge">Bayesian-ranked</span></div>')
    st.caption("Top-rated films per genre, scored using the IMDB Bayesian Weighted Rating formula.")

    selected_genre = st.selectbox(
        "Pick a genre",
        all_genres,
        index=all_genres.index("Action") if "Action" in all_genres else 0,
        key="discover_genre",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 5])
    with c1:
        surprise = st.button("🎲 Surprise Me", key="surprise_btn")

    if surprise:
        rand_movie = random.choice(genre_top.get(selected_genre, []))
        if rand_movie:
            st.session_state["surprise_pick"] = rand_movie["title"]
            push_recently_viewed(rand_movie["title"], safe_year(rand_movie.get("year")))
            st.info(f"🎲 Random pick: **{rand_movie['title']}** — head to *Find Similar* to explore!")

    if selected_genre and genre_top:
        top_movies = genre_top.get(selected_genre, [])
        if top_movies:
            render_card_grid(top_movies[:8], cols=4)
        else:
            st.html('<div class="no-res">No movies found for this genre.</div>')
    else:
        st.html('<div class="info-box">Run <b>python build_model.py</b> to generate the genre index.</div>')


# ════════════════════════════════════════════════════════
#  TAB 2 — FIND SIMILAR
# ════════════════════════════════════════════════════════

with tab_recommend:
    st.html('<div class="section-heading">Find Similar Movies</div>')

    # Pre-fill from Surprise Me
    default_search = st.session_state.get("surprise_pick", "")

    search = st.text_input(
        "Search title",
        value=default_search,
        placeholder="e.g.  Inception,  The Dark Knight,  Parasite …",
        key="search_input",
        label_visibility="collapsed",
    )

    if search:
        hits = movies[movies["title"].str.contains(search, case=False, na=False)]["title"]
    else:
        hits = movies["title"]

    if hits.empty:
        st.warning("No titles matched — try a shorter or different term.")
        st.stop()

    selected = st.selectbox("Select a movie", sorted(hits.unique()), key="movie_select", label_visibility="collapsed")

    # Show selected movie info inline
    base_matches = movies[movies["title"].str.lower() == selected.lower()]
    if not base_matches.empty:
        b = base_matches.iloc[0]
        b_year    = safe_year(b.get("year"))
        b_rating  = float(b["vote_average"]) if pd.notna(b.get("vote_average")) else 0
        b_genres  = str(b.get("genres_display",""))
        b_dir     = str(b.get("director_display","")).strip()
        b_dir_str = f"· Dir. {b_dir}" if b_dir and b_dir not in ("Unknown","nan") else ""
        st.caption(
            f"**{selected}** ({b_year or 'N/A'}) "
            f"· ⭐ {b_rating:.1f} "
            f"· {b_genres} {b_dir_str}"
        )
        if b_year:
            push_recently_viewed(selected, b_year)

    go = st.button("🎬 Find Similar Movies", type="primary", key="go_btn")

    if go:
        with st.spinner("Computing hybrid recommendations …"):
            results = recommend_hybrid(selected, n=n_results * 4)

        if results.empty:
            st.html('<div class="no-res">Movie not found in the catalog.</div>')
            st.stop()

        # Apply filters
        if genre_filter:
            results = results[results["genres_display"].apply(
                lambda g: any(gf in str(g) for gf in genre_filter)
            )]
        results = results[results["vote_average"] >= min_rating]
        results = results[
            results["year"].isna() |
            results["year"].between(year_range[0], year_range[1])
        ]
        results = results.head(n_results).reset_index(drop=True)

        if results.empty:
            st.html('<div class="no-res">No results match your filters — try widening them.</div>')
            st.stop()

        base_row = movies[movies["title"].str.lower() == selected.lower()].iloc[0].to_dict()
        st.html(
            f'<div class="section-heading">Because you like <span class="hl">{selected}</span>'
            f'<span class="badge">{len(results)} results</span></div>'
        )

        movie_dicts = results.to_dict("records")
        sim_scores  = results["_score"].tolist()

        COLS = 4
        rows = [movie_dicts[i:i+COLS] for i in range(0, len(movie_dicts), COLS)]
        for row in rows:
            cols = st.columns(COLS)
            for col, movie in zip(cols, row):
                idx     = movie_dicts.index(movie)
                sim     = sim_scores[idx]
                why_str = why_recommended(base_row, movie)
                with col:
                    st.html(movie_card_html(movie, sim_score=sim, why=why_str))

        # ── Director filmography section ──────────────────────────────────────
        director = str(base_row.get("director_display","")).strip()
        if director and director not in ("Unknown","nan") and director in director_map:
            filmography = [
                m for m in director_map[director]
                if m.get("title","").lower() != selected.lower()
            ][:8]

            if filmography:
                st.html(f'<div class="dir-header">More from <span>{director}</span></div>')
                dir_cols = st.columns(min(4, len(filmography)))
                for col, movie in zip(dir_cols, filmography[:4]):
                    with col:
                        st.html(movie_card_html(movie, show_score_bar=False))

        # ── Expandable detail ─────────────────────────────────────────────────
        with st.expander("📋 Full recommendation table"):
            display_df = results[["title", "year", "director_display", "genres_display", "vote_average", "_score"]].copy()
            display_df.columns = ["Title", "Year", "Director", "Genres", "Rating", "Match Score"]
            display_df["Match Score"] = display_df["Match Score"].map(lambda x: f"{x:.3f}")
            display_df["Rating"]      = display_df["Rating"].map(lambda x: f"{x:.1f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.html(
    '<div style="text-align:center;font-size:0.75rem;color:#2a3d55;padding:0.5rem">'
    'CineMatch · Built with Streamlit · scikit-learn TF-IDF + Cosine Similarity · '
    'TMDB 5000 + The Movies Dataset (Kaggle) · Developed by Aiman Hafeez'
    '</div>'
)