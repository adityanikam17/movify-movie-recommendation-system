"""
Movify — Streamlit frontend.

A single-page Streamlit app (with lightweight client-side "routing" via
``st.query_params``) that talks to a FastAPI backend for:

* keyword search with autocomplete suggestions
* a categorized home feed (trending / popular / top rated / ...)
* movie detail pages with TF-IDF and genre-based recommendations

The backend's exact response shape for ``/tmdb/search`` can vary (raw TMDB
payload vs. a pre-shaped list of cards), so the parsing layer below is
written to tolerate both without the UI code needing to know the difference.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("movie_recommender")

# Prefer an environment variable / Streamlit secret over a hardcoded value so
# the same code works locally and in deployment without editing source.
API_BASE = os.getenv("MOVIE_API_BASE", "https://movie-rec-466x.onrender.com").rstrip("/")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/1f2937/ffffff?text=No+Poster"

REQUEST_TIMEOUT_S = 25
CACHE_TTL_SEARCH_S = 30
CACHE_TTL_FEED_S = 300
CACHE_TTL_DETAILS_S = 600

HOME_CATEGORIES = ["trending", "popular", "top_rated", "now_playing", "upcoming"]
MIN_SEARCH_CHARS = 2

st.set_page_config(page_title="Movify", page_icon="🎬", layout="wide")

T = TypeVar("T")

# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------

st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title {
    font-size: 0.9rem; font-weight: 600; line-height: 1.2rem;
    height: 2.4rem; overflow: hidden; margin: 0.4rem 0 0.3rem 0;
}
.card {
    border: 1px solid rgba(0,0,0,0.08); border-radius: 16px;
    padding: 16px; background: rgba(255,255,255,0.7);
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------------


class APIError(Exception):
    """Raised when the backend API is unreachable or returns an error."""


def _build_session() -> requests.Session:
    """A requests.Session with connection pooling and automatic retries
    on transient (5xx) failures, so a single flaky response doesn't break
    the whole page."""
    session = requests.Session()
    retry_policy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _build_session()


def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET a JSON resource from the backend, raising ``APIError`` with a
    user-friendly message on any failure."""
    url = f"{API_BASE}{path}"
    try:
        response = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise APIError("The server took too long to respond. Please try again.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise APIError("Could not reach the movie service. Is the backend running?") from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        raise APIError(f"The server returned an error (HTTP {status}).") from exc
    except ValueError as exc:  # JSON decoding failure
        raise APIError("The server sent back something that wasn't valid JSON.") from exc


def safe_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T | None, str | None]:
    """Run a (possibly cached) API call and turn exceptions into a
    ``(result, error_message)`` tuple so UI code stays simple."""
    try:
        return func(*args, **kwargs), None
    except APIError as exc:
        logger.warning("API call failed: %s", exc)
        return None, str(exc)


@st.cache_data(ttl=CACHE_TTL_SEARCH_S, show_spinner=False)
def fetch_search_results(query: str) -> Any:
    return api_get("/tmdb/search", params={"query": query})


@st.cache_data(ttl=CACHE_TTL_FEED_S, show_spinner=False)
def fetch_home_feed(category: str, limit: int = 24) -> Any:
    return api_get("/home", params={"category": category, "limit": limit})


@st.cache_data(ttl=CACHE_TTL_DETAILS_S, show_spinner=False)
def fetch_movie_details(tmdb_id: int) -> Any:
    return api_get(f"/movie/id/{tmdb_id}")


@st.cache_data(ttl=CACHE_TTL_DETAILS_S, show_spinner=False)
def fetch_recommendation_bundle(title: str, tfidf_top_n: int = 12, genre_limit: int = 12) -> Any:
    return api_get(
        "/movie/search",
        params={"query": title, "tfidf_top_n": tfidf_top_n, "genre_limit": genre_limit},
    )


@st.cache_data(ttl=CACHE_TTL_DETAILS_S, show_spinner=False)
def fetch_genre_recommendations(tmdb_id: int, limit: int = 18) -> Any:
    return api_get("/recommend/genre", params={"tmdb_id": tmdb_id, "limit": limit})


# ----------------------------------------------------------------------------
# Data model + parsing
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MovieCard:
    """A normalized, display-ready movie record. Every place in the UI that
    renders a poster grid works with this type, regardless of which backend
    endpoint (or TMDB response shape) the data originally came from."""

    tmdb_id: int
    title: str
    poster_url: str | None = None
    release_date: str = ""

    @property
    def display_poster(self) -> str:
        return self.poster_url or PLACEHOLDER_POSTER

    @property
    def year(self) -> str:
        return (self.release_date or "")[:4]

    @property
    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


def parse_search_results(data: Any, keyword: str) -> list[MovieCard]:
    """Normalize ``/tmdb/search`` results into ``MovieCard``s and filter by
    keyword. Supports both a raw TMDB payload (``{"results": [...]}`` with
    ``id`` / ``poster_path``) and an already-shaped list of cards
    (``tmdb_id`` / ``poster_url``), since the backend has returned both
    shapes at different times.
    """
    items: list[MovieCard] = []

    if isinstance(data, dict) and "results" in data:
        for movie in data.get("results") or []:
            title = (movie.get("title") or "").strip()
            tmdb_id = movie.get("id")
            if not title or not tmdb_id:
                continue
            poster_path = movie.get("poster_path")
            items.append(
                MovieCard(
                    tmdb_id=int(tmdb_id),
                    title=title,
                    poster_url=f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None,
                    release_date=movie.get("release_date", ""),
                )
            )
    elif isinstance(data, list):
        for movie in data:
            tmdb_id = movie.get("tmdb_id") or movie.get("id")
            title = (movie.get("title") or "").strip()
            if not title or not tmdb_id:
                continue
            items.append(
                MovieCard(
                    tmdb_id=int(tmdb_id),
                    title=title,
                    poster_url=movie.get("poster_url"),
                    release_date=movie.get("release_date", ""),
                )
            )

    keyword_l = keyword.strip().lower()
    if not keyword_l:
        return items

    matched = [m for m in items if keyword_l in m.title.lower()]
    # Never show a blank page just because the substring match was too
    # strict — fall back to the unfiltered list from the backend.
    return matched or items


def parse_tfidf_items(items: Any) -> list[MovieCard]:
    cards: list[MovieCard] = []
    for entry in items or []:
        tmdb = entry.get("tmdb") or {}
        tmdb_id = tmdb.get("tmdb_id")
        if not tmdb_id:
            continue
        cards.append(
            MovieCard(
                tmdb_id=int(tmdb_id),
                title=tmdb.get("title") or entry.get("title") or "Untitled",
                poster_url=tmdb.get("poster_url"),
            )
        )
    return cards


def parse_genre_items(items: Any) -> list[MovieCard]:
    cards: list[MovieCard] = []
    for entry in items or []:
        tmdb_id = entry.get("tmdb_id") or entry.get("id")
        if not tmdb_id:
            continue
        cards.append(
            MovieCard(
                tmdb_id=int(tmdb_id),
                title=entry.get("title") or "Untitled",
                poster_url=entry.get("poster_url"),
                release_date=entry.get("release_date", ""),
            )
        )
    return cards


# ----------------------------------------------------------------------------
# Routing (query-param backed view state)
# ----------------------------------------------------------------------------


def init_session_state() -> None:
    st.session_state.setdefault("view", "home")
    st.session_state.setdefault("selected_tmdb_id", None)

    qp_view = st.query_params.get("view")
    qp_id = st.query_params.get("id")

    if qp_view in ("home", "details"):
        st.session_state.view = qp_view

    if qp_id:
        try:
            st.session_state.selected_tmdb_id = int(qp_id)
            st.session_state.view = "details"
        except ValueError:
            logger.warning("Ignoring invalid 'id' query param: %r", qp_id)


def go_home() -> None:
    st.session_state.view = "home"
    st.session_state.selected_tmdb_id = None
    st.query_params.clear()
    st.query_params["view"] = "home"
    st.rerun()


def go_to_details(tmdb_id: int) -> None:
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# ----------------------------------------------------------------------------
# Reusable UI components
# ----------------------------------------------------------------------------


def render_poster_grid(cards: list[MovieCard], columns: int, key_prefix: str) -> None:
    if not cards:
        st.info("No movies to show.")
        return

    for row_start in range(0, len(cards), columns):
        row = cards[row_start : row_start + columns]
        grid_cols = st.columns(columns)
        for col, card in zip(grid_cols, row):
            with col:
                st.image(card.display_poster, use_container_width=True)
                st.markdown(f"<div class='movie-title'>{card.title}</div>", unsafe_allow_html=True)
                if st.button("Open", key=f"{key_prefix}-{card.tmdb_id}-{row_start}", use_container_width=True):
                    go_to_details(card.tmdb_id)


def render_api_error(message: str, retry_label: str = "Retry") -> None:
    st.error(f"⚠️ {message}")
    if st.button(retry_label):
        st.cache_data.clear()
        st.rerun()


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------


def render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        st.markdown("## 🎬 Movify")
        if st.button("🏠 Home", use_container_width=True):
            go_home()

        st.markdown("---")
        st.markdown("### Home Feed")
        category = st.selectbox(
            "Category",
            HOME_CATEGORIES,
            index=0,
            format_func=lambda c: c.replace("_", " ").title(),
        )
        columns = st.slider("Grid columns", min_value=4, max_value=8, value=6)

        st.markdown("---")
        with st.expander("Connection"):
            st.caption(f"API base: `{API_BASE}`")

    return category, columns


def render_header() -> None:
    st.title("🎬 Movify")
    st.markdown(
        "<div class='small-muted'>Type a keyword for suggestions and matching results, "
        "or open a movie for details and recommendations.</div>",
        unsafe_allow_html=True,
    )
    st.divider()


def render_search_results(query: str, columns: int) -> None:
    if len(query) < MIN_SEARCH_CHARS:
        st.caption(f"Type at least {MIN_SEARCH_CHARS} characters for suggestions.")
        return

    with st.spinner("Searching…"):
        data, err = safe_call(fetch_search_results, query)

    if err:
        render_api_error(f"Search failed: {err}")
        return

    cards = parse_search_results(data, query)

    if cards:
        labels = ["-- Select a movie --"] + [c.label for c in cards[:10]]
        label_to_id = {c.label: c.tmdb_id for c in cards[:10]}
        selected = st.selectbox("Suggestions", labels, index=0)
        if selected != "-- Select a movie --":
            go_to_details(label_to_id[selected])
    else:
        st.info("No suggestions found. Try another keyword.")

    st.markdown("### Results")
    render_poster_grid(cards, columns=columns, key_prefix="search")


def render_home_feed(category: str, columns: int) -> None:
    st.markdown(f"### 🏠 Home — {category.replace('_', ' ').title()}")

    with st.spinner("Loading feed…"):
        raw_cards, err = safe_call(fetch_home_feed, category, 24)

    if err:
        render_api_error(f"Could not load the home feed: {err}")
        return

    if not raw_cards:
        st.info("Nothing to show for this category right now.")
        return

    cards = [
        MovieCard(
            tmdb_id=int(c["tmdb_id"]),
            title=c.get("title", "Untitled"),
            poster_url=c.get("poster_url"),
        )
        for c in raw_cards
        if c.get("tmdb_id")
    ]
    render_poster_grid(cards, columns=columns, key_prefix="home")


def render_home_view(category: str, columns: int) -> None:
    query = st.text_input(
        "Search by movie title (keyword)",
        placeholder="Type: avenger, batman, love...",
    ).strip()
    st.divider()

    if query:
        render_search_results(query, columns)
    else:
        render_home_feed(category, columns)


def render_recommendations(title: str, tmdb_id: int, columns: int) -> None:
    st.divider()
    st.markdown("### ✅ Recommendations")

    if not title:
        st.warning("No title available to compute recommendations.")
        return

    with st.spinner("Finding similar movies…"):
        bundle, err = safe_call(fetch_recommendation_bundle, title)

    if bundle and not err:
        st.markdown("#### 🔎 Similar Movies (TF-IDF)")
        render_poster_grid(
            parse_tfidf_items(bundle.get("tfidf_recommendations")),
            columns=columns,
            key_prefix="rec-tfidf",
        )

        st.markdown("#### 🎭 More Like This (Genre)")
        render_poster_grid(
            parse_genre_items(bundle.get("genre_recommendations", [])),
            columns=columns,
            key_prefix="rec-genre",
        )
        return

    st.info("TF-IDF recommendations unavailable — showing genre matches instead.")
    with st.spinner("Finding genre matches…"):
        genre_cards, err2 = safe_call(fetch_genre_recommendations, tmdb_id, 18)

    if err2 or not genre_cards:
        st.warning("No recommendations available right now.")
        return

    render_poster_grid(parse_genre_items(genre_cards), columns=columns, key_prefix="rec-genre-fallback")


def render_details_view(columns: int) -> None:
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            go_home()
        return

    header_col, back_col = st.columns([3, 1])
    with header_col:
        st.markdown("### 📄 Movie Details")
    with back_col:
        if st.button("← Back to Home", use_container_width=True):
            go_home()

    with st.spinner("Loading details…"):
        data, err = safe_call(fetch_movie_details, tmdb_id)

    if err or not data:
        render_api_error(f"Could not load details: {err or 'Unknown error'}")
        return

    poster_col, info_col = st.columns([1, 2.4], gap="large")

    with poster_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(data.get("poster_url") or PLACEHOLDER_POSTER, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title', '')}")
        release = data.get("release_date") or "-"
        genres = ", ".join(g["name"] for g in data.get("genres", [])) or "-"
        st.markdown(f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Genres: {genres}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], use_container_width=True)

    render_recommendations(title=(data.get("title") or "").strip(), tmdb_id=tmdb_id, columns=columns)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    init_session_state()
    category, columns = render_sidebar()
    render_header()

    if st.session_state.view == "details":
        render_details_view(columns)
    else:
        render_home_view(category, columns)


if __name__ == "__main__":
    main()