from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from training.retrieval_samples import build_retrieval_samples
from features.user_history import build_history_prefixes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "split"


def test_history_contains_only_earlier_positive_movies(sample_tables):
    train = sample_tables["ratings"]
    original = train.copy(deep=True)

    events = build_history_prefixes(train, max_history=2)

    for _, user_events in events.groupby("user_id", sort=False):
        liked_movies = []
        for event in user_events.itertuples(index=False):
            assert event.hist_movie_id == liked_movies[-2:]
            if event.rating >= 4.0:
                liked_movies.append(event.movie_id)
    assert_frame_equal(train, original)


def test_history_keeps_input_order_when_timestamps_match(source_tables):
    ratings = source_tables["ratings"]
    tied_rows = ratings[ratings.duplicated(["user_id", "timestamp"], keep=False)]
    assert not tied_rows.empty
    first = tied_rows.iloc[0]
    selected = tied_rows.loc[
        (tied_rows["user_id"] == first["user_id"])
        & (tied_rows["timestamp"] == first["timestamp"])
    ].head(2)

    events = build_history_prefixes(selected, positive_threshold=1.0)

    assert events.iloc[0]["hist_movie_id"] == []
    assert events.iloc[1]["hist_movie_id"] == [selected.iloc[0]["movie_id"]]


def test_retrieval_samples_have_prior_history_and_features(sample_tables):
    train = sample_tables["ratings"]
    samples = build_retrieval_samples(
        train, sample_tables["users"], sample_tables["movies"], max_history=2
    )
    genres_by_movie = sample_tables["movies"].set_index("movie_id")["genres"].to_dict()

    assert not samples.empty
    assert (samples["rating"] >= 4.0).all()
    assert {"gender", "age", "genres", "hist_movie_id", "hist_genres"} <= set(samples)
    for event in samples.itertuples(index=False):
        assert event.hist_movie_id
        assert event.movie_id not in event.hist_movie_id
        assert event.hist_genres == [
            genres_by_movie[movie_id].split("|")[0] for movie_id in event.hist_movie_id
        ]


def test_retrieval_samples_require_prior_positive_event(sample_tables):
    first_events = sample_tables["ratings"].groupby("user_id").head(1)

    with pytest.raises(ValueError, match="召回训练样本为空"):
        build_retrieval_samples(
            first_events, sample_tables["users"], sample_tables["movies"]
        )


def test_history_rejects_missing_fields(sample_tables):
    train = sample_tables["ratings"].drop(columns="rating")

    with pytest.raises(ValueError, match="train 缺少字段"):
        build_history_prefixes(train)


def test_retrieval_samples_use_only_training_interactions(source_tables):
    train = pd.read_pickle(SPLIT_DIR / "train.pkl")
    validation = pd.read_pickle(SPLIT_DIR / "validation.pkl")
    test = pd.read_pickle(SPLIT_DIR / "test.pkl")
    train = train.loc[train["user_id"].isin(train["user_id"].unique()[:2])]

    samples = build_retrieval_samples(
        train, source_tables["users"], source_tables["movies"]
    )
    held_out = pd.concat([validation, test], ignore_index=True)
    sample_pairs = set(zip(samples["user_id"], samples["movie_id"]))
    held_out_pairs = set(zip(held_out["user_id"], held_out["movie_id"]))

    assert sample_pairs.isdisjoint(held_out_pairs)
    assert samples["hist_movie_id"].map(len).between(1, 10).all()
