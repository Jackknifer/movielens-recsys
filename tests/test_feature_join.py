from pathlib import Path

import pandas as pd
import pytest
from pandas.errors import MergeError
from pandas.testing import assert_frame_equal

from features.feature_join import join_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_join_features_preserves_records_and_adds_metadata(sample_tables):
    records = sample_tables["ratings"]
    original = records.copy(deep=True)

    result = join_features(records, sample_tables["users"], sample_tables["movies"])

    assert len(result) == len(records)
    assert_frame_equal(result[records.columns].reset_index(drop=True), original.reset_index(drop=True))
    assert_frame_equal(records, original)
    assert {"gender", "age", "occupation", "zip_code", "genres", "isAdult", "startYear"} <= set(result)
    assert "averageRating" not in result
    assert "numVotes" not in result
    first = result.iloc[0]
    user = sample_tables["users"].set_index("user_id").loc[first["user_id"]]
    movie = sample_tables["movies"].set_index("movie_id").loc[first["movie_id"]]
    assert first["gender"] == user["gender"]
    assert first["genres"] == movie["genres"]


@pytest.mark.parametrize("table_name, id_column", [("users", "user_id"), ("movies", "movie_id")])
def test_join_features_rejects_missing_reference(sample_tables, table_name, id_column):
    tables = {name: table.copy() for name, table in sample_tables.items()}
    missing_id = tables[table_name][id_column].iloc[0]
    tables[table_name] = tables[table_name].loc[tables[table_name][id_column] != missing_id]

    with pytest.raises(ValueError, match=f"不存在的 {id_column}"):
        join_features(tables["ratings"], tables["users"], tables["movies"])


def test_join_features_rejects_duplicate_movie_id(sample_tables):
    movies = sample_tables["movies"]
    duplicate_movies = pd.concat([movies, movies.iloc[[0]]], ignore_index=True)

    with pytest.raises(MergeError, match="not a many-to-one merge"):
        join_features(sample_tables["ratings"], sample_tables["users"], duplicate_movies)


def test_join_features_on_full_train(source_tables):
    train = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "split" / "train.pkl")

    result = join_features(train, source_tables["users"], source_tables["movies"])

    assert len(result) == len(train)
    assert result[["user_id", "movie_id", "rating", "timestamp"]].reset_index(drop=True).equals(
        train[["user_id", "movie_id", "rating", "timestamp"]].reset_index(drop=True)
    )
    assert result["gender"].notna().all()
    assert result["genres"].notna().all()
