import pandas as pd
from pandas.testing import assert_frame_equal

from data.split import split_by_user_time


def test_split_expected_holdouts(sample_tables):
    ratings = sample_tables["ratings"].copy()
    original = ratings.copy(deep=True)
    result = split_by_user_time(ratings)
    assert len(result.train) == 6
    assert len(result.validation) == 2
    assert len(result.test) == 2
    assert result.cold_start.empty
    for user_id, group in ratings.groupby("user_id"):
        ordered = group.sort_values("timestamp", kind="stable")
        for name, expected in (
            ("train", ordered.iloc[:-2]),
            ("validation", ordered.iloc[[-2]]),
            ("test", ordered.iloc[[-1]]),
        ):
            actual = getattr(result, name)
            assert_frame_equal(
                actual[actual.user_id == user_id].reset_index(drop=True),
                expected.reset_index(drop=True),
            )
    assert_frame_equal(ratings, original)


def test_split_short_history(sample_tables):
    # 从真实评分中截取一个用户的两条历史，检查短历史分支。
    ratings = sample_tables["ratings"]
    first_user, second_user = ratings.user_id.unique()
    short_history = ratings[ratings.user_id == second_user].head(2)
    selected = pd.concat([
        ratings[ratings.user_id == first_user], short_history
    ], ignore_index=True)
    result = split_by_user_time(selected)
    assert_frame_equal(
        result.cold_start,
        short_history.sort_values("timestamp", kind="stable").reset_index(drop=True),
    )
    assert set(result.train.user_id) == {first_user}
    assert set(result.validation.user_id) == {first_user}
    assert set(result.test.user_id) == {first_user}
    assert sum(len(getattr(result, name)) for name in (
        "train", "validation", "test", "cold_start"
    )) == len(selected)


def test_split_shuffled_and_tied_timestamps(source_tables):
    # 选择真实数据中具有相同时间戳的一个用户，保持原始时间值。
    ratings = source_tables["ratings"]
    tied_rows = ratings[ratings.duplicated(["user_id", "timestamp"], keep=False)]
    assert not tied_rows.empty
    user_id = tied_rows.iloc[0].user_id
    selected = ratings[ratings.user_id == user_id].sample(frac=1, random_state=42)
    expected = selected.sort_values("timestamp", kind="stable").reset_index(drop=True)
    result = split_by_user_time(selected)
    assert_frame_equal(result.train, expected.iloc[:-2].reset_index(drop=True))
    assert_frame_equal(result.validation, expected.iloc[[-2]].reset_index(drop=True))
    assert_frame_equal(result.test, expected.iloc[[-1]].reset_index(drop=True))
    combined = pd.concat([result.train, result.validation, result.test], ignore_index=True)
    assert_frame_equal(combined, expected)
    assert not combined.duplicated(["user_id", "movie_id"]).any()
