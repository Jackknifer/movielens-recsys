import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data.load import MovieLensData, load_movielens


def test_load_valid_data(sample_tables, sample_dir):
    # 将真实 ID 转为文本，验证 loader 能恢复整数类型和原有数值。
    ratings = sample_tables["ratings"].copy()
    ratings[["user_id", "movie_id", "timestamp"]] = ratings[
        ["user_id", "movie_id", "timestamp"]
    ].astype(str)
    ratings.to_pickle(sample_dir / "ratings.pkl")

    result = load_movielens(sample_dir)

    assert isinstance(result, MovieLensData)
    for name in ("users", "movies", "ratings"):
        expected = sample_tables[name].copy()
        for column in ("user_id", "movie_id", "timestamp"):
            if column in expected:
                expected[column] = expected[column].astype("int64")
        assert_frame_equal(getattr(result, name), expected)


def test_load_rejects_missing_column(sample_tables, sample_dir):
    # 只删除评分字段，检查异常是否准确指向该字段。
    sample_tables["ratings"].drop(columns="rating").to_pickle(
        sample_dir / "ratings.pkl"
    )
    with pytest.raises(ValueError, match="ratings 缺少字段.*rating"):
        load_movielens(sample_dir)


def test_load_rejects_duplicate_user(sample_tables, sample_dir):
    # 重复保存一条真实用户记录，验证主键唯一性检查。
    users = sample_tables["users"]
    pd.concat([users, users.iloc[[0]]], ignore_index=True).to_pickle(
        sample_dir / "users.pkl"
    )
    with pytest.raises(ValueError, match="users.user_id 存在重复值"):
        load_movielens(sample_dir)
