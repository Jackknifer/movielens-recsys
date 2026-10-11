import pandas as pd


USER_FEATURE_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]
MOVIE_FEATURE_COLUMNS = ["movie_id", "genres", "isAdult", "startYear"]


def join_features(
    records: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """根据 user_id 和 movie_id，为每条记录连接用户、电影特征。"""
    required_columns = (
        ("records", records, ["user_id", "movie_id"]),
        ("users", users, USER_FEATURE_COLUMNS),
        ("movies", movies, MOVIE_FEATURE_COLUMNS),
    )
    for name, table, columns in required_columns:
        missing = set(columns) - set(table.columns)
        if missing:
            raise ValueError(f"{name} 缺少字段：{sorted(missing)}")

    feature_columns = set(USER_FEATURE_COLUMNS[1:] + MOVIE_FEATURE_COLUMNS[1:])
    if feature_columns.intersection(records.columns):
        raise ValueError("records 已包含待连接的特征字段")

    # 每条记录都应当能在两张属性表中找到对应编号。
    if not records["user_id"].isin(users["user_id"]).all():
        raise ValueError("records 包含 users 中不存在的 user_id")
    if not records["movie_id"].isin(movies["movie_id"]).all():
        raise ValueError("records 包含 movies 中不存在的 movie_id")

    # 左连接保留每条记录及其原有字段；右表编号必须唯一，避免记录数量增加。
    joined = records.merge(
        users[USER_FEATURE_COLUMNS],
        on="user_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    joined = joined.merge(
        movies[MOVIE_FEATURE_COLUMNS],
        on="movie_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    return joined
