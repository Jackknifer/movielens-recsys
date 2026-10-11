import numpy as np
import pandas as pd

from retrieval.popular_recall import _validate_movie_ids
from features.feature_join import USER_FEATURE_COLUMNS


def build_user_contexts(
    queries: pd.DataFrame, history: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame,
    max_history: int = 10, positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """根据用户编号和推荐时间构造历史，允许用户没有交互记录。"""
    if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
        raise ValueError("max_history 必须是正整数")
    if not np.isfinite(positive_threshold) or not 1 <= positive_threshold <= 5:
        raise ValueError("positive_threshold 必须在 1 到 5 之间")
    if set(queries.columns) != {"user_id", "as_of_timestamp"} or queries.empty:
        raise ValueError("queries 只需要非空的 user_id 和 as_of_timestamp")
    _validate_movie_ids(queries["user_id"].to_numpy())
    required = {"user_id", "movie_id", "rating", "timestamp"}
    if not required.issubset(history.columns) or history[list(required)].isna().any().any():
        raise ValueError("history 缺少必要字段或存在空值")
    for name, table in (("queries", queries), ("history", history)):
        column = "as_of_timestamp" if name == "queries" else "timestamp"
        if not pd.api.types.is_integer_dtype(table[column]) or table[column].isna().any() or (table[column] < 0).any():
            raise ValueError(f"{name}.{column} 必须是非负整数")
    if not history["rating"].between(1, 5).all() or history.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("history 的评分范围或用户电影唯一性无效")
    for column in ("user_id", "movie_id"):
        values = history[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or (values < 1).any():
            raise ValueError(f"history.{column} 必须是正整数")
    _validate_movie_ids(users["user_id"].to_numpy())
    _validate_movie_ids(movies["movie_id"].to_numpy())
    if users[USER_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("用户属性存在空值")
    if not queries["user_id"].isin(users["user_id"]).all():
        raise ValueError("queries 中的用户缺少属性记录")
    if not history["user_id"].isin(users["user_id"]).all() or not history["movie_id"].isin(movies["movie_id"]).all():
        raise ValueError("history 包含属性表以外的编号")
    if not movies["genres"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("电影类型必须是非空字符串")

    contexts = queries.merge(users[USER_FEATURE_COLUMNS], on="user_id", validate="one_to_one", sort=False)
    times = queries.set_index("user_id")["as_of_timestamp"]
    selected = history.loc[history["user_id"].isin(times.index)].copy()
    selected = selected.loc[selected["timestamp"] <= selected["user_id"].map(times)]
    selected = selected.assign(event_position=np.arange(len(selected))).sort_values(
        ["user_id", "timestamp", "event_position"], kind="stable",
    )
    groups = {user_id: group for user_id, group in selected.groupby("user_id", sort=False)}
    genre_map = movies.set_index("movie_id")["genres"].to_dict()
    histories, seen, lengths, genres = [], [], [], []
    for user_id in contexts["user_id"]:
        events = groups.get(user_id)
        liked = [] if events is None else events.loc[events["rating"] >= positive_threshold, "movie_id"].tolist()
        recent = liked[-max_history:]
        histories.append(recent)
        seen.append([] if events is None else sorted(set(events["movie_id"])))
        lengths.append(len(liked))
        genres.append([genre_map[item].split("|")[0] for item in recent])
    contexts["hist_movie_id"], contexts["seen_movie_ids"] = histories, seen
    contexts["history_length"], contexts["hist_genres"] = lengths, genres
    return contexts
