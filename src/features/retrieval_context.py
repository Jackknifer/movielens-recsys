import numpy as np
import pandas as pd

from features.feature_join import join_features
from features.user_history import REQUIRED_COLUMNS


def build_retrieval_contexts(
    history: pd.DataFrame,
    heldout: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    max_history: int = 10,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """为每个保留目标构造用户历史，并保存需要排除的全部已看电影。"""
    if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
        raise ValueError("max_history 必须是正整数")
    if not np.isfinite(positive_threshold) or not 1 <= positive_threshold <= 5:
        raise ValueError("positive_threshold 必须在 1 到 5 之间")
    for name, table in (("history", history), ("heldout", heldout)):
        missing = REQUIRED_COLUMNS - set(table.columns)
        if missing:
            raise ValueError(f"{name} 缺少字段：{sorted(missing)}")
        if table.empty or table[list(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(f"{name} 不能为空，必要字段不能存在空值")
        for column in ("user_id", "movie_id", "timestamp"):
            if not pd.api.types.is_integer_dtype(table[column]):
                raise TypeError(f"{name}.{column} 必须使用整数")
        if not pd.api.types.is_numeric_dtype(table["rating"]):
            raise TypeError(f"{name}.rating 必须使用数值")
        if not table["rating"].between(1, 5).all():
            raise ValueError(f"{name}.rating 必须在 1 到 5 之间")
    if heldout["user_id"].duplicated().any():
        raise ValueError("heldout 中每个用户只能有一个目标")
    if not heldout["rating"].ge(positive_threshold).all():
        raise ValueError("heldout 只能包含达到 positive_threshold 的目标")

    # 这里只读取被评估用户的历史，不把 heldout 加入历史表。
    selected = history.loc[history["user_id"].isin(heldout["user_id"])].copy()
    if not heldout["user_id"].isin(selected["user_id"]).all():
        raise ValueError("被评估用户缺少历史，需要使用独立的冷启动流程")
    target_times = heldout.set_index("user_id")["timestamp"]
    if selected["timestamp"].gt(selected["user_id"].map(target_times)).any():
        raise ValueError("history 包含晚于评估目标的交互")

    # 同一时间戳使用输入位置确定顺序，与项目的数据划分规则一致。
    selected["_context_event_order"] = np.arange(len(selected))
    selected = selected.sort_values(["user_id", "timestamp", "_context_event_order"])
    history_by_user = {}
    seen_by_user = {}
    lengths_by_user = {}
    for user_id, events in selected.groupby("user_id", sort=False):
        liked = events.loc[events["rating"] >= positive_threshold, "movie_id"].tolist()
        if not liked:
            raise ValueError(f"user_id={user_id} 缺少喜欢的电影历史")
        history_by_user[user_id] = liked[-max_history:]
        lengths_by_user[user_id] = len(liked)
        # 低评分电影也属于已看电影，召回时同样需要排除。
        seen_by_user[user_id] = sorted(set(events["movie_id"]))

    contexts = join_features(heldout.reset_index(drop=True), users, movies)
    contexts["hist_movie_id"] = contexts["user_id"].map(history_by_user)
    contexts["seen_movie_ids"] = contexts["user_id"].map(seen_by_user)
    contexts["history_length"] = contexts["user_id"].map(lengths_by_user)
    if any(
        row.movie_id in row.seen_movie_ids
        for row in contexts.itertuples(index=False)
    ):
        raise ValueError("评估目标已经出现在历史中，请检查数据划分")
    genres_by_movie = movies.set_index("movie_id")["genres"].to_dict()
    contexts["hist_genres"] = contexts["hist_movie_id"].map(
        lambda movie_ids: [genres_by_movie[movie_id].split("|")[0] for movie_id in movie_ids]
    )
    contexts["as_of_timestamp"] = contexts["timestamp"]
    return contexts
