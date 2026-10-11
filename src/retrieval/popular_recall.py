import numpy as np
import pandas as pd

from retrieval.vector_recall import RESULT_COLUMNS


def _validate_movie_ids(movie_ids: np.ndarray) -> None:
    if movie_ids.ndim != 1 or not len(movie_ids):
        raise ValueError("movie_ids 必须是非空的一维数组")
    if not np.issubdtype(movie_ids.dtype, np.integer) or (movie_ids < 1).any():
        raise ValueError("movie_ids 必须包含正整数编号")
    if len(np.unique(movie_ids)) != len(movie_ids):
        raise ValueError("movie_ids 不能重复")


def _validate_contexts(contexts: pd.DataFrame, k: int, require_history: bool = False) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k 必须是正整数")
    required = {"user_id", "seen_movie_ids"}
    if require_history:
        required.add("hist_movie_id")
    if not required.issubset(contexts.columns):
        raise ValueError(f"contexts 缺少字段：{sorted(required - set(contexts.columns))}")
    if contexts.empty or contexts["user_id"].duplicated().any():
        raise ValueError("contexts 不能为空，每个用户只能出现一次")
    _validate_movie_ids(contexts["user_id"].to_numpy())
    for context in contexts.itertuples(index=False):
        columns = ("seen_movie_ids", "hist_movie_id") if require_history else ("seen_movie_ids",)
        for column in columns:
            values = getattr(context, column)
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"{column} 必须保存为列表或 tuple")
            if values:
                _validate_movie_ids(np.asarray(values))
        if require_history and not set(context.hist_movie_id).issubset(context.seen_movie_ids):
            raise ValueError("hist_movie_id 必须属于 seen_movie_ids")


def _rank_candidates(
    user_id: int, scores: pd.DataFrame, seen_movie_ids: list[int], k: int, source: str,
) -> pd.DataFrame:
    available = scores.loc[~scores["movie_id"].isin(seen_movie_ids)]
    selected = available.sort_values(
        ["source_score", "movie_id"], ascending=[False, True], kind="stable",
    ).head(k).copy()
    selected["user_id"] = user_id
    selected["source"] = source
    selected["source_rank"] = np.arange(1, len(selected) + 1)
    return selected[RESULT_COLUMNS]


def build_popularity_index(
    train: pd.DataFrame, movie_ids: np.ndarray, positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """只统计传入训练集中的高评分次数，候选目录由调用方指定。"""
    _validate_movie_ids(movie_ids)
    if not np.isfinite(positive_threshold) or not 1 <= positive_threshold <= 5:
        raise ValueError("positive_threshold 必须在 1 到 5 之间")
    required = {"user_id", "movie_id", "rating"}
    if train.empty or not required.issubset(train.columns):
        raise ValueError("train 必须包含 user_id、movie_id 和 rating，且不能为空")
    if train[list(required)].isna().any().any():
        raise ValueError("train 的必要字段存在空值")
    for column in ("user_id", "movie_id"):
        values = train[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or (values < 1).any():
            raise ValueError(f"train.{column} 必须包含正整数编号")
    ratings = train["rating"].to_numpy()
    if not np.issubdtype(ratings.dtype, np.number) or not np.isfinite(ratings).all():
        raise ValueError("train.rating 必须包含有限数值")
    if not train["rating"].between(1, 5).all():
        raise ValueError("train.rating 必须在 1 到 5 之间")
    if train.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("train 存在重复的用户电影组合")

    positives = train.loc[train["rating"] >= positive_threshold]
    counts = positives.groupby("movie_id").size().reindex(movie_ids, fill_value=0)
    if counts.sum() == 0:
        raise ValueError("候选目录内没有高评分训练交互")
    index = pd.DataFrame({"movie_id": movie_ids, "popularity": counts.to_numpy()})
    return index.sort_values(["popularity", "movie_id"], ascending=[False, True]).reset_index(drop=True)


def recall_popular(contexts: pd.DataFrame, index: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """按高评分次数推荐电影，并排除全部已看电影。"""
    _validate_contexts(contexts, k)
    _validate_movie_ids(index["movie_id"].to_numpy())
    popularity = index["popularity"].to_numpy()
    if not np.issubdtype(popularity.dtype, np.integer) or (popularity < 0).any():
        raise ValueError("popularity 必须是非负整数")
    # 没有高评分记录的电影保留在目录中，其热门得分为 0。
    scores = index.rename(columns={"popularity": "source_score"})
    results = [
        _rank_candidates(row.user_id, scores, row.seen_movie_ids, k, "popular")
        for row in contexts.itertuples(index=False)
    ]
    return pd.concat(results, ignore_index=True)
