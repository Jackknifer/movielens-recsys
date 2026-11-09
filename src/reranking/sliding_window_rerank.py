import numpy as np
import pandas as pd

from features.movie_attributes import movie_diversity_attributes


def validate_window_parameters(window_size: int, max_genre_count: int, max_decade_count: int, top_k: int) -> None:
    for name, value in (("window_size", window_size), ("max_genre_count", max_genre_count),
                        ("max_decade_count", max_decade_count), ("top_k", top_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")
    if max_genre_count > window_size or max_decade_count > window_size:
        raise ValueError("类型和年代的数量上限不能超过窗口长度")


def sliding_window_order(
    genres: np.ndarray, decades: np.ndarray, window_size: int = 7,
    max_genre_count: int = 4, max_decade_count: int = 4, top_k: int = 20,
) -> tuple[np.ndarray, dict]:
    validate_window_parameters(window_size, max_genre_count, max_decade_count, top_k)
    genres, decades = np.asarray(genres), np.asarray(decades)
    if genres.ndim != 1 or decades.shape != genres.shape:
        raise ValueError("类型和年代需要具有相同长度的一维数组")
    for values in (genres, decades):
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
            raise ValueError("类型和年代编码必须是非负整数")
    order = np.arange(len(genres))
    events = {"swap_positions": [], "unchanged_violation_positions": []}
    if not len(order):
        return order, events
    genre_counts = np.zeros(int(genres.max()) + 1, dtype=int)
    decade_counts = np.zeros(int(decades.max()) + 1, dtype=int)
    for position in range(min(top_k, len(order))):
        remaining = order[position:]
        allowed = (genre_counts[genres[remaining]] < max_genre_count) & (decade_counts[decades[remaining]] < max_decade_count)
        allowed &= (genre_counts.max() <= max_genre_count) & (decade_counts.max() <= max_decade_count)
        if not allowed[0]:
            replacements = np.flatnonzero(allowed)
            if len(replacements):
                replacement = position + int(replacements[0])
                order[position], order[replacement] = order[replacement], order[position]
                events["swap_positions"].append(position + 1)
            else:
                events["unchanged_violation_positions"].append(position + 1)
        selected = order[position]
        genre_counts[genres[selected]] += 1
        decade_counts[decades[selected]] += 1
        # 下一位置只保留最近 window_size - 1 部电影的计数。
        if position + 1 >= window_size:
            expired = order[position + 1 - window_size]
            genre_counts[genres[expired]] -= 1
            decade_counts[decades[expired]] -= 1
    return order, events


def rerank_candidates(
    scored: pd.DataFrame, movies: pd.DataFrame, top_k: int = 20,
    window_size: int = 7, max_genre_count: int = 4, max_decade_count: int = 4,
) -> pd.DataFrame:
    validate_window_parameters(window_size, max_genre_count, max_decade_count, top_k)
    required = ["user_id", "movie_id", "rank", "ranking_score"]
    if not scored.index.is_unique or scored.duplicated(["user_id", "movie_id"]).any() or scored[required].isna().any().any():
        raise ValueError("重排候选包含重复编号或空值")
    if not np.isfinite(scored["ranking_score"].to_numpy(dtype=float)).all():
        raise ValueError("重排分数必须包含有限数值")
    metadata = movie_diversity_attributes(movies).set_index("movie_id")
    if not scored["movie_id"].isin(metadata.index).all():
        raise ValueError("重排候选缺少电影属性")
    if scored.empty:
        return scored.copy()
    for column in ("primary_genre", "decade"):
        metadata[column] = pd.factorize(metadata[column], sort=False)[0]
    groups = []
    for _, group in scored.groupby("user_id", sort=False):
        ranks = group["rank"].to_numpy()
        if not np.issubdtype(ranks.dtype, np.integer) or not np.array_equal(np.sort(ranks), np.arange(1, len(group) + 1)):
            raise ValueError("重排输入的 rank 必须从 1 开始连续编号")
        group = group.sort_values("rank")
        attributes = metadata.loc[group["movie_id"]]
        order, _ = sliding_window_order(attributes["primary_genre"].to_numpy(), attributes["decade"].to_numpy(),
                                        window_size, max_genre_count, max_decade_count, top_k)
        ordered = group.iloc[order].copy()
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        groups.append(ordered)
    return pd.concat(groups, ignore_index=True)
