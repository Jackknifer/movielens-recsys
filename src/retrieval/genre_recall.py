from dataclasses import dataclass

import numpy as np
import pandas as pd

from retrieval.popular_recall import (
    _rank_candidates, _validate_contexts, _validate_movie_ids, build_popularity_index,
)
from retrieval.vector_recall import RESULT_COLUMNS


@dataclass(frozen=True)
class GenreIndex:
    candidates: pd.DataFrame
    genres_by_movie: dict[int, list[str]]


def build_genre_index(
    train: pd.DataFrame, movies: pd.DataFrame, movie_ids: np.ndarray,
    positive_threshold: float = 4.0,
) -> GenreIndex:
    """在每种电影类型内部，把高评分次数转换为概率。"""
    popularity = build_popularity_index(train, movie_ids, positive_threshold)
    _validate_movie_ids(movies["movie_id"].to_numpy())
    if not movies["genres"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("movies.genres 必须包含非空字符串")
    genres = movies.set_index("movie_id")["genres"].str.split("|", regex=False)
    if genres.map(lambda values: any(not value for value in values) or len(set(values)) != len(values)).any():
        raise ValueError("每部电影的 genres 不能包含空类型或重复类型")
    if not set(movie_ids).issubset(genres.index):
        raise ValueError("movies 缺少候选电影的类型")

    candidates = popularity.copy()
    candidates["genre"] = candidates["movie_id"].map(genres)
    candidates = candidates.explode("genre", ignore_index=True)
    totals = candidates.groupby("genre")["popularity"].transform("sum")
    candidates = candidates.loc[totals > 0].copy()
    candidates["genre_probability"] = candidates["popularity"] / totals.loc[candidates.index]
    return GenreIndex(candidates, genres.to_dict())


def user_genre_weights(history: list[int], genres_by_movie: dict[int, list[str]]) -> pd.Series:
    """每部历史电影贡献相同总权重，多个类型平分这份权重。"""
    if not history:
        return pd.Series(dtype=np.float64)
    weights: dict[str, float] = {}
    for movie_id in history:
        movie_genres = genres_by_movie[movie_id]
        for genre in movie_genres:
            weights[genre] = weights.get(genre, 0.0) + 1 / len(movie_genres)
    return pd.Series(weights, dtype=np.float64) / len(history)


def recall_genres(contexts: pd.DataFrame, index: GenreIndex, k: int = 20) -> pd.DataFrame:
    """根据最近高评分历史的类型权重，计算类型热门电影的得分。"""
    _validate_contexts(contexts, k, require_history=True)
    results = []
    for context in contexts.itertuples(index=False):
        weights = user_genre_weights(context.hist_movie_id, index.genres_by_movie)
        candidates = index.candidates.copy()
        candidates["source_score"] = (
            candidates["genre"].map(weights).fillna(0) * candidates["genre_probability"]
        )
        # 同一部电影可以匹配多个类型，先求和再确定排名。
        scores = candidates.groupby("movie_id", as_index=False)["source_score"].sum()
        scores = scores.loc[scores["source_score"] > 0]
        results.append(_rank_candidates(
            context.user_id, scores, context.seen_movie_ids, k, "genre_interest",
        ))
    return pd.concat(results, ignore_index=True)[RESULT_COLUMNS]
