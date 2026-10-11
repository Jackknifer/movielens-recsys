import numpy as np
import pandas as pd

from retrieval.genre_recall import GenreIndex, recall_genres
from retrieval.item_similarity_recall import recall_item_similarity
from retrieval.popular_recall import _validate_contexts, recall_popular
from retrieval.vector_recall import RESULT_COLUMNS, RetrievalArtifacts, recall_users


def classify_users(contexts: pd.DataFrame, known_users: np.ndarray, min_history: int = 5) -> pd.Series:
    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 1:
        raise ValueError("min_history 必须是正整数")
    _validate_contexts(contexts, 1, require_history=True)
    lengths = contexts["history_length"]
    if not pd.api.types.is_integer_dtype(lengths) or lengths.isna().any() or (lengths < 0).any():
        raise ValueError("history_length 必须是非负整数")
    if (lengths < contexts["hist_movie_id"].map(len)).any():
        raise ValueError("history_length 不能小于当前历史长度")
    if ((lengths == 0) != contexts["hist_movie_id"].map(lambda values: not values)).any():
        raise ValueError("history_length 与电影历史是否为空必须一致")
    groups = pd.Series("short_history", index=contexts.index, name="user_group")
    groups.loc[lengths == 0] = "no_history"
    warm = (lengths >= min_history) & contexts["user_id"].isin(known_users)
    groups.loc[warm] = "warm"
    groups.loc[(lengths > 0) & ~contexts["user_id"].isin(known_users)] = "unseen_user"
    return groups


def dispatch_recall(
    contexts: pd.DataFrame, artifacts: RetrievalArtifacts, popularity: pd.DataFrame,
    genres: GenreIndex, similarity: pd.DataFrame, k: int = 100,
    min_history: int = 5, recent_items: int = 3, batch_size: int = 128,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """按历史数量和模型用户词表选择召回通道。"""
    _validate_contexts(contexts, k, require_history=True)
    if k > len(artifacts.movie_ids):
        raise ValueError("k 不能超过候选电影数量")
    known_users = artifacts.encoders["user_id"].categories_[0].astype(np.int64)
    contexts = contexts.copy()
    contexts["user_group"] = classify_users(contexts, known_users, min_history)
    contexts["is_cold_start"] = contexts["user_group"] != "warm"
    warm = contexts.loc[contexts["user_group"] == "warm"]
    with_history = contexts.loc[contexts["history_length"] > 0]
    results = {name: pd.DataFrame(columns=RESULT_COLUMNS) for name in (
        "youtubednn", "item_similarity", "genre_interest", "popular",
    )}
    results["popular"] = recall_popular(contexts, popularity, k)
    if not with_history.empty:
        results["genre_interest"] = recall_genres(with_history, genres, k)
    if not warm.empty:
        results["youtubednn"] = recall_users(warm, artifacts, k, batch_size)
        results["item_similarity"] = recall_item_similarity(warm, similarity, k, recent_items)
    return contexts, results
