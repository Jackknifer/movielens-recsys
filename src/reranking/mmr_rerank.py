import numpy as np
import pandas as pd


def prepare_rerank_inputs(scores: np.ndarray, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or embeddings.ndim != 2 or len(embeddings) != len(scores) or not embeddings.shape[1]:
        raise ValueError("分数需要非空一维数组，电影向量需要具有相同行数的二维数组")
    if not np.isfinite(scores).all() or not np.isfinite(embeddings).all():
        raise ValueError("分数与电影向量必须是有限数值")
    similarity = embeddings @ embeddings.T
    return scores.copy(), similarity


def validate_rerank_inputs(relevance: np.ndarray, similarity: np.ndarray, top_k: int) -> None:
    if relevance.ndim != 1 or not len(relevance) or similarity.shape != (len(relevance), len(relevance)):
        raise ValueError("相关性与相似度矩阵的维度不匹配")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(relevance):
        raise ValueError("top_k 必须是正整数，且不能超过候选数量")
    if not np.isfinite(relevance).all() or not np.isfinite(similarity).all():
        raise ValueError("相关性与相似度矩阵必须包含有限数值")
    if not np.allclose(similarity, similarity.T, atol=1e-12, rtol=0):
        raise ValueError("相似度矩阵必须对称")


def complete_order(selected: list[int], candidate_count: int) -> np.ndarray:
    available = np.ones(candidate_count, dtype=bool)
    available[selected] = False
    return np.concatenate([np.asarray(selected, dtype=int), np.flatnonzero(available)])


def mmr_order(
    relevance: np.ndarray, similarity: np.ndarray, relevance_weight: float = 0.35,
    top_k: int = 20,
) -> np.ndarray:
    validate_rerank_inputs(relevance, similarity, top_k)
    if not np.isfinite(relevance_weight) or not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight 必须位于 0 和 1 之间")
    available = np.ones(len(relevance), dtype=bool)
    redundancy = np.full(len(relevance), -np.inf, dtype=np.float64)
    selected = []
    for position in range(top_k):
        # 空集合时，使用相关性最高的电影作为第一部电影。
        objective = relevance if position == 0 else relevance_weight * relevance - (1 - relevance_weight) * redundancy
        index = int(np.argmax(np.where(available, objective, -np.inf)))
        selected.append(index)
        available[index] = False
        redundancy = np.maximum(redundancy, similarity[index])
    return complete_order(selected, len(relevance))


def rerank_candidates(
    scored: pd.DataFrame, movie_ids: np.ndarray, item_vectors: np.ndarray,
    top_k: int = 20, relevance_weight: float = 0.35,
) -> pd.DataFrame:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k 必须是正整数")
    if isinstance(relevance_weight, bool) or not np.isfinite(relevance_weight) or not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight 必须位于 0 和 1 之间")
    if movie_ids.ndim != 1 or not len(movie_ids) or len(np.unique(movie_ids)) != len(movie_ids):
        raise ValueError("电影目录必须是一维数组，且包含不同的电影编号")
    if item_vectors.ndim != 2 or len(item_vectors) != len(movie_ids) or not item_vectors.shape[1] or not np.isfinite(item_vectors).all():
        raise ValueError("电影向量必须具有正确的维度，且包含有限数值")
    required = ["user_id", "movie_id", "rank", "ranking_score"]
    if not scored.index.is_unique or scored.duplicated(["user_id", "movie_id"]).any() or scored[required].isna().any().any():
        raise ValueError("重排候选包含重复编号或空值")
    if not np.isfinite(scored["ranking_score"].to_numpy(dtype=float)).all():
        raise ValueError("重排分数必须包含有限数值")
    if not scored["movie_id"].isin(movie_ids).all():
        raise ValueError("重排候选缺少电影向量")
    if scored.empty:
        return scored.copy()
    positions = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    groups = []
    for _, group in scored.groupby("user_id", sort=False):
        ranks = group["rank"].to_numpy()
        if not np.issubdtype(ranks.dtype, np.integer) or not np.array_equal(np.sort(ranks), np.arange(1, len(group) + 1)):
            raise ValueError("重排输入的 rank 必须从 1 开始连续编号")
        group = group.sort_values("rank")
        embeddings = item_vectors[positions.loc[group["movie_id"]].to_numpy()]
        relevance, similarity = prepare_rerank_inputs(group["ranking_score"].to_numpy(), embeddings)
        order = mmr_order(relevance, similarity, relevance_weight, min(top_k, len(group)))
        ordered = group.iloc[order].copy()
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        groups.append(ordered)
    return pd.concat(groups, ignore_index=True)
