import numpy as np
import pandas as pd

from retrieval.genre_recall import user_genre_weights


CHANNEL_NAMES = ("youtubednn", "item_similarity", "genre_interest", "popular")
RANKING_NUMERIC_COLUMNS = [
    f"{channel}_{kind}" for channel in CHANNEL_NAMES for kind in ("present", "score", "reciprocal_rank")
] + [
    "source_count", "merge_reciprocal_rank", "log_history_length", "genre_affinity",
    "recent_mean_similarity", "recent_max_similarity", "last_movie_similarity",
    "log_popularity", "retrieval_known_item", "history_vector_coverage",
]
FEATURE_VERSION = "candidate_context_v1"


def with_recall_metadata(records: pd.DataFrame, merged: pd.DataFrame, channels: dict) -> pd.DataFrame:
    keys = ["user_id", "movie_id"]
    if set(channels) != set(CHANNEL_NAMES) or records.duplicated(keys).any():
        raise ValueError("召回通道或评分记录的唯一性无效")
    result = records.reset_index(drop=True).copy()
    index = pd.MultiIndex.from_frame(result[keys])
    ranks = merged.set_index(keys)["candidate_rank"].reindex(index)
    result["candidate_rank"] = ranks.fillna(0).to_numpy(dtype=np.int64)
    scores, positions = [{} for _ in range(len(result))], [{} for _ in range(len(result))]
    for channel in CHANNEL_NAMES:
        rows = channels[channel].set_index(keys)[["source_score", "source_rank"]].reindex(index)
        for position in np.flatnonzero(rows["source_rank"].notna().to_numpy()):
            scores[position][channel] = float(rows.iloc[position]["source_score"])
            positions[position][channel] = int(rows.iloc[position]["source_rank"])
    result["source_scores"], result["source_ranks"] = scores, positions
    return result


def build_context_features(records: pd.DataFrame, contexts: pd.DataFrame, movies: pd.DataFrame, retrieval, popularity: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "movie_id", "candidate_rank", "source_scores", "source_ranks"}
    if records.empty or not required.issubset(records.columns):
        raise ValueError("上下文排序特征需要非空候选和召回来源字段")
    if contexts["user_id"].duplicated().any() or not records["user_id"].isin(contexts["user_id"]).all():
        raise ValueError("每个候选用户需要唯一的上下文")
    result = records.reset_index(drop=True).copy()
    values = np.zeros((len(result), len(RANKING_NUMERIC_COLUMNS)), dtype=np.float32)
    columns = {name: position for position, name in enumerate(RANKING_NUMERIC_COLUMNS)}
    for position, row in enumerate(result.itertuples(index=False)):
        scores, ranks = row.source_scores, row.source_ranks
        if not isinstance(scores, dict) or not isinstance(ranks, dict) or set(scores) != set(ranks) or not set(scores).issubset(CHANNEL_NAMES):
            raise ValueError("召回分数和排名必须使用相同的通道字段")
        for channel in CHANNEL_NAMES:
            if channel in scores:
                rank, score = ranks[channel], scores[channel]
                if rank < 1 or not np.isfinite(score) or (channel == "popular" and score < 0):
                    raise ValueError("召回分数或排名无效")
                values[position, columns[f"{channel}_present"]] = 1
                values[position, columns[f"{channel}_score"]] = np.log1p(score) if channel == "popular" else score
                values[position, columns[f"{channel}_reciprocal_rank"]] = 1 / rank
        if row.candidate_rank < 0:
            raise ValueError("candidate_rank 不能为负数")
        values[position, columns["source_count"]] = len(scores)
        values[position, columns["merge_reciprocal_rank"]] = 0 if row.candidate_rank == 0 else 1 / row.candidate_rank
    catalog = pd.Index(retrieval.movie_ids)
    genres = movies.set_index("movie_id")["genres"].str.split("|", regex=False).to_dict()
    popular = popularity.set_index("movie_id")["popularity"]
    context_index = contexts.set_index("user_id")
    for user_id, group in result.groupby("user_id", sort=False):
        positions = group.index.to_numpy()
        context = context_index.loc[user_id]
        history = context["hist_movie_id"]
        weights = user_genre_weights(history, genres)
        values[positions, columns["log_history_length"]] = np.log1p(context["history_length"])
        values[positions, columns["genre_affinity"]] = [sum(weights.get(genre, 0) for genre in genres[item]) / len(genres[item]) for item in group["movie_id"]]
        item_positions = catalog.get_indexer(group["movie_id"])
        known = item_positions >= 0
        values[positions, columns["retrieval_known_item"]] = known
        # 目录外电影的向量特征为 0，并通过独立字段记录是否具有向量。
        candidate_vectors = np.zeros((len(group), retrieval.item_vectors.shape[1]), dtype=np.float32)
        candidate_vectors[known] = retrieval.item_vectors[item_positions[known]]
        history_positions = catalog.get_indexer(history)
        available = history_positions >= 0
        values[positions, columns["history_vector_coverage"]] = available.mean() if history else 0
        if available.any():
            similarities = candidate_vectors @ retrieval.item_vectors[history_positions[available]].T
            values[positions, columns["recent_mean_similarity"]] = similarities.mean(axis=1)
            values[positions, columns["recent_max_similarity"]] = similarities.max(axis=1)
        if history and history_positions[-1] >= 0:
            values[positions, columns["last_movie_similarity"]] = candidate_vectors @ retrieval.item_vectors[history_positions[-1]]
        values[positions, columns["log_popularity"]] = np.log1p(popular.reindex(group["movie_id"], fill_value=0).to_numpy(dtype=float))
    if not np.isfinite(values).all():
        raise ValueError("上下文排序特征存在无效数值")
    result[RANKING_NUMERIC_COLUMNS] = values
    return result
