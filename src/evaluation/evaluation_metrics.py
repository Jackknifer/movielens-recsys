import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from reranking.diversity_rerank import movie_diversity_attributes
from evaluation.evaluate_retrieval import evaluate_top_k


def binary_metrics(records: pd.DataFrame, probability_scores: bool = True) -> dict:
    """AUC 与 LogLoss 使用明确评分标签；GAUC 只统计具有两种标签的用户。"""
    if records.empty:
        return {"samples": 0, "auc": None, "gauc": None, "log_loss": None, "eligible_gauc_users": 0, "eligible_gauc_samples": 0}
    labels, scores = records["label"].to_numpy(), records["ranking_score"].to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("二分类标签或预测分数无效")
    if probability_scores and ((scores < 0) | (scores > 1)).any():
        raise ValueError("预测概率必须在 0 到 1 之间")
    weighted_auc, eligible_samples, eligible_users = 0.0, 0, 0
    for _, group in records.groupby("user_id", sort=False):
        if group["label"].nunique() == 2:
            weighted_auc += len(group) * roc_auc_score(group["label"], group["ranking_score"])
            eligible_samples += len(group)
            eligible_users += 1
    return {
        "samples": len(records), "positive_samples": int((labels == 1).sum()),
        "negative_samples": int((labels == 0).sum()),
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "gauc": float(weighted_auc / eligible_samples) if eligible_samples else None,
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])) if probability_scores else None,
        "eligible_gauc_users": eligible_users, "eligible_gauc_samples": eligible_samples,
    }


def top_k_metrics(contexts: pd.DataFrame, targets: pd.DataFrame, results: pd.DataFrame, movie_ids: np.ndarray, k: int) -> dict:
    if targets.empty:
        return {"k": k, "evaluated_users": 0, "hits": 0, "recall_at_k": None, "ndcg_at_k": None, "mrr_at_k": None}
    selected_contexts = contexts.merge(targets[["user_id", "movie_id"]], on="user_id", validate="one_to_one")
    selected_results = results.loc[results["user_id"].isin(targets["user_id"]), ["user_id", "movie_id", "rank"]]
    return evaluate_top_k(selected_contexts, selected_results.rename(columns={"rank": "source_rank"}), movie_ids, k)


def diversity_metrics(
    results: pd.DataFrame, movies: pd.DataFrame, popularity: pd.DataFrame,
    movie_ids: np.ndarray, item_vectors: np.ndarray, k: int,
) -> dict:
    selected = results.loc[results["rank"] <= k]
    metadata = movie_diversity_attributes(movies).set_index("movie_id")
    movie_genres = movies.set_index("movie_id")["genres"].str.split("|", regex=False).to_dict()
    catalog_genres = {genre for item in movie_ids for genre in movie_genres[item]}
    recommended_genres = {genre for item in selected["movie_id"] for genre in movie_genres[item]}
    if not selected["movie_id"].isin(movie_ids).all():
        raise ValueError("多样性评估包含目录外电影")
    probabilities = (popularity.set_index("movie_id")["popularity"].reindex(movie_ids) + 1)
    probabilities = probabilities / probabilities.sum()
    positions = {int(item): index for index, item in enumerate(movie_ids)}
    distances, genre_runs, decade_runs = [], [], []
    for _, group in selected.groupby("user_id", sort=False):
        items = group.sort_values("rank")["movie_id"].tolist()
        if len(items) > 1:
            vectors = item_vectors[[positions[item] for item in items]]
            similarity = np.clip(vectors @ vectors.T, -1, 1)
            distances.append(float((1 - similarity)[np.triu_indices(len(items), 1)].mean()))
        for column, runs in (("primary_genre", genre_runs), ("decade", decade_runs)):
            values = metadata.loc[items, column].tolist()
            longest, current, previous = 0, 0, None
            for value in values:
                current = current + 1 if value == previous else 1
                longest, previous = max(longest, current), value
            runs.append(longest)
    return {
        "k": k, "recommendations": len(selected), "users": int(selected["user_id"].nunique()),
        "catalog_coverage": float(selected["movie_id"].nunique() / len(movie_ids)),
        "genre_coverage": float(len(recommended_genres) / len(catalog_genres)),
        "novelty": float(-np.log2(selected["movie_id"].map(probabilities)).mean()) if len(selected) else None,
        "intra_list_diversity": float(np.mean(distances)) if distances else None,
        "max_genre_run": max(genre_runs, default=0), "max_decade_run": max(decade_runs, default=0),
    }


def source_statistics(channels: dict[str, pd.DataFrame], merged: pd.DataFrame, final: pd.DataFrame, targets: pd.DataFrame) -> dict:
    proposals = pd.concat(channels.values(), ignore_index=True)
    unique = proposals[["user_id", "movie_id"]].drop_duplicates()
    channel_hits = {}
    target_by_user = targets.set_index("user_id")["movie_id"]
    for name, rows in channels.items():
        channel_hits[name] = set(rows.loc[rows["movie_id"].eq(rows["user_id"].map(target_by_user)), "user_id"])
    details = {}
    for name, rows in channels.items():
        others = set().union(*(hits for source, hits in channel_hits.items() if source != name))
        details[name] = {
            "proposed_candidates": len(rows), "users": int(rows["user_id"].nunique()),
            "target_hits": len(channel_hits[name]), "exclusive_target_hits": len(channel_hits[name] - others),
            "merged_primary_candidates": int(merged["source"].eq(name).sum()),
            "final_primary_candidates": int(final["source"].eq(name).sum()),
            "final_primary_fraction": float(final["source"].eq(name).mean()) if len(final) else None,
            "final_candidates_with_source": int(final["sources"].map(lambda sources: name in sources).sum()),
        }
    return {
        "proposals": len(proposals), "unique_candidates": len(unique),
        "duplicates": len(proposals) - len(unique),
        "duplicate_fraction": float(1 - len(unique) / len(proposals)) if len(proposals) else None,
        "merged_candidates": len(merged), "budget_excluded_unique_candidates": len(unique) - len(merged),
        "channels": details,
    }
