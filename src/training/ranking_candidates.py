from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval.cold_start import dispatch_recall
from retrieval.genre_recall import build_genre_index
from retrieval.item_similarity_recall import build_item_similarity_index
from retrieval.merge_recall import snake_merge
from retrieval.popular_recall import build_popularity_index
from features.ranking_features import build_context_features
from training.train_retrieval import train_retrieval
from features.user_context import build_user_contexts
from retrieval.vector_recall import load_retrieval_artifacts
from features.feature_join import join_features
from training.retrieval_samples import build_retrieval_samples


@dataclass(frozen=True)
class RecallState:
    retrieval: object
    popularity: pd.DataFrame
    genres: object
    similarity: pd.DataFrame
    users: pd.DataFrame
    movies: pd.DataFrame
    history: pd.DataFrame
    config: dict


def build_recall_state(directory: Path, users: pd.DataFrame, movies: pd.DataFrame, history: pd.DataFrame, config: dict) -> RecallState:
    retrieval = load_retrieval_artifacts(directory)
    if config["candidate_budget"] > len(retrieval.movie_ids) or config["neighbors_per_item"] >= len(retrieval.movie_ids):
        raise ValueError("召回目录不足以支持当前候选数量和近邻数量")
    popularity = build_popularity_index(history, retrieval.movie_ids, config["positive_threshold"])
    genres = build_genre_index(history, movies, retrieval.movie_ids, config["positive_threshold"])
    similarity = build_item_similarity_index(retrieval.movie_ids, retrieval.item_vectors, config["neighbors_per_item"])
    return RecallState(retrieval, popularity, genres, similarity, users, movies, history, config)


def generate_candidates(state, queries: pd.DataFrame, history: pd.DataFrame | None = None) -> dict:
    config = state.config
    contexts = build_user_contexts(queries, state.history if history is None else history, state.users, state.movies, config["max_history"], config["positive_threshold"])
    contexts, channels = dispatch_recall(contexts, state.retrieval, state.popularity, state.genres, state.similarity, config["candidate_budget"], config["min_history"], config["recent_items"], config["batch_size"])
    return {"contexts": contexts, "channels": channels, "merged": snake_merge(contexts, channels, config["candidate_budget"])}


def candidate_records(result: dict, state) -> pd.DataFrame:
    records = join_features(result["merged"], state.users, state.movies)
    return build_context_features(records, result["contexts"], state.movies, state.retrieval, state.popularity)


def split_ranking_history(train: pd.DataFrame, history_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(history_fraction) or not 0 < history_fraction < 1:
        raise ValueError("排序历史比例必须位于 0 和 1 之间")
    ordered = train.assign(event_position=np.arange(len(train))).sort_values(["user_id", "timestamp", "event_position"], kind="stable")
    sizes = ordered.groupby("user_id")["movie_id"].transform("size")
    if sizes.min() < 2:
        raise ValueError("排序候选训练需要每位用户至少两条训练交互")
    cutoffs = np.floor(sizes * history_fraction).astype(int).clip(lower=1)
    prefix = ordered.groupby("user_id").cumcount() < cutoffs
    return ordered.loc[prefix, train.columns].copy(), ordered.loc[~prefix, train.columns].copy()


def build_candidate_training(train: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame, config: dict, directory: Path) -> tuple[pd.DataFrame, dict]:
    history, targets = split_ranking_history(train, config["ranking_history_fraction"])
    samples = build_retrieval_samples(history, users, movies, config["max_history"], config["positive_threshold"])
    retrieval_report = train_retrieval(
        samples, directory / "retrieval", max_history=config["max_history"], emb_dim=config["retrieval_emb_dim"],
        num_sampled=config["num_sampled"], temperature=config["temperature"], epochs=config["retrieval_epochs"],
        batch_size=config["batch_size"], learning_rate=config.get("retrieval_learning_rate", config["learning_rate"]), seed=config.get("retrieval_seed", config["seed"]),
    )
    state = build_recall_state(directory / "retrieval", users, movies, history, config)
    queries = history.groupby("user_id", as_index=False)["timestamp"].max().rename(columns={"timestamp": "as_of_timestamp"})
    result = generate_candidates(state, queries)
    records = candidate_records(result, state)
    keys = ["user_id", "movie_id"]
    positive_keys = pd.MultiIndex.from_frame(targets.loc[targets["rating"] >= config["positive_threshold"], keys])
    negative_keys = pd.MultiIndex.from_frame(targets.loc[targets["rating"] <= config["negative_threshold"], keys])
    index = pd.MultiIndex.from_frame(records[keys])
    records["label"] = index.isin(positive_keys).astype(np.int32)
    records["sample_kind"] = np.where(records["label"] == 1, "window_positive", np.where(index.isin(negative_keys), "window_low_rating", "window_unobserved"))
    eligible = records.groupby("user_id")["label"].sum()
    selected = eligible.index[eligible > 0]
    report = {
        "policy": "per_user_training_prefix_then_target_window", "history_fraction": config["ranking_history_fraction"],
        "history_rows": len(history), "target_rows": len(targets), "queries": len(queries),
        "eligible_queries": len(selected), "queries_without_positive_candidates": int((eligible == 0).sum()),
        "candidate_rows": len(records), "retrieval_report": retrieval_report,
        "label_semantics": "observed_high_rating_in_training_target_window",
        "uses_validation_or_test_labels": False,
    }
    if not len(selected):
        raise ValueError("训练候选中没有高评分目标")
    directory.mkdir(parents=True, exist_ok=True)
    history.to_pickle(directory / "history.pkl")
    targets.to_pickle(directory / "targets.pkl")
    queries.to_pickle(directory / "queries.pkl")
    result["contexts"].to_pickle(directory / "contexts.pkl")
    result["merged"].to_pickle(directory / "candidates.pkl")
    return records.loc[records["user_id"].isin(selected)].reset_index(drop=True), report
