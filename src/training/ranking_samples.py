import numpy as np
import pandas as pd

from training.negative_sampling import sample_unobserved, validate_training_interactions
from features.feature_join import join_features


def build_observed_samples(
    interactions: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame,
    positive_threshold: float = 4.0, negative_threshold: float = 2.0,
) -> pd.DataFrame:
    if not 1 <= negative_threshold < positive_threshold <= 5:
        raise ValueError("评分阈值必须满足 1 <= negative_threshold < positive_threshold <= 5")
    validate_training_interactions(interactions)
    positive = interactions["rating"] >= positive_threshold
    negative = interactions["rating"] <= negative_threshold
    selected = interactions.loc[positive | negative, ["user_id", "movie_id"]].copy()
    selected["label"] = positive.loc[selected.index].astype(np.int32)
    selected["sample_kind"] = np.where(selected["label"] == 1, "positive", "explicit_negative")
    return join_features(selected.reset_index(drop=True), users, movies)


def build_ranking_samples(
    train: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame,
    positive_threshold: float = 4.0, negative_threshold: float = 2.0,
    negatives_per_positive: int = 1, seed: int = 42,
) -> pd.DataFrame:
    """组合高评分、低评分和未交互训练样本，再连接属性。"""
    observed = build_observed_samples(train, users, movies, positive_threshold, negative_threshold)
    random_negatives = sample_unobserved(
        train, np.sort(train["movie_id"].unique()), negatives_per_positive, positive_threshold, seed,
    )
    sampled = join_features(random_negatives, users, movies)
    samples = pd.concat([observed, sampled], ignore_index=True)
    if set(samples["label"]) != {0, 1}:
        raise ValueError("排序训练样本必须同时包含两种标签")
    if samples.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("排序训练样本出现重复用户电影组合")
    return samples
