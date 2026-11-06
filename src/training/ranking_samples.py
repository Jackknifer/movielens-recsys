import numpy as np
import pandas as pd

from features.feature_join import join_features


def validate_training_interactions(train: pd.DataFrame) -> None:
    required = {"user_id", "movie_id", "rating", "timestamp"}
    if train.empty or not required.issubset(train.columns) or train[list(required)].isna().any().any():
        raise ValueError("train 必须非空，且包含完整的评分交互字段")
    for column in ("user_id", "movie_id", "timestamp"):
        values = train[column].to_numpy()
        lower = 0 if column == "timestamp" else 1
        if not np.issubdtype(values.dtype, np.integer) or (values < lower).any():
            raise ValueError(f"train.{column} 的整数范围无效")
    if not train["rating"].between(1, 5).all() or train.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("train 的评分范围或交互唯一性无效")


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
