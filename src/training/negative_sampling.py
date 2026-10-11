import numpy as np
import pandas as pd

from retrieval.popular_recall import _validate_movie_ids


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


def sample_unobserved(
    train: pd.DataFrame, movie_ids: np.ndarray, negatives_per_positive: int = 1,
    positive_threshold: float = 4.0, seed: int = 42,
) -> pd.DataFrame:
    """从训练目录中选择未交互电影，作为声明了假设的训练负样本。"""
    validate_training_interactions(train)
    _validate_movie_ids(movie_ids)
    if not np.isfinite(positive_threshold) or not 1 <= positive_threshold <= 5:
        raise ValueError("positive_threshold 必须在 1 到 5 之间")
    if isinstance(negatives_per_positive, bool) or not isinstance(negatives_per_positive, int) or negatives_per_positive < 0:
        raise ValueError("negatives_per_positive 必须是非负整数")
    if not set(movie_ids).issubset(train["movie_id"]):
        raise ValueError("负采样目录必须只包含训练阶段出现的电影")
    generator = np.random.default_rng(seed)
    rows = []
    catalog = np.sort(movie_ids)
    for user_id, events in train.groupby("user_id", sort=True):
        available = catalog[~np.isin(catalog, events["movie_id"])]
        count = min(len(available), int((events["rating"] >= positive_threshold).sum()) * negatives_per_positive)
        for movie_id in generator.choice(available, size=count, replace=False):
            rows.append((int(user_id), int(movie_id), 0, "unobserved"))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "label", "sample_kind"])
