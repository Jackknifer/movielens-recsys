import pandas as pd

from features.feature_join import join_features
from features.user_history import build_history_prefixes


def build_retrieval_samples(
    train: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    max_history: int = 10,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """使用训练集生成“已有喜欢的电影 → 下一部喜欢的电影”样本。"""
    events = build_history_prefixes(train, max_history, positive_threshold)

    # 每条样本的目标电影需要得到较高评分，且用户此前已有可用历史。
    selected = events.loc[
        (events["rating"] >= positive_threshold)
        & events["hist_movie_id"].map(bool)
    ].copy()
    if selected.empty:
        raise ValueError("召回训练样本为空")

    samples = join_features(selected, users, movies)
    genres_by_movie = movies.set_index("movie_id")["genres"].to_dict()

    # 每部历史电影取第一个类型；顺序与 hist_movie_id 保持一致。
    samples["hist_genres"] = samples["hist_movie_id"].map(
        lambda movie_ids: [genres_by_movie[movie_id].split("|")[0] for movie_id in movie_ids]
    )
    return samples
