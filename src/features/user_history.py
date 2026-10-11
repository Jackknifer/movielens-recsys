import pandas as pd


REQUIRED_COLUMNS = {"user_id", "movie_id", "rating", "timestamp"}
EVENT_ORDER_COLUMN = "_history_event_order"


def build_history_prefixes(
    train: pd.DataFrame,
    max_history: int = 10,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """为训练集中的每次交互，记录该次交互之前用户喜欢的电影。"""
    missing = REQUIRED_COLUMNS - set(train.columns)
    if missing:
        raise ValueError(f"train 缺少字段：{sorted(missing)}")
    if train.empty:
        raise ValueError("train 不能为空")
    if EVENT_ORDER_COLUMN in train.columns or "hist_movie_id" in train.columns:
        raise ValueError("train 已包含用户历史使用的字段")
    if train[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("train 的必要字段存在空值")
    if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
        raise ValueError("max_history 必须是正整数")
    if not 1 <= positive_threshold <= 5:
        raise ValueError("positive_threshold 必须在 1 到 5 之间")

    ordered = train.copy()
    # 输入位置用于确定同一用户、同一时间戳下的交互顺序。
    ordered[EVENT_ORDER_COLUMN] = range(len(ordered))
    ordered = ordered.sort_values(
        ["user_id", "timestamp", EVENT_ORDER_COLUMN]
    ).reset_index(drop=True)

    histories: list[list[int]] = []
    for _, user_events in ordered.groupby("user_id", sort=False):
        liked_movies: list[int] = []
        for event in user_events.itertuples(index=False):
            # 先记录历史，再考虑当前电影，保证目标电影不会进入自己的历史。
            histories.append(liked_movies[-max_history:].copy())
            if event.rating >= positive_threshold:
                liked_movies.append(event.movie_id)

    ordered["hist_movie_id"] = histories
    return ordered.drop(columns=EVENT_ORDER_COLUMN)
