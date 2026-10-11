import numpy as np
import pandas as pd

from retrieval.popular_recall import _validate_contexts
from retrieval.vector_recall import RESULT_COLUMNS


MERGED_COLUMNS = RESULT_COLUMNS + ["sources", "source_scores", "source_ranks", "candidate_rank"]


def snake_merge(contexts: pd.DataFrame, channels: dict[str, pd.DataFrame], k: int = 100) -> pd.DataFrame:
    """按照通道顺序往返选取电影，保存重复电影的全部来源。"""
    _validate_contexts(contexts, k)
    if not channels:
        raise ValueError("channels 至少需要一个召回通道")
    by_channel = []
    for name, results in channels.items():
        if not set(RESULT_COLUMNS).issubset(results.columns):
            raise ValueError(f"{name} 缺少召回字段")
        if results[RESULT_COLUMNS].isna().any().any() or results.duplicated(["user_id", "movie_id"]).any():
            raise ValueError(f"{name} 存在空值或重复电影")
        if not results["user_id"].isin(contexts["user_id"]).all() or not results["source"].eq(name).all():
            raise ValueError(f"{name} 的用户范围或来源名称无效")
        if not np.isfinite(results["source_score"].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} 的得分存在无效数值")
        users = {}
        for user_id, group in results.groupby("user_id", sort=False):
            ranks = group["source_rank"].to_numpy()
            if not np.issubdtype(ranks.dtype, np.integer) or not np.array_equal(np.sort(ranks), np.arange(1, len(group) + 1)):
                raise ValueError("source_rank 必须从 1 开始连续编号")
            users[user_id] = group.sort_values("source_rank").to_dict("records")
        by_channel.append(users)

    rows = []
    for context in contexts.itertuples(index=False):
        lists = [users.get(context.user_id, []) for users in by_channel]
        all_sources = {}
        for candidates in lists:
            for candidate in candidates:
                all_sources.setdefault(candidate["movie_id"], []).append(candidate)
        pointers = [0] * len(lists)
        current, direction, selected = 0, 1, set(context.seen_movie_ids)
        count = 0
        while count < k and any(pointer < len(items) for pointer, items in zip(pointers, lists)):
            if pointers[current] < len(lists[current]):
                candidate = lists[current][pointers[current]]
                pointers[current] += 1
                movie_id = candidate["movie_id"]
                if movie_id not in selected:
                    selected.add(movie_id)
                    count += 1
                    provenance = all_sources[movie_id]
                    rows.append({
                        **candidate, "sources": [row["source"] for row in provenance],
                        "source_scores": {row["source"]: float(row["source_score"]) for row in provenance},
                        "source_ranks": {row["source"]: int(row["source_rank"]) for row in provenance},
                        "candidate_rank": count,
                    })
            current += direction
            if current == len(lists):
                current, direction = len(lists) - 1, -1
            elif current < 0:
                current, direction = 0, 1
    return pd.DataFrame(rows, columns=MERGED_COLUMNS)
