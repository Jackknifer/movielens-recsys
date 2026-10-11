import numpy as np
import pandas as pd


def movie_diversity_attributes(movies: pd.DataFrame) -> pd.DataFrame:
    if movies["movie_id"].duplicated().any() or movies[["movie_id", "genres", "startYear"]].isna().any().any():
        raise ValueError("电影属性存在重复编号或空值")
    if not movies["genres"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("genres 必须是非空字符串")
    attributes = movies[["movie_id"]].copy()
    attributes["primary_genre"] = movies["genres"].str.split("|", regex=False).str[0]
    years = pd.to_numeric(movies["startYear"].replace("__MISSING__", np.nan), errors="raise")
    if ((years.dropna() % 1 != 0) | (years.dropna() < 0)).any():
        raise ValueError("startYear 必须是有效年份或 __MISSING__")
    attributes["decade"] = (years // 10 * 10).astype("Int64").astype(str).replace("<NA>", "__MISSING__")
    return attributes


def rerank_candidates(
    scored: pd.DataFrame, movies: pd.DataFrame, top_k: int = 20,
    max_genre_run: int = 2, max_decade_run: int = 2,
) -> pd.DataFrame:
    """约束展示列表的连续类型和年代，完整候选集合与模型分数保持不变。"""
    for name, value in (("top_k", top_k), ("max_genre_run", max_genre_run), ("max_decade_run", max_decade_run)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")
    if not scored.index.is_unique:
        raise ValueError("重排输入的行索引必须唯一")
    if scored.duplicated(["user_id", "movie_id"]).any() or scored[["movie_id", "rank", "ranking_score"]].isna().any().any():
        raise ValueError("重排输入存在重复电影或空值")
    if not np.isfinite(scored["ranking_score"].to_numpy(dtype=float)).all():
        raise ValueError("重排分数存在无效数值")
    attributes = movie_diversity_attributes(movies).set_index("movie_id")
    if not scored["movie_id"].isin(attributes.index).all():
        raise ValueError("重排候选缺少电影属性")
    if scored.empty:
        return scored.copy()
    groups = []
    for user_id, group in scored.groupby("user_id", sort=False):
        ranks = group["rank"].to_numpy()
        if not np.issubdtype(ranks.dtype, np.integer) or not np.array_equal(np.sort(ranks), np.arange(1, len(group) + 1)):
            raise ValueError("重排输入的 rank 必须从 1 开始连续编号")
        remaining = group.sort_values("rank").index.tolist()
        chosen, last_genre, last_decade = [], None, None
        genre_run, decade_run = 0, 0
        for _ in range(min(top_k, len(remaining))):
            selected = None
            for index in remaining:
                movie_id = scored.at[index, "movie_id"]
                genre, decade = attributes.loc[movie_id, ["primary_genre", "decade"]]
                if genre == last_genre and genre_run >= max_genre_run:
                    continue
                if decade == last_decade and decade_run >= max_decade_run:
                    continue
                selected = index
                break
            if selected is None:
                raise ValueError(f"user_id={user_id} 的候选无法继续满足展示列表的重排限制")
            remaining.remove(selected)
            chosen.append(selected)
            genre, decade = attributes.loc[scored.at[selected, "movie_id"], ["primary_genre", "decade"]]
            genre_run = genre_run + 1 if genre == last_genre else 1
            decade_run = decade_run + 1 if decade == last_decade else 1
            last_genre, last_decade = genre, decade
        ordered = scored.loc[chosen + remaining].copy()
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        groups.append(ordered)
    return pd.concat(groups, ignore_index=True)
