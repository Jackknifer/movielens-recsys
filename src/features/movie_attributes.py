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
