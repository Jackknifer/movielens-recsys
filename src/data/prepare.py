from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from pandas.testing import assert_frame_equal


MISSING_CATEGORY = "__MISSING__"


def prepare_movielens(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """准备 loader 的输入文件，明确表示电影元数据中的缺失类别。"""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("输入目录和输出目录必须不同")

    users = pd.read_pickle(source_dir / "users.pkl")
    movies = pd.read_pickle(source_dir / "movies.pkl")
    ratings = pd.read_pickle(source_dir / "ratings.pkl")
    # 本项目统一使用 MovieLens 五星评分，并逐条核对官方原始记录。
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise") / 2
    with ZipFile(source_dir / "ml-1m.zip") as archive:
        with archive.open("ml-1m/ratings.dat") as source:
            official = pd.read_csv(
                source, sep="::", engine="python",
                names=["user_id", "movie_id", "rating", "timestamp"],
            )
    normalized = ratings.astype({
        "user_id": "int64", "movie_id": "int64", "timestamp": "int64"
    })
    official["rating"] = official["rating"].astype("float64")
    keys = ["user_id", "movie_id"]
    assert_frame_equal(
        normalized.sort_values(keys).reset_index(drop=True),
        official.sort_values(keys).reset_index(drop=True),
    )
    missing_counts = {}
    for column in ("isAdult", "startYear"):
        missing_counts[column] = int(movies[column].isna().sum())
        # 缺失属性作为独立类别保存，不能推断电影的年龄限制或发行年份。
        movies[column] = movies[column].astype(object).fillna(MISSING_CATEGORY)

    output_dir.mkdir(parents=True, exist_ok=True)
    users.to_pickle(output_dir / "users.pkl")
    movies.to_pickle(output_dir / "movies.pkl")
    normalized.to_pickle(output_dir / "ratings.pkl")
    return missing_counts
