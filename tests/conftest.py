from pathlib import Path
import shutil

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config) -> None:
    # pytest 创建本次测试目录前，需要保证项目工作目录存在。
    (PROJECT_ROOT / ".work").mkdir(exist_ok=True)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus) -> None:
    # 成功后清理本次 pytest 目录；失败时保留模型和数据，供检查使用。
    if exitstatus == 0:
        directory = session.config._tmp_path_factory.getbasetemp()
        if directory.parent.resolve() == (PROJECT_ROOT / ".work").resolve() and directory.name.startswith("pytest"):
            shutil.rmtree(directory)


@pytest.fixture(scope="session")
def source_tables() -> dict[str, pd.DataFrame]:
    # 使用项目准备好的真实 MovieLens 表；缺少数据时直接报告失败。
    input_dir = PROJECT_ROOT / "data" / "processed" / "input"
    tables = {
        name: pd.read_pickle(input_dir / f"{name}.pkl")
        for name in ("users", "movies", "ratings")
    }
    # 原始用户和电影编号以文本保存，选取关联记录前统一为整数。
    for table in tables.values():
        for column in ("user_id", "movie_id", "timestamp"):
            if column in table:
                table[column] = pd.to_numeric(table[column], errors="raise").astype("int64")
    return tables


@pytest.fixture
def sample_tables(source_tables) -> dict[str, pd.DataFrame]:
    ratings = source_tables["ratings"]
    counts = ratings.groupby("user_id").size()
    user_ids = counts[counts >= 5].index[:2]
    if len(user_ids) != 2:
        raise ValueError("测试数据需要两个至少有五条评分的用户")
    sample = ratings[ratings.user_id.isin(user_ids)].groupby("user_id").head(5)
    return {
        "users": source_tables["users"].loc[
            source_tables["users"].user_id.isin(user_ids)
        ].copy(),
        "movies": source_tables["movies"].loc[
            source_tables["movies"].movie_id.isin(sample.movie_id)
        ].copy(),
        "ratings": sample.copy(),
    }


@pytest.fixture
def sample_dir(sample_tables, tmp_path) -> Path:
    # pytest 的工作目录由 pyproject.toml 固定在项目 .work/pytest 内。
    for name, frame in sample_tables.items():
        frame.to_pickle(tmp_path / f"{name}.pkl")
    return tmp_path
