# 本模块负责读取 MovieLens 数据，并在数据进入后续流程前完成基础校验。
# 这里不进行特征编码、时间划分、样本构造或模型训练。

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


# users.pkl 至少需要包含这些字段。
# 后续流程会使用 user_id 连接评分记录，其他字段用于构造用户特征。
USER_REQUIRED_COLUMNS = {
    "user_id",
    "gender",
    "age",
    "occupation",
    "zip_code",
}

# movies.pkl 至少需要包含这些字段。
# movie_id 用于连接评分记录，其他字段用于排序和重排。
MOVIE_REQUIRED_COLUMNS = {
    "movie_id",
    "genres",
    "isAdult",
    "startYear",
}

# ratings.pkl 至少需要包含这些字段。
# 每一行表示某个用户在某个时间对某部电影给出的评分。
RATING_REQUIRED_COLUMNS = {
    "user_id",
    "movie_id",
    "rating",
    "timestamp",
}


@dataclass(frozen=True)
class MovieLensData:
    """集中保存完成基础校验后的三张 MovieLens 数据表。

    frozen=True 会阻止 users、movies、ratings 属性被重新赋值。
    DataFrame 内部的数据仍然可以修改，后续处理需要修改时应当调用 copy()。
    """

    users: pd.DataFrame
    movies: pd.DataFrame
    ratings: pd.DataFrame


def load_movielens(data_dir: Path) -> MovieLensData:
    """读取并校验 MovieLens 数据。

    Args:
        data_dir: 保存 users.pkl、movies.pkl 和 ratings.pkl 的目录。

    Returns:
        包含 users、movies、ratings 三张 DataFrame 的 MovieLensData。

    Raises:
        NotADirectoryError: data_dir 不存在或不是目录。
        FileNotFoundError: 三个必需文件中至少有一个不存在。
        TypeError: pkl 文件保存的对象不是 pandas DataFrame。
        ValueError: 数据存在字段、空值、类型、范围、重复或引用问题。
    """

    # 调用方可能传入字符串。统一转换为 Path 后，路径连接和文件检查
    # 都可以使用 Path 提供的方法完成。
    data_dir = Path(data_dir)

    # 数据根路径必须是一个已经存在的目录。
    if not data_dir.is_dir():
        raise NotADirectoryError(f"数据目录不存在: {data_dir}")

    # 建立表名到文件路径的映射，后续可以统一检查和读取。
    file_paths = {
        "users": data_dir / "users.pkl",
        "movies": data_dir / "movies.pkl",
        "ratings": data_dir / "ratings.pkl",
    }

    # 遍历三个路径，只收集不存在或不是普通文件的路径。
    # 一次收集全部缺失项，可以让异常信息同时列出多个文件。
    missing_paths = [
        path
        for path in file_paths.values()
        if not path.is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(f"缺少数据文件: {missing_paths}")

    # pickle 可以保存 Python 对象，因此这里只读取可信来源的 pkl 文件。
    # 读取完成后，三个变量预期都是 pandas DataFrame。
    users = pd.read_pickle(file_paths["users"])
    movies = pd.read_pickle(file_paths["movies"])
    ratings = pd.read_pickle(file_paths["ratings"])

    # 将表名和读取结果放入同一个字典，减少重复的类型检查代码。
    tables = {
        "users": users,
        "movies": movies,
        "ratings": ratings,
    }

    # pd.read_pickle() 可以读取多种 Python 对象。
    # 后续代码需要 DataFrame 的列选择、分组和连接功能，因此这里明确检查类型。
    for table_name, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"{table_name}.pkl 必须保存 pandas DataFrame"
            )

    # 检查三张表是否包含后续流程需要的全部字段。
    # 这里只要求必需字段存在，文件中允许保留其他字段。
    _require_columns(
        table_name="users",
        table=users,
        required_columns=USER_REQUIRED_COLUMNS,
    )
    _require_columns(
        table_name="movies",
        table=movies,
        required_columns=MOVIE_REQUIRED_COLUMNS,
    )
    _require_columns(
        table_name="ratings",
        table=ratings,
        required_columns=RATING_REQUIRED_COLUMNS,
    )

    # 后面的类型转换会修改 DataFrame。
    # 创建副本可以让修改只作用于本函数准备返回的数据。
    users = users.copy()
    movies = movies.copy()
    ratings = ratings.copy()

    # 必需字段不能包含空值。
    # 空值会影响 ID 连接、时间排序、评分范围检查和模型输入。
    _require_no_nulls(
        table_name="users",
        table=users,
        columns=USER_REQUIRED_COLUMNS,
    )
    _require_no_nulls(
        table_name="movies",
        table=movies,
        columns=MOVIE_REQUIRED_COLUMNS,
    )
    _require_no_nulls(
        table_name="ratings",
        table=ratings,
        columns=RATING_REQUIRED_COLUMNS,
    )

    # ID 和 timestamp 会参与表间连接、用户分组、时间排序和数组索引。
    # 统一转换为 int64 可以防止整数 1 和字符串 "1" 被当成两个值。
    users["user_id"] = _to_integer_series(
        users["user_id"],
        column_name="users.user_id",
    )
    movies["movie_id"] = _to_integer_series(
        movies["movie_id"],
        column_name="movies.movie_id",
    )
    ratings["user_id"] = _to_integer_series(
        ratings["user_id"],
        column_name="ratings.user_id",
    )
    ratings["movie_id"] = _to_integer_series(
        ratings["movie_id"],
        column_name="ratings.movie_id",
    )
    ratings["timestamp"] = _to_integer_series(
        ratings["timestamp"],
        column_name="ratings.timestamp",
    )

    # rating 允许整数或小数，所以这里只要求每个值都能转换为数值。
    # errors="raise" 会让非法字符串直接触发异常。
    ratings["rating"] = pd.to_numeric(
        ratings["rating"],
        errors="raise",
    )

    # 使用带名称的属性保存三张表，避免调用方依赖元组位置记忆顺序。
    data = MovieLensData(
        users=users,
        movies=movies,
        ratings=ratings,
    )

    # 文件、对象类型、字段、空值和基础类型已经检查完成。
    # 这里继续检查重复记录、取值范围和表间引用。
    validate_movielens(data)

    return data


def _require_columns(
    table_name: str,
    table: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """检查 DataFrame 是否包含指定字段。

    required_columns 和 table.columns 都转换为集合后，可以使用集合差集
    得到数据表中缺少的字段。
    """

    # 例如必需字段为 {"user_id", "rating"}，现有字段只有 {"user_id"}，
    # 差集结果就是 {"rating"}。
    missing_columns = required_columns - set(table.columns)

    if missing_columns:
        raise ValueError(
            f"{table_name} 缺少字段: {sorted(missing_columns)}"
        )


def _require_no_nulls(
    table_name: str,
    table: pd.DataFrame,
    columns: set[str],
) -> None:
    """检查指定字段是否存在空值，并报告各字段的空值数量。"""

    # table[list(columns)] 选出需要检查的字段。
    # isna() 将空值标记为 True，sum() 按字段统计 True 的数量。
    null_counts = table[list(columns)].isna().sum()

    # 只保留空值数量大于 0 的字段，使异常信息只展示存在问题的字段。
    invalid_counts = null_counts[null_counts > 0]

    if not invalid_counts.empty:
        raise ValueError(
            f"{table_name} 存在空值: {invalid_counts.to_dict()}"
        )


def _to_integer_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """把 Series 转换为 int64，同时拒绝非法字符串和非整数数值。"""

    # errors="raise" 表示遇到 "abc" 这类无法转换的值时直接抛出异常。
    numeric = pd.to_numeric(series, errors="raise")

    # 对 1 取余可以检查小数部分：3.0 % 1 为 0，3.5 % 1 为 0.5。
    # 这样可以防止 astype("int64") 把 3.5 截断成 3。
    if (numeric % 1 != 0).any():
        raise ValueError(f"{column_name} 必须全部为整数")

    return numeric.astype("int64")


def validate_movielens(data: MovieLensData) -> None:
    """检查三张数据表的内容范围、唯一性和表间引用。"""

    # 使用短变量名读取容器中的三张表，方便后面的校验表达式阅读。
    users = data.users
    movies = data.movies
    ratings = data.ratings

    # 三张表都必须至少包含一行数据。
    if users.empty:
        raise ValueError("users 数据表为空")

    if movies.empty:
        raise ValueError("movies 数据表为空")

    if ratings.empty:
        raise ValueError("ratings 数据表为空")

    # user_id 是用户表主键，同一个 user_id 只能出现一次。
    if users["user_id"].duplicated().any():
        raise ValueError("users.user_id 存在重复值")

    # movie_id 是电影表主键，同一个 movie_id 只能出现一次。
    if movies["movie_id"].duplicated().any():
        raise ValueError("movies.movie_id 存在重复值")

    # 当前项目把每个用户对每部电影的评分视为一条记录。
    # 相同 user_id 和 movie_id 再次出现时，无法确定应使用哪一条评分。
    if ratings.duplicated(
        subset=["user_id", "movie_id"]
    ).any():
        raise ValueError("ratings 存在重复的用户电影组合")

    # 原始 ID 必须大于 0；模型所需的 padding 和未知编号由编码模块管理。
    if (users["user_id"] <= 0).any():
        raise ValueError("users.user_id 必须大于 0")

    if (movies["movie_id"] <= 0).any():
        raise ValueError("movies.movie_id 必须大于 0")

    # MovieLens timestamp 使用非负 Unix 时间戳。
    if (ratings["timestamp"] < 0).any():
        raise ValueError("ratings.timestamp 不能小于 0")

    # MovieLens-1M 的评分范围是 1 到 5，between() 默认包含两个边界。
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("ratings.rating 必须位于 1 到 5 之间")

    # 用户表给出全部合法用户编号，评分表中的用户编号必须属于这个集合。
    known_user_ids = set(users["user_id"])
    rating_user_ids = set(ratings["user_id"])
    unknown_user_ids = rating_user_ids - known_user_ids

    if unknown_user_ids:
        # 最多展示十个未知编号，避免异常信息过长。
        example_ids = sorted(unknown_user_ids)[:10]
        raise ValueError(
            f"ratings 引用了未知 user_id: {example_ids}"
        )

    # 电影表给出全部合法电影编号，评分表中的电影编号必须属于这个集合。
    known_movie_ids = set(movies["movie_id"])
    rating_movie_ids = set(ratings["movie_id"])
    unknown_movie_ids = rating_movie_ids - known_movie_ids

    if unknown_movie_ids:
        # 与用户编号检查一致，最多展示十个未知编号。
        example_ids = sorted(unknown_movie_ids)[:10]
        raise ValueError(
            f"ratings 引用了未知 movie_id: {example_ids}"
        )
