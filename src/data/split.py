# 本模块负责按照每个用户的交互时间划分训练集、验证集和测试集。
# 输入应当使用 load.py 返回并完成基础校验的 ratings DataFrame。

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_integer_dtype


# 时间划分至少需要用户编号和时间戳。
# rating、movie_id 以及其他附加字段会原样保留在输出中。
SPLIT_REQUIRED_COLUMNS = {
    "user_id",
    "timestamp",
}

# 该字段只在函数内部使用，用于记录每行数据进入函数时的位置。
# 当同一用户的多个交互具有相同 timestamp 时，它可以提供稳定的次序。
EVENT_ORDER_COLUMN = "_event_order"


@dataclass(frozen=True)
class TemporalSplit:
    """保存按用户时间顺序划分后的四份数据。

    train:
        满足最少交互数量的用户中，验证记录和测试记录之前的交互。
    validation:
        每个满足条件用户的倒数第二条交互。
    test:
        每个满足条件用户的最后一条交互。
    cold_start:
        交互数量少于 minimum_interactions 的用户的全部交互。

    frozen=True 只阻止四个属性被重新赋值，DataFrame 内部仍然可以修改。
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    cold_start: pd.DataFrame


def split_by_user_time(
    ratings: pd.DataFrame,
    minimum_interactions: int = 3,
) -> TemporalSplit:
    """按照每个用户的时间顺序划分评分记录。

    对于交互数量达到 minimum_interactions 的用户：
        - 最后一条交互进入 test。
        - 倒数第二条交互进入 validation。
        - 其余较早交互进入 train。

    交互数量不足的用户无法同时提供训练历史、验证标签和测试标签，
    因此其全部记录进入 cold_start。

    Args:
        ratings: 至少包含 user_id 和 timestamp 的评分数据。
        minimum_interactions: 用户进入三段时间划分所需的最少交互数量。

    Returns:
        包含 train、validation、test 和 cold_start 的 TemporalSplit。

    Raises:
        TypeError: ratings 不是 DataFrame，或关键字段不是整数类型。
        ValueError: 输入为空、字段缺失、存在空值，或参数小于 3。
        RuntimeError: 划分结果出现记录遗漏、重复或时间次序问题。
    """

    _validate_split_input(
        ratings=ratings,
        minimum_interactions=minimum_interactions,
    )

    # 使用副本，避免排序和内部字段影响调用方持有的原始 DataFrame。
    ordered = ratings.copy()

    # range(len(ordered)) 为每一行生成唯一位置编号：0、1、2、……。
    # 这个编号只负责处理相同 timestamp 下的次序，不会出现在最终输出中。
    ordered[EVENT_ORDER_COLUMN] = range(len(ordered))

    # 排序键依次为 user_id、timestamp 和原始位置编号。
    # timestamp 相同时使用原始位置确定次序。
    # mergesort 是稳定排序，可以保持相等排序键之间的既有顺序。
    ordered = ordered.sort_values(
        by=["user_id", "timestamp", EVENT_ORDER_COLUMN],
        kind="mergesort",
    )

    train_groups: list[pd.DataFrame] = []
    validation_groups: list[pd.DataFrame] = []
    test_groups: list[pd.DataFrame] = []
    cold_start_groups: list[pd.DataFrame] = []

    # sort=False 表示按照 ordered 中已经确定的用户次序进行分组。
    for _, user_ratings in ordered.groupby("user_id", sort=False):
        if len(user_ratings) < minimum_interactions:
            # 记录数量不足时，整个用户进入 cold_start。
            cold_start_groups.append(user_ratings)
            continue

        # iloc[:-2] 取得最后两行之前的全部记录。
        train_groups.append(user_ratings.iloc[:-2])

        # 使用双层方括号可以保留 DataFrame 类型。
        # iloc[-2] 会返回 Series，iloc[[-2]] 会返回单行 DataFrame。
        validation_groups.append(user_ratings.iloc[[-2]])
        test_groups.append(user_ratings.iloc[[-1]])

    # 至少需要一个用户进入训练、验证和测试划分。
    # 如果没有符合条件的用户，后续模型训练无法继续。
    if not train_groups:
        raise ValueError(
            "没有用户达到 minimum_interactions，无法生成训练集"
        )

    # 将各用户的数据重新连接成四张 DataFrame。
    # ignore_index=True 会为每张结果表重新生成连续行索引。
    train = pd.concat(train_groups, ignore_index=True)
    validation = pd.concat(validation_groups, ignore_index=True)
    test = pd.concat(test_groups, ignore_index=True)
    cold_start = _concat_or_empty(
        groups=cold_start_groups,
        template=ordered,
    )

    # 在删除内部位置编号之前，使用它检查记录覆盖和时间次序。
    _validate_split_result(
        original_size=len(ordered),
        train=train,
        validation=validation,
        test=test,
        cold_start=cold_start,
    )

    # EVENT_ORDER_COLUMN 只服务于内部校验，调用方应当收到与输入相同的字段。
    train = _remove_internal_column(train)
    validation = _remove_internal_column(validation)
    test = _remove_internal_column(test)
    cold_start = _remove_internal_column(cold_start)

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        cold_start=cold_start,
    )


def _validate_split_input(
    ratings: pd.DataFrame,
    minimum_interactions: int,
) -> None:
    """检查时间划分所需的输入条件。"""

    if not isinstance(ratings, pd.DataFrame):
        raise TypeError("ratings 必须是 pandas DataFrame")

    if ratings.empty:
        raise ValueError("ratings 数据表为空")

    if not isinstance(minimum_interactions, int):
        raise TypeError("minimum_interactions 必须是整数")

    # 至少需要三条交互，才能分别提供训练记录、验证标签和测试标签。
    if minimum_interactions < 3:
        raise ValueError("minimum_interactions 必须大于或等于 3")

    # 该名称由本模块用于内部位置编号，输入数据不能占用。
    if EVENT_ORDER_COLUMN in ratings.columns:
        raise ValueError(
            f"ratings 不能包含内部字段: {EVENT_ORDER_COLUMN}"
        )

    missing_columns = SPLIT_REQUIRED_COLUMNS - set(ratings.columns)
    if missing_columns:
        raise ValueError(
            f"ratings 缺少时间划分字段: {sorted(missing_columns)}"
        )

    null_counts = ratings[list(SPLIT_REQUIRED_COLUMNS)].isna().sum()
    invalid_counts = null_counts[null_counts > 0]
    if not invalid_counts.empty:
        raise ValueError(
            f"ratings 的时间划分字段存在空值: {invalid_counts.to_dict()}"
        )

    # load.py 已经将这两个字段转换为 int64。
    # 保留检查可以防止其他调用方绕过 load.py 传入不符合要求的数据。
    if not is_integer_dtype(ratings["user_id"]):
        raise TypeError("ratings.user_id 必须是整数类型")

    if not is_integer_dtype(ratings["timestamp"]):
        raise TypeError("ratings.timestamp 必须是整数类型")


def _concat_or_empty(
    groups: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    """连接多个分组；没有分组时返回与模板字段和类型相同的空表。"""

    if groups:
        return pd.concat(groups, ignore_index=True)

    # iloc[0:0] 保留 template 的字段名称和数据类型，但不包含任何记录。
    return template.iloc[0:0].copy().reset_index(drop=True)


def _remove_internal_column(table: pd.DataFrame) -> pd.DataFrame:
    """删除内部位置编号，并重新生成连续行索引。"""

    return table.drop(columns=[EVENT_ORDER_COLUMN]).reset_index(drop=True)


def _validate_split_result(
    original_size: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    cold_start: pd.DataFrame,
) -> None:
    """检查划分结果是否完整、互斥并保持用户时间次序。"""

    all_parts = [train, validation, test, cold_start]
    combined = pd.concat(all_parts, ignore_index=True)

    # 四份数据的记录总数必须与输入完全一致。
    if len(combined) != original_size:
        raise RuntimeError(
            "时间划分后的记录数量与输入记录数量不一致"
        )

    # 每条输入记录拥有唯一 EVENT_ORDER_COLUMN。
    # 该字段出现重复说明同一条记录进入了多个集合。
    if combined[EVENT_ORDER_COLUMN].duplicated().any():
        raise RuntimeError("同一条评分记录进入了多个数据集合")

    expected_event_orders = set(range(original_size))
    actual_event_orders = set(combined[EVENT_ORDER_COLUMN])

    # 两个集合相等，说明每条输入记录都进入了一个输出集合。
    if actual_event_orders != expected_event_orders:
        raise RuntimeError("时间划分结果存在遗漏记录")

    validation_user_ids = set(validation["user_id"])
    test_user_ids = set(test["user_id"])

    # 每个进入验证集的用户都必须有对应的测试记录，反向也一样。
    if validation_user_ids != test_user_ids:
        raise RuntimeError("验证集和测试集的用户集合不一致")

    # 每个满足条件的用户只能有一条验证记录和一条测试记录。
    if validation["user_id"].duplicated().any():
        raise RuntimeError("同一用户在验证集中出现多条记录")

    if test["user_id"].duplicated().any():
        raise RuntimeError("同一用户在测试集中出现多条记录")

    # 按用户检查：训练记录发生在验证记录之前，验证记录发生在测试记录之前。
    for user_id in validation_user_ids:
        user_train = train[train["user_id"] == user_id]
        user_validation = validation[validation["user_id"] == user_id]
        user_test = test[test["user_id"] == user_id]

        if user_train.empty:
            raise RuntimeError(f"用户 {user_id} 没有训练记录")

        # 排序键由 timestamp 和原始位置共同组成。
        # Python 会依次比较元组中的两个值。
        train_last_key = (
            user_train.iloc[-1]["timestamp"],
            user_train.iloc[-1][EVENT_ORDER_COLUMN],
        )
        validation_key = (
            user_validation.iloc[0]["timestamp"],
            user_validation.iloc[0][EVENT_ORDER_COLUMN],
        )
        test_key = (
            user_test.iloc[0]["timestamp"],
            user_test.iloc[0][EVENT_ORDER_COLUMN],
        )

        if not train_last_key < validation_key < test_key:
            raise RuntimeError(
                f"用户 {user_id} 的训练、验证和测试次序异常"
            )
