import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def _to_column(values: pd.Series) -> pd.DataFrame:
    """把一列类别值转换为 sklearn 所需的二维表。"""
    if not isinstance(values, pd.Series):
        raise TypeError("values 必须是 pandas Series")
    if values.empty:
        raise ValueError("待编码数据不能为空")
    if values.isna().any():
        raise ValueError("请在编码之前明确处理缺失值")

    # 同一字段统一转为字符串，保证训练和预测使用相同的类别表示。
    # to_frame 将形状从 (记录数,) 转为 (记录数, 1)。
    return values.astype(str).to_frame(name="value")


def fit_category_encoder(train_values: pd.Series) -> OrdinalEncoder:
    """只根据训练阶段可见的一列数据建立类别编号。"""
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )
    # fit 记录训练数据出现过的类别。验证和测试只能使用 transform。
    encoder.fit(_to_column(train_values))
    return encoder


def encode_category(
    values: pd.Series,
    encoder: OrdinalEncoder,
) -> pd.Series:
    """使用已有编码器，将一列类别转换为模型使用的整数编号。"""
    encoded = encoder.transform(_to_column(values))

    # sklearn 的已知类别从 0 开始，未知类别为 -1。
    # 全部加 2 后：0 留给序列 padding，1 表示未知，已知类别从 2 开始。
    # [:, 0] 取出二维结果的第一列，恢复为一维数组。
    codes = encoded[:, 0] + 2
    return pd.Series(
        codes,
        index=values.index,
        name=f"{values.name}_encoded",
        dtype="int64",
    )
