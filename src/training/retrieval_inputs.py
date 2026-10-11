import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.utils import pad_sequences

from features.encode import encode_category, fit_category_encoder


USER_FEATURE_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]


def _validate_samples(samples: pd.DataFrame, require_target: bool = True) -> None:
    required = set(USER_FEATURE_COLUMNS + ["hist_movie_id", "hist_genres"])
    if require_target:
        required.add("movie_id")
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples 缺少字段：{sorted(missing)}")
    if samples.empty:
        raise ValueError("samples 不能为空")
    for movie_ids, genres in zip(samples["hist_movie_id"], samples["hist_genres"]):
        if not isinstance(movie_ids, (list, tuple)) or not isinstance(genres, (list, tuple)):
            raise TypeError("两列历史都必须保存列表或 tuple")
        if not movie_ids or len(movie_ids) != len(genres):
            raise ValueError("电影历史和类型历史必须非空且长度相同")


def fit_retrieval_encoders(samples: pd.DataFrame) -> dict[str, OrdinalEncoder]:
    """只使用训练样本建立词表，目标电影和历史电影共用一个词表。"""
    _validate_samples(samples)
    encoders = {
        name: fit_category_encoder(samples[name]) for name in USER_FEATURE_COLUMNS
    }

    # 第一部喜欢的电影可能只出现在历史中，也需要获得有效的电影编号。
    movie_values = pd.concat(
        [samples["movie_id"], samples["hist_movie_id"].explode()], ignore_index=True
    )
    encoders["movie_id"] = fit_category_encoder(movie_values)
    encoders["genres"] = fit_category_encoder(samples["hist_genres"].explode())
    return encoders


def _encode_sequences(
    sequences: pd.Series,
    encoder: OrdinalEncoder,
    max_history: int,
) -> np.ndarray:
    # explode 把列表展开为一列，一次编码后按原来的列表长度重新分组。
    codes = encode_category(sequences.explode(), encoder).to_numpy(dtype=np.int32)
    encoded_sequences = []
    offset = 0
    for sequence in sequences:
        length = len(sequence)
        encoded_sequences.append(codes[offset:offset + length])
        offset += length

    # 左侧补 0；超长时保留最近的 max_history 条记录。
    return pad_sequences(
        encoded_sequences, maxlen=max_history, padding="pre", truncating="pre",
        value=0, dtype="int32",
    )


def build_user_inputs(
    samples: pd.DataFrame,
    encoders: dict[str, OrdinalEncoder],
    max_history: int = 10,
) -> dict[str, np.ndarray]:
    """转换用户属性和历史，推荐时无需提供目标电影。"""
    _validate_samples(samples, require_target=False)
    if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
        raise ValueError("max_history 必须是正整数")

    # 单值特征形状为 (样本数, 1)，历史特征形状为 (样本数, max_history)。
    features = {
        name: encode_category(samples[name], encoders[name]).to_numpy(dtype=np.int32)[:, None]
        for name in USER_FEATURE_COLUMNS
    }
    features["hist_movie_id"] = _encode_sequences(
        samples["hist_movie_id"], encoders["movie_id"], max_history
    )
    features["hist_genres"] = _encode_sequences(
        samples["hist_genres"], encoders["genres"], max_history
    )
    return features


def build_retrieval_inputs(
    samples: pd.DataFrame,
    encoders: dict[str, OrdinalEncoder],
    max_history: int = 10,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """训练时分别返回用户输入和目标电影编号。"""
    _validate_samples(samples)
    features = build_user_inputs(samples, encoders, max_history)
    # 目标电影单独返回，用户模型的输入中只包含用户信息和历史。
    targets = encode_category(samples["movie_id"], encoders["movie_id"]).to_numpy(
        dtype=np.int64
    )[:, None]
    return features, targets
