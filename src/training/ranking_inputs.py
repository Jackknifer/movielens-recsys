import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from features.ranking_features import RANKING_NUMERIC_COLUMNS
from features.encode import encode_category, fit_category_encoder
from features.feature_join import MOVIE_FEATURE_COLUMNS, USER_FEATURE_COLUMNS


RANKING_FEATURE_COLUMNS = USER_FEATURE_COLUMNS + MOVIE_FEATURE_COLUMNS


def ranking_feature_frame(records: pd.DataFrame) -> pd.DataFrame:
    missing = set(RANKING_FEATURE_COLUMNS) - set(records.columns)
    if missing or records.empty:
        raise ValueError(f"排序输入不能为空，缺少字段：{sorted(missing)}")
    frame = records[RANKING_FEATURE_COLUMNS].copy()
    if frame.isna().any().any():
        raise ValueError("排序输入的属性存在空值")
    if not frame["genres"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("genres 必须是非空字符串")
    frame["genres"] = frame["genres"].str.split("|", regex=False).str[0]
    return frame


def fit_ranking_encoders(samples: pd.DataFrame) -> dict[str, OrdinalEncoder]:
    frame = ranking_feature_frame(samples)
    return {name: fit_category_encoder(frame[name]) for name in RANKING_FEATURE_COLUMNS}


def build_ranking_inputs(records: pd.DataFrame, encoders: dict[str, OrdinalEncoder], numeric_scaler: StandardScaler | None = None) -> dict[str, np.ndarray]:
    frame = ranking_feature_frame(records)
    if set(encoders) != set(RANKING_FEATURE_COLUMNS):
        raise ValueError("排序编码器的字段不符合输入要求")
    inputs = {
        name: encode_category(frame[name], encoders[name]).to_numpy(dtype=np.int32)[:, None]
        for name in RANKING_FEATURE_COLUMNS
    }
    if numeric_scaler is not None:
        numeric = records[RANKING_NUMERIC_COLUMNS]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("排序数值特征存在无效数值")
        inputs["context_features"] = numeric_scaler.transform(numeric).astype(np.float32)
    return inputs


def unknown_rates(features: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float((values == 1).mean()) for name, values in features.items() if name in RANKING_FEATURE_COLUMNS}
