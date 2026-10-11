import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder

from models.youtubednn import L2Normalize, MaskedMean
from training.retrieval_inputs import USER_FEATURE_COLUMNS, build_user_inputs


RESULT_COLUMNS = ["user_id", "movie_id", "source", "source_score", "source_rank"]


@dataclass(frozen=True)
class RetrievalArtifacts:
    user_tower: tf.keras.Model
    encoders: dict[str, OrdinalEncoder]
    movie_ids: np.ndarray
    item_vectors: np.ndarray
    max_history: int
    training_report: dict


def _validate_catalog(movie_ids: np.ndarray, item_vectors: np.ndarray) -> None:
    if movie_ids.ndim != 1 or not len(movie_ids):
        raise ValueError("movie_ids 必须是非空的一维数组")
    if not np.issubdtype(movie_ids.dtype, np.integer) or (movie_ids < 1).any():
        raise ValueError("movie_ids 必须包含正整数编号")
    if len(np.unique(movie_ids)) != len(movie_ids):
        raise ValueError("movie_ids 存在重复电影")
    if item_vectors.ndim != 2 or item_vectors.shape[0] != len(movie_ids):
        raise ValueError("电影编号数量与向量行数不相同")
    if item_vectors.shape[1] < 1 or not np.isfinite(item_vectors).all():
        raise ValueError("电影向量的维度或数值无效")
    if not np.allclose(np.linalg.norm(item_vectors, axis=1), 1, atol=1e-5):
        raise ValueError("电影向量的 L2 长度应当为 1")


def load_retrieval_artifacts(directory: Path) -> RetrievalArtifacts:
    """读取训练产物，检查电影编号、向量和用户模型是否使用相同规则。"""
    directory = Path(directory)
    with (directory / "training_report.json").open(encoding="utf-8") as source:
        report = json.load(source)
    with (directory / "encoders.pkl").open("rb") as source:
        encoders = pickle.load(source)
    movie_ids = np.load(directory / "movie_ids.npy", allow_pickle=False)
    item_vectors = np.load(directory / "item_embeddings.npy", allow_pickle=False)
    _validate_catalog(movie_ids, item_vectors)

    for name in USER_FEATURE_COLUMNS + ["movie_id", "genres"]:
        if not isinstance(encoders[name], OrdinalEncoder):
            raise TypeError(f"{name} 必须使用 OrdinalEncoder")
        if len(encoders[name].categories_[0]) + 2 != report["vocab_sizes"][name]:
            raise ValueError(f"{name} 的词表长度与训练报告不相同")
    expected_ids = encoders["movie_id"].categories_[0].astype(np.int64)
    if not np.array_equal(movie_ids, expected_ids):
        raise ValueError("movie_ids 的顺序与训练词表不相同")
    if item_vectors.shape != (report["movies"], report["emb_dim"]):
        raise ValueError("电影向量的形状与训练报告不相同")
    max_history = report["max_history"]
    if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
        raise ValueError("训练报告中的 max_history 必须是正整数")

    user_tower = tf.keras.models.load_model(
        directory / "user_tower.keras", compile=False,
        custom_objects={"MaskedMean": MaskedMean, "L2Normalize": L2Normalize},
    )
    expected_features = set(USER_FEATURE_COLUMNS + ["hist_movie_id", "hist_genres"])
    if set(user_tower.input_names) != expected_features:
        raise ValueError("用户模型的输入字段不符合召回输入规则")
    for tensor in user_tower.inputs:
        name = tensor.name.split(":")[0]
        width = max_history if name.startswith("hist_") else 1
        if tensor.shape.as_list()[1:] != [width]:
            raise ValueError(f"用户模型的 {name} 输入长度与训练报告不相同")
    if user_tower.output_shape[-1] != item_vectors.shape[1]:
        raise ValueError("用户向量与电影向量的维度不相同")
    movie_embedding = user_tower.get_layer("movie_embedding")
    expected_vectors = tf.math.l2_normalize(movie_embedding.embeddings[2:], axis=-1).numpy()
    if expected_vectors.shape != item_vectors.shape or not np.allclose(
        item_vectors, expected_vectors, atol=1e-6
    ):
        raise ValueError("导出的电影向量与用户模型中的 movie_embedding 不相同")
    return RetrievalArtifacts(user_tower, encoders, movie_ids, item_vectors, max_history, report)


def top_k_recall(
    user_ids: np.ndarray,
    user_vectors: np.ndarray,
    movie_ids: np.ndarray,
    item_vectors: np.ndarray,
    seen_movie_ids: list[list[int]],
    k: int = 20,
) -> pd.DataFrame:
    """计算全部候选电影的内积，排除已看电影后，按得分返回前 k 个。"""
    _validate_catalog(movie_ids, item_vectors)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(movie_ids):
        raise ValueError("k 必须是正整数，且不能超过候选电影数量")
    if user_ids.ndim != 1 or not len(user_ids) or len(np.unique(user_ids)) != len(user_ids):
        raise ValueError("user_ids 必须是非空的一维数组，用户编号不能重复")
    if not np.issubdtype(user_ids.dtype, np.integer) or (user_ids < 1).any():
        raise ValueError("user_ids 必须包含正整数编号")
    if user_vectors.shape != (len(user_ids), item_vectors.shape[1]):
        raise ValueError("用户向量数量或维度不符合输入要求")
    if not np.isfinite(user_vectors).all():
        raise ValueError("用户向量存在无效数值")
    if not np.allclose(np.linalg.norm(user_vectors, axis=1), 1, atol=1e-5):
        raise ValueError("用户向量的 L2 长度应当为 1")
    if len(seen_movie_ids) != len(user_ids):
        raise ValueError("每个用户都需要提供一份已看电影列表")
    if any(not isinstance(seen, (list, tuple)) for seen in seen_movie_ids):
        raise TypeError("已看电影必须保存为列表或 tuple")

    # 行对应用户，列对应 movie_ids；已归一化向量的内积等于余弦相似度。
    scores = user_vectors @ item_vectors.T
    rows = []
    for position, user_id in enumerate(user_ids):
        available = np.flatnonzero(~np.isin(movie_ids, seen_movie_ids[position]))
        # 得分相同时按电影编号升序排列，保证结果顺序可以重复验证。
        order = np.lexsort((movie_ids[available], -scores[position, available]))
        selected = available[order[:k]]
        for rank, item_position in enumerate(selected, start=1):
            rows.append({
                "user_id": int(user_id), "movie_id": int(movie_ids[item_position]),
                "source": "youtubednn", "source_score": float(scores[position, item_position]),
                "source_rank": rank,
            })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def recall_users(
    contexts: pd.DataFrame,
    artifacts: RetrievalArtifacts,
    k: int = 20,
    batch_size: int = 128,
) -> pd.DataFrame:
    """使用已有编码器与用户模型分批召回，控制得分矩阵的大小。"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    if contexts.empty or contexts["user_id"].duplicated().any():
        raise ValueError("contexts 不能为空，每个用户只能出现一次")
    results = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts.iloc[start:start + batch_size]
        features = build_user_inputs(batch, artifacts.encoders, artifacts.max_history)
        user_vectors = artifacts.user_tower(features, training=False).numpy()
        results.append(top_k_recall(
            batch["user_id"].to_numpy(), user_vectors,
            artifacts.movie_ids, artifacts.item_vectors,
            batch["seen_movie_ids"].tolist(), k,
        ))
    return pd.concat(results, ignore_index=True)
