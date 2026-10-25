import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tensorflow_recommenders.layers.feature_interaction import Cross

from models.deepfm import FactorizationMachine, LinearSum
from models.rankmixer import MODEL_IMPLEMENTATION_VERSION, NumericColumns, TokenMixing, validate_rankmixer_model
from training.ranking_inputs import RANKING_FEATURE_COLUMNS, build_ranking_inputs, unknown_rates
from features.ranking_features import FEATURE_VERSION, RANKING_NUMERIC_COLUMNS, build_context_features
from features.feature_join import join_features


@dataclass(frozen=True)
class RankingArtifacts:
    model: tf.keras.Model
    encoders: dict[str, OrdinalEncoder]
    training_report: dict
    numeric_scaler: StandardScaler | None = None


def load_ranking_artifacts(directory: Path) -> RankingArtifacts:
    directory = Path(directory)
    with (directory / "training_report.json").open(encoding="utf-8") as source:
        report = json.load(source)
    with (directory / "encoders.pkl").open("rb") as source:
        encoders = pickle.load(source)
    if set(encoders) != set(RANKING_FEATURE_COLUMNS) or report["feature_columns"] != RANKING_FEATURE_COLUMNS:
        raise ValueError("排序产物的字段不符合输入要求")
    feature_version = report.get("feature_version", "categorical_v1")
    if feature_version not in {"categorical_v1", FEATURE_VERSION}:
        raise ValueError("无法读取当前排序特征版本")
    scaler = None
    if feature_version == FEATURE_VERSION:
        with (directory / "numeric_scaler.pkl").open("rb") as source:
            scaler = pickle.load(source)
        if not isinstance(scaler, StandardScaler) or scaler.feature_names_in_.tolist() != RANKING_NUMERIC_COLUMNS or report["numeric_columns"] != RANKING_NUMERIC_COLUMNS:
            raise ValueError("排序数值特征与训练报告不一致")
        if not np.array_equal(scaler.mean_, report["numeric_mean"]) or not np.array_equal(scaler.scale_, report["numeric_scale"]):
            raise ValueError("排序标准化参数与训练报告不一致")
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
            raise ValueError("排序标准化参数无效")
    model = tf.keras.models.load_model(
        directory / "ranker.keras", compile=False,
        custom_objects={"FactorizationMachine": FactorizationMachine, "LinearSum": LinearSum, "Cross": Cross, "NumericColumns": NumericColumns, "TokenMixing": TokenMixing},
    )
    model_type = report.get("model_type", "deepfm")
    if model_type not in {"deepfm", "dcnv2", "rankmixer"} or model.name != model_type:
        raise ValueError("排序模型类型与训练报告不一致")
    if model_type == "dcnv2":
        cross_layers = [layer for layer in model.layers if isinstance(layer, Cross)]
        if report["architecture"] != "parallel" or report["cross_projection_dim"] is not None:
            raise ValueError("DCN-V2 需要使用并行结构和完整矩阵交叉层")
        if len(cross_layers) != report["cross_layers"] or any(layer.get_config()["projection_dim"] is not None for layer in cross_layers):
            raise ValueError("DCN-V2 的交叉层与训练报告不一致")
    if model_type == "rankmixer":
        if scaler is None or report["model_implementation_version"] != MODEL_IMPLEMENTATION_VERSION or report["score_semantics"] != "uncalibrated_ranking_logit":
            raise ValueError("RankMixer 版本、特征或得分含义无效")
        validate_rankmixer_model(model, report["model_config"], report["token_groups"])
    expected_inputs = set(RANKING_FEATURE_COLUMNS) | ({"context_features"} if scaler is not None else set())
    if set(model.input_names) != expected_inputs or model.output_shape[1:] != (1,):
        raise ValueError("排序模型的输入或输出形状无效")
    for tensor in model.inputs:
        width = len(RANKING_NUMERIC_COLUMNS) if tensor.name.split(":")[0] == "context_features" else 1
        if tensor.shape.as_list()[1:] != [width]:
            raise ValueError("排序模型的字段长度不符合输入要求")
    for name, encoder in encoders.items():
        if not isinstance(encoder, OrdinalEncoder) or encoder.categories_[0].tolist() != report["categories"][name]:
            raise ValueError(f"{name} 的词表与训练报告不一致")
        if len(encoder.categories_[0]) + 2 != report["vocab_sizes"][name]:
            raise ValueError(f"{name} 的词表长度无效")
        layer = model.get_layer(f"{name}_embedding")
        if layer.input_dim != report["vocab_sizes"][name] or layer.output_dim != report["emb_dim"]:
            raise ValueError(f"{name} 的模型向量形状与报告不一致")
    if any(not np.isfinite(weight.numpy()).all() for weight in model.weights):
        raise ValueError("排序模型的参数存在无效数值")
    return RankingArtifacts(model, encoders, report, scaler)


def predict_records(records: pd.DataFrame, artifacts: RankingArtifacts, batch_size: int = 128) -> tuple[np.ndarray, dict]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    features = build_ranking_inputs(records, artifacts.encoders, artifacts.numeric_scaler)
    scores = []
    for start in range(0, len(records), batch_size):
        batch = {name: values[start:start + batch_size] for name, values in features.items()}
        scores.append(artifacts.model(batch, training=False).numpy().reshape(-1))
    scores = np.concatenate(scores)
    if not np.isfinite(scores).all():
        raise ValueError("ranking_score 必须是有限数值")
    if artifacts.numeric_scaler is None and ((scores < 0) | (scores > 1)).any():
        raise ValueError("二分类模型的 ranking_score 必须在 0 到 1 之间")
    return scores, unknown_rates(features)


def score_candidates(
    candidates: pd.DataFrame, artifacts: RankingArtifacts, users: pd.DataFrame,
    movies: pd.DataFrame, batch_size: int = 128, *, contexts: pd.DataFrame | None = None,
    retrieval=None, popularity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    if candidates.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("排序候选存在重复电影")
    results = candidates.copy()
    if results.empty:
        results["ranking_score"] = pd.Series(dtype=float)
        results["rank"] = pd.Series(dtype=np.int64)
        return results, {name: None for name in RANKING_FEATURE_COLUMNS}
    records = join_features(candidates, users, movies)
    if artifacts.numeric_scaler is not None:
        if contexts is None or retrieval is None or popularity is None:
            raise ValueError("上下文排序需要用户历史、召回向量和热门统计")
        records = build_context_features(records, contexts, movies, retrieval, popularity)
    results["ranking_score"], rates = predict_records(records, artifacts, batch_size)
    results = results.sort_values(["user_id", "ranking_score", "movie_id"], ascending=[True, False, True], kind="stable")
    results["rank"] = results.groupby("user_id").cumcount() + 1
    return results.reset_index(drop=True), rates
