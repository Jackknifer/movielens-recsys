import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from data.load import validate_movielens, MovieLensData
from evaluation.evaluate_retrieval import _file_sha256
from features.ranking_features import FEATURE_VERSION, RANKING_NUMERIC_COLUMNS, build_context_features, with_recall_metadata
from pipeline.artifact_store import load_bundle, require_empty_directory, write_json
from training.ranking_candidates import build_recall_state, candidate_records, generate_candidates
from training.ranking_inputs import RANKING_FEATURE_COLUMNS, build_ranking_inputs
from training.ranking_samples import build_observed_samples


DATASET_VERSION = "ranking_dataset_v1"


@dataclass(frozen=True)
class RankingDataset:
    directory: Path
    manifest: dict
    encoders: dict
    scaler: object

    def frame(self, name: str) -> pd.DataFrame:
        filename = f"{name}.pkl"
        if filename not in self.manifest["files"]:
            raise ValueError(f"数据快照中没有记录：{name}")
        return pd.read_pickle(self.directory / filename)

    def inputs(self, split: str) -> dict[str, np.ndarray]:
        filename = f"{split}-inputs.npz"
        if filename not in self.manifest["files"]:
            raise ValueError(f"数据快照中没有记录输入：{split}")
        with np.load(self.directory / filename, allow_pickle=False) as source:
            result = {name: source[name] for name in source.files}
        expected = set(RANKING_FEATURE_COLUMNS) | {"context_features"}
        if set(result) != expected:
            raise ValueError("数据快照的输入字段无效")
        count = len(self.frame(f"{split}-records"))
        for name, values in result.items():
            width = len(RANKING_NUMERIC_COLUMNS) if name == "context_features" else 1
            dtype = np.dtype("float32") if name == "context_features" else np.dtype("int32")
            if values.shape != (count, width) or values.dtype != dtype or not np.isfinite(values).all():
                raise ValueError(f"数据快照的输入形状或数值无效：{name}")
            values.setflags(write=False)
        return result

    def prepared_training(self) -> dict:
        return {
            "encoders": self.encoders, "scaler": self.scaler,
            "train_features": self.inputs("train"), "validation_features": self.inputs("validation"),
            "training_keys": self.frame("train-records")[["user_id", "movie_id", "candidate_rank"]],
        }


def load_ranking_dataset(directory: Path) -> RankingDataset:
    directory = Path(directory)
    with (directory / "manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest["dataset_version"] != DATASET_VERSION or manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError("无法读取当前排序数据快照版本")
    if manifest["feature_columns"] != RANKING_FEATURE_COLUMNS or manifest["numeric_columns"] != RANKING_NUMERIC_COLUMNS:
        raise ValueError("数据快照的特征顺序与当前代码不一致")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file() and path != directory / "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("数据快照文件集合与 Manifest 不一致")
    for filename, digest in manifest["files"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("数据快照的文件路径无效")
        if _file_sha256(directory / relative) != digest:
            raise ValueError(f"数据快照文件摘要不一致：{filename}")
    with (directory / "encoders.pkl").open("rb") as source:
        encoders = pickle.load(source)
    with (directory / "numeric_scaler.pkl").open("rb") as source:
        scaler = pickle.load(source)
    if list(encoders) != RANKING_FEATURE_COLUMNS or scaler.feature_names_in_.tolist() != RANKING_NUMERIC_COLUMNS:
        raise ValueError("数据快照的编码器或标准化字段无效")
    return RankingDataset(directory, manifest, encoders, scaler)


def _save_inputs(directory: Path, name: str, records: pd.DataFrame, encoders: dict, scaler) -> None:
    records = records.sort_values(["user_id", "candidate_rank"], kind="stable").reset_index(drop=True)
    features = build_ranking_inputs(records, encoders, scaler)
    np.savez(directory / f"{name}-inputs.npz", **features)
    columns = ["user_id", "movie_id", "candidate_rank"]
    if "label" in records:
        columns += ["label", "sample_kind"]
    records[columns].to_pickle(directory / f"{name}-records.pkl")


def export_ranking_dataset(source_dir: Path, output_dir: Path, project_root: Path) -> dict:
    source_dir, project_root = Path(source_dir), Path(project_root)
    output_dir = require_empty_directory(output_dir)
    bundle = load_bundle(source_dir)
    if bundle.ranking.numeric_scaler is None:
        raise ValueError("固定排序数据需要带候选上下文的完整模型")
    for filename, expected in bundle.manifest["source_sha256"].items():
        if _file_sha256(Path(filename)) != expected:
            raise ValueError(f"来源数据摘要不一致：{filename}")
    output_dir.mkdir(parents=True)
    encoders, scaler = bundle.ranking.encoders, bundle.ranking.numeric_scaler
    for filename, value in (("encoders.pkl", encoders), ("numeric_scaler.pkl", scaler)):
        with (output_dir / filename).open("wb") as destination:
            pickle.dump(value, destination)

    # 使用已经保存的前段召回模型恢复排序训练候选。
    training_dir = source_dir / "ranking" / "training_context"
    history = pd.read_pickle(training_dir / "history.pkl")
    targets = pd.read_pickle(training_dir / "targets.pkl")
    queries = pd.read_pickle(training_dir / "queries.pkl")
    validate_movielens(MovieLensData(bundle.users, bundle.movies, history))
    state = build_recall_state(training_dir / "retrieval", bundle.users, bundle.movies, history, bundle.config)
    result = generate_candidates(state, queries)
    records = candidate_records(result, state)
    keys = ["user_id", "movie_id"]
    positive = pd.MultiIndex.from_frame(targets.loc[targets["rating"] >= bundle.config["positive_threshold"], keys])
    negative = pd.MultiIndex.from_frame(targets.loc[targets["rating"] <= bundle.config["negative_threshold"], keys])
    index = pd.MultiIndex.from_frame(records[keys])
    records["label"] = index.isin(positive).astype(np.int32)
    records["sample_kind"] = np.where(records["label"] == 1, "window_positive", np.where(index.isin(negative), "window_low_rating", "window_unobserved"))
    eligible = records.groupby("user_id")["label"].sum()
    records = records.loc[records["user_id"].isin(eligible.index[eligible > 0])].reset_index(drop=True)
    report = bundle.ranking.training_report
    if len(records) != report["samples"] or records["user_id"].nunique() != report["users"] or records["sample_kind"].value_counts().to_dict() != report["sample_kinds"]:
        raise ValueError("恢复的训练候选数量与原报告不一致")
    original_sample = pd.read_pickle(source_dir / "ranking" / "training_sample.pkl")
    pd.testing.assert_frame_equal(records.iloc[:len(original_sample)].reset_index(drop=True), original_sample.reset_index(drop=True), check_exact=False, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(records[RANKING_NUMERIC_COLUMNS].mean().to_numpy(), scaler.mean_, rtol=1e-6, atol=1e-7)
    _save_inputs(output_dir, "train", records, encoders, scaler)
    training_summary = {"samples": len(records), "users": int(records["user_id"].nunique()), "excluded_queries": int((eligible == 0).sum()), "sample_kinds": report["sample_kinds"]}
    print(json.dumps({"stage": "training_snapshot", **training_summary}, ensure_ascii=False), flush=True)
    del records, result, state

    source_files = dict(bundle.manifest["source_sha256"])
    split_summary = {}
    for split in ("validation", "test"):
        target_path = project_root / "data" / "processed" / "split" / f"{split}.pkl"
        heldout = pd.read_pickle(target_path)
        heldout = heldout.loc[heldout["user_id"].isin(bundle.history["user_id"].unique())].sort_values("user_id").copy()
        queries = heldout[["user_id", "timestamp"]].rename(columns={"timestamp": "as_of_timestamp"})
        result = generate_candidates(bundle, queries)
        records = candidate_records(result, bundle)
        _save_inputs(output_dir, split, records, encoders, scaler)
        result["contexts"].to_pickle(output_dir / f"{split}-contexts.pkl")
        result["merged"].to_pickle(output_dir / f"{split}-candidates.pkl")
        heldout.to_pickle(output_dir / f"{split}-heldout.pkl")
        heldout.loc[heldout["rating"] >= bundle.config["positive_threshold"]].to_pickle(output_dir / f"{split}-targets.pkl")
        for name, frame in result["channels"].items():
            frame.to_pickle(output_dir / f"{split}-channel-{name}.pkl")
        observed = build_observed_samples(heldout, bundle.users, bundle.movies, bundle.config["positive_threshold"], bundle.config["negative_threshold"])
        observed = with_recall_metadata(observed, result["merged"], result["channels"])
        observed = build_context_features(observed, result["contexts"], bundle.movies, bundle.retrieval, bundle.popularity)
        _save_inputs(output_dir, f"{split}-observed", observed, encoders, scaler)
        source_files[str(target_path.resolve())] = _file_sha256(target_path)
        split_summary[split] = {"users": len(queries), "candidates": len(records), "positive_target_users": int((heldout["rating"] >= bundle.config["positive_threshold"]).sum())}
        print(json.dumps({"stage": f"{split}_snapshot", **split_summary[split]}, ensure_ascii=False), flush=True)
        del records, result, observed

    # 独立保存原排序权重与召回文件，供新格式读取和比较。
    reference_dir = output_dir / "reference-bundle"
    shutil.copytree(source_dir, reference_dir)
    reference_manifest = dict(bundle.manifest)
    reference_manifest.update(
        schema_version=3,
        runtime_compatibility={"feature_version": FEATURE_VERSION, "model_implementation_version": 1, "tensorflow_major_minor": ".".join(tf.__version__.split(".")[:2])},
        source_bundle={"manifest_sha256": _file_sha256(source_dir / "manifest.json"), "path": str(source_dir.resolve())},
    )
    write_json(reference_dir / "manifest.json", reference_manifest)
    manifest = {
        "dataset_version": DATASET_VERSION, "feature_version": FEATURE_VERSION,
        "feature_columns": RANKING_FEATURE_COLUMNS, "numeric_columns": RANKING_NUMERIC_COLUMNS,
        "source_bundle_manifest_sha256": _file_sha256(source_dir / "manifest.json"),
        "source_code_sha256": bundle.manifest["code_sha256"], "source_sha256": source_files,
        "training": training_summary, "splits": split_summary, "include_validation_history": False,
        "files": {str(path.relative_to(output_dir)): _file_sha256(path) for path in sorted(output_dir.rglob("*")) if path.is_file()},
    }
    write_json(output_dir / "manifest.json", manifest)
    load_ranking_dataset(output_dir)
    return manifest
