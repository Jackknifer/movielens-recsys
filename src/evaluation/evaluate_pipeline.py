import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from pipeline.artifact_store import load_bundle, require_empty_directory, write_json, write_result_csv
from evaluation.evaluate_retrieval import _file_sha256
from evaluation.evaluation_metrics import binary_metrics, diversity_metrics, source_statistics, top_k_metrics
from ranking.ranking_predict import predict_records
from training.ranking_samples import build_observed_samples
from features.ranking_features import build_context_features, with_recall_metadata
from pipeline.run_pipeline import PROJECT_ROOT, recommend_from_bundle, save_recommendation_run
from training.train_ranking import predict_inputs
from training.ranking_inputs import unknown_rates
from reranking.diversity_rerank import rerank_candidates


def recommend_from_snapshot(bundle, dataset, split: str) -> dict:
    with (dataset.directory / "reference-bundle" / "manifest.json").open(encoding="utf-8") as source:
        reference = json.load(source)
    expected = {name: digest for name, digest in reference["files"].items() if name.startswith(("retrieval/", "indexes/"))}
    if any(bundle.manifest["files"].get(name) != digest for name, digest in expected.items()):
        raise ValueError("评估模型的召回与固定候选来源不一致")
    contexts = dataset.frame(f"{split}-contexts")
    candidates = dataset.frame(f"{split}-candidates").sort_values(["user_id", "candidate_rank"], kind="stable").reset_index(drop=True)
    records = dataset.frame(f"{split}-records")
    pd.testing.assert_frame_equal(candidates[["user_id", "movie_id", "candidate_rank"]], records)
    inputs = dataset.inputs(split)
    ranked = candidates.copy()
    ranked["ranking_score"] = predict_inputs(tf.function(bundle.ranking.model, reduce_retracing=True), inputs)
    ranked = ranked.sort_values(["user_id", "ranking_score", "movie_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    ranked["rank"] = ranked.groupby("user_id").cumcount() + 1
    config = bundle.config
    reranked = rerank_candidates(ranked, bundle.movies, config["top_k"], config["max_genre_run"], config["max_decade_run"])
    final = reranked.loc[reranked["rank"] <= config["top_k"]].merge(contexts[["user_id", "is_cold_start", "as_of_timestamp", "user_group"]], on="user_id", validate="many_to_one")
    channels = {name: dataset.frame(f"{split}-channel-{name}") for name in ("youtubednn", "item_similarity", "genre_interest", "popular")}
    return {"contexts": contexts, "channels": channels, "merged": candidates, "ranked": ranked, "reranked": reranked, "final": final, "unknown_rates": unknown_rates(inputs)}


def run_pipeline_evaluation(
    bundle_dir: Path, output_dir: Path, split: str = "validation", ks: tuple[int, ...] = (5, 10, 20),
    include_validation_history: bool = False, project_root: Path = PROJECT_ROOT,
    *, ranking_dataset_dir: Path | None = None, save_details: bool = True,
) -> dict:
    """固定模型与候选预算，分别评估召回、排序和展示列表。"""
    if split not in {"validation", "test"} or (include_validation_history and split != "test"):
        raise ValueError("验证历史只能加入 test 评估")
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks 必须包含正整数")
    output_dir = require_empty_directory(output_dir)
    bundle = load_bundle(bundle_dir)
    if max(ks) > bundle.config["top_k"]:
        raise ValueError("评估的 k 不能超过展示列表长度")
    for filename, expected in bundle.manifest["source_sha256"].items():
        if _file_sha256(Path(filename)) != expected:
            raise ValueError(f"数据版本与训练时不一致：{filename}")
    split_dir = Path(project_root) / "data" / "processed" / "split"
    target_path = split_dir / f"{split}.pkl"
    heldout = pd.read_pickle(target_path)
    selected_users = bundle.history["user_id"].unique()
    heldout = heldout.loc[heldout["user_id"].isin(selected_users)].sort_values("user_id").copy()
    if heldout.empty or heldout["user_id"].duplicated().any():
        raise ValueError("当前评估需要每个训练用户最多一个保留目标")
    queries = heldout[["user_id", "timestamp"]].rename(columns={"timestamp": "as_of_timestamp"})
    history = bundle.history
    extra_files = [target_path]
    if include_validation_history:
        validation_path = split_dir / "validation.pkl"
        validation = pd.read_pickle(validation_path)
        validation = validation.loc[validation["user_id"].isin(selected_users)]
        history = pd.concat([history, validation], ignore_index=True)
        extra_files.append(validation_path)
    # 推荐函数只接收用户与时间；标签在返回候选之后进入评估。
    dataset = None
    if ranking_dataset_dir is None:
        result = recommend_from_bundle(bundle, queries, history)
    else:
        from training.ranking_dataset import load_ranking_dataset
        if include_validation_history:
            raise ValueError("固定数据快照使用已经记录的推荐历史")
        dataset = load_ranking_dataset(ranking_dataset_dir)
        pd.testing.assert_frame_equal(heldout.reset_index(drop=True), dataset.frame(f"{split}-heldout").reset_index(drop=True))
        result = recommend_from_snapshot(bundle, dataset, split)
    positive = heldout.loc[heldout["rating"] >= bundle.config["positive_threshold"]]
    contexts, movie_ids = result["contexts"], bundle.retrieval.movie_ids
    evaluation_ks = sorted(set(ks) | {bundle.config["candidate_budget"]})
    stages = {}
    stage_frames = {
        "merged": result["merged"].rename(columns={"candidate_rank": "rank"}),
        "ranked": result["ranked"], "reranked": result["reranked"],
    }
    for name, frame in stage_frames.items():
        stages[name] = {
            "top_k": [top_k_metrics(contexts, positive, frame, movie_ids, k) for k in evaluation_ks],
            "diversity": [diversity_metrics(frame, bundle.movies, bundle.popularity, movie_ids, bundle.retrieval.item_vectors, k) for k in sorted(set(ks))],
        }
    single_channels = {}
    for name, frame in result["channels"].items():
        single_channels[name] = [top_k_metrics(contexts, positive, frame.rename(columns={"source_rank": "rank"}), movie_ids, k) for k in evaluation_ks]
    groups = {}
    for name in ("no_history", "short_history", "unseen_user", "warm"):
        group_contexts = contexts.loc[contexts["user_group"] == name]
        group_targets = positive.loc[positive["user_id"].isin(group_contexts["user_id"])]
        groups[name] = {
            "users": len(group_contexts), "positive_target_users": len(group_targets),
            "stages": {stage: [top_k_metrics(group_contexts, group_targets, frame, movie_ids, k) for k in sorted(set(ks))] for stage, frame in stage_frames.items()},
        }
    observed = build_observed_samples(
        heldout, bundle.users, bundle.movies, bundle.config["positive_threshold"], bundle.config["negative_threshold"],
    )
    binary_unknown = None
    if len(observed):
        if dataset is not None:
            observed = dataset.frame(f"{split}-observed-records").copy()
            observed_inputs = dataset.inputs(f"{split}-observed")
            observed["ranking_score"] = predict_inputs(tf.function(bundle.ranking.model, reduce_retracing=True), observed_inputs)
            binary_unknown = unknown_rates(observed_inputs)
        elif bundle.ranking.numeric_scaler is not None:
            observed = with_recall_metadata(observed, result["merged"], result["channels"])
            observed = build_context_features(observed, contexts, bundle.movies, bundle.retrieval, bundle.popularity)
            observed["ranking_score"], binary_unknown = predict_records(observed, bundle.ranking, bundle.config["batch_size"])
        else:
            observed["ranking_score"], binary_unknown = predict_records(observed, bundle.ranking, bundle.config["batch_size"])
    else:
        observed["ranking_score"] = pd.Series(dtype=float)
    observed_metrics = binary_metrics(observed, probability_scores=bundle.ranking.numeric_scaler is None)
    report = {
        "split": split, "split_policy": bundle.manifest["split_policy"],
        "include_validation_history": include_validation_history,
        "evaluated_users": len(contexts), "positive_target_users": len(positive),
        "candidate_catalog": "all_exported_known_movie_ids", "catalog_size": len(movie_ids),
        "per_channel_budget": bundle.config["candidate_budget"], "merged_budget": bundle.config["candidate_budget"],
        "config": bundle.config, "stages": stages, "single_channels": single_channels,
        "ranking_model": bundle.ranking.training_report.get("model_type", "deepfm"),
        "groups": groups, "source_statistics": source_statistics(result["channels"], result["merged"], result["final"], positive),
        "binary_metrics": observed_metrics, "binary_population": "observed_high_or_low_heldout_ratings",
        "ranking_score_semantics": bundle.ranking.training_report.get("score_semantics", "predicted_sample_label_probability"),
        "binary_unknown_rates": binary_unknown, "ranking_unknown_rates": result["unknown_rates"],
        "input_sha256": {str(path.resolve()): _file_sha256(path) for path in extra_files + [Path(bundle_dir) / "manifest.json"]},
    }
    if dataset is not None:
        report["ranking_dataset_sha256"] = _file_sha256(dataset.directory / "manifest.json")
    if save_details:
        save_recommendation_run(result, output_dir, bundle, bundle.config["top_k"])
    else:
        output_dir.mkdir(parents=True)
        for name in ("ranked", "reranked"):
            write_result_csv(output_dir / f"{name}.csv.gz", result[name][["user_id", "movie_id", "candidate_rank", "ranking_score", "rank"]])
    write_result_csv(output_dir / "observed_predictions.csv.gz", observed)
    write_json(output_dir / "evaluation_report.json", report)
    return report
