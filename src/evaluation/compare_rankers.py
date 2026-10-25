import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate_retrieval import _file_sha256
from pipeline.artifact_store import require_empty_directory, write_json
from training.ranking_dataset import load_ranking_dataset


def per_user_metrics(records: pd.DataFrame, targets: pd.DataFrame, scored: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    keys = ["user_id", "movie_id", "candidate_rank"]
    if scored.duplicated(["user_id", "movie_id"]).any() or not np.isfinite(scored["ranking_score"]).all():
        raise ValueError("比较结果包含重复候选或无效得分")
    actual = scored.sort_values(["user_id", "candidate_rank"], kind="stable")[keys].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, records[keys].reset_index(drop=True))
    ranks = scored.sort_values(["user_id", "rank"], kind="stable")
    expected_rank = ranks.groupby("user_id").cumcount().to_numpy() + 1
    if not np.array_equal(expected_rank, ranks["rank"].to_numpy()):
        raise ValueError("比较结果的排名不连续")
    matched = targets[["user_id", "movie_id"]].merge(scored[["user_id", "movie_id", "rank"]], on=["user_id", "movie_id"], how="left", validate="one_to_one").sort_values("user_id")
    positions = matched["rank"].fillna(np.inf).to_numpy(dtype=float)
    hit = positions <= k
    return pd.DataFrame({
        "user_id": matched["user_id"].to_numpy(), f"recall_at_{k}": hit.astype(float),
        f"ndcg_at_{k}": np.where(hit, 1 / np.log2(positions + 1), 0.0),
        f"mrr_at_{k}": np.where(hit, 1 / positions, 0.0),
    })


def paired_bootstrap(differences: np.ndarray, repetitions: int = 10000, seed: int = 42) -> dict:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or repetitions < 1:
        raise ValueError("成对重采样需要非空、有限的一维差值")
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        indexes = generator.integers(0, len(values), size=(count, len(values)))
        estimates[start:start + count] = values[indexes].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"difference": float(values.mean()), "ci95": [float(lower), float(upper)], "users": len(values), "bootstrap_repetitions": repetitions}


def summarize_experiment(experiment_dir: Path, dataset_dir: Path, output_dir: Path) -> dict:
    experiment_dir = Path(experiment_dir)
    output_dir = require_empty_directory(output_dir)
    dataset = load_ranking_dataset(dataset_dir)
    with (experiment_dir / "experiment.json").open(encoding="utf-8") as source:
        experiment = json.load(source)
    if experiment["dataset_sha256"] != _file_sha256(dataset.directory / "manifest.json"):
        raise ValueError("实验与排序数据快照摘要不一致")
    records, targets = dataset.frame("test-records"), dataset.frame("test-targets")
    all_rows, user_frames, fingerprints = [], {}, {}
    output_dir.mkdir(parents=True)
    for job in experiment["final_runs"]:
        directory = experiment_dir / "runs" / job["run_id"]
        with (directory / "ranking" / "training_report.json").open(encoding="utf-8") as source:
            training = json.load(source)
        with (directory / "evaluation" / "evaluation_report.json").open(encoding="utf-8") as source:
            evaluation = json.load(source)
        if evaluation["ranking_dataset_sha256"] != experiment["dataset_sha256"]:
            raise ValueError("模型评估使用了不同的排序候选")
        frame = pd.read_csv(directory / "evaluation" / "ranked.csv.gz")
        metrics = per_user_metrics(records, targets, frame)
        measured = next(item for item in evaluation["stages"]["ranked"]["top_k"] if item["k"] == 20)
        for name in ("recall_at_20", "ndcg_at_20", "mrr_at_20"):
            np.testing.assert_allclose(metrics[name].mean(), measured[name.replace("_20", "_k")], rtol=1e-12, atol=1e-12)
        variant, seed = job["variant"], job["seed"]
        fingerprint = (seed, training["initial_embedding_sha256"], tuple(row["query_order_sha256"] for row in training["history"]))
        fingerprints.setdefault(seed, []).append(fingerprint)
        metrics.to_csv(output_dir / f"{variant}-seed-{seed}-users.csv", index=False)
        user_frames[(variant, seed)] = metrics.set_index("user_id")
        reranked = next(item for item in evaluation["stages"]["reranked"]["top_k"] if item["k"] == 20)
        diversity = next(item for item in evaluation["stages"]["reranked"]["diversity"] if item["k"] == 20)
        all_rows.append({
            "variant": variant, "seed": seed, "learning_rate": training["learning_rate"],
            "recall_at_20": measured["recall_at_k"], "ndcg_at_20": measured["ndcg_at_k"], "mrr_at_20": measured["mrr_at_k"], "hits": measured["hits"],
            "reranked_recall_at_20": reranked["recall_at_k"], "reranked_ndcg_at_20": reranked["ndcg_at_k"],
            "catalog_coverage": diversity["catalog_coverage"], "intra_list_diversity": diversity["intra_list_diversity"],
            "auc": evaluation["binary_metrics"]["auc"], "parameter_count": training["parameter_count"],
            "non_embedding_parameters": training["non_embedding_parameters"], "selected_epoch": training["selected_epoch"],
            "matrix_macs_per_candidate": training["matrix_macs_per_candidate"],
            "executed_epochs": training["executed_epochs"], "training_seconds": training["elapsed_seconds"],
            "peak_rss_mib": training["peak_rss_mib"], "model_bytes": training["model_bytes"],
        })
    for entries in fingerprints.values():
        if len({entry[1] for entry in entries}) != 1:
            raise ValueError("相同种子的模型没有使用相同初始 Embedding")
        for epoch in range(min(len(entry[2]) for entry in entries)):
            if len({entry[2][epoch] for entry in entries}) != 1:
                raise ValueError("相同种子的训练查询顺序不一致")
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_dir / "runs.csv", index=False)
    summaries = []
    for variant, group in frame.groupby("variant", sort=False):
        if sorted(group["seed"].tolist()) != sorted(experiment["protocol"]["seeds"]):
            raise ValueError("每种模型必须包含全部规定的随机种子")
        summary = {"variant": variant, "runs": len(group)}
        for name in frame.select_dtypes(include="number").columns:
            if name != "seed":
                summary[f"{name}_mean"] = float(group[name].mean())
                summary[f"{name}_std"] = float(group[name].std(ddof=1))
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False)
    comparisons = {}
    seeds = experiment["protocol"]["seeds"]
    for baseline in frame["variant"].unique():
        if baseline == "rankmixer":
            continue
        details = {}
        for metric in ("ndcg_at_20", "recall_at_20", "mrr_at_20"):
            differences = []
            for seed in seeds:
                first, second = user_frames[("rankmixer", seed)], user_frames[(baseline, seed)]
                pd.testing.assert_index_equal(first.index, second.index)
                differences.append((first[metric] - second[metric]).to_numpy())
            details[metric] = paired_bootstrap(np.mean(differences, axis=0))
        details["hit_changes"] = [{
            "seed": seed,
            "gained_users": int(((user_frames[("rankmixer", seed)]["recall_at_20"] == 1) & (user_frames[(baseline, seed)]["recall_at_20"] == 0)).sum()),
            "lost_users": int(((user_frames[("rankmixer", seed)]["recall_at_20"] == 0) & (user_frames[(baseline, seed)]["recall_at_20"] == 1)).sum()),
        } for seed in seeds]
        comparisons[f"rankmixer_vs_{baseline}"] = details
    report = {
        "dataset_sha256": experiment["dataset_sha256"], "protocol": experiment["protocol"],
        "evaluated_users": len(targets), "candidate_budget": 100, "display_k": 20,
        "summaries": summaries, "paired_comparisons": comparisons,
        "bootstrap_population": "per_user_difference_averaged_over_fixed_training_seeds",
        "candidate_sets_verified": True, "initial_embeddings_verified": True, "query_orders_verified": True,
    }
    write_json(output_dir / "comparison.json", report)
    return report
