import argparse
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.artifact_store import export_bundle, load_bundle, load_retrieval_source, require_empty_directory, write_json, write_result_csv
from reranking.diversity_rerank import rerank_candidates
from evaluation.evaluate_retrieval import _file_sha256
from ranking.ranking_predict import score_candidates
from training.ranking_candidates import build_candidate_training, build_recall_state, candidate_records, generate_candidates
from training.train_ranking import train_ranking
from training.train_retrieval import train_retrieval
from data.load import load_movielens
from training.retrieval_samples import build_retrieval_samples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEYS = {
    "max_users", "positive_threshold", "negative_threshold", "negatives_per_positive", "max_history",
    "min_history", "retrieval_emb_dim", "num_sampled", "temperature", "retrieval_epochs", "ranking_emb_dim",
    "ranking_hidden_units", "ranking_cross_layers", "ranking_dropout", "ranking_l2", "ranking_epochs", "batch_size", "learning_rate",
    "seed", "candidate_budget", "neighbors_per_item", "recent_items", "top_k", "max_genre_run", "max_decade_run",
    "ranking_history_fraction", "ranking_query_batch_size", "ranking_patience",
}


def validate_config(config: dict) -> None:
    optional = {"ranking_model", "ranking_model_config", "retrieval_seed", "retrieval_learning_rate"}
    if not isinstance(config, dict) or not CONFIG_KEYS.issubset(config) or set(config) - CONFIG_KEYS - optional:
        raise ValueError("配置需要包含 offline.yaml 中的全部字段")
    model_type = config.get("ranking_model", "dcnv2")
    if model_type not in {"dcnv2", "rankmixer"}:
        raise ValueError("排序模型类型无效")
    if model_type == "rankmixer":
        from models.rankmixer import validate_rankmixer_config
        validate_rankmixer_config(config.get("ranking_model_config"))
    elif config.get("ranking_model_config"):
        raise ValueError("DCNv2 使用隐藏层和 Cross 配置")
    if "retrieval_seed" in config and (isinstance(config["retrieval_seed"], bool) or not isinstance(config["retrieval_seed"], int) or config["retrieval_seed"] < 0):
        raise ValueError("retrieval_seed 必须是非负整数")
    if "retrieval_learning_rate" in config and (not np.isfinite(config["retrieval_learning_rate"]) or config["retrieval_learning_rate"] <= 0):
        raise ValueError("retrieval_learning_rate 必须是有限正数")
    integer_keys = {
        "max_history", "min_history", "retrieval_emb_dim", "num_sampled", "retrieval_epochs", "ranking_emb_dim",
        "ranking_epochs", "ranking_cross_layers", "batch_size", "candidate_budget", "neighbors_per_item", "recent_items", "top_k",
        "max_genre_run", "max_decade_run",
        "ranking_query_batch_size", "ranking_patience",
    }
    for name in integer_keys:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")
    for name in ("seed", "negatives_per_positive", "max_users"):
        value = config[name]
        if name == "max_users" and value is None:
            continue
        lower = 1 if name == "max_users" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} 的整数范围无效")
    if not 1 <= config["negative_threshold"] < config["positive_threshold"] <= 5:
        raise ValueError("评分阈值无效")
    if not np.isfinite(config["ranking_history_fraction"]) or not 0 < config["ranking_history_fraction"] < 1:
        raise ValueError("ranking_history_fraction 必须位于 0 和 1 之间")
    for name in ("temperature", "learning_rate"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} 必须是有限正数")
    if not 0 <= config["ranking_dropout"] < 1 or not np.isfinite(config["ranking_l2"]) or config["ranking_l2"] < 0:
        raise ValueError("排序正则化参数无效")
    if not isinstance(config["ranking_hidden_units"], list) or not config["ranking_hidden_units"]:
        raise ValueError("ranking_hidden_units 必须是非空列表")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in config["ranking_hidden_units"]):
        raise ValueError("ranking_hidden_units 必须包含正整数")
    if config["candidate_budget"] < config["top_k"] or config["max_history"] < config["recent_items"]:
        raise ValueError("候选数量或历史长度不足以支持当前配置")


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    validate_config(config)
    return config


def train_context_ranking(state, train: pd.DataFrame, validation: pd.DataFrame, scratch: Path) -> None:
    config = state.config
    training_dir = scratch / "candidate-training"
    samples, data_report = build_candidate_training(train, state.users, state.movies, config, training_dir)
    queries = validation[["user_id", "timestamp"]].rename(columns={"timestamp": "as_of_timestamp"})
    result = generate_candidates(state, queries)
    records = candidate_records(result, state)
    targets = validation.loc[validation["rating"] >= config["positive_threshold"]]
    train_ranking(
        samples, scratch / "ranking", records, result["contexts"], targets, state.retrieval.movie_ids,
        emb_dim=config["ranking_emb_dim"], hidden_units=tuple(config["ranking_hidden_units"]),
        dropout=config["ranking_dropout"], l2=config["ranking_l2"], epochs=config["ranking_epochs"],
        batch_size=config["batch_size"], learning_rate=config["learning_rate"], seed=config["seed"],
        cross_layers=config["ranking_cross_layers"], query_batch_size=config["ranking_query_batch_size"],
        patience=config["ranking_patience"], selection_k=config["top_k"],
        model_type=config.get("ranking_model", "dcnv2"), model_config=config.get("ranking_model_config"),
    )
    context_dir = scratch / "ranking" / "training_context"
    context_dir.mkdir()
    shutil.copytree(training_dir / "retrieval", context_dir / "retrieval")
    for filename in ("history.pkl", "targets.pkl", "queries.pkl", "contexts.pkl"):
        shutil.copy2(training_dir / filename, context_dir / filename)
    samples.iloc[:256].to_pickle(scratch / "ranking" / "training_sample.pkl")
    data_report["training_context_sha256"] = {str(path.relative_to(context_dir)): _file_sha256(path) for path in context_dir.rglob("*") if path.is_file()}
    write_json(scratch / "ranking" / "candidate_training_report.json", data_report)


def run_training(config: dict, output_dir: Path, project_root: Path = PROJECT_ROOT) -> dict:
    validate_config(config)
    output_dir = require_empty_directory(output_dir)
    project_root = Path(project_root)
    data = load_movielens(project_root / "data" / "processed" / "input")
    split_dir = project_root / "data" / "processed" / "split"
    train = pd.read_pickle(split_dir / "train.pkl")
    if config["max_users"] is not None:
        user_ids = np.sort(train["user_id"].unique())[:config["max_users"]]
        train = train.loc[train["user_id"].isin(user_ids)].copy()
    retrieval_samples = build_retrieval_samples(train, data.users, data.movies, config["max_history"], config["positive_threshold"])
    validation = pd.read_pickle(split_dir / "validation.pkl")
    validation = validation.loc[validation["user_id"].isin(train["user_id"])]
    scratch = project_root / ".work" / f"training-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True)
    train_retrieval(
        retrieval_samples, scratch / "retrieval", max_history=config["max_history"], emb_dim=config["retrieval_emb_dim"],
        num_sampled=config["num_sampled"], temperature=config["temperature"], epochs=config["retrieval_epochs"],
        batch_size=config["batch_size"], learning_rate=config.get("retrieval_learning_rate", config["learning_rate"]), seed=config.get("retrieval_seed", config["seed"]),
    )
    state = build_recall_state(scratch / "retrieval", data.users, data.movies, train, config)
    train_context_ranking(state, train, validation, scratch)
    source_files = [project_root / "data" / "processed" / "input" / f"{name}.pkl" for name in ("users", "movies")]
    source_files += [split_dir / "train.pkl", split_dir / "validation.pkl"]
    manifest = export_bundle(output_dir, scratch / "retrieval", scratch / "ranking", data.users, data.movies, train, config, source_files, project_root)
    load_bundle(output_dir)
    shutil.rmtree(scratch)
    return manifest


def run_ranking_training(
    source_dir: Path, output_dir: Path, cross_layers: int = 3, epochs: int | None = None,
    project_root: Path = PROJECT_ROOT,
    *, config_path: Path | None = None, ranking_dataset_dir: Path | None = None,
) -> dict:
    source_dir, project_root = Path(source_dir), Path(project_root)
    output_dir = require_empty_directory(output_dir)
    source_manifest, data = load_retrieval_source(source_dir)
    config = {"ranking_history_fraction": 0.8, "ranking_query_batch_size": 32, "ranking_patience": 3, **source_manifest["config"], "ranking_cross_layers": cross_layers}
    if config_path is not None:
        requested = load_config(config_path)
        ranking_keys = {"learning_rate", "seed", "ranking_model", "ranking_model_config"} | {name for name in requested if name.startswith("ranking_")}
        config.update({name: value for name, value in requested.items() if name in ranking_keys})
    retrieval_report = load_bundle(source_dir).retrieval.training_report
    config["retrieval_seed"] = retrieval_report["seed"]
    config["retrieval_learning_rate"] = retrieval_report["learning_rate"]
    if "ranking_patience" not in source_manifest["config"]:
        config["ranking_epochs"] = 10
    if epochs is not None:
        config["ranking_epochs"] = epochs
    validate_config(config)
    validation_path = project_root / "data" / "processed" / "split" / "validation.pkl"
    if str(validation_path.resolve()) not in source_manifest["source_sha256"]:
        raise ValueError("原模型目录缺少验证数据的来源记录")
    validation = pd.read_pickle(validation_path)
    validation = validation.loc[validation["user_id"].isin(data.ratings["user_id"])]
    scratch = project_root / ".work" / f"training-ranking-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True)
    if ranking_dataset_dir is None:
        state = build_recall_state(source_dir / "retrieval", data.users, data.movies, data.ratings, config)
        train_context_ranking(state, data.ratings, validation, scratch)
    else:
        from training.ranking_dataset import load_ranking_dataset
        dataset = load_ranking_dataset(ranking_dataset_dir)
        with (dataset.directory / "reference-bundle" / "manifest.json").open(encoding="utf-8") as source:
            reference = json.load(source)
        expected = {name: digest for name, digest in reference["files"].items() if name.startswith(("retrieval/", "indexes/"))}
        if any(source_manifest["files"].get(name) != digest for name, digest in expected.items()):
            raise ValueError("排序数据快照与来源召回不一致")
        train_ranking(
            dataset.frame("train-records"), scratch / "ranking", dataset.frame("validation-records"),
            dataset.frame("validation-contexts"), dataset.frame("validation-targets"),
            load_bundle(source_dir).retrieval.movie_ids, emb_dim=config["ranking_emb_dim"],
            hidden_units=tuple(config["ranking_hidden_units"]), cross_layers=config["ranking_cross_layers"],
            dropout=config["ranking_dropout"], l2=config["ranking_l2"], epochs=config["ranking_epochs"],
            batch_size=config["batch_size"], learning_rate=config["learning_rate"], seed=config["seed"],
            query_batch_size=config["ranking_query_batch_size"], patience=config["ranking_patience"],
            selection_k=config["top_k"], model_type=config.get("ranking_model", "dcnv2"),
            model_config=config.get("ranking_model_config"), prepared=dataset.prepared_training(),
        )
        write_json(scratch / "ranking" / "dataset_reference.json", {"manifest_sha256": _file_sha256(dataset.directory / "manifest.json"), "dataset_version": dataset.manifest["dataset_version"]})
    origin = {
        "bundle": str(source_dir.resolve()), "manifest_sha256": _file_sha256(source_dir / "manifest.json"),
        "files": {name: digest for name, digest in source_manifest["files"].items() if name.startswith(("retrieval/", "indexes/"))},
    }
    manifest = export_bundle(
        output_dir, source_dir / "retrieval", scratch / "ranking", data.users, data.movies, data.ratings,
        config, [Path(filename) for filename in source_manifest["source_sha256"]], project_root,
        indexes_dir=source_dir / "indexes", retrieval_origin=origin,
    )
    load_bundle(output_dir)
    shutil.rmtree(scratch)
    return manifest


def recommend_from_bundle(bundle, queries: pd.DataFrame, history: pd.DataFrame | None = None, top_k: int | None = None) -> dict:
    config = bundle.config
    validate_config(config)
    top_k = config["top_k"] if top_k is None else top_k
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= config["candidate_budget"]:
        raise ValueError("top_k 必须是正整数，且不能超过候选数量")
    result = generate_candidates(bundle, queries, history)
    contexts, channels, merged = result["contexts"], result["channels"], result["merged"]
    ranked, rates = score_candidates(merged, bundle.ranking, bundle.users, bundle.movies, config["batch_size"], contexts=contexts, retrieval=bundle.retrieval, popularity=bundle.popularity)
    reranked = rerank_candidates(ranked, bundle.movies, top_k, config["max_genre_run"], config["max_decade_run"])
    final = reranked.loc[reranked["rank"] <= top_k].merge(
        contexts[["user_id", "is_cold_start", "as_of_timestamp", "user_group"]], on="user_id", validate="many_to_one",
    )
    return {"contexts": contexts, "channels": channels, "merged": merged, "ranked": ranked, "reranked": reranked, "final": final, "unknown_rates": rates}


def save_recommendation_run(result: dict, output_dir: Path, bundle, top_k: int) -> dict:
    output_dir = require_empty_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["contexts"].to_pickle(output_dir / "contexts.pkl")
    for name in ("merged", "ranked", "reranked", "final"):
        write_result_csv(output_dir / f"{name}.csv.gz", result[name])
    for name, rows in result["channels"].items():
        write_result_csv(output_dir / f"recall-{name}.csv.gz", rows)
    counts = result["final"].groupby("user_id").size().reindex(result["contexts"]["user_id"], fill_value=0)
    report = {
        "bundle": str(bundle.directory.resolve()), "config": bundle.config, "top_k": top_k,
        "users": len(result["contexts"]), "recommendations": len(result["final"]),
        "min_per_user": int(counts.min()), "max_per_user": int(counts.max()),
        "users_without_candidates": int((counts == 0).sum()),
        "user_groups": result["contexts"]["user_group"].value_counts().to_dict(),
        "ranking_unknown_rates": result["unknown_rates"],
    }
    write_json(output_dir / "recommendation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="训练、推荐和评估完整离线电影推荐流程")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "offline.yaml")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--max-users", type=int)
    train.add_argument("--epochs", type=int)
    ranking = commands.add_parser("train-ranking")
    ranking.add_argument("--source-bundle", type=Path, required=True)
    ranking.add_argument("--output-dir", type=Path, required=True)
    ranking.add_argument("--cross-layers", type=int, default=3)
    ranking.add_argument("--epochs", type=int)
    ranking.add_argument("--config", type=Path)
    ranking.add_argument("--ranking-dataset", type=Path)
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--bundle", type=Path, required=True)
    recommend.add_argument("--queries", type=Path, required=True)
    recommend.add_argument("--output-dir", type=Path, required=True)
    recommend.add_argument("--top-k", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    evaluate.add_argument("--include-validation-history", action="store_true")
    evaluate.add_argument("--ranking-dataset", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        if args.max_users is not None:
            config["max_users"] = args.max_users
        if args.epochs is not None:
            config["retrieval_epochs"] = config["ranking_epochs"] = args.epochs
        manifest = run_training(config, args.output_dir)
        report = {name: manifest[name] for name in ("training_users", "training_rows", "catalog_size")}
    elif args.command == "train-ranking":
        manifest = run_ranking_training(args.source_bundle, args.output_dir, args.cross_layers, args.epochs, config_path=args.config, ranking_dataset_dir=args.ranking_dataset)
        report = {name: manifest[name] for name in ("training_users", "training_rows", "catalog_size", "retrieval_origin")}
    elif args.command == "recommend":
        bundle = load_bundle(args.bundle)
        queries = pd.read_csv(args.queries)
        result = recommend_from_bundle(bundle, queries, top_k=args.top_k)
        report = save_recommendation_run(result, args.output_dir, bundle, bundle.config["top_k"] if args.top_k is None else args.top_k)
    else:
        from evaluation.evaluate_pipeline import run_pipeline_evaluation
        report = run_pipeline_evaluation(args.bundle, args.output_dir, args.split, tuple(args.ks), args.include_validation_history, ranking_dataset_dir=args.ranking_dataset)
        report = {"split": report["split"], "stages": report["stages"], "binary_metrics": report["binary_metrics"]}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
