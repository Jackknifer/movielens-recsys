import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import tensorflow as tf

from evaluation.compare_rankers import summarize_experiment
from evaluation.evaluate_pipeline import run_pipeline_evaluation
from evaluation.evaluate_retrieval import _file_sha256
from models.rankmixer import validate_rankmixer_config
from pipeline.artifact_store import export_bundle, load_bundle, runtime_code_sha256, write_json
from training.ranking_dataset import load_ranking_dataset
from training.train_ranking import train_ranking


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_VARIANTS = ("dcnv2", "dcnv2_matched", "rankmixer")
ABLATION_VARIANTS = ("rankmixer_no_mixing", "rankmixer_shared_ffn", "rankmixer_shared_ffn_matched")
VARIANTS = PRIMARY_VARIANTS + ABLATION_VARIANTS
PROTOCOL = {"seeds": [42, 2025, 2026], "learning_rates": [0.0003, 0.001, 0.003], "epochs": 10, "patience": 3, "query_batch_size": 32, "selection": "validation_ndcg_then_recall", "top_k": 20, "threads": 4}


def training_code_sha256() -> dict:
    files = runtime_code_sha256(PROJECT_ROOT)
    included = {"src/training/train_ranking.py", "src/training/ranking_inputs.py", "src/features/encode.py"}
    return {name: digest for name, digest in files.items() if name in included or name.startswith("src/models/")}


def variant_config(variant: str, seed: int, learning_rate: float, source_config: dict) -> dict:
    if variant not in VARIANTS:
        raise ValueError("实验模型名称无效")
    config = {**source_config, "seed": seed, "learning_rate": learning_rate,
              "ranking_epochs": PROTOCOL["epochs"], "ranking_patience": PROTOCOL["patience"],
              "ranking_query_batch_size": PROTOCOL["query_batch_size"],
              "ranking_model": "dcnv2", "ranking_hidden_units": [64, 32]}
    config.pop("ranking_model_config", None)
    if variant == "dcnv2_matched":
        config["ranking_hidden_units"] = [112, 64]
    elif variant.startswith("rankmixer"):
        model_config = validate_rankmixer_config()
        if variant == "rankmixer_no_mixing":
            model_config["mixing"] = False
        if variant in {"rankmixer_shared_ffn", "rankmixer_shared_ffn_matched"}:
            model_config["shared_ffn"] = True
        if variant == "rankmixer_shared_ffn_matched":
            model_config["ffn_hidden_dim"] = 640
        config.update(ranking_model="rankmixer", ranking_model_config=model_config)
    return config


def job_specification(variant: str, seed: int, learning_rate: float) -> dict:
    rate = format(learning_rate, ".4f").replace(".", "p")
    return {"run_id": f"{variant}-lr-{rate}-seed-{seed}", "variant": variant, "seed": seed, "learning_rate": learning_rate}


def worker(args) -> None:
    dataset = load_ranking_dataset(args.dataset)
    reference = load_bundle(dataset.directory / "reference-bundle")
    config = variant_config(args.variant, args.seed, args.learning_rate, reference.config)
    config["retrieval_seed"] = reference.retrieval.training_report["seed"]
    config["retrieval_learning_rate"] = reference.retrieval.training_report["learning_rate"]
    job = job_specification(args.variant, args.seed, args.learning_rate)
    directory = args.output_dir / "runs" / job["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    if args.worker == "train":
        with (args.output_dir / "experiment.json").open(encoding="utf-8") as source:
            experiment = json.load(source)
        if experiment["training_code_sha256"] != training_code_sha256():
            raise ValueError("训练代码在实验期间发生变化")
        report = train_ranking(
            dataset.frame("train-records"), directory / "ranking", dataset.frame("validation-records"),
            dataset.frame("validation-contexts"), dataset.frame("validation-targets"), reference.retrieval.movie_ids,
            emb_dim=config["ranking_emb_dim"], hidden_units=tuple(config["ranking_hidden_units"]),
            dropout=config["ranking_dropout"], l2=config["ranking_l2"], epochs=config["ranking_epochs"],
            batch_size=config["batch_size"], learning_rate=args.learning_rate, seed=args.seed,
            cross_layers=config["ranking_cross_layers"], query_batch_size=config["ranking_query_batch_size"],
            patience=config["ranking_patience"], selection_k=20, model_type=config["ranking_model"],
            model_config=config.get("ranking_model_config"), prepared=dataset.prepared_training(),
        )
        write_json(directory / "training_complete.json", {**job, "dataset_sha256": _file_sha256(dataset.directory / "manifest.json"), "training_report_sha256": _file_sha256(directory / "ranking" / "training_report.json"), "selected_validation": report["selected_validation"], "config": config, "code_sha256": runtime_code_sha256(PROJECT_ROOT), "training_code_sha256": training_code_sha256()})
    else:
        with (directory / "training_complete.json").open(encoding="utf-8") as source:
            complete = json.load(source)
        if complete["config"] != config:
            raise ValueError("评估配置与训练配置不一致")
        source_files = [Path(name) for name in reference.manifest["source_sha256"]] + [dataset.directory / "manifest.json"]
        origin = {"manifest_sha256": _file_sha256(reference.directory / "manifest.json"), "files": {name: digest for name, digest in reference.manifest["files"].items() if name.startswith(("retrieval/", "indexes/"))}}
        if (directory / "bundle" / "manifest.json").exists():
            existing = load_bundle(directory / "bundle")
            if existing.config != config or _file_sha256(existing.directory / "ranking" / "ranker.keras") != _file_sha256(directory / "ranking" / "ranker.keras"):
                raise ValueError("恢复评估的模型与训练产物不一致")
        else:
            export_bundle(directory / "bundle", reference.directory / "retrieval", directory / "ranking", reference.users, reference.movies, reference.history, config, source_files, PROJECT_ROOT, indexes_dir=reference.directory / "indexes", retrieval_origin=origin)
        report = run_pipeline_evaluation(directory / "bundle", directory / "evaluation", "test", (5, 10, 20), ranking_dataset_dir=dataset.directory, save_details=False)
        write_json(directory / "evaluation_complete.json", {**job, "evaluation_report_sha256": _file_sha256(directory / "evaluation" / "evaluation_report.json"), "ranking_dataset_sha256": report["ranking_dataset_sha256"]})


def run_job(job: dict, phase: str, args, experiment: dict) -> None:
    directory = args.output_dir / "runs" / job["run_id"]
    completion_name = "training_complete.json" if phase == "train" else "evaluation_complete.json"
    completion_path = directory / completion_name
    if completion_path.exists():
        with completion_path.open(encoding="utf-8") as source:
            completed = json.load(source)
        filename = "ranking/training_report.json" if phase == "train" else "evaluation/evaluation_report.json"
        digest_key = "training_report_sha256" if phase == "train" else "evaluation_report_sha256"
        if completed[digest_key] != _file_sha256(directory / filename):
            raise ValueError("已完成的实验文件摘要不一致")
        if phase == "train" and (completed["dataset_sha256"] != experiment["dataset_sha256"] or completed["training_code_sha256"] != experiment["training_code_sha256"]):
            raise ValueError("恢复实验的模型代码或数据不一致")
        return
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--dataset", str(args.dataset), "--output-dir", str(args.output_dir), "--variant", job["variant"], "--seed", str(job["seed"]), "--learning-rate", str(job["learning_rate"])]
    print(json.dumps({"phase": phase, "status": "started", **job}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    with (directory / f"{phase}.log").open("w", encoding="utf-8") as log:
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "1"}
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"phase": phase, "status": "completed", "seconds": time.perf_counter() - started, **job}, ensure_ascii=False), flush=True)


def run_experiment(args) -> None:
    dataset = load_ranking_dataset(args.dataset)
    manifest_path = args.output_dir / "experiment.json"
    fingerprint = {"dataset_sha256": _file_sha256(dataset.directory / "manifest.json"), "protocol": PROTOCOL, "code_sha256": runtime_code_sha256(PROJECT_ROOT), "training_code_sha256": training_code_sha256()}
    if args.resume:
        with manifest_path.open(encoding="utf-8") as source:
            experiment = json.load(source)
        if any(experiment[name] != fingerprint[name] for name in ("dataset_sha256", "protocol", "training_code_sha256")):
            raise ValueError("恢复实验的协议、代码或数据与已有记录不一致")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("实验输出目录必须为空")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        experiment = {**fingerprint, "search_runs": [], "final_runs": [], "selected": {}}
        write_json(manifest_path, experiment)
    for variant in PRIMARY_VARIANTS:
        candidates = [job_specification(variant, 42, rate) for rate in PROTOCOL["learning_rates"]]
        for job in candidates:
            run_job(job, "train", args, experiment)
        choices = []
        for job in candidates:
            with (args.output_dir / "runs" / job["run_id"] / "training_complete.json").open(encoding="utf-8") as source:
                choices.append((job, json.load(source)["selected_validation"]))
        selected, _ = max(choices, key=lambda item: (item[1]["ndcg_at_k"], item[1]["recall_at_k"], -item[0]["learning_rate"]))
        experiment["selected"][variant] = selected
        for job in candidates:
            if job not in experiment["search_runs"]:
                experiment["search_runs"].append(job)
        write_json(manifest_path, experiment)
    final = []
    for variant in PRIMARY_VARIANTS:
        rate = experiment["selected"][variant]["learning_rate"]
        final.extend(job_specification(variant, seed, rate) for seed in PROTOCOL["seeds"])
    for variant in ABLATION_VARIANTS:
        rate = experiment["selected"]["rankmixer"]["learning_rate"]
        final.extend(job_specification(variant, seed, rate) for seed in PROTOCOL["seeds"])
    experiment["final_runs"] = final
    write_json(manifest_path, experiment)
    for job in final:
        run_job(job, "train", args, experiment)
    for job in final:
        run_job(job, "evaluate", args, experiment)
    summary_dir = args.output_dir / "summary"
    if not (summary_dir / "comparison.json").exists():
        summarize_experiment(args.output_dir, dataset.directory, summary_dir)
    benchmark_dir = args.output_dir / "benchmark"
    if not (benchmark_dir / "benchmark.json").exists():
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "benchmark_rankers.py"), "--dataset", str(dataset.directory), "--experiment-dir", str(args.output_dir), "--output-dir", str(benchmark_dir)]
        with (args.output_dir / "benchmark.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"status": "complete", "training_runs": len({job["run_id"] for job in experiment["search_runs"] + final}), "test_runs": len(final), "summary": str(summary_dir)}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定候选下比较 DCNv2、RankMixer 和结构消融")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("train", "evaluate"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(PROTOCOL["threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.worker:
        if args.variant is None or args.seed is None or args.learning_rate is None:
            raise ValueError("实验工作进程需要模型、随机种子和学习率")
        worker(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
