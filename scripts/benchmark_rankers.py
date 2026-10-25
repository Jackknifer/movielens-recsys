import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from pipeline.artifact_store import load_bundle, require_empty_directory, write_json
from ranking.ranking_predict import load_ranking_artifacts, score_candidates
from training.ranking_dataset import load_ranking_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="在相同真实候选上轮换测量排序模型耗时")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1:
        raise ValueError("预热和测量次数必须为正数")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    destination = require_empty_directory(args.output_dir)
    dataset = load_ranking_dataset(args.dataset)
    reference = load_bundle(dataset.directory / "reference-bundle")
    with (args.experiment_dir / "experiment.json").open(encoding="utf-8") as source:
        experiment = json.load(source)
    all_inputs = dataset.inputs("test")
    candidates = dataset.frame("test-candidates").sort_values(["user_id", "candidate_rank"], kind="stable").reset_index(drop=True)
    contexts = dataset.frame("test-contexts")
    user = int(candidates.iloc[0]["user_id"])
    selected = candidates.loc[candidates["user_id"] == user].copy()
    context = contexts.loc[contexts["user_id"] == user]
    if len(selected) != 100 or len(candidates) < 3200:
        raise ValueError("计时需要真实的完整候选列表")
    artifacts, predictors, load_seconds = {}, {}, {}
    for variant, job in experiment["selected"].items():
        started = time.perf_counter()
        artifact = load_ranking_artifacts(args.experiment_dir / "runs" / job["run_id"] / "ranking")
        load_seconds[variant] = time.perf_counter() - started
        artifacts[variant] = artifact
        predictors[variant] = tf.function(artifact.model, reduce_retracing=True)
    scenarios = {
        "model_100_candidates": (100, {name: values[:100] for name, values in all_inputs.items()}),
        "model_3200_candidates": (3200, {name: values[:3200] for name, values in all_inputs.items()}),
        "ranking_stage_100_candidates": (100, None),
    }
    measurements = {}
    variants = list(artifacts)
    for scenario, (count, inputs) in scenarios.items():
        raw = {variant: [] for variant in variants}
        for iteration in range(args.warmup + args.repetitions):
            order = variants[iteration % len(variants):] + variants[:iteration % len(variants)]
            for variant in order:
                started = time.perf_counter()
                if inputs is None:
                    scores, _ = score_candidates(selected, artifacts[variant], reference.users, reference.movies,
                        reference.config["batch_size"], contexts=context, retrieval=reference.retrieval, popularity=reference.popularity)
                    values = scores["ranking_score"].to_numpy()
                else:
                    values = predictors[variant](inputs, training=False).numpy()
                elapsed = (time.perf_counter() - started) * 1000
                if not np.isfinite(values).all():
                    raise ValueError("计时产生无效预测")
                if iteration >= args.warmup:
                    raw[variant].append(elapsed)
        measurements[scenario] = {}
        for variant, times in raw.items():
            median, tail = np.quantile(times, [0.5, 0.95])
            measurements[scenario][variant] = {"p50_ms": float(median), "p95_ms": float(tail), "candidates_per_second_at_p50": float(count / (median / 1000)), "raw_ms": times}
        print(json.dumps({"scenario": scenario, "measurements": {name: {key: value for key, value in result.items() if key != "raw_ms"} for name, result in measurements[scenario].items()}}, ensure_ascii=False), flush=True)
    processor = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True).stdout.strip() if platform.system() == "Darwin" else platform.processor()
    report = {
        "warmup": args.warmup, "repetitions": args.repetitions, "seed": 42,
        "variant_order": "rotated_each_iteration", "tensorflow": tf.__version__,
        "python": platform.python_version(), "platform": platform.platform(), "processor": processor,
        "physical_devices": [device.name for device in tf.config.list_physical_devices()],
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "load_seconds": load_seconds,
        "measurements": measurements,
    }
    destination.mkdir(parents=True)
    write_json(destination / "benchmark.json", report)


if __name__ == "__main__":
    main()
