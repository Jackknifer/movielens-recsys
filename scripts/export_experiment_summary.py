import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出包含完整实验结果的公开摘要")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("公开摘要文件已经存在")
    directory = args.experiment_dir
    dataset = read_json(args.dataset / "manifest.json")
    experiment = read_json(directory / "experiment.json")
    comparison = read_json(directory / "summary" / "comparison.json")
    benchmark = read_json(directory / "benchmark" / "benchmark.json")
    if file_sha256(args.dataset / "manifest.json") != experiment["dataset_sha256"] or comparison["dataset_sha256"] != experiment["dataset_sha256"]:
        raise ValueError("实验结果与数据快照摘要不一致")
    source_files = ["experiment.json", "summary/comparison.json", "summary/runs.csv", "benchmark/benchmark.json"]
    jobs = {job["run_id"]: job for job in experiment["search_runs"] + experiment["final_runs"]}
    search, structures = [], {}
    for run_id, job in jobs.items():
        relative = f"runs/{run_id}"
        completion = read_json(directory / relative / "training_complete.json")
        report = read_json(directory / relative / "ranking" / "training_report.json")
        if completion["dataset_sha256"] != experiment["dataset_sha256"] or completion["training_report_sha256"] != file_sha256(directory / relative / "ranking" / "training_report.json"):
            raise ValueError("训练报告与完成记录不一致")
        source_files.extend([f"{relative}/training_complete.json", f"{relative}/ranking/training_report.json"])
        structures[job["variant"]] = {name: report[name] for name in ("model_type", "emb_dim", "dropout", "l2", "parameter_count", "embedding_parameters", "non_embedding_parameters", "matrix_macs_per_candidate", "model_config", "token_groups", "hidden_units", "cross_layers") if name in report}
        if job in experiment["search_runs"]:
            search.append({**job, "selected_epoch": report["selected_epoch"], "selected_validation": report["selected_validation"]})
    top_k_results = []
    for job in experiment["final_runs"]:
        relative = f"runs/{job['run_id']}"
        completion = read_json(directory / relative / "evaluation_complete.json")
        if completion["ranking_dataset_sha256"] != experiment["dataset_sha256"] or completion["evaluation_report_sha256"] != file_sha256(directory / relative / "evaluation" / "evaluation_report.json"):
            raise ValueError("测试报告与完成记录不一致")
        evaluation = read_json(directory / relative / "evaluation" / "evaluation_report.json")
        top_k_results.append({**job, "stages": {stage: results["top_k"] for stage, results in evaluation["stages"].items()}})
        source_files.extend([f"{relative}/evaluation_complete.json", f"{relative}/evaluation/evaluation_report.json"])
    runs = pd.read_csv(directory / "summary" / "runs.csv")
    expected = {(job["variant"], job["seed"]) for job in experiment["final_runs"]}
    if set(zip(runs["variant"], runs["seed"])) != expected or len(runs) != len(expected):
        raise ValueError("公开摘要需要包含全部最终测试")
    benchmark = {**benchmark, "measurements": {
        scenario: {variant: {name: value for name, value in measurements.items() if name != "raw_ms"} for variant, measurements in variants.items()}
        for scenario, variants in benchmark["measurements"].items()
    }}
    result = {
        "dataset": "MovieLens-1M", "training_runs": len(jobs), "test_runs": len(runs),
        "data": {name: dataset[name] for name in ("training", "splits", "include_validation_history")},
        "comparison": comparison, "runs": runs.to_dict(orient="records"), "top_k_results": top_k_results,
        "validation_search": search, "selected": experiment["selected"],
        "structures": structures, "benchmark": benchmark,
        "training_code_sha256": experiment["training_code_sha256"],
        "evidence_sha256": {name: file_sha256(directory / name) for name in source_files},
        "interpretation": {
            "population": f"测试请求共 {dataset['splits']['test']['users']:,} 位用户，排名指标统计其中 {dataset['splits']['test']['positive_target_users']:,} 位高评分目标用户。每位用户使用相同的 {comparison['candidate_budget']} 个候选，展示 {comparison['display_k']} 部电影。",
            "selection": "学习率与训练轮次仅使用验证集选择。消融实验沿用 RankMixer 的学习率。",
            "uncertainty": "标准差来自 3 个训练种子。95% 区间对跨种子平均后的逐用户差值进行 10,000 次成对重采样，表示固定种子下的用户抽样不确定性。",
            "training_cost": "训练耗时包括模型构建、训练、每轮验证和保存读取检查；峰值内存统计整个训练工作进程。",
            "compute_budget": "矩阵 MAC 统计 Dense、EinsumDense 和完整矩阵 Cross 的乘加次数。预测耗时另行测量。",
            "scope": "这是当前 MovieLens 基准上的受控比较。测试集此前已经公开用于项目评估，结论仅适用于记录的数据、候选和配置。",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"training_runs": len(jobs), "test_runs": len(runs), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
