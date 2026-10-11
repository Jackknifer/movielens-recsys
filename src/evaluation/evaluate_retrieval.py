import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.retrieval_context import build_retrieval_contexts
from retrieval.vector_recall import load_retrieval_artifacts, recall_users
from data.load import load_movielens


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_top_k(
    contexts: pd.DataFrame,
    recommendations: pd.DataFrame,
    movie_ids: np.ndarray,
    k: int = 20,
) -> dict:
    """每个用户只有一个保留目标，未命中和目录外目标都计入分母。"""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k 必须是正整数")
    if contexts.empty or contexts["user_id"].duplicated().any():
        raise ValueError("contexts 不能为空，每个用户只能有一个目标")
    if movie_ids.ndim != 1 or not len(movie_ids) or len(np.unique(movie_ids)) != len(movie_ids):
        raise ValueError("候选电影目录必须非空，电影编号不能重复")
    required = {"user_id", "movie_id", "source_rank"}
    if not required.issubset(recommendations.columns):
        raise ValueError(f"recommendations 缺少字段：{sorted(required - set(recommendations.columns))}")
    if recommendations[list(required)].isna().any().any():
        raise ValueError("推荐结果的必要字段存在空值")
    if recommendations.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("同一用户的推荐结果存在重复电影")
    if not recommendations["user_id"].isin(contexts["user_id"]).all():
        raise ValueError("推荐结果包含未参加评估的用户")
    if not recommendations["movie_id"].isin(movie_ids).all():
        raise ValueError("推荐结果包含候选目录以外的电影")
    seen_by_user = contexts.set_index("user_id")["seen_movie_ids"].to_dict()
    for user_id, group in recommendations.groupby("user_id", sort=False):
        ranks = group["source_rank"].to_numpy()
        if not np.issubdtype(ranks.dtype, np.integer):
            raise TypeError("source_rank 必须使用整数")
        if not np.array_equal(np.sort(ranks), np.arange(1, len(group) + 1)):
            raise ValueError("每个用户的 source_rank 必须从 1 开始连续编号")
        if group["movie_id"].isin(seen_by_user[user_id]).any():
            raise ValueError("推荐结果包含用户已看电影")

    selected = recommendations.loc[recommendations["source_rank"] <= k]
    targets = contexts.set_index("user_id")["movie_id"]
    hits = selected.loc[selected["movie_id"].eq(selected["user_id"].map(targets))]
    ranks = hits.set_index("user_id")["source_rank"].reindex(targets.index, fill_value=0)
    hit_mask = ranks.to_numpy() > 0
    ndcg = np.zeros(len(contexts), dtype=np.float64)
    reciprocal_rank = np.zeros(len(contexts), dtype=np.float64)
    # 单个相关物品的理想 DCG 为 1；未命中用户的两项得分保持为 0。
    ndcg[hit_mask] = 1 / np.log2(ranks.to_numpy()[hit_mask] + 1)
    reciprocal_rank[hit_mask] = 1 / ranks.to_numpy()[hit_mask]
    reachable = contexts["movie_id"].isin(movie_ids)
    return {
        "k": k, "evaluated_users": len(contexts), "hits": int(hit_mask.sum()),
        "recall_at_k": float(hit_mask.mean()), "hit_rate_at_k": float(hit_mask.mean()),
        "ndcg_at_k": float(ndcg.mean()), "mrr_at_k": float(reciprocal_rank.mean()),
        "catalog_size": len(movie_ids), "unavailable_targets": int((~reachable).sum()),
        "catalog_target_coverage": float(reachable.mean()),
        "recommendation_coverage": float(selected["movie_id"].nunique() / len(movie_ids)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_evaluation(
    artifact_dir: Path,
    output_dir: Path,
    split: str = "validation",
    ks: tuple[int, ...] = (5, 10, 20),
    positive_threshold: float = 4.0,
    include_validation_history: bool = False,
    max_users: int | None = None,
    batch_size: int = 128,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """读取保留数据与已训练模型，保存推荐结果和可以追踪输入的指标报告。"""
    if split not in {"validation", "test"}:
        raise ValueError("split 必须是 validation 或 test")
    if include_validation_history and split != "test":
        raise ValueError("只有 test 评估可以加入 validation 历史")
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks 必须包含正整数")
    if max_users is not None and (
        isinstance(max_users, bool) or not isinstance(max_users, int) or max_users < 1
    ):
        raise ValueError("max_users 必须是正整数")
    if not np.isfinite(positive_threshold) or not 1 <= positive_threshold <= 5:
        raise ValueError("positive_threshold 必须在 1 到 5 之间")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"请选择空的输出目录：{output_dir}")
    artifact_dir, project_root = Path(artifact_dir), Path(project_root)
    artifacts = load_retrieval_artifacts(artifact_dir)
    if max(ks) > len(artifacts.movie_ids):
        raise ValueError("评估的 k 不能超过候选电影数量")
    data = load_movielens(project_root / "data" / "processed" / "input")
    split_dir = project_root / "data" / "processed" / "split"
    history = pd.read_pickle(split_dir / "train.pkl")
    heldout = pd.read_pickle(split_dir / f"{split}.pkl")
    heldout_rows = len(heldout)

    # 小规模训练模型只评估训练词表中的用户，报告明确记录这个评估范围。
    known_users = artifacts.encoders["user_id"].categories_[0].astype(np.int64)
    heldout = heldout.loc[heldout["user_id"].isin(known_users)]
    known_user_rows = len(heldout)
    heldout = heldout.loc[heldout["rating"] >= positive_threshold].sort_values("user_id")
    positive_target_rows = len(heldout)
    if max_users is not None:
        heldout = heldout.head(max_users)
    if heldout.empty:
        raise ValueError("没有符合评估范围的高评分保留目标")
    history_files = [split_dir / "train.pkl"]
    if include_validation_history:
        validation_path = split_dir / "validation.pkl"
        history = pd.concat([history, pd.read_pickle(validation_path)], ignore_index=True)
        history_files.append(validation_path)
    history = history.loc[history["user_id"].isin(heldout["user_id"])]
    contexts = build_retrieval_contexts(
        history, heldout, data.users, data.movies, artifacts.max_history, positive_threshold,
    )
    recommendations = recall_users(contexts, artifacts, max(ks), batch_size)
    report = {
        "split": split, "split_policy": "per_user_leave_two_out",
        "positive_threshold": positive_threshold,
        "include_validation_history": include_validation_history,
        "population": "known_training_users_with_positive_heldout_targets",
        "candidate_catalog": "all_exported_known_movie_ids",
        "heldout_rows": heldout_rows, "known_user_rows": known_user_rows,
        "positive_target_rows": positive_target_rows, "evaluated_users": len(contexts),
        "max_users": max_users, "batch_size": batch_size,
        "artifact_dir": str(artifact_dir.resolve()),
        "training_report": artifacts.training_report,
        "metrics": [evaluate_top_k(contexts, recommendations, artifacts.movie_ids, k) for k in sorted(set(ks))],
    }
    input_files = history_files + [split_dir / f"{split}.pkl"]
    input_files += [project_root / "data" / "processed" / "input" / f"{name}.pkl" for name in ("users", "movies")]
    input_files += [artifact_dir / name for name in (
        "training_report.json", "encoders.pkl", "movie_ids.npy", "item_embeddings.npy", "user_tower.keras",
    )]
    report["input_sha256"] = {str(path.resolve()): _file_sha256(path) for path in input_files}
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts.to_pickle(output_dir / "contexts.pkl")
    recommendations.to_csv(output_dir / "recommendations.csv.gz", index=False, compression="gzip")
    with (output_dir / "evaluation_report.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 YoutubeDNN 的完整已编码电影目录召回")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / ".work" / "retrieval-study")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / ".work" / "retrieval-evaluation")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--include-validation-history", action="store_true")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    report = run_evaluation(
        args.artifact_dir, args.output_dir, args.split, tuple(args.ks),
        args.positive_threshold, args.include_validation_history, args.max_users, args.batch_size,
    )
    print(json.dumps({"split": report["split"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
