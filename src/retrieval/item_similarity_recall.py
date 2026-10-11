import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate_retrieval import _file_sha256, evaluate_top_k
from retrieval.genre_recall import build_genre_index, recall_genres
from retrieval.popular_recall import _rank_candidates, _validate_contexts, build_popularity_index, recall_popular
from features.retrieval_context import build_retrieval_contexts
from retrieval.vector_recall import RESULT_COLUMNS, _validate_catalog, load_retrieval_artifacts, recall_users
from data.load import load_movielens


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMILARITY_COLUMNS = ["seed_movie_id", "movie_id", "similarity", "neighbor_rank"]


def build_item_similarity_index(
    movie_ids: np.ndarray, item_vectors: np.ndarray, neighbors_per_item: int = 50,
) -> pd.DataFrame:
    """计算完整电影目录的余弦相似度，为每部电影保存最近邻。"""
    _validate_catalog(movie_ids, item_vectors)
    if (
        isinstance(neighbors_per_item, bool) or not isinstance(neighbors_per_item, int)
        or not 1 <= neighbors_per_item < len(movie_ids)
    ):
        raise ValueError("neighbors_per_item 必须是正整数，且小于候选电影数量")
    rows = []
    for position, seed_movie_id in enumerate(movie_ids):
        similarities = item_vectors @ item_vectors[position]
        available = np.flatnonzero(movie_ids != seed_movie_id)
        order = np.lexsort((movie_ids[available], -similarities[available]))
        for rank, neighbor in enumerate(available[order[:neighbors_per_item]], start=1):
            rows.append((int(seed_movie_id), int(movie_ids[neighbor]), float(similarities[neighbor]), rank))
    return pd.DataFrame(rows, columns=SIMILARITY_COLUMNS)


def recall_item_similarity(
    contexts: pd.DataFrame, index: pd.DataFrame, k: int = 20, recent_items: int = 3,
) -> pd.DataFrame:
    """聚合最近若干部电影的近邻，使用各历史电影给出的最大相似度。"""
    _validate_contexts(contexts, k, require_history=True)
    if isinstance(recent_items, bool) or not isinstance(recent_items, int) or recent_items < 1:
        raise ValueError("recent_items 必须是正整数")
    if index.empty or not set(SIMILARITY_COLUMNS).issubset(index.columns):
        raise ValueError("物品相似索引不能为空，且必须包含全部必要字段")
    if index[SIMILARITY_COLUMNS].isna().any().any():
        raise ValueError("物品相似索引存在空值")
    for column in ("seed_movie_id", "movie_id", "neighbor_rank"):
        values = index[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or (values < 1).any():
            raise ValueError(f"{column} 必须包含正整数")
    similarities = index["similarity"].to_numpy()
    if not np.isfinite(similarities).all() or (np.abs(similarities) > 1 + 1e-5).any():
        raise ValueError("similarity 必须是有效的余弦相似度")
    if index.duplicated(["seed_movie_id", "movie_id"]).any():
        raise ValueError("每部历史电影的近邻不能重复")
    if index["seed_movie_id"].eq(index["movie_id"]).any():
        raise ValueError("物品相似索引不能包含电影自身")
    for _, group in index.groupby("seed_movie_id", sort=False):
        if not np.array_equal(np.sort(group["neighbor_rank"]), np.arange(1, len(group) + 1)):
            raise ValueError("neighbor_rank 必须从 1 开始连续编号")

    results = []
    for context in contexts.itertuples(index=False):
        # 历史按时间升序保存；目录外历史没有可查询的电影向量。
        seeds = context.hist_movie_id[-recent_items:]
        neighbors = index.loc[index["seed_movie_id"].isin(seeds)]
        scores = neighbors.groupby("movie_id", as_index=False)["similarity"].max()
        scores = scores.rename(columns={"similarity": "source_score"})
        results.append(_rank_candidates(
            context.user_id, scores, context.seen_movie_ids, k, "item_similarity",
        ))
    return pd.concat(results, ignore_index=True)[RESULT_COLUMNS]


def run_recall_comparison(
    evaluation_dir: Path, output_dir: Path, ks: tuple[int, ...] = (5, 10, 20),
    neighbors_per_item: int = 50, recent_items: int = 3, project_root: Path = PROJECT_ROOT,
) -> dict:
    """复用已保存的评估设置，在相同目录和用户上下文下比较四路召回。"""
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks 必须包含正整数")
    evaluation_dir, output_dir, project_root = Path(evaluation_dir), Path(output_dir), Path(project_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"请选择空的输出目录：{output_dir}")
    report_path = evaluation_dir / "evaluation_report.json"
    context_path = evaluation_dir / "contexts.pkl"
    with report_path.open(encoding="utf-8") as source:
        evaluation = json.load(source)
    for filename, expected_hash in evaluation["input_sha256"].items():
        if _file_sha256(Path(filename)) != expected_hash:
            raise ValueError(f"输入文件的 SHA-256 与评估记录不一致：{filename}")

    artifact_dir = Path(evaluation["artifact_dir"])
    split_dir = project_root / "data" / "processed" / "split"
    input_dir = project_root / "data" / "processed" / "input"
    split = evaluation["split"]
    if split not in {"validation", "test"}:
        raise ValueError("评估记录的 split 必须是 validation 或 test")
    if evaluation["include_validation_history"] and split != "test":
        raise ValueError("只有 test 评估可以加入 validation 历史")
    required_files = [split_dir / "train.pkl", split_dir / f"{split}.pkl"]
    required_files += [input_dir / f"{name}.pkl" for name in ("users", "movies")]
    required_files += [artifact_dir / name for name in (
        "training_report.json", "encoders.pkl", "movie_ids.npy", "item_embeddings.npy", "user_tower.keras",
    )]
    if evaluation["include_validation_history"]:
        required_files.append(split_dir / "validation.pkl")
    if not {str(path.resolve()) for path in required_files}.issubset(evaluation["input_sha256"]):
        raise ValueError("评估记录没有包含当前数据与模型文件的 SHA-256")
    artifacts = load_retrieval_artifacts(artifact_dir)
    if artifacts.training_report != evaluation["training_report"]:
        raise ValueError("训练报告与评估记录不一致")
    if max(ks) > len(artifacts.movie_ids):
        raise ValueError("k 不能超过候选电影数量")
    data = load_movielens(input_dir)
    all_train = pd.read_pickle(split_dir / "train.pkl")
    known_users = artifacts.encoders["user_id"].categories_[0].astype(np.int64)
    train = all_train.loc[all_train["user_id"].isin(known_users)].copy()
    if set(train["user_id"]) != set(known_users):
        raise ValueError("训练交互缺少模型词表中的用户")
    threshold = evaluation["positive_threshold"]
    heldout = pd.read_pickle(split_dir / f"{split}.pkl")
    heldout = heldout.loc[
        heldout["user_id"].isin(known_users) & (heldout["rating"] >= threshold)
    ].sort_values("user_id")
    if evaluation["max_users"] is not None:
        heldout = heldout.head(evaluation["max_users"])
    history = train
    if evaluation["include_validation_history"]:
        history = pd.concat([train, pd.read_pickle(split_dir / "validation.pkl")], ignore_index=True)
    contexts = build_retrieval_contexts(
        history, heldout, data.users, data.movies, artifacts.max_history, threshold,
    )
    pd.testing.assert_frame_equal(contexts, pd.read_pickle(context_path))
    if len(contexts) != evaluation["evaluated_users"]:
        raise ValueError("评估用户数量与记录不一致")

    popularity = build_popularity_index(train, artifacts.movie_ids, threshold)
    genres = build_genre_index(train, data.movies, artifacts.movie_ids, threshold)
    similarity = build_item_similarity_index(artifacts.movie_ids, artifacts.item_vectors, neighbors_per_item)
    k = max(ks)
    recommendations = {
        "popular": recall_popular(contexts, popularity, k),
        "genre_interest": recall_genres(contexts, genres, k),
        "item_similarity": recall_item_similarity(contexts, similarity, k, recent_items),
        "youtubednn": recall_users(contexts, artifacts, k, evaluation["batch_size"]),
    }
    channels = {}
    for source, results in recommendations.items():
        counts = results.groupby("user_id").size().reindex(contexts["user_id"], fill_value=0)
        channels[source] = {
            "candidates": len(results), "min_per_user": int(counts.min()),
            "max_per_user": int(counts.max()), "mean_per_user": float(counts.mean()),
            "users_without_candidates": int((counts == 0).sum()),
            "metrics": [evaluate_top_k(contexts, results, artifacts.movie_ids, value) for value in sorted(set(ks))],
        }
    selected_seeds = [item for history_ids in contexts["hist_movie_id"] for item in history_ids[-recent_items:]]
    report = {
        "evaluation": evaluation, "training_users": len(known_users), "training_rows": len(train),
        "candidate_budget_per_channel": k, "neighbors_per_item": neighbors_per_item,
        "recent_items": recent_items, "item_similarity_aggregation": "max",
        "selected_history_items": len(selected_seeds),
        "unavailable_history_items": int((~np.isin(selected_seeds, artifacts.movie_ids)).sum()),
        "channels": channels,
        "input_sha256": {
            **evaluation["input_sha256"], str(report_path.resolve()): _file_sha256(report_path),
            str(context_path.resolve()): _file_sha256(context_path),
        },
        "code_sha256": {
            str(path.relative_to(project_root)): _file_sha256(path)
            for path in sorted((project_root / "src").glob("**/*.py"))
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    popularity.to_pickle(output_dir / "popularity_index.pkl")
    genres.candidates.to_pickle(output_dir / "genre_index.pkl")
    similarity.to_pickle(output_dir / "item_similarity_index.pkl")
    for source, results in recommendations.items():
        results.to_csv(output_dir / f"{source}.csv.gz", index=False, compression="gzip")
    with (output_dir / "comparison_report.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="使用相同的评估设置比较四路召回")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--neighbors-per-item", type=int, default=50)
    parser.add_argument("--recent-items", type=int, default=3)
    args = parser.parse_args()
    report = run_recall_comparison(
        args.evaluation_dir, args.output_dir, tuple(args.ks), args.neighbors_per_item, args.recent_items,
    )
    print(json.dumps(report["channels"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
