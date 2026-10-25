import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tensorflow as tf

from evaluation.evaluate_retrieval import _file_sha256
from retrieval.genre_recall import GenreIndex, build_genre_index
from retrieval.item_similarity_recall import build_item_similarity_index
from retrieval.popular_recall import build_popularity_index
from ranking.ranking_predict import RankingArtifacts, load_ranking_artifacts
from retrieval.vector_recall import RetrievalArtifacts, load_retrieval_artifacts
from data.load import MovieLensData, validate_movielens
from features.ranking_features import FEATURE_VERSION


@dataclass(frozen=True)
class RecommendationBundle:
    directory: Path
    retrieval: RetrievalArtifacts
    ranking: RankingArtifacts
    users: pd.DataFrame
    movies: pd.DataFrame
    history: pd.DataFrame
    popularity: pd.DataFrame
    genres: GenreIndex
    similarity: pd.DataFrame
    config: dict
    manifest: dict


def write_json(path: Path, value: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)


def write_result_csv(path: Path, results: pd.DataFrame) -> None:
    serializable = results.copy()
    for column in ("sources", "source_scores", "source_ranks", "hist_movie_id", "seen_movie_ids", "hist_genres"):
        if column in serializable:
            serializable[column] = serializable[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    serializable.to_csv(path, index=False)


def require_empty_directory(directory: Path) -> Path:
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"请选择空的输出目录：{directory}")
    return directory


def runtime_code_sha256(project_root: Path) -> dict[str, str]:
    project_root = Path(project_root)
    return {
        str(path.relative_to(project_root)): _file_sha256(path)
        for path in sorted((project_root / "src").glob("**/*.py"))
    }


def validate_code_files(manifest: dict) -> None:
    if manifest["schema_version"] == 3:
        compatibility = manifest["runtime_compatibility"]
        if compatibility["feature_version"] not in {"categorical_v1", FEATURE_VERSION} or compatibility["model_implementation_version"] != 1 or compatibility["tensorflow_major_minor"] != ".".join(tf.__version__.split(".")[:2]):
            raise ValueError("模型目录与当前运行版本不兼容")
        return
    project_root = Path(__file__).resolve().parents[2]
    for filename, expected in manifest["code_sha256"].items():
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "src" or path.suffix != ".py":
            raise ValueError(f"代码路径无效：{filename}")
        if _file_sha256(project_root / path) != expected:
            raise ValueError(f"代码版本与模型产物不一致：{filename}")


def export_bundle(
    directory: Path, retrieval_dir: Path, ranking_dir: Path, users: pd.DataFrame,
    movies: pd.DataFrame, train: pd.DataFrame, config: dict, source_files: list[Path], project_root: Path,
    *, indexes_dir: Path | None = None, retrieval_origin: dict | None = None,
) -> dict:
    directory = require_empty_directory(directory)
    retrieval = load_retrieval_artifacts(retrieval_dir)
    ranking = load_ranking_artifacts(ranking_dir)
    if config["candidate_budget"] > len(retrieval.movie_ids) or config["neighbors_per_item"] >= len(retrieval.movie_ids):
        raise ValueError("候选数量或近邻数量超过训练电影目录的范围")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(retrieval_dir, directory / "retrieval")
    shutil.copytree(ranking_dir, directory / "ranking")
    data_dir = directory / "indexes"
    if indexes_dir is not None:
        shutil.copytree(indexes_dir, data_dir)
    else:
        popularity = build_popularity_index(train, retrieval.movie_ids, config["positive_threshold"])
        genres = build_genre_index(train, movies, retrieval.movie_ids, config["positive_threshold"])
        similarity = build_item_similarity_index(retrieval.movie_ids, retrieval.item_vectors, config["neighbors_per_item"])
        data_dir.mkdir()
        for name, table in {"users": users, "movies": movies, "history": train, "popularity": popularity, "similarity": similarity}.items():
            table.to_pickle(data_dir / f"{name}.pkl")
        with (data_dir / "genres.pkl").open("wb") as destination:
            pickle.dump(genres, destination)
    manifest = {
        "schema_version": 3, "config": config, "split_policy": "per_user_leave_two_out",
        "runtime_compatibility": {"feature_version": ranking.training_report.get("feature_version", "categorical_v1"), "model_implementation_version": 1, "tensorflow_major_minor": ".".join(tf.__version__.split(".")[:2])},
        "training_users": int(train["user_id"].nunique()), "training_rows": len(train),
        "catalog_size": len(retrieval.movie_ids),
        "files": {str(path.relative_to(directory)): _file_sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()},
        "source_sha256": {str(path.resolve()): _file_sha256(path) for path in source_files},
        "code_sha256": runtime_code_sha256(project_root),
    }
    if retrieval_origin is not None:
        for filename, expected in retrieval_origin["files"].items():
            if manifest["files"].get(filename) != expected:
                raise ValueError(f"复用召回产物的 SHA-256 不一致：{filename}")
        manifest["retrieval_origin"] = retrieval_origin
    write_json(directory / "manifest.json", manifest)
    return manifest


def validate_bundle_files(directory: Path) -> dict:
    directory = Path(directory)
    with (directory / "manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest["schema_version"] not in {2, 3}:
        raise ValueError("无法读取当前 Manifest 版本")
    actual_files = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file() and path.name != "manifest.json"}
    if actual_files != set(manifest["files"]):
        raise ValueError("模型目录的文件集合与 Manifest 不一致")
    for filename, expected in manifest["files"].items():
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts or _file_sha256(directory / path) != expected:
            raise ValueError(f"模型产物的 SHA-256 校验失败：{filename}")
    return manifest


def load_retrieval_source(directory: Path) -> tuple[dict, MovieLensData]:
    """复用训练产物时检查文件、源数据和召回代码，记录原模型目录的版本。"""
    directory = Path(directory)
    manifest = validate_bundle_files(directory)
    for filename, expected in manifest["source_sha256"].items():
        if _file_sha256(Path(filename)) != expected:
            raise ValueError(f"源数据的 SHA-256 校验失败：{filename}")
    validate_code_files(manifest)
    retrieval = load_retrieval_artifacts(directory / "retrieval")
    tables = [pd.read_pickle(directory / "indexes" / f"{name}.pkl") for name in ("users", "movies", "history")]
    data = MovieLensData(*tables)
    validate_movielens(data)
    if len(data.ratings) != manifest["training_rows"] or data.ratings["user_id"].nunique() != manifest["training_users"]:
        raise ValueError("召回历史快照的数量与 Manifest 不一致")
    if len(retrieval.movie_ids) != manifest["catalog_size"]:
        raise ValueError("召回电影目录的数量与 Manifest 不一致")
    return manifest, data


def load_bundle(directory: Path) -> RecommendationBundle:
    directory = Path(directory)
    manifest = validate_bundle_files(directory)
    validate_code_files(manifest)
    retrieval = load_retrieval_artifacts(directory / "retrieval")
    ranking = load_ranking_artifacts(directory / "ranking")
    config = manifest["config"]
    retrieval_settings = {
        "max_history": "max_history", "emb_dim": "retrieval_emb_dim", "num_sampled": "num_sampled",
        "temperature": "temperature", "epochs": "retrieval_epochs", "batch_size": "batch_size",
        "learning_rate": "learning_rate", "seed": "seed",
    }
    if "retrieval_seed" in config:
        retrieval_settings["seed"] = "retrieval_seed"
    if "retrieval_learning_rate" in config:
        retrieval_settings["learning_rate"] = "retrieval_learning_rate"
    ranking_settings = {
        "emb_dim": "ranking_emb_dim", "dropout": "ranking_dropout",
        "l2": "ranking_l2", "epochs": "ranking_epochs", "batch_size": "batch_size",
        "learning_rate": "learning_rate", "seed": "seed",
    }
    model_type = ranking.training_report.get("model_type", "deepfm")
    if model_type in {"deepfm", "dcnv2"}:
        ranking_settings["hidden_units"] = "ranking_hidden_units"
    if model_type == "dcnv2":
        ranking_settings["cross_layers"] = "ranking_cross_layers"
    if model_type == "rankmixer":
        from models.rankmixer import validate_rankmixer_config
        if config.get("ranking_model") != model_type or validate_rankmixer_config(config["ranking_model_config"]) != ranking.training_report["model_config"]:
            raise ValueError("RankMixer 模型配置与 Manifest 不一致")
    if ranking.numeric_scaler is not None:
        ranking_settings.update(query_batch_size="ranking_query_batch_size", patience="ranking_patience", selection_k="top_k")
    for report, settings in ((retrieval.training_report, retrieval_settings), (ranking.training_report, ranking_settings)):
        if any(report[name] != config[key] for name, key in settings.items()):
            raise ValueError("模型训练设置与 Manifest 不一致")
    data_dir = directory / "indexes"
    users, movies, history = (pd.read_pickle(data_dir / f"{name}.pkl") for name in ("users", "movies", "history"))
    validate_movielens(MovieLensData(users, movies, history))
    if len(history) != manifest["training_rows"] or history["user_id"].nunique() != manifest["training_users"]:
        raise ValueError("历史快照的数量与 Manifest 不一致")
    if len(retrieval.movie_ids) != manifest["catalog_size"]:
        raise ValueError("电影目录的数量与 Manifest 不一致")
    with (data_dir / "genres.pkl").open("rb") as source:
        genres = pickle.load(source)
    if not isinstance(genres, GenreIndex):
        raise TypeError("类型索引必须使用 GenreIndex")
    return RecommendationBundle(
        directory, retrieval, ranking, users, movies, history,
        pd.read_pickle(data_dir / "popularity.pkl"), genres, pd.read_pickle(data_dir / "similarity.pkl"),
        manifest["config"], manifest,
    )
