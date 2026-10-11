import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow_recommenders.layers.feature_interaction import Cross

from pipeline.artifact_store import export_bundle, load_bundle, load_retrieval_source, validate_bundle_files, write_json
from retrieval.cold_start import classify_users
from models.deepfm import FactorizationMachine, build_deepfm
from models.dcnv2 import build_dcnv2
from reranking.diversity_rerank import movie_diversity_attributes, rerank_candidates
from evaluation.evaluate_pipeline import run_pipeline_evaluation
from evaluation.evaluate_retrieval import _file_sha256
from evaluation.evaluation_metrics import binary_metrics, diversity_metrics, top_k_metrics
from retrieval.merge_recall import snake_merge
from training.negative_sampling import sample_unobserved
from training.ranking_inputs import RANKING_FEATURE_COLUMNS, build_ranking_inputs, fit_ranking_encoders
from training.ranking_candidates import candidate_records
from features.ranking_features import RANKING_NUMERIC_COLUMNS, build_context_features
from ranking.ranking_predict import load_ranking_artifacts, predict_records, score_candidates
from training.ranking_samples import build_observed_samples, build_ranking_samples
from pipeline.run_pipeline import load_config, recommend_from_bundle, run_ranking_training, run_training
from features.user_context import build_user_contexts
from retrieval.vector_recall import recall_users
from training.train_ranking import listwise_loss
from features.feature_join import join_features
from training.retrieval_inputs import build_user_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline_bundle(tmp_path_factory):
    config = load_config(PROJECT_ROOT / "config" / "offline.yaml")
    config.update(max_users=20, retrieval_epochs=1, ranking_epochs=1)
    directory = tmp_path_factory.mktemp("pipeline-parent") / "bundle"
    run_training(config, directory)
    return load_bundle(directory)


@pytest.fixture(scope="module")
def pipeline_queries(pipeline_bundle):
    validation = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "split" / "validation.pkl")
    selected = validation.loc[validation["user_id"].isin(pipeline_bundle.history["user_id"])]
    return selected[["user_id", "timestamp"]].rename(columns={"timestamp": "as_of_timestamp"})


@pytest.fixture(scope="module")
def pipeline_result(pipeline_bundle, pipeline_queries):
    return recommend_from_bundle(pipeline_bundle, pipeline_queries)


def test_target_free_context_and_model_inputs(pipeline_bundle, pipeline_queries, pipeline_result):
    contexts = pipeline_result["contexts"]
    assert "movie_id" not in contexts and "rating" not in contexts
    for row in contexts.itertuples(index=False):
        events = pipeline_bundle.history.loc[lambda table: table["user_id"] == row.user_id].sort_values("timestamp", kind="stable")
        events = events.loc[events["timestamp"] <= row.as_of_timestamp]
        assert row.hist_movie_id == events.loc[events["rating"] >= 4, "movie_id"].tolist()[-10:]
        assert set(row.seen_movie_ids) == set(events["movie_id"])
    features = build_user_inputs(contexts, pipeline_bundle.retrieval.encoders)
    assert set(features) == set(pipeline_bundle.retrieval.user_tower.input_names)
    expected = recall_users(contexts, pipeline_bundle.retrieval, k=100)
    pd.testing.assert_frame_equal(expected, pipeline_result["channels"]["youtubednn"])


def test_real_zero_and_short_history_routes(pipeline_bundle):
    train = pipeline_bundle.history
    user_id = int(train.iloc[0]["user_id"])
    events = train.loc[train["user_id"] == user_id].sort_values("timestamp", kind="stable")
    first_time = int(events["timestamp"].min())
    queries = pd.DataFrame({"user_id": [user_id], "as_of_timestamp": [first_time - 1]})
    cold = recommend_from_bundle(pipeline_bundle, queries)
    assert cold["contexts"].iloc[0]["user_group"] == "no_history"
    assert len(cold["final"]) == 20
    assert all(cold["channels"][name].empty for name in ("genre_interest", "item_similarity", "youtubednn"))
    first_positive = events.loc[events["rating"] >= 4].iloc[:1]
    queries["as_of_timestamp"] = int(first_positive.iloc[0]["timestamp"])
    short = recommend_from_bundle(pipeline_bundle, queries, history=first_positive)
    assert short["contexts"].iloc[0]["user_group"] == "short_history"
    assert not short["channels"]["genre_interest"].empty
    assert short["channels"]["item_similarity"].empty and short["channels"]["youtubednn"].empty


def test_unseen_real_user_uses_attribute_and_history_routes(pipeline_bundle):
    train = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "split" / "train.pkl")
    user_id = next(item for item in train["user_id"].unique() if item not in set(pipeline_bundle.history["user_id"]))
    events = train.loc[train["user_id"] == user_id]
    queries = pd.DataFrame({"user_id": [int(user_id)], "as_of_timestamp": [int(events["timestamp"].max())]})
    result = recommend_from_bundle(pipeline_bundle, queries, events)
    assert result["contexts"].iloc[0]["user_group"] == "unseen_user"
    assert result["channels"]["youtubednn"].empty
    assert len(result["final"]) == 20


def test_snake_merge_preserves_all_sources_and_budget(pipeline_result):
    result = pipeline_result
    assert not result["merged"].duplicated(["user_id", "movie_id"]).any()
    for context in result["contexts"].itertuples(index=False):
        rows = result["merged"].loc[lambda table: table["user_id"] == context.user_id]
        assert len(rows) == 100
        assert not rows["movie_id"].isin(context.seen_movie_ids).any()
        assert rows["candidate_rank"].tolist() == list(range(1, 101))
        for row in rows.itertuples(index=False):
            sources = []
            for name, channel in result["channels"].items():
                candidate = channel.loc[(channel["user_id"] == row.user_id) & (channel["movie_id"] == row.movie_id)]
                if len(candidate):
                    sources.append(name)
                    assert row.source_scores[name] == candidate.iloc[0]["source_score"]
                    assert row.source_ranks[name] == candidate.iloc[0]["source_rank"]
            assert row.sources == sources
    assert result["merged"]["sources"].map(len).max() > 1


def test_snake_order_matches_independent_bidirectional_traversal(pipeline_result):
    context = pipeline_result["contexts"].iloc[:1]
    user_id = context.iloc[0]["user_id"]
    channels = {name: rows.loc[rows["user_id"] == user_id] for name, rows in pipeline_result["channels"].items()}
    lists = [rows.sort_values("source_rank")["movie_id"].tolist() for rows in channels.values()]
    pointers = [0] * len(lists)
    expected, seen = [], set(context.iloc[0]["seen_movie_ids"])
    forward = list(range(len(lists)))
    while len(expected) < 100:
        for position in forward + forward[::-1]:
            if pointers[position] < len(lists[position]):
                item = lists[position][pointers[position]]
                pointers[position] += 1
                if item not in seen:
                    expected.append(item)
                    seen.add(item)
                    if len(expected) == 100:
                        break
    assert snake_merge(context, channels, 100)["movie_id"].tolist() == expected


def test_real_negative_sampling_is_deterministic_and_excludes_all_interactions(pipeline_bundle):
    train = pipeline_bundle.history
    catalog = np.sort(train["movie_id"].unique())
    first = sample_unobserved(train, catalog, seed=42)
    second = sample_unobserved(train, catalog, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first.merge(train[["user_id", "movie_id"]], on=["user_id", "movie_id"]).empty
    assert set(first["movie_id"]).issubset(catalog)
    assert not first.duplicated(["user_id", "movie_id"]).any()
    samples = build_ranking_samples(train, pipeline_bundle.users, pipeline_bundle.movies)
    for kind, condition in (("positive", train["rating"] >= 4), ("explicit_negative", train["rating"] <= 2)):
        actual = samples.loc[samples["sample_kind"] == kind, ["user_id", "movie_id"]]
        expected = train.loc[condition, ["user_id", "movie_id"]]
        assert set(map(tuple, actual.to_numpy())) == set(map(tuple, expected.to_numpy()))


def test_fm_matches_explicit_pairwise_products_of_real_feature_embeddings(pipeline_bundle):
    records = build_observed_samples(pipeline_bundle.history, pipeline_bundle.users, pipeline_bundle.movies).iloc[:4]
    features = build_ranking_inputs(records, pipeline_bundle.ranking.encoders)
    model = pipeline_bundle.ranking.model
    embeddings = np.concatenate([model.get_layer(f"{name}_embedding")(values).numpy() for name, values in features.items()], axis=1)
    expected = sum((embeddings[:, i] * embeddings[:, j]).sum(axis=1) for i in range(embeddings.shape[1]) for j in range(i + 1, embeddings.shape[1]))
    actual = FactorizationMachine()(tf.convert_to_tensor(embeddings)).numpy().reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_ranking_reloads_and_scores_same_candidates(pipeline_bundle, pipeline_result):
    reloaded = load_bundle(pipeline_bundle.directory)
    scores, _ = score_candidates(pipeline_result["merged"], reloaded.ranking, reloaded.users, reloaded.movies, batch_size=37,
        contexts=pipeline_result["contexts"], retrieval=reloaded.retrieval, popularity=reloaded.popularity)
    pd.testing.assert_frame_equal(scores, pipeline_result["ranked"], atol=1e-6, rtol=1e-6)
    before = pipeline_bundle.ranking.encoders["movie_id"].categories_[0].copy()
    observed = candidate_records(pipeline_result, pipeline_bundle)
    predict_records(observed.iloc[:20], pipeline_bundle.ranking)
    np.testing.assert_array_equal(before, pipeline_bundle.ranking.encoders["movie_id"].categories_[0])


def test_dcnv2_cross_matches_numpy_on_real_trained_embeddings(pipeline_bundle):
    model = pipeline_bundle.ranking.model
    report = pipeline_bundle.ranking.training_report
    assert model.name == report["model_type"] == "dcnv2"
    assert report["architecture"] == "parallel" and report["cross_projection_dim"] is None
    layers = [layer for layer in model.layers if isinstance(layer, Cross)]
    assert len(layers) == report["cross_layers"] == 3
    records = pd.read_pickle(pipeline_bundle.directory / "ranking/training_sample.pkl").iloc[:8]
    inputs = build_ranking_inputs(records, pipeline_bundle.ranking.encoders, pipeline_bundle.ranking.numeric_scaler)
    base = tf.keras.Model(model.inputs, model.get_layer("base_features").output)(inputs).numpy()
    expected = base.copy()
    for layer in layers:
        kernel, bias = layer.get_weights()
        assert kernel.shape == (166, 166)
        expected = base * (expected @ kernel + bias) + expected
    actual = tf.keras.Model(model.inputs, model.get_layer("cross_2").output)(inputs).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert model.get_layer("combined_features").output_shape == (None, 198)
    assert report["serialization_max_abs_error"] <= 1e-6


def test_ranking_only_training_reuses_identical_retrieval_and_indexes(pipeline_bundle, pipeline_queries, pipeline_result, tmp_path):
    directory = tmp_path / "retrained-ranking"
    manifest = run_ranking_training(pipeline_bundle.directory, directory, epochs=4)
    original = pipeline_bundle.manifest
    reused = {name: digest for name, digest in original["files"].items() if name.startswith(("retrieval/", "indexes/"))}
    assert manifest["retrieval_origin"]["files"] == reused
    assert manifest["retrieval_origin"]["manifest_sha256"] == _file_sha256(pipeline_bundle.directory / "manifest.json")
    assert all(manifest["files"][name] == digest for name, digest in reused.items())
    replacement = load_bundle(directory)
    result = recommend_from_bundle(replacement, pipeline_queries)
    pd.testing.assert_frame_equal(pipeline_result["merged"], result["merged"])
    for name in result["channels"]:
        pd.testing.assert_frame_equal(pipeline_result["channels"][name], result["channels"][name])
    assert replacement.ranking.training_report["sample_kinds"] == pipeline_bundle.ranking.training_report["sample_kinds"]
    assert replacement.ranking.training_report["categories"] == pipeline_bundle.ranking.training_report["categories"]
    report = replacement.ranking.training_report
    best = max(report["history"], key=lambda row: (row["validation_ndcg"], row["validation_recall"]))
    assert report["selected_epoch"] == best["epoch"]
    assert report["selected_validation"]["ndcg_at_k"] == best["validation_ndcg"]


def test_real_deepfm_training_can_supply_retrieval_and_reload_scores(pipeline_bundle, tmp_path):
    config = pipeline_bundle.config
    records = build_observed_samples(pipeline_bundle.history, pipeline_bundle.users, pipeline_bundle.movies)
    assert set(records["label"]) == {0, 1}
    encoders = fit_ranking_encoders(records)
    vocab_sizes = {name: len(encoder.categories_[0]) + 2 for name, encoder in encoders.items()}
    features = build_ranking_inputs(records, encoders)
    tf.keras.utils.set_random_seed(config["seed"])
    model = build_deepfm(
        vocab_sizes, config["ranking_emb_dim"], tuple(config["ranking_hidden_units"]),
        config["ranking_dropout"], config["ranking_l2"],
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="binary_crossentropy")
    history = model.fit(
        features, records["label"].to_numpy(), epochs=config["ranking_epochs"],
        batch_size=config["batch_size"], verbose=0,
    )
    assert np.isfinite(history.history["loss"]).all()
    before = model({name: values[:16] for name, values in features.items()}, training=False).numpy().reshape(-1)
    ranking_dir = tmp_path / "ranking"
    ranking_dir.mkdir()
    model.save(ranking_dir / "ranker.keras")
    with (ranking_dir / "encoders.pkl").open("wb") as destination:
        pickle.dump(encoders, destination)
    write_json(ranking_dir / "training_report.json", {
        "model_type": model.name, "feature_version": "categorical_v1",
        "feature_columns": RANKING_FEATURE_COLUMNS, "vocab_sizes": vocab_sizes,
        "categories": {name: encoder.categories_[0].tolist() for name, encoder in encoders.items()},
        "emb_dim": config["ranking_emb_dim"], "hidden_units": config["ranking_hidden_units"],
        "dropout": config["ranking_dropout"], "l2": config["ranking_l2"],
        "epochs": config["ranking_epochs"], "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"], "seed": config["seed"],
        "rows": len(records), "loss": history.history["loss"],
    })
    directory = tmp_path / "deepfm-bundle"
    export_bundle(
        directory, pipeline_bundle.directory / "retrieval", ranking_dir,
        pipeline_bundle.users, pipeline_bundle.movies, pipeline_bundle.history, config,
        [Path(filename) for filename in pipeline_bundle.manifest["source_sha256"]], PROJECT_ROOT,
        indexes_dir=pipeline_bundle.directory / "indexes",
    )
    manifest, data = load_retrieval_source(directory)
    assert manifest["training_users"] == data.ratings["user_id"].nunique() == config["max_users"]
    ranking = load_ranking_artifacts(directory / "ranking")
    assert ranking.model.name == "deepfm"
    scores, _ = predict_records(records.iloc[:16], ranking)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    np.testing.assert_allclose(scores, before, atol=1e-6, rtol=1e-6)
    assert load_bundle(directory).ranking.model.name == "deepfm"


def test_ranking_only_training_checks_source_assets_before_training(pipeline_bundle, tmp_path):
    directory = tmp_path / "corrupted-source"
    shutil.copytree(pipeline_bundle.directory, directory)
    path = directory / "indexes" / "popularity.pkl"
    popularity = pd.read_pickle(path)
    popularity.loc[0, "popularity"] += 1
    popularity.to_pickle(path)
    with pytest.raises(ValueError, match="SHA-256"):
        run_ranking_training(directory, tmp_path / "replacement")
    assert not (tmp_path / "replacement").exists()


def test_dcnv2_rejects_invalid_cross_layer_count(pipeline_bundle):
    for count in (0, -1, True):
        with pytest.raises(ValueError, match="正整数"):
            build_dcnv2(pipeline_bundle.ranking.training_report["vocab_sizes"], cross_layers=count)


def test_bundle_records_validated_dcnv2_model_type(pipeline_bundle):
    assert validate_bundle_files(pipeline_bundle.directory) == pipeline_bundle.manifest
    assert pipeline_bundle.config["ranking_cross_layers"] == pipeline_bundle.ranking.training_report["cross_layers"]


def test_candidate_training_history_and_targets_are_separate_real_training_events(pipeline_bundle):
    directory = pipeline_bundle.directory / "ranking/training_context"
    history = pd.read_pickle(directory / "history.pkl")
    targets = pd.read_pickle(directory / "targets.pkl")
    queries = pd.read_pickle(directory / "queries.pkl")
    contexts = pd.read_pickle(directory / "contexts.pkl")
    keys = ["user_id", "movie_id"]
    assert history.merge(targets, on=keys).empty
    recovered = pd.concat([history, targets]).sort_values(keys).reset_index(drop=True)
    expected = pipeline_bundle.history.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(recovered, expected)
    times = queries.set_index("user_id")["as_of_timestamp"]
    assert history["timestamp"].le(history["user_id"].map(times)).all()
    assert targets["timestamp"].ge(targets["user_id"].map(times)).all()
    expected_counts = np.floor(pipeline_bundle.history.groupby("user_id").size() * 0.8).astype(int)
    pd.testing.assert_series_equal(history.groupby("user_id").size(), expected_counts)
    for row in contexts.itertuples(index=False):
        future = targets.loc[targets["user_id"] == row.user_id, "movie_id"]
        assert not set(future).intersection(row.seen_movie_ids)
    with (pipeline_bundle.directory / "ranking/candidate_training_report.json").open() as source:
        report = json.load(source)
    assert report["uses_validation_or_test_labels"] is False
    assert report["history_rows"] == len(history) and report["target_rows"] == len(targets)
    for filename, expected_digest in report["training_context_sha256"].items():
        assert _file_sha256(directory / filename) == expected_digest
    sample = pd.read_pickle(pipeline_bundle.directory / "ranking/training_sample.pkl")
    positives = targets.loc[targets["rating"] >= 4, keys]
    assert sample.loc[sample["label"] == 1, keys].merge(positives, on=keys).shape[0] == int(sample["label"].sum())


def test_context_features_use_real_recall_scores_and_change_with_history(pipeline_bundle, pipeline_result):
    records = candidate_records(pipeline_result, pipeline_bundle)
    assert len(RANKING_NUMERIC_COLUMNS) == 22
    assert np.isfinite(records[RANKING_NUMERIC_COLUMNS].to_numpy()).all()
    row = records.iloc[0]
    for channel, rank in row["source_ranks"].items():
        assert row[f"{channel}_reciprocal_rank"] == pytest.approx(1 / rank)
        score = row["source_scores"][channel]
        assert row[f"{channel}_score"] == pytest.approx(np.log1p(score) if channel == "popular" else score)
    assert row["merge_reciprocal_rank"] == pytest.approx(1 / row["candidate_rank"])
    user_id = int(row["user_id"])
    first_time = int(pipeline_bundle.history.loc[pipeline_bundle.history["user_id"] == user_id, "timestamp"].min())
    cold = recommend_from_bundle(pipeline_bundle, pd.DataFrame({"user_id": [user_id], "as_of_timestamp": [first_time - 1]}))
    cold_records = candidate_records(cold, pipeline_bundle)
    common = records.loc[records["user_id"] == user_id].merge(cold_records, on=["user_id", "movie_id"], suffixes=("_warm", "_cold"))
    assert not common.empty
    assert common["log_history_length_warm"].gt(0).all() and common["log_history_length_cold"].eq(0).all()
    assert common["genre_affinity_cold"].eq(0).all() and common["recent_mean_similarity_cold"].eq(0).all()
    assert common["youtubednn_present_cold"].eq(0).all()
    original = pipeline_bundle.ranking.numeric_scaler.mean_.copy()
    build_ranking_inputs(cold_records, pipeline_bundle.ranking.encoders, pipeline_bundle.ranking.numeric_scaler)
    np.testing.assert_array_equal(original, pipeline_bundle.ranking.numeric_scaler.mean_)


def test_listwise_loss_matches_independent_numpy_on_real_candidate_groups(pipeline_bundle):
    records = pd.read_pickle(pipeline_bundle.directory / "ranking/training_sample.pkl")
    users = records["user_id"].unique()[:2]
    records = records.loc[records["user_id"].isin(users)]
    inputs = build_ranking_inputs(records, pipeline_bundle.ranking.encoders, pipeline_bundle.ranking.numeric_scaler)
    logits = pipeline_bundle.ranking.model(inputs, training=False).numpy().reshape(-1)
    lengths = records.groupby("user_id", sort=False).size().to_numpy(dtype=np.int64)
    labels = records["label"].to_numpy(dtype=np.float32)
    expected, offset = [], 0
    for length in lengths:
        group_scores = logits[offset:offset + length].astype(float)
        group_labels = labels[offset:offset + length]
        log_probabilities = group_scores - np.logaddexp.reduce(group_scores)
        expected.append(-np.sum(group_labels / group_labels.sum() * log_probabilities))
        offset += length
    actual = listwise_loss(tf.constant(logits), tf.constant(labels), tf.constant(lengths))
    assert float(actual.numpy()) == pytest.approx(np.mean(expected), rel=1e-6)


def test_context_ranker_requires_declared_features(pipeline_bundle, pipeline_result):
    with pytest.raises(ValueError, match="上下文排序需要"):
        score_candidates(pipeline_result["merged"], pipeline_bundle.ranking, pipeline_bundle.users, pipeline_bundle.movies)
    records = candidate_records(pipeline_result, pipeline_bundle).iloc[:8]
    with pytest.raises(KeyError):
        predict_records(records.drop(columns="genre_affinity"), pipeline_bundle.ranking)


def test_reranking_preserves_candidates_and_scores_and_obeys_prefix_limits(pipeline_bundle, pipeline_result):
    ranked, reranked = pipeline_result["ranked"], pipeline_result["reranked"]
    keys = ["user_id", "movie_id"]
    original = ranked.drop(columns="rank").sort_values(keys).reset_index(drop=True)
    actual = reranked.drop(columns="rank").sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(original, actual)
    metrics = diversity_metrics(reranked, pipeline_bundle.movies, pipeline_bundle.popularity, pipeline_bundle.retrieval.movie_ids, pipeline_bundle.retrieval.item_vectors, 20)
    assert metrics["max_genre_run"] <= 2 and metrics["max_decade_run"] <= 2
    assert 0 <= metrics["intra_list_diversity"] <= 2


def test_reranking_rejects_unavailable_prefix_constraints(pipeline_bundle, pipeline_result):
    attributes = movie_diversity_attributes(pipeline_bundle.movies).set_index("movie_id")
    scored = pipeline_result["ranked"]
    user = scored.loc[scored["user_id"] == scored.iloc[0]["user_id"]].copy()
    user["primary_genre"] = user["movie_id"].map(attributes["primary_genre"])
    group = next(group for _, group in user.groupby("primary_genre") if len(group) >= 3)
    group = group.iloc[:3].drop(columns="primary_genre")
    group["rank"] = np.arange(1, len(group) + 1)
    with pytest.raises(ValueError, match="重排限制"):
        rerank_candidates(group, pipeline_bundle.movies, top_k=3, max_genre_run=1)


def test_binary_metrics_match_real_observed_predictions(pipeline_bundle):
    records = pd.read_pickle(pipeline_bundle.directory / "ranking/training_sample.pkl")
    records["ranking_score"], _ = predict_records(records, pipeline_bundle.ranking)
    metrics = binary_metrics(records, probability_scores=False)
    assert metrics["auc"] == pytest.approx(roc_auc_score(records["label"], records["ranking_score"]))
    assert metrics["log_loss"] is None
    eligible = [group for _, group in records.groupby("user_id") if group["label"].nunique() == 2]
    expected = sum(len(group) * roc_auc_score(group["label"], group["ranking_score"]) for group in eligible) / sum(len(group) for group in eligible)
    assert metrics["gauc"] == pytest.approx(expected)
    singleton = records.groupby("user_id").head(1)
    assert binary_metrics(singleton, probability_scores=False)["gauc"] is None


def test_full_evaluation_reports_fixed_budget_and_real_stages(pipeline_bundle, tmp_path):
    report = run_pipeline_evaluation(pipeline_bundle.directory, tmp_path / "evaluation")
    assert report["evaluated_users"] == 20 and report["positive_target_users"] == 12
    assert report["per_channel_budget"] == report["merged_budget"] == 100
    ceiling = [report["stages"][name]["top_k"][-1]["recall_at_k"] for name in ("merged", "ranked", "reranked")]
    assert ceiling[0] == ceiling[1] == ceiling[2]
    assert report["binary_metrics"]["gauc"] is None
    assert report["groups"]["no_history"]["users"] == 0
    assert report["groups"]["no_history"]["stages"]["merged"][0]["recall_at_k"] is None
    with (tmp_path / "evaluation" / "evaluation_report.json").open() as source:
        assert json.load(source) == report


def test_test_evaluation_updates_history_without_changing_indexes(pipeline_bundle, tmp_path):
    before = pipeline_bundle.popularity.copy(deep=True)
    report = run_pipeline_evaluation(pipeline_bundle.directory, tmp_path / "test", split="test", include_validation_history=True)
    contexts = pd.read_pickle(tmp_path / "test" / "contexts.pkl")
    validation = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "split" / "validation.pkl").set_index("user_id")
    for row in contexts.itertuples(index=False):
        assert int(validation.loc[row.user_id, "movie_id"]) in row.seen_movie_ids
    assert report["include_validation_history"]
    pd.testing.assert_frame_equal(before, load_bundle(pipeline_bundle.directory).popularity)


def test_bundle_checks_actual_file_corruption(pipeline_bundle, tmp_path):
    directory = tmp_path / "changed-bundle"
    shutil.copytree(pipeline_bundle.directory, directory)
    vectors = np.load(directory / "retrieval" / "item_embeddings.npy", allow_pickle=False)
    np.save(directory / "retrieval" / "item_embeddings.npy", vectors[::-1])
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(directory)


def test_invalid_requests_fail_before_recommendation(pipeline_bundle, pipeline_queries):
    invalid = pipeline_queries.iloc[:1].copy()
    invalid["movie_id"] = int(pipeline_bundle.retrieval.movie_ids[0])
    with pytest.raises(ValueError, match="queries"):
        recommend_from_bundle(pipeline_bundle, invalid)
    with pytest.raises(ValueError, match="top_k"):
        recommend_from_bundle(pipeline_bundle, pipeline_queries, top_k=101)
    with pytest.raises(FileExistsError):
        run_training(pipeline_bundle.config, pipeline_bundle.directory)
