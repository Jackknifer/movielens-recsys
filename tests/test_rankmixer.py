import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from evaluation.compare_rankers import paired_bootstrap, per_user_metrics
from evaluation.evaluate_pipeline import recommend_from_snapshot, run_pipeline_evaluation
from models.ranker_factory import build_ranker, initialize_common_embeddings, matrix_macs_per_candidate
from models.rankmixer import TokenMixing, build_rankmixer, token_group_specification, validate_rankmixer_config
from pipeline.artifact_store import load_bundle
from pipeline.run_pipeline import load_config, recommend_from_bundle, run_ranking_training, run_training
from training.ranking_dataset import export_ranking_dataset, load_ranking_dataset
from training.ranking_inputs import RANKING_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ranking_dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("rankmixer")
    config = load_config(ROOT / "config" / "offline.yaml")
    config.update(max_users=20, retrieval_epochs=1, ranking_epochs=1)
    run_training(config, directory / "original")
    export_ranking_dataset(directory / "original", directory / "dataset", ROOT)
    return load_ranking_dataset(directory / "dataset")


@pytest.fixture(scope="module")
def rankmixer_bundle(ranking_dataset):
    destination = ranking_dataset.directory.parent / "rankmixer"
    run_ranking_training(ranking_dataset.directory / "reference-bundle", destination, epochs=1,
                         config_path=ROOT / "config" / "rankmixer.yaml", ranking_dataset_dir=ranking_dataset.directory)
    return load_bundle(destination)


def fresh_model(dataset, **config):
    vocab = {name: len(encoder.categories_[0]) + 2 for name, encoder in dataset.encoders.items()}
    tf.keras.utils.set_random_seed(42)
    return build_rankmixer(vocab, model_config=config)


def test_token_mixing_moves_real_feature_groups(ranking_dataset):
    model = fresh_model(ranking_dataset)
    inputs = {name: values[:16] for name, values in ranking_dataset.inputs("train").items()}
    tokens = tf.keras.Model(model.inputs, model.get_layer("feature_tokens").output)(inputs).numpy()
    actual = TokenMixing(8, 40, 8)(tokens).numpy()
    expected = np.empty_like(tokens)
    for destination in range(8):
        for source in range(8):
            expected[:, destination, source * 5:(source + 1) * 5] = tokens[:, source, destination * 5:(destination + 1) * 5]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(TokenMixing(8, 40, 8)(actual).numpy(), tokens)


def test_per_token_ffn_has_independent_real_parameters(ranking_dataset):
    model = fresh_model(ranking_dataset)
    inputs = {name: values[:16] for name, values in ranking_dataset.inputs("train").items()}
    tokens = tf.keras.Model(model.inputs, model.get_layer("mix_norm_0").output)(inputs)
    layer = model.get_layer("ffn_in_0")
    before = layer(tokens).numpy()
    kernel = layer.kernel.numpy()
    kernel[0, 0, 0] += 1
    layer.kernel.assign(kernel)
    after = layer(tokens).numpy()
    np.testing.assert_array_equal(before[:, 1:], after[:, 1:])
    assert np.max(np.abs(before[:, 0] - after[:, 0])) > 0


@pytest.mark.parametrize("config,parameters,macs", [
    ({}, 111641, 109080),
    ({"mixing": False}, 111641, 109080),
    ({"shared_ffn": True}, 20361, 109080),
    ({"shared_ffn": True, "ffn_hidden_dim": 640}, 111081, 825880),
])
def test_rankmixer_budget_matches_real_model(ranking_dataset, config, parameters, macs):
    model = fresh_model(ranking_dataset, **config)
    embedding = sum(model.get_layer(f"{name}_embedding").count_params() for name in RANKING_FEATURE_COLUMNS)
    assert model.count_params() - embedding == parameters
    assert matrix_macs_per_candidate(model) == macs


def test_model_families_share_seeded_embedding_initialization(ranking_dataset):
    vocab = {name: len(encoder.categories_[0]) + 2 for name, encoder in ranking_dataset.encoders.items()}
    first, second = build_ranker(vocab), build_ranker(vocab, model_type="rankmixer")
    assert initialize_common_embeddings(first, vocab, 16, 42) == initialize_common_embeddings(second, vocab, 16, 42)
    for name in vocab:
        np.testing.assert_array_equal(first.get_layer(f"{name}_embedding").get_weights()[0], second.get_layer(f"{name}_embedding").get_weights()[0])


def test_invalid_rankmixer_shapes_fail():
    for config in ({"token_count": 7}, {"num_heads": 4}, {"hidden_dim": 41}, {"num_blocks": True}, {"ffn_hidden_dim": 0}, {"mixing": 1}):
        with pytest.raises(ValueError):
            validate_rankmixer_config(config)


def test_trained_rankmixer_reloads_in_independent_process(ranking_dataset, rankmixer_bundle, tmp_path):
    inputs = {name: values[:32] for name, values in ranking_dataset.inputs("validation").items()}
    before = rankmixer_bundle.ranking.model(inputs, training=False).numpy()
    output = tmp_path / "reloaded.npy"
    subprocess.run([sys.executable, str(ROOT / "tests" / "helpers" / "reload_ranker.py"), str(ranking_dataset.directory), str(rankmixer_bundle.directory / "ranking"), str(output)], cwd=ROOT,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    np.testing.assert_allclose(np.load(output, allow_pickle=False), before, rtol=1e-6, atol=1e-6)
    report = rankmixer_bundle.ranking.training_report
    assert report["token_groups"] == token_group_specification()
    assert report["model_type"] == "rankmixer"
    assert report["serialization_max_abs_error"] <= 1e-6


def test_frozen_and_live_rankmixer_recommendations_match(ranking_dataset, rankmixer_bundle):
    cached = recommend_from_snapshot(rankmixer_bundle, ranking_dataset, "validation")
    queries = cached["contexts"][["user_id", "as_of_timestamp"]]
    live = recommend_from_bundle(rankmixer_bundle, queries)
    pd.testing.assert_frame_equal(cached["merged"], live["merged"], check_exact=False, rtol=1e-6, atol=1e-7)
    keys = ["user_id", "movie_id"]
    first, second = cached["ranked"].sort_values(keys), live["ranked"].sort_values(keys)
    np.testing.assert_allclose(first["ranking_score"], second["ranking_score"], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(first["rank"], second["rank"])


def test_rankmixer_real_cold_start_predictions(rankmixer_bundle):
    history = rankmixer_bundle.history
    user = int(history.iloc[0]["user_id"])
    first = int(history.loc[history["user_id"] == user, "timestamp"].min())
    result = recommend_from_bundle(rankmixer_bundle, pd.DataFrame({"user_id": [user], "as_of_timestamp": [first - 1]}))
    assert result["contexts"].iloc[0]["user_group"] == "no_history"
    assert len(result["final"]) == 20
    assert np.isfinite(result["final"]["ranking_score"]).all()


def test_full_rankmixer_flow_uses_real_history_groups(tmp_path):
    config = load_config(ROOT / "config" / "rankmixer.yaml")
    config.update(max_users=20, retrieval_epochs=1, ranking_epochs=1)
    destination = tmp_path / "full-rankmixer"
    run_training(config, destination)
    bundle = load_bundle(destination)
    query_path = tmp_path / "queries.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "create_example_queries.py"),
                    "--bundle", str(destination), "--output", str(query_path)], cwd=ROOT,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    result = recommend_from_bundle(bundle, pd.read_csv(query_path))
    assert result["contexts"]["user_group"].tolist() == ["no_history", "short_history", "warm"]
    assert len(result["final"]) == 60
    assert np.isfinite(result["final"]["ranking_score"]).all()
    report = run_pipeline_evaluation(destination, tmp_path / "validation", ks=(20,))
    assert report["ranking_model"] == "rankmixer"


def test_real_paired_metrics_agree_with_evaluation(ranking_dataset, rankmixer_bundle, tmp_path):
    output = tmp_path / "evaluation"
    report = run_pipeline_evaluation(rankmixer_bundle.directory, output, split="test", ks=(20,), ranking_dataset_dir=ranking_dataset.directory, save_details=False)
    scored = pd.read_csv(output / "ranked.csv.gz")
    metrics = per_user_metrics(ranking_dataset.frame("test-records"), ranking_dataset.frame("test-targets"), scored)
    expected = next(item for item in report["stages"]["ranked"]["top_k"] if item["k"] == 20)
    assert metrics["recall_at_20"].mean() == expected["recall_at_k"]
    assert metrics["ndcg_at_20"].mean() == pytest.approx(expected["ndcg_at_k"])
    differences = metrics["ndcg_at_20"].to_numpy() - metrics["recall_at_20"].to_numpy()
    first, second = paired_bootstrap(differences, 200), paired_bootstrap(differences, 200)
    assert first == second and first["ci95"][0] <= first["difference"] <= first["ci95"][1]
    with pytest.raises(AssertionError):
        per_user_metrics(ranking_dataset.frame("test-records"), ranking_dataset.frame("test-targets"), scored.iloc[:-1])
