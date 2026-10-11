import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from training.retrieval_inputs import build_retrieval_inputs, fit_retrieval_encoders
from training.train_retrieval import train_retrieval
from models.youtubednn import build_youtubednn
from features.encode import encode_category
from training.retrieval_samples import build_retrieval_samples
from evaluation.evaluate_retrieval import evaluate_top_k
from evaluation.evaluate_retrieval import run_evaluation
from retrieval.genre_recall import build_genre_index, recall_genres, user_genre_weights
from retrieval.item_similarity_recall import build_item_similarity_index, recall_item_similarity, run_recall_comparison
from retrieval.popular_recall import build_popularity_index, recall_popular
from features.retrieval_context import build_retrieval_contexts
from retrieval.vector_recall import load_retrieval_artifacts, recall_users, top_k_recall
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retrieval_examples(source_tables):
    train = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "split" / "train.pkl")
    selected = train.loc[train["user_id"].isin(train["user_id"].unique()[:2])]
    return build_retrieval_samples(selected, source_tables["users"], source_tables["movies"])


def test_inputs_share_movie_codes_and_keep_latest_history(retrieval_examples):
    samples = retrieval_examples
    encoders = fit_retrieval_encoders(samples)
    features, targets = build_retrieval_inputs(samples, encoders, max_history=4)
    expected_movies = set(samples["movie_id"].astype(str)) | set(samples["hist_movie_id"].explode().astype(str))

    assert set(encoders["movie_id"].categories_[0]) == expected_movies
    assert "movie_id" not in features
    assert features["hist_movie_id"].shape == (len(samples), 4)
    assert targets.shape == (len(samples), 1)
    assert (targets >= 2).all()
    for position, movie_ids in enumerate(samples["hist_movie_id"]):
        expected = encode_category(pd.Series(movie_ids[-4:]), encoders["movie_id"]).to_numpy()
        actual = features["hist_movie_id"][position]
        np.testing.assert_array_equal(actual[-len(expected):], expected)
        assert (actual[:4 - len(expected)] == 0).all()


def test_transform_preserves_vocabulary_for_another_real_user(retrieval_examples, source_tables):
    encoders = fit_retrieval_encoders(retrieval_examples)
    original_movies = encoders["movie_id"].categories_[0].copy()
    train = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "split" / "train.pkl")
    other_user = train["user_id"].unique()[2]
    samples = build_retrieval_samples(
        train.loc[train["user_id"] == other_user], source_tables["users"], source_tables["movies"]
    )
    features, targets = build_retrieval_inputs(samples, encoders)
    known = samples["movie_id"].astype(str).isin(original_movies).to_numpy()

    assert (features["user_id"] == 1).all()
    assert (~known).any()
    assert (targets[~known] == 1).all()
    np.testing.assert_array_equal(encoders["movie_id"].categories_[0], original_movies)


def test_inputs_reject_mismatched_history_lengths(retrieval_examples):
    invalid = retrieval_examples.copy(deep=True)
    invalid.at[invalid.index[0], "hist_genres"] = []

    with pytest.raises(ValueError, match="长度相同"):
        fit_retrieval_encoders(invalid)


def test_user_tower_ignores_padding_positions(retrieval_examples):
    encoders = fit_retrieval_encoders(retrieval_examples)
    features, targets = build_retrieval_inputs(retrieval_examples, encoders, max_history=4)
    vocab_sizes = {name: len(encoder.categories_[0]) + 2 for name, encoder in encoders.items()}
    _, user_tower, item_tower = build_youtubednn(vocab_sizes, max_history=4, emb_dim=8, num_sampled=4)
    shifted = {name: values.copy() for name, values in features.items()}
    for name in ("hist_movie_id", "hist_genres"):
        shifted[name] = np.roll(shifted[name], 1, axis=1)

    vectors = user_tower(features, training=False).numpy()
    np.testing.assert_allclose(vectors, user_tower(shifted, training=False).numpy(), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    item_vectors = item_tower({"movie_id": targets}, training=False).numpy()
    np.testing.assert_allclose(np.linalg.norm(item_vectors, axis=1), 1, atol=1e-6)


def test_sampled_softmax_updates_known_movie_embeddings(retrieval_examples):
    encoders = fit_retrieval_encoders(retrieval_examples)
    features, targets = build_retrieval_inputs(retrieval_examples, encoders, max_history=4)
    vocab_sizes = {name: len(encoder.categories_[0]) + 2 for name, encoder in encoders.items()}
    trainer, _, _ = build_youtubednn(vocab_sizes, max_history=4, emb_dim=8, num_sampled=4)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    before = trainer.movie_embedding.embeddings.numpy().copy()

    result = trainer.train_on_batch(features, targets, return_dict=True)
    after = trainer.movie_embedding.embeddings.numpy()

    assert np.isfinite(result["loss"])
    assert not np.array_equal(before[2:], after[2:])
    np.testing.assert_array_equal(before[:2], after[:2])


def test_exported_models_keep_movie_mapping_and_vectors(retrieval_examples, tmp_path):
    report = train_retrieval(
        retrieval_examples, tmp_path, max_history=4, emb_dim=8,
        num_sampled=4, epochs=1, batch_size=32,
    )
    with (tmp_path / "encoders.pkl").open("rb") as source:
        encoders = pickle.load(source)
    movie_ids = np.load(tmp_path / "movie_ids.npy")
    vectors = np.load(tmp_path / "item_embeddings.npy")
    item_tower = tf.keras.models.load_model(tmp_path / "item_tower.keras", compile=False)
    user_tower = tf.keras.models.load_model(tmp_path / "user_tower.keras", compile=False)
    codes = np.arange(2, len(movie_ids) + 2, dtype=np.int32)[:, None]
    features, _ = build_retrieval_inputs(retrieval_examples, encoders, max_history=4)

    np.testing.assert_array_equal(movie_ids, encoders["movie_id"].categories_[0].astype(np.int64))
    np.testing.assert_allclose(vectors, item_tower({"movie_id": codes}).numpy(), atol=1e-6)
    assert np.isfinite(user_tower(features).numpy()).all()
    assert report["samples"] == len(retrieval_examples)
    assert np.isfinite(report["sampled_softmax_loss"]).all()


def test_training_requires_an_empty_output_directory(retrieval_examples, tmp_path):
    retrieval_examples.to_pickle(tmp_path / "existing-samples.pkl")

    with pytest.raises(FileExistsError):
        train_retrieval(retrieval_examples, tmp_path)


@pytest.fixture(scope="module")
def evaluation_tables(source_tables):
    split_dir = PROJECT_ROOT / "data" / "processed" / "split"
    train = pd.read_pickle(split_dir / "train.pkl")
    user_ids = train["user_id"].unique()[:20]
    return {
        "train": train.loc[train["user_id"].isin(user_ids)].copy(),
        "validation": pd.read_pickle(split_dir / "validation.pkl").loc[
            lambda table: table["user_id"].isin(user_ids)
        ].copy(),
        "test": pd.read_pickle(split_dir / "test.pkl").loc[
            lambda table: table["user_id"].isin(user_ids)
        ].copy(),
        "users": source_tables["users"], "movies": source_tables["movies"],
    }


@pytest.fixture(scope="module")
def evaluation_contexts(evaluation_tables):
    tables = evaluation_tables
    heldout = tables["validation"].loc[tables["validation"]["rating"] >= 4]
    return build_retrieval_contexts(
        tables["train"], heldout, tables["users"], tables["movies"], max_history=4,
    )


@pytest.fixture(scope="module")
def evaluation_artifact_dir(evaluation_tables, tmp_path_factory):
    tables = evaluation_tables
    samples = build_retrieval_samples(tables["train"], tables["users"], tables["movies"])
    directory = tmp_path_factory.mktemp("retrieval-evaluation")
    train_retrieval(
        samples, directory, max_history=4, emb_dim=8,
        num_sampled=4, epochs=1, batch_size=128,
    )
    return directory


@pytest.fixture(scope="module")
def evaluation_artifacts(evaluation_artifact_dir):
    return load_retrieval_artifacts(evaluation_artifact_dir)


def test_evaluation_context_uses_only_past_positive_history(evaluation_tables, evaluation_contexts):
    for context in evaluation_contexts.itertuples(index=False):
        past = evaluation_tables["train"].loc[
            evaluation_tables["train"]["user_id"] == context.user_id
        ].sort_values("timestamp", kind="stable")
        liked = past.loc[past["rating"] >= 4, "movie_id"].tolist()
        assert context.hist_movie_id == liked[-4:]
        assert context.history_length == len(liked)
        assert set(context.seen_movie_ids) == set(past["movie_id"])
        assert context.movie_id not in context.seen_movie_ids
        assert past["timestamp"].max() <= context.as_of_timestamp
        genres = evaluation_tables["movies"].set_index("movie_id")["genres"]
        assert context.hist_genres == [genres.loc[item].split("|")[0] for item in liked[-4:]]
    low_rating_movies = evaluation_tables["train"].loc[
        evaluation_tables["train"]["rating"] < 4, ["user_id", "movie_id"]
    ]
    seen = evaluation_contexts.set_index("user_id")["seen_movie_ids"]
    checked = low_rating_movies.loc[low_rating_movies["user_id"].isin(seen.index)]
    assert not checked.empty
    assert all(row.movie_id in seen.loc[row.user_id] for row in checked.itertuples(index=False))


def test_test_context_can_include_validation_history(evaluation_tables):
    tables = evaluation_tables
    history = pd.concat([tables["train"], tables["validation"]], ignore_index=True)
    heldout = tables["test"].loc[tables["test"]["rating"] >= 4]
    contexts = build_retrieval_contexts(
        history, heldout, tables["users"], tables["movies"], max_history=4,
    )
    validation = tables["validation"].set_index("user_id")
    for context in contexts.itertuples(index=False):
        previous = validation.loc[context.user_id]
        assert previous["movie_id"] in context.seen_movie_ids
        if previous["rating"] >= 4:
            assert context.hist_movie_id[-1] == previous["movie_id"]
        else:
            assert previous["movie_id"] not in context.hist_movie_id


def test_context_rejects_real_future_interactions(evaluation_tables):
    tables = evaluation_tables
    heldout = tables["validation"].loc[tables["validation"]["rating"] >= 4]
    times = heldout.set_index("user_id")["timestamp"]
    future = tables["test"].loc[
        tables["test"]["timestamp"] > tables["test"]["user_id"].map(times)
    ]
    assert not future.empty
    history = pd.concat([tables["train"], future], ignore_index=True)
    with pytest.raises(ValueError, match="晚于"):
        build_retrieval_contexts(history, heldout, tables["users"], tables["movies"])


def test_recall_matches_full_catalog_scores_and_filters_seen(evaluation_contexts, evaluation_artifacts):
    artifacts = evaluation_artifacts
    results = recall_users(evaluation_contexts, artifacts, k=10, batch_size=3)
    features, _ = build_retrieval_inputs(evaluation_contexts, artifacts.encoders, max_history=4)
    vectors = artifacts.user_tower(features, training=False).numpy()
    for position, context in enumerate(evaluation_contexts.itertuples(index=False)):
        expected = pd.DataFrame({
            "movie_id": artifacts.movie_ids,
            "score": artifacts.item_vectors @ vectors[position],
        })
        expected = expected.loc[~expected["movie_id"].isin(context.seen_movie_ids)]
        expected = expected.sort_values(["score", "movie_id"], ascending=[False, True]).head(10)
        actual = results.loc[results["user_id"] == context.user_id]
        np.testing.assert_array_equal(actual["movie_id"], expected["movie_id"])
        np.testing.assert_allclose(actual["source_score"], expected["score"], atol=1e-6)
        assert actual["source_rank"].tolist() == list(range(1, len(actual) + 1))
        assert set(actual["source"]) == {"youtubednn"}


def test_metrics_match_independent_rank_calculation(evaluation_contexts, evaluation_artifacts):
    artifacts = evaluation_artifacts
    catalog_size = len(artifacts.movie_ids)
    results = recall_users(evaluation_contexts, artifacts, k=catalog_size)
    for k in (10, 20, catalog_size):
        report = evaluate_top_k(evaluation_contexts, results, artifacts.movie_ids, k=k)
        ranks = []
        for context in evaluation_contexts.itertuples(index=False):
            items = results.loc[
                (results["user_id"] == context.user_id) & (results["source_rank"] <= k), "movie_id"
            ].tolist()
            ranks.append(items.index(context.movie_id) + 1 if context.movie_id in items else 0)
        count = len(ranks)
        assert report["evaluated_users"] == count
        assert report["recall_at_k"] == sum(rank > 0 for rank in ranks) / count
        assert report["ndcg_at_k"] == pytest.approx(
            sum(1 / np.log2(rank + 1) for rank in ranks if rank) / count
        )
        assert report["mrr_at_k"] == pytest.approx(sum(1 / rank for rank in ranks if rank) / count)
        if k == catalog_size:
            assert report["hits"] > 0
            assert any(rank > 1 for rank in ranks)


def test_metrics_keep_catalog_outside_targets_in_denominator(evaluation_contexts, evaluation_artifacts):
    artifacts = evaluation_artifacts
    reachable = evaluation_contexts["movie_id"].isin(artifacts.movie_ids)
    assert reachable.any()
    removed_target = evaluation_contexts.loc[reachable, "movie_id"].iloc[0]
    keep = artifacts.movie_ids != removed_target
    features, _ = build_retrieval_inputs(evaluation_contexts, artifacts.encoders, max_history=4)
    vectors = artifacts.user_tower(features, training=False).numpy()
    # 从真实目录中排除一个真实目标，检查目录覆盖不足时的评估分母。
    k = int(keep.sum())
    results = top_k_recall(
        evaluation_contexts["user_id"].to_numpy(), vectors,
        artifacts.movie_ids[keep], artifacts.item_vectors[keep],
        evaluation_contexts["seen_movie_ids"].tolist(), k=k,
    )
    report = evaluate_top_k(evaluation_contexts, results, artifacts.movie_ids[keep], k=k)
    assert report["evaluated_users"] == len(evaluation_contexts)
    unavailable = ~evaluation_contexts["movie_id"].isin(artifacts.movie_ids[keep])
    assert report["unavailable_targets"] == int(unavailable.sum())
    assert report["catalog_target_coverage"] == pytest.approx(1 - unavailable.mean())
    assert report["hits"] > 0
    assert report["recall_at_k"] == report["hits"] / len(evaluation_contexts)


def test_metrics_reject_recommendations_containing_seen_movies(evaluation_contexts, evaluation_artifacts):
    artifacts = evaluation_artifacts
    context = evaluation_contexts.iloc[:1]
    features, _ = build_retrieval_inputs(context, artifacts.encoders, max_history=4)
    vectors = artifacts.user_tower(features, training=False).numpy()
    results = top_k_recall(
        context["user_id"].to_numpy(), vectors,
        artifacts.movie_ids, artifacts.item_vectors, [[]], k=len(artifacts.movie_ids),
    )
    assert results["movie_id"].isin(context.iloc[0]["seen_movie_ids"]).any()
    with pytest.raises(ValueError, match="已看电影"):
        evaluate_top_k(context, results, artifacts.movie_ids, k=10)


def test_loader_rejects_changed_movie_id_order(evaluation_artifact_dir, tmp_path):
    directory = tmp_path / "changed-mapping"
    shutil.copytree(evaluation_artifact_dir, directory)
    movie_ids = np.load(directory / "movie_ids.npy", allow_pickle=False)
    np.save(directory / "movie_ids.npy", movie_ids[::-1])
    with pytest.raises(ValueError, match="顺序"):
        load_retrieval_artifacts(directory)


def test_empty_available_catalog_counts_as_a_miss(evaluation_contexts, evaluation_artifacts):
    artifacts = evaluation_artifacts
    context = evaluation_contexts.iloc[:1]
    keep = np.isin(artifacts.movie_ids, context.iloc[0]["seen_movie_ids"])
    assert keep.any()
    features, _ = build_retrieval_inputs(context, artifacts.encoders, max_history=4)
    vectors = artifacts.user_tower(features, training=False).numpy()
    k = min(10, int(keep.sum()))
    results = top_k_recall(
        context["user_id"].to_numpy(), vectors,
        artifacts.movie_ids[keep], artifacts.item_vectors[keep],
        context["seen_movie_ids"].tolist(), k=k,
    )
    report = evaluate_top_k(context, results, artifacts.movie_ids[keep], k=k)
    assert results.empty
    assert report["evaluated_users"] == 1
    assert report["recall_at_k"] == report["ndcg_at_k"] == report["mrr_at_k"] == 0


@pytest.fixture(scope="module")
def recall_indexes(evaluation_tables, evaluation_artifacts):
    artifacts = evaluation_artifacts
    train, movies = evaluation_tables["train"], evaluation_tables["movies"]
    return {
        "popular": build_popularity_index(train, artifacts.movie_ids),
        "genre": build_genre_index(train, movies, artifacts.movie_ids),
        "item": build_item_similarity_index(artifacts.movie_ids, artifacts.item_vectors, neighbors_per_item=50),
    }


def test_popularity_counts_only_real_positive_training_events(evaluation_tables, evaluation_artifacts, recall_indexes):
    train = evaluation_tables["train"]
    catalog = evaluation_artifacts.movie_ids
    actual = recall_indexes["popular"].set_index("movie_id")["popularity"]
    expected = train.loc[train["rating"] >= 4, "movie_id"].value_counts().reindex(catalog, fill_value=0)
    np.testing.assert_array_equal(actual.reindex(catalog), expected)
    assert set(actual.index) == set(catalog)
    assert actual.sum() == ((train["rating"] >= 4) & train["movie_id"].isin(catalog)).sum()
    assert (train["rating"] < 4).any()


def test_popular_recall_matches_counts_and_filters_all_seen(evaluation_contexts, recall_indexes):
    index = recall_indexes["popular"]
    counts = index.set_index("movie_id")["popularity"].to_dict()
    results = recall_popular(evaluation_contexts, index, k=20)
    for context in evaluation_contexts.itertuples(index=False):
        available = set(counts) - set(context.seen_movie_ids)
        expected = sorted(available, key=lambda item: (-counts[item], item))[:20]
        actual = results.loc[results["user_id"] == context.user_id]
        assert actual["movie_id"].tolist() == expected
        assert actual["source_score"].tolist() == [counts[item] for item in expected]
        assert actual["source_rank"].tolist() == list(range(1, len(expected) + 1))


def test_genre_index_and_multigenre_history_weights(evaluation_tables, recall_indexes):
    index = recall_indexes["genre"]
    probabilities = index.candidates.groupby("genre")["genre_probability"].sum()
    np.testing.assert_allclose(probabilities, 1, atol=1e-12)
    movie_id = next(item for item, genres in index.genres_by_movie.items() if len(genres) > 1)
    assert movie_id in set(evaluation_tables["movies"]["movie_id"])
    genres = index.genres_by_movie[movie_id]
    weights = user_genre_weights([movie_id], index.genres_by_movie)
    assert set(weights.index) == set(genres)
    np.testing.assert_allclose(weights, 1 / len(genres), atol=1e-12)
    assert weights.sum() == pytest.approx(1)


def test_genre_recall_matches_independent_weighted_counts(evaluation_contexts, recall_indexes):
    index = recall_indexes["genre"]
    counts = recall_indexes["popular"].set_index("movie_id")["popularity"].to_dict()
    genre_totals = {}
    for movie_id, count in counts.items():
        for genre in index.genres_by_movie[movie_id]:
            genre_totals[genre] = genre_totals.get(genre, 0) + count
    results = recall_genres(evaluation_contexts, index, k=20)
    for context in evaluation_contexts.itertuples(index=False):
        interests = {}
        for movie_id in context.hist_movie_id:
            genres = index.genres_by_movie[movie_id]
            for genre in genres:
                interests[genre] = interests.get(genre, 0) + 1 / (len(genres) * len(context.hist_movie_id))
        scores = {}
        for movie_id, count in counts.items():
            score = sum(
                interests.get(genre, 0) * count / genre_totals[genre]
                for genre in index.genres_by_movie[movie_id] if genre_totals[genre] > 0
            )
            if score > 0 and movie_id not in context.seen_movie_ids:
                scores[movie_id] = score
        expected = sorted(scores, key=lambda item: (-scores[item], item))[:20]
        actual = results.loc[results["user_id"] == context.user_id]
        assert actual["movie_id"].tolist() == expected
        np.testing.assert_allclose(actual["source_score"], [scores[item] for item in expected], atol=1e-12)


def test_item_neighbors_match_sklearn_cosine_similarity(evaluation_artifacts, recall_indexes):
    artifacts = evaluation_artifacts
    scores = cosine_similarity(artifacts.item_vectors)
    for position in (0, len(artifacts.movie_ids) // 2, len(artifacts.movie_ids) - 1):
        seed = artifacts.movie_ids[position]
        actual = recall_indexes["item"].loc[lambda table: table["seed_movie_id"] == seed]
        available = [i for i, movie_id in enumerate(artifacts.movie_ids) if movie_id != seed]
        expected = sorted(available, key=lambda i: (-scores[position, i], artifacts.movie_ids[i]))[:50]
        assert actual["movie_id"].tolist() == artifacts.movie_ids[expected].tolist()
        np.testing.assert_allclose(actual["similarity"], scores[position, expected], atol=1e-6)
        assert actual["neighbor_rank"].tolist() == list(range(1, 51))
        assert seed not in set(actual["movie_id"])


def test_item_recall_uses_recent_seeds_and_max_similarity(evaluation_contexts, recall_indexes):
    neighbors = {}
    for row in recall_indexes["item"].itertuples(index=False):
        neighbors.setdefault(row.seed_movie_id, []).append((row.movie_id, row.similarity))
    results = recall_item_similarity(evaluation_contexts, recall_indexes["item"], k=20, recent_items=3)
    for context in evaluation_contexts.itertuples(index=False):
        scores = {}
        for seed in context.hist_movie_id[-3:]:
            for movie_id, score in neighbors.get(seed, []):
                if movie_id not in context.seen_movie_ids:
                    scores[movie_id] = max(scores.get(movie_id, -np.inf), score)
        expected = sorted(scores, key=lambda item: (-scores[item], item))[:20]
        actual = results.loc[results["user_id"] == context.user_id]
        assert actual["movie_id"].tolist() == expected
        np.testing.assert_allclose(actual["source_score"], [scores[item] for item in expected], atol=1e-12)
        assert actual["source_rank"].tolist() == list(range(1, len(expected) + 1))


def test_real_user_before_first_interaction_has_only_popular_candidates(evaluation_tables, recall_indexes):
    # 使用真实用户在首次交互之前的状态，检查零历史路径。
    user_id = int(evaluation_tables["train"].iloc[0]["user_id"])
    contexts = pd.DataFrame({"user_id": [user_id], "hist_movie_id": [[]], "seen_movie_ids": [[]]})
    assert len(recall_popular(contexts, recall_indexes["popular"], k=20)) == 20
    assert recall_genres(contexts, recall_indexes["genre"], k=20).empty
    assert recall_item_similarity(contexts, recall_indexes["item"], k=20).empty


def test_item_recall_returns_empty_when_real_history_has_no_exported_vectors(
    evaluation_contexts, evaluation_artifacts,
):
    context = evaluation_contexts.iloc[:1]
    artifacts = evaluation_artifacts
    keep = ~np.isin(artifacts.movie_ids, context.iloc[0]["hist_movie_id"])
    index = build_item_similarity_index(artifacts.movie_ids[keep], artifacts.item_vectors[keep], 50)
    results = recall_item_similarity(context, index, k=20)
    assert results.empty
    report = evaluate_top_k(context, results, artifacts.movie_ids[keep], k=20)
    assert report["hits"] == 0


def test_new_channels_reject_invalid_history_and_self_neighbors(evaluation_contexts, recall_indexes):
    invalid = evaluation_contexts.iloc[:1].copy(deep=True)
    invalid.at[invalid.index[0], "seen_movie_ids"] = []
    with pytest.raises(ValueError, match="必须属于"):
        recall_genres(invalid, recall_indexes["genre"])
    with pytest.raises(ValueError, match="必须属于"):
        recall_item_similarity(invalid, recall_indexes["item"])
    index = recall_indexes["item"].copy()
    index.at[index.index[0], "movie_id"] = index.iloc[0]["seed_movie_id"]
    with pytest.raises(ValueError, match="电影自身"):
        recall_item_similarity(evaluation_contexts, index)


def test_four_channel_comparison_is_reproducible_and_checks_contexts(evaluation_artifact_dir, tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    run_evaluation(evaluation_artifact_dir, evaluation_dir, project_root=PROJECT_ROOT)
    first_dir, second_dir = tmp_path / "first-comparison", tmp_path / "second-comparison"
    first = run_recall_comparison(evaluation_dir, first_dir, project_root=PROJECT_ROOT)
    second = run_recall_comparison(evaluation_dir, second_dir, project_root=PROJECT_ROOT)
    assert first == second
    assert first["training_users"] == 20
    assert len(first["channels"]) == 4
    assert first["evaluation"]["evaluated_users"] == 12
    for source, channel in first["channels"].items():
        pd.testing.assert_frame_equal(pd.read_csv(first_dir / f"{source}.csv.gz"), pd.read_csv(second_dir / f"{source}.csv.gz"))
        assert channel["max_per_user"] <= 20
        assert all(metric["evaluated_users"] == 12 for metric in channel["metrics"])
        assert channel["metrics"][-1]["catalog_size"] == 778
    contexts = pd.read_pickle(evaluation_dir / "contexts.pkl")
    contexts = contexts.iloc[::-1].reset_index(drop=True)
    contexts.to_pickle(evaluation_dir / "contexts.pkl")
    with pytest.raises(AssertionError):
        run_recall_comparison(evaluation_dir, tmp_path / "changed-context", project_root=PROJECT_ROOT)
