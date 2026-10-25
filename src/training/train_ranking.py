import json
import pickle
import hashlib
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.preprocessing import StandardScaler

from models.ranker_factory import build_ranker, initialize_common_embeddings, matrix_macs_per_candidate
from models.rankmixer import MODEL_IMPLEMENTATION_VERSION, token_group_specification, validate_rankmixer_config
from evaluation.evaluation_metrics import top_k_metrics
from features.ranking_features import FEATURE_VERSION, RANKING_NUMERIC_COLUMNS
from training.ranking_inputs import build_ranking_inputs, fit_ranking_encoders, unknown_rates


def listwise_loss(logits, labels, row_lengths):
    scores = tf.RaggedTensor.from_row_lengths(tf.reshape(logits, [-1]), row_lengths).to_tensor(default_value=-1e9)
    targets = tf.RaggedTensor.from_row_lengths(labels, row_lengths).to_tensor(default_value=0.0)
    totals = tf.reduce_sum(targets, axis=1, keepdims=True)
    tf.debugging.assert_positive(totals, message="每份训练候选需要至少一个高评分目标")
    distribution = tf.stop_gradient(targets / totals)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=distribution, logits=scores))


def predict_inputs(model, features: dict, batch_size: int = 4096) -> np.ndarray:
    count = len(next(iter(features.values())))
    scores = [model({name: values[start:start + batch_size] for name, values in features.items()}, training=False).numpy().reshape(-1) for start in range(0, count, batch_size)]
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise ValueError("排序预测存在无效数值")
    return result


def validation_metrics(model, features: dict, records: pd.DataFrame, contexts: pd.DataFrame, targets: pd.DataFrame, catalog: np.ndarray, k: int) -> dict:
    scored = records[["user_id", "movie_id"]].copy()
    scored["ranking_score"] = predict_inputs(model, features)
    scored = scored.sort_values(["user_id", "ranking_score", "movie_id"], ascending=[True, False, True], kind="stable")
    scored["rank"] = scored.groupby("user_id").cumcount() + 1
    return top_k_metrics(contexts, targets, scored, catalog, k)


def train_ranking(
    samples: pd.DataFrame, output_dir: Path, validation: pd.DataFrame,
    validation_contexts: pd.DataFrame, validation_targets: pd.DataFrame, catalog: np.ndarray,
    emb_dim: int = 16, hidden_units: tuple[int, ...] = (64, 32), dropout: float = 0.1,
    l2: float = 1e-6, epochs: int = 10, batch_size: int = 128,
    learning_rate: float = 0.001, seed: int = 42, cross_layers: int = 3,
    query_batch_size: int = 32, patience: int = 3, selection_k: int = 20,
    model_type: str = "dcnv2", model_config: dict | None = None, prepared: dict | None = None,
) -> dict:
    started = time.perf_counter()
    integers = (epochs, batch_size, query_batch_size, patience, selection_k)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("训练轮次、批次大小、等待轮次和学习率必须有效")
    if samples.empty or not samples["label"].isin([0, 1]).all() or samples.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("列表训练样本的标签或唯一性无效")
    if validation.empty or validation_targets.empty:
        raise ValueError("模型选择需要非空验证候选和高评分目标")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"请选择空的输出目录：{output_dir}")
    if prepared is not None:
        pd.testing.assert_frame_equal(samples[["user_id", "movie_id", "candidate_rank"]].reset_index(drop=True), prepared["training_keys"].reset_index(drop=True))
    samples = samples.sort_values(["user_id", "candidate_rank"], kind="stable").reset_index(drop=True)
    if prepared is not None:
        pd.testing.assert_frame_equal(samples[["user_id", "movie_id", "candidate_rank"]], prepared["training_keys"].reset_index(drop=True))
    groups = samples.groupby("user_id", sort=False)
    if (groups["label"].sum() == 0).any():
        raise ValueError("每份训练候选需要至少一个高评分目标")
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    if prepared is None:
        encoders = fit_ranking_encoders(samples)
        scaler = StandardScaler().fit(samples[RANKING_NUMERIC_COLUMNS])
        features = build_ranking_inputs(samples, encoders, scaler)
        valid_features = build_ranking_inputs(validation, encoders, scaler)
    else:
        encoders, scaler = prepared["encoders"], prepared["scaler"]
        features, valid_features = prepared["train_features"], prepared["validation_features"]
        if any(len(values) != len(samples) for values in features.values()) or any(len(values) != len(validation) for values in valid_features.values()):
            raise ValueError("固定排序输入与候选记录的数量不一致")
    vocab_sizes = {name: len(encoder.categories_[0]) + 2 for name, encoder in encoders.items()}
    model = build_ranker(vocab_sizes, model_type, emb_dim, hidden_units, dropout, l2, cross_layers, len(RANKING_NUMERIC_COLUMNS), model_config, seed)
    embedding_digest = initialize_common_embeddings(model, vocab_sizes, emb_dim, seed) if prepared is not None else None
    scorer = tf.keras.Model(model.inputs, model.get_layer("ranking_logit").output)
    compiled_predictor = tf.function(model, reduce_retracing=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function(reduce_retracing=True)
    def train_step(batch_features, batch_labels, lengths):
        with tf.GradientTape() as tape:
            logits = scorer(batch_features, training=True)
            loss = listwise_loss(logits, batch_labels, lengths) + tf.add_n(scorer.losses)
        tf.debugging.assert_all_finite(loss, "排序训练损失无效")
        gradients = tape.gradient(loss, scorer.trainable_variables)
        for variable, gradient in zip(scorer.trainable_variables, gradients):
            if gradient is None:
                raise ValueError(f"排序参数没有梯度：{variable.name}")
            tf.debugging.assert_all_finite(gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient, "排序梯度无效")
        optimizer.apply_gradients(zip(gradients, scorer.trainable_variables))
        return loss

    indexes = list(groups.indices.values())
    labels = samples["label"].to_numpy(dtype=np.float32)
    generator = np.random.default_rng(seed)
    history, best_weights, best_key, selected_epoch, waiting = [], None, None, 0, 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        order = generator.permutation(len(indexes))
        total_loss = 0.0
        for start in range(0, len(order), query_batch_size):
            selected = [indexes[position] for position in order[start:start + query_batch_size]]
            positions = np.concatenate(selected)
            lengths = np.asarray([len(group) for group in selected], dtype=np.int64)
            batch_features = {name: values[positions] for name, values in features.items()}
            loss = train_step(batch_features, labels[positions], lengths)
            total_loss += float(loss.numpy()) * len(selected)
        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        metrics = validation_metrics(compiled_predictor, valid_features, validation, validation_contexts, validation_targets, catalog, selection_k)
        record = {"epoch": epoch, "loss": total_loss / len(indexes), "validation_recall": metrics["recall_at_k"], "validation_ndcg": metrics["ndcg_at_k"], "validation_hits": metrics["hits"], "training_seconds": training_seconds, "validation_seconds": time.perf_counter() - validation_started, "query_order_sha256": hashlib.sha256(order.tobytes()).hexdigest()}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        key = (metrics["ndcg_at_k"], metrics["recall_at_k"])
        if best_key is None or key > best_key:
            best_key, best_weights, selected_epoch, waiting = key, model.get_weights(), epoch, 0
        else:
            waiting += 1
        if waiting >= patience:
            break
    model.set_weights(best_weights)
    restored_metrics = validation_metrics(compiled_predictor, valid_features, validation, validation_contexts, validation_targets, catalog, selection_k)
    if (restored_metrics["ndcg_at_k"], restored_metrics["recall_at_k"]) != best_key:
        raise ValueError("保存模型的验证指标与选中轮次不一致")
    report = {
        "model_type": model_type, "architecture": "parallel" if model_type == "dcnv2" else "post_norm_token_mixing", "feature_version": FEATURE_VERSION,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "objective": "listwise_softmax", "score_semantics": "uncalibrated_ranking_logit",
        "cross_layers": cross_layers, "cross_projection_dim": None,
        "samples": len(samples), "users": int(samples["user_id"].nunique()),
        "sample_kinds": samples["sample_kind"].value_counts().to_dict(),
        "feature_columns": list(encoders), "numeric_columns": RANKING_NUMERIC_COLUMNS,
        "numeric_mean": scaler.mean_.tolist(), "numeric_scale": scaler.scale_.tolist(),
        "vocab_sizes": vocab_sizes, "categories": {name: encoder.categories_[0].tolist() for name, encoder in encoders.items()},
        "emb_dim": emb_dim, "hidden_units": list(hidden_units), "dropout": dropout, "l2": l2,
        "epochs": epochs, "executed_epochs": len(history), "selected_epoch": selected_epoch,
        "selection_metric": "validation_ndcg_then_recall", "selection_k": selection_k, "selected_validation": restored_metrics,
        "patience": patience, "query_batch_size": query_batch_size, "batch_size": batch_size,
        "learning_rate": learning_rate, "seed": seed, "validation_samples": len(validation),
        "validation_unknown_rates": unknown_rates(valid_features), "history": history,
        "tensorflow_version": tf.__version__, "tfrs_version": tfrs.__version__,
        "initial_embedding_sha256": embedding_digest, "parameter_count": model.count_params(),
        "matrix_macs_per_candidate": matrix_macs_per_candidate(model),
    }
    embedding_parameters = sum(model.get_layer(f"{name}_embedding").count_params() for name in encoders)
    report.update(embedding_parameters=embedding_parameters, non_embedding_parameters=model.count_params() - embedding_parameters)
    if model_type == "rankmixer":
        report.update(model_config=validate_rankmixer_config(model_config), token_groups=token_group_specification())
        for name in ("cross_layers", "cross_projection_dim", "hidden_units"):
            del report[name]
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "ranker.keras")
    restored = tf.keras.models.load_model(output_dir / "ranker.keras", compile=False)
    check_inputs = {name: values[:32] for name, values in features.items()}
    before, after = model(check_inputs, training=False).numpy(), restored(check_inputs, training=False).numpy()
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
    report["serialization_max_abs_error"] = float(np.max(np.abs(before - after)))
    report["elapsed_seconds"] = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mib"] = peak / (1024**2 if platform.system() == "Darwin" else 1024)
    report["model_bytes"] = (output_dir / "ranker.keras").stat().st_size
    for filename, value in (("encoders.pkl", encoders), ("numeric_scaler.pkl", scaler)):
        with (output_dir / filename).open("wb") as destination:
            pickle.dump(value, destination)
    with (output_dir / "training_report.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
    return report
