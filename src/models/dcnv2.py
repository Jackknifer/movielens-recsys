import math

import tensorflow as tf
from tensorflow_recommenders.layers.feature_interaction import Cross

from training.ranking_inputs import RANKING_FEATURE_COLUMNS


def build_dcnv2(
    vocab_sizes: dict[str, int], emb_dim: int = 16, hidden_units: tuple[int, ...] = (64, 32),
    dropout: float = 0.1, l2: float = 1e-6, cross_layers: int = 3, numeric_dim: int = 0,
) -> tf.keras.Model:
    """完整矩阵 Cross Network 与 DNN 并行，共用字段向量。"""
    if set(vocab_sizes) != set(RANKING_FEATURE_COLUMNS):
        raise ValueError("vocab_sizes 必须包含全部排序字段")
    integers = list(vocab_sizes.values()) + [emb_dim, cross_layers] + list(hidden_units)
    if not hidden_units or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers):
        raise ValueError("词表长度、向量维度和层数必须是正整数")
    if any(size < 3 for size in vocab_sizes.values()):
        raise ValueError("词表需要包含保留编号和有效类别")
    if not math.isfinite(dropout) or not 0 <= dropout < 1 or not math.isfinite(l2) or l2 < 0:
        raise ValueError("dropout 或 l2 的范围无效")
    if isinstance(numeric_dim, bool) or not isinstance(numeric_dim, int) or numeric_dim < 0:
        raise ValueError("numeric_dim 必须是非负整数")
    regularizer = tf.keras.regularizers.l2(l2)
    inputs, embeddings = {}, []
    for name in RANKING_FEATURE_COLUMNS:
        inputs[name] = tf.keras.Input(shape=(1,), dtype="int32", name=name)
        embeddings.append(tf.keras.layers.Embedding(
            vocab_sizes[name], emb_dim, embeddings_regularizer=regularizer, name=f"{name}_embedding",
        )(inputs[name]))
    shared = tf.keras.layers.Concatenate(axis=1, name="shared_embeddings")(embeddings)
    categorical = tf.keras.layers.Flatten(name="categorical_features")(shared)
    if numeric_dim:
        inputs["context_features"] = tf.keras.Input(shape=(numeric_dim,), dtype="float32", name="context_features")
        base = tf.keras.layers.Concatenate(name="base_features")([categorical, inputs["context_features"]])
    else:
        base = tf.keras.layers.Activation("linear", name="base_features")(categorical)
    crossed = base
    for position in range(cross_layers):
        crossed = Cross(
            projection_dim=None, kernel_regularizer=regularizer, bias_regularizer=regularizer,
            name=f"cross_{position}",
        )(base, crossed)
    deep = base
    for position, units in enumerate(hidden_units):
        deep = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizer, name=f"deep_{position}")(deep)
        deep = tf.keras.layers.Dropout(dropout, name=f"dropout_{position}")(deep)
    combined = tf.keras.layers.Concatenate(name="combined_features")([crossed, deep])
    logits = tf.keras.layers.Dense(1, name="ranking_logit")(combined)
    output = tf.keras.layers.Activation("linear" if numeric_dim else "sigmoid", name="ranking_score")(logits)
    return tf.keras.Model(inputs=inputs, outputs=output, name="dcnv2")
