import tensorflow as tf

from training.ranking_inputs import RANKING_FEATURE_COLUMNS


@tf.keras.utils.register_keras_serializable(package="offline_recsys")
class FactorizationMachine(tf.keras.layers.Layer):
    def call(self, embeddings):
        summed_square = tf.square(tf.reduce_sum(embeddings, axis=1))
        square_summed = tf.reduce_sum(tf.square(embeddings), axis=1)
        return 0.5 * tf.reduce_sum(summed_square - square_summed, axis=1, keepdims=True)


@tf.keras.utils.register_keras_serializable(package="offline_recsys")
class LinearSum(tf.keras.layers.Layer):
    def call(self, values):
        return tf.reduce_sum(values, axis=1)


def build_deepfm(
    vocab_sizes: dict[str, int], emb_dim: int = 16, hidden_units: tuple[int, ...] = (64, 32),
    dropout: float = 0.1, l2: float = 1e-6,
) -> tf.keras.Model:
    """一阶项、FM 二阶项和 DNN 共用同一组特征，FM 与 DNN 共用向量。"""
    if set(vocab_sizes) != set(RANKING_FEATURE_COLUMNS) or any(size < 3 for size in vocab_sizes.values()):
        raise ValueError("vocab_sizes 必须包含全部排序字段和有效的词表长度")
    if not isinstance(emb_dim, int) or emb_dim < 1 or not hidden_units or any(unit < 1 for unit in hidden_units):
        raise ValueError("emb_dim 和 hidden_units 必须包含正整数")
    if not 0 <= dropout < 1 or not 0 <= l2 < float("inf"):
        raise ValueError("dropout 或 l2 的范围无效")
    inputs, embeddings, linear_terms = {}, [], []
    regularizer = tf.keras.regularizers.l2(l2)
    for name in RANKING_FEATURE_COLUMNS:
        inputs[name] = tf.keras.Input(shape=(1,), dtype="int32", name=name)
        embeddings.append(tf.keras.layers.Embedding(
            vocab_sizes[name], emb_dim, embeddings_regularizer=regularizer, name=f"{name}_embedding",
        )(inputs[name]))
        linear_terms.append(tf.keras.layers.Embedding(
            vocab_sizes[name], 1, embeddings_regularizer=regularizer, name=f"{name}_linear",
        )(inputs[name]))
    shared = tf.keras.layers.Concatenate(axis=1, name="shared_embeddings")(embeddings)
    fm = FactorizationMachine(name="fm_logit")(shared)
    linear = LinearSum(name="linear_logit")(tf.keras.layers.Concatenate(axis=1)(linear_terms))
    deep = tf.keras.layers.Flatten()(shared)
    for position, units in enumerate(hidden_units):
        deep = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizer, name=f"deep_{position}")(deep)
        deep = tf.keras.layers.Dropout(dropout)(deep)
    deep = tf.keras.layers.Dense(1, name="deep_logit")(deep)
    logits = tf.keras.layers.Add(name="combined_logit")([linear, fm, deep])
    output = tf.keras.layers.Activation("sigmoid", name="ranking_score")(logits)
    return tf.keras.Model(inputs=inputs, outputs=output, name="deepfm")
