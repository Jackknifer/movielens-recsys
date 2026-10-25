import math

import tensorflow as tf

from features.ranking_features import RANKING_NUMERIC_COLUMNS
from training.ranking_inputs import RANKING_FEATURE_COLUMNS


MODEL_IMPLEMENTATION_VERSION = 1
TOKEN_GROUPS = (
    ("user_identity", ("user_id",), ()),
    ("user_profile", ("gender", "age", "occupation", "zip_code"), ()),
    ("movie_identity", ("movie_id",), ()),
    ("movie_attributes", ("genres", "isAdult", "startYear"), ()),
    ("recall_channels", (), tuple(RANKING_NUMERIC_COLUMNS[:12])),
    ("merge_and_interest", (), ("source_count", "merge_reciprocal_rank", "log_history_length", "genre_affinity")),
    ("vector_relations", (), ("recent_mean_similarity", "recent_max_similarity", "last_movie_similarity", "retrieval_known_item", "history_vector_coverage")),
    ("popularity", (), ("log_popularity",)),
)
DEFAULT_CONFIG = {
    "tokenization_version": "semantic_v1", "token_count": 8, "hidden_dim": 40,
    "num_heads": 8, "num_blocks": 2, "ffn_expansion": 2, "ffn_hidden_dim": None,
    "mixing": True, "shared_ffn": False, "layer_norm_epsilon": 1e-5,
}


def token_group_specification() -> list[dict]:
    return [{"name": name, "categorical_fields": list(categories), "numeric_fields": list(numeric)} for name, categories, numeric in TOKEN_GROUPS]


def validate_rankmixer_config(config: dict | None = None) -> dict:
    if config is not None and not isinstance(config, dict):
        raise ValueError("RankMixer 配置必须是字典")
    supplied = {} if config is None else config
    if set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError("RankMixer 配置包含未知字段")
    result = {**DEFAULT_CONFIG, **supplied}
    if result["tokenization_version"] != "semantic_v1" or result["token_count"] != len(TOKEN_GROUPS):
        raise ValueError("RankMixer 的特征分组版本或 token 数量无效")
    for name in ("token_count", "hidden_dim", "num_heads", "num_blocks", "ffn_expansion"):
        if isinstance(result[name], bool) or not isinstance(result[name], int) or result[name] < 1:
            raise ValueError(f"{name} 必须是正整数")
    if result["num_heads"] != result["token_count"] or result["hidden_dim"] % result["num_heads"]:
        raise ValueError("RankMixer 头数必须等于 token 数量，表示维度必须能被头数整除")
    hidden = result["ffn_hidden_dim"]
    if hidden is not None and (isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1):
        raise ValueError("ffn_hidden_dim 必须为空或正整数")
    if any(not isinstance(result[name], bool) for name in ("mixing", "shared_ffn")):
        raise ValueError("RankMixer 的结构开关必须是布尔值")
    epsilon = result["layer_norm_epsilon"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("LayerNorm 的 epsilon 必须是有限正数")
    return result


@tf.keras.utils.register_keras_serializable(package="movielens_recsys")
class NumericColumns(tf.keras.layers.Layer):
    def __init__(self, positions, **kwargs):
        super().__init__(**kwargs)
        self.positions = list(positions)
        if not self.positions or any(isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(RANKING_NUMERIC_COLUMNS) for position in self.positions):
            raise ValueError("数值特征位置无效")

    def call(self, inputs):
        return tf.gather(inputs, self.positions, axis=-1)

    def get_config(self):
        return {**super().get_config(), "positions": self.positions}


@tf.keras.utils.register_keras_serializable(package="movielens_recsys")
class TokenMixing(tf.keras.layers.Layer):
    def __init__(self, token_count: int, hidden_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        for value in (token_count, hidden_dim, num_heads):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Token Mixing 的形状参数必须是正整数")
        if token_count != num_heads or hidden_dim % num_heads:
            raise ValueError("Token Mixing 的头数、token 数量与维度不兼容")
        self.token_count, self.hidden_dim, self.num_heads = token_count, hidden_dim, num_heads

    def call(self, inputs):
        inputs = tf.ensure_shape(inputs, [None, self.token_count, self.hidden_dim])
        batch = tf.shape(inputs)[0]
        heads = tf.reshape(inputs, [batch, self.token_count, self.num_heads, self.hidden_dim // self.num_heads])
        mixed = tf.transpose(heads, [0, 2, 1, 3])
        return tf.reshape(mixed, [batch, self.token_count, self.hidden_dim])

    def get_config(self):
        return {**super().get_config(), "token_count": self.token_count, "hidden_dim": self.hidden_dim, "num_heads": self.num_heads}


def build_rankmixer(vocab_sizes: dict[str, int], emb_dim: int = 16, dropout: float = 0.1,
                   l2: float = 1e-6, numeric_dim: int = 22, model_config: dict | None = None,
                   seed: int = 42) -> tf.keras.Model:
    config = validate_rankmixer_config(model_config)
    if set(vocab_sizes) != set(RANKING_FEATURE_COLUMNS):
        raise ValueError("RankMixer 词表必须包含全部排序类别字段")
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 3 for size in vocab_sizes.values()):
        raise ValueError("RankMixer 词表长度必须包含保留编号和有效类别")
    if isinstance(emb_dim, bool) or not isinstance(emb_dim, int) or emb_dim < 1 or numeric_dim != len(RANKING_NUMERIC_COLUMNS):
        raise ValueError("RankMixer 的向量维度或数值特征维度无效")
    if not math.isfinite(dropout) or not 0 <= dropout < 1 or not math.isfinite(l2) or l2 < 0:
        raise ValueError("RankMixer 的 Dropout 或 L2 参数无效")
    tokens, dimension = config["token_count"], config["hidden_dim"]
    hidden = config["ffn_hidden_dim"] or config["ffn_expansion"] * dimension
    regularizer = tf.keras.regularizers.l2(l2)
    inputs, embeddings = {}, {}
    for name in RANKING_FEATURE_COLUMNS:
        inputs[name] = tf.keras.Input(shape=(1,), dtype="int32", name=name)
        embedding = tf.keras.layers.Embedding(vocab_sizes[name], emb_dim, embeddings_regularizer=regularizer, name=f"{name}_embedding")(inputs[name])
        embeddings[name] = tf.keras.layers.Flatten(name=f"{name}_vector")(embedding)
    inputs["context_features"] = tf.keras.Input(shape=(numeric_dim,), dtype="float32", name="context_features")
    projected = []
    for name, categories, numeric in TOKEN_GROUPS:
        pieces = [embeddings[field] for field in categories]
        if numeric:
            positions = [RANKING_NUMERIC_COLUMNS.index(field) for field in numeric]
            pieces.append(NumericColumns(positions, name=f"{name}_numeric")(inputs["context_features"]))
        grouped = pieces[0] if len(pieces) == 1 else tf.keras.layers.Concatenate(name=f"{name}_fields")(pieces)
        token = tf.keras.layers.Dense(dimension, kernel_regularizer=regularizer, name=f"{name}_projection")(grouped)
        projected.append(tf.keras.layers.Reshape((1, dimension), name=f"{name}_token")(token))
    current = tf.keras.layers.Concatenate(axis=1, name="feature_tokens")(projected)
    for block in range(config["num_blocks"]):
        if config["mixing"]:
            mixed = TokenMixing(tokens, dimension, config["num_heads"], name=f"token_mixing_{block}")(current)
            current = tf.keras.layers.Add(name=f"mix_residual_{block}")([current, mixed])
        current = tf.keras.layers.LayerNormalization(axis=-1, epsilon=config["layer_norm_epsilon"], name=f"mix_norm_{block}")(current)
        residual = current
        # 按单个 token 的矩阵维度初始化，独立参数使用不同的随机种子。
        limit = math.sqrt(6 / (dimension + hidden))
        first_initializer = tf.keras.initializers.RandomUniform(-limit, limit, seed=seed + 2 * block + 1)
        second_initializer = tf.keras.initializers.RandomUniform(-limit, limit, seed=seed + 2 * block + 2)
        if config["shared_ffn"]:
            current = tf.keras.layers.Dense(hidden, activation="gelu", kernel_initializer=first_initializer, kernel_regularizer=regularizer, name=f"ffn_in_{block}")(current)
            current = tf.keras.layers.Dense(dimension, kernel_initializer=second_initializer, kernel_regularizer=regularizer, name=f"ffn_out_{block}")(current)
        else:
            current = tf.keras.layers.EinsumDense("btd,tdh->bth", output_shape=(tokens, hidden), bias_axes="th", activation="gelu", kernel_initializer=first_initializer, kernel_regularizer=regularizer, name=f"ffn_in_{block}")(current)
            current = tf.keras.layers.EinsumDense("bth,thd->btd", output_shape=(tokens, dimension), bias_axes="td", kernel_initializer=second_initializer, kernel_regularizer=regularizer, name=f"ffn_out_{block}")(current)
        current = tf.keras.layers.Dropout(dropout, name=f"ffn_dropout_{block}")(current)
        current = tf.keras.layers.Add(name=f"ffn_residual_{block}")([residual, current])
        current = tf.keras.layers.LayerNormalization(axis=-1, epsilon=config["layer_norm_epsilon"], name=f"ffn_norm_{block}")(current)
    pooled = tf.keras.layers.GlobalAveragePooling1D(name="pooled_features")(current)
    logits = tf.keras.layers.Dense(1, name="ranking_logit")(pooled)
    output = tf.keras.layers.Activation("linear", name="ranking_score")(logits)
    return tf.keras.Model(inputs, output, name="rankmixer")


def validate_rankmixer_model(model: tf.keras.Model, config: dict, groups: list[dict]) -> None:
    config = validate_rankmixer_config(config)
    if groups != token_group_specification() or model.name != "rankmixer":
        raise ValueError("RankMixer 特征分组或模型名称与训练报告不一致")
    if model.get_layer("feature_tokens").output_shape[1:] != (config["token_count"], config["hidden_dim"]):
        raise ValueError("RankMixer token 形状与训练报告不一致")
    mixing = [layer for layer in model.layers if isinstance(layer, TokenMixing)]
    expected = config["num_blocks"] if config["mixing"] else 0
    if len(mixing) != expected or any((layer.token_count, layer.hidden_dim, layer.num_heads) != (config["token_count"], config["hidden_dim"], config["num_heads"]) for layer in mixing):
        raise ValueError("RankMixer 混合层与训练报告不一致")
    hidden = config["ffn_hidden_dim"] or config["ffn_expansion"] * config["hidden_dim"]
    for block in range(config["num_blocks"]):
        first, second = model.get_layer(f"ffn_in_{block}"), model.get_layer(f"ffn_out_{block}")
        prefix = [] if config["shared_ffn"] else [config["token_count"]]
        if first.kernel.shape.as_list() != prefix + [config["hidden_dim"], hidden] or second.kernel.shape.as_list() != prefix + [hidden, config["hidden_dim"]]:
            raise ValueError("RankMixer FFN 参数形状与训练报告不一致")
        for name in (f"mix_norm_{block}", f"ffn_norm_{block}"):
            layer = model.get_layer(name)
            if layer.epsilon != config["layer_norm_epsilon"] or list(layer.axis) != [2]:
                raise ValueError("RankMixer LayerNorm 配置与训练报告不一致")
    if not isinstance(model.get_layer("pooled_features"), tf.keras.layers.GlobalAveragePooling1D):
        raise ValueError("RankMixer 必须使用 token 平均聚合")
