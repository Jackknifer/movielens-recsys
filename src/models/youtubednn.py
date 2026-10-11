import tensorflow as tf

from training.retrieval_inputs import USER_FEATURE_COLUMNS


@tf.keras.utils.register_keras_serializable(package="offline_recsys")
class MaskedMean(tf.keras.layers.Layer):
    """对历史向量求平均，编号为 0 的 padding 不参与计算。"""

    def call(self, inputs):
        ids, embeddings = inputs
        mask = tf.cast(ids != 0, embeddings.dtype)[..., None]
        total = tf.reduce_sum(embeddings * mask, axis=1)
        count = tf.reduce_sum(mask, axis=1)
        return tf.math.divide_no_nan(total, count)


@tf.keras.utils.register_keras_serializable(package="offline_recsys")
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)


class YoutubeDNNTrainer(tf.keras.Model):
    """每个 batch 使用 sampled softmax 训练用户模型和电影 Embedding。"""

    def __init__(self, user_tower, item_tower, movie_embedding, num_sampled, temperature, seed):
        super().__init__(name="youtubednn")
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.movie_embedding = movie_embedding
        self.num_classes = movie_embedding.input_dim - 2
        self.num_sampled = num_sampled
        self.temperature = temperature
        self.seed = seed
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        return self.user_tower(inputs, training=training)

    def train_step(self, data):
        features, targets, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("本训练入口要求每条样本使用相同权重")

        # 类别编号从 2 开始。减去 2 后，符合 sampled softmax 的类别范围。
        labels = tf.cast(targets, tf.int64) - 2
        tf.debugging.assert_greater_equal(labels, tf.constant(0, dtype=tf.int64))
        tf.debugging.assert_less(labels, tf.cast(self.num_classes, tf.int64))
        with tf.GradientTape() as tape:
            user_vectors = self(features, training=True)
            # padding 和未知编号不参加目标类别采样；训练和导出共用归一化电影向量。
            item_vectors = tf.math.l2_normalize(self.movie_embedding.embeddings[2:], axis=-1)
            losses = tf.nn.sampled_softmax_loss(
                weights=item_vectors / self.temperature,
                biases=tf.zeros(self.num_classes),
                labels=labels,
                inputs=user_vectors,
                num_sampled=self.num_sampled,
                num_classes=self.num_classes,
                seed=self.seed,
            )
            loss = tf.debugging.check_numerics(tf.reduce_mean(losses), "训练 loss 存在无效数值")

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss, sample_weight=tf.cast(tf.shape(targets)[0], tf.float32))
        return {"loss": self.loss_tracker.result()}


def build_youtubednn(
    vocab_sizes: dict[str, int],
    max_history: int = 10,
    emb_dim: int = 16,
    num_sampled: int = 20,
    temperature: float = 0.1,
    seed: int = 42,
) -> tuple[YoutubeDNNTrainer, tf.keras.Model, tf.keras.Model]:
    """构建训练模型、用户模型和电影模型，三个模型共享训练参数。"""
    for name in USER_FEATURE_COLUMNS + ["movie_id", "genres"]:
        if vocab_sizes[name] < 3:
            raise ValueError(f"{name} 的词表需要包含已知类别")
    if max_history < 1 or emb_dim < 1 or temperature <= 0:
        raise ValueError("历史长度、向量维度和 temperature 必须为正数")
    if not 1 <= num_sampled < vocab_sizes["movie_id"] - 2:
        raise ValueError("num_sampled 必须小于已知电影数量，并大于或等于 1")

    inputs = {}
    user_embeddings = []
    for name in USER_FEATURE_COLUMNS:
        inputs[name] = tf.keras.Input(shape=(1,), dtype="int32", name=name)
        embedding = tf.keras.layers.Embedding(vocab_sizes[name], emb_dim, name=f"{name}_embedding")
        user_embeddings.append(tf.keras.layers.Flatten()(embedding(inputs[name])))

    inputs["hist_movie_id"] = tf.keras.Input(shape=(max_history,), dtype="int32", name="hist_movie_id")
    inputs["hist_genres"] = tf.keras.Input(shape=(max_history,), dtype="int32", name="hist_genres")
    movie_embedding = tf.keras.layers.Embedding(vocab_sizes["movie_id"], emb_dim, name="movie_embedding")
    genre_embedding = tf.keras.layers.Embedding(vocab_sizes["genres"], emb_dim, name="genre_embedding")
    for name, embedding in (("hist_movie_id", movie_embedding), ("hist_genres", genre_embedding)):
        user_embeddings.append(MaskedMean()([inputs[name], embedding(inputs[name])]))

    # 用户属性与两组历史向量连接后，通过 Dense 学习最终的用户向量。
    hidden = tf.keras.layers.Concatenate()(user_embeddings)
    hidden = tf.keras.layers.Dense(64, activation="relu")(hidden)
    hidden = tf.keras.layers.Dense(32, activation="relu")(hidden)
    user_vector = L2Normalize()(tf.keras.layers.Dense(emb_dim)(hidden))
    user_tower = tf.keras.Model(inputs, user_vector, name="user_tower")

    # 电影模型和历史电影使用同一个 Embedding，保证它们处于同一个向量空间。
    item_input = tf.keras.Input(shape=(1,), dtype="int32", name="movie_id")
    item_vector = L2Normalize()(tf.keras.layers.Flatten()(movie_embedding(item_input)))
    item_tower = tf.keras.Model({"movie_id": item_input}, item_vector, name="item_tower")
    trainer = YoutubeDNNTrainer(user_tower, item_tower, movie_embedding, num_sampled, temperature, seed)
    return trainer, user_tower, item_tower
