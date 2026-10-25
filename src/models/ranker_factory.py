import hashlib

import numpy as np
import tensorflow as tf
from tensorflow_recommenders.layers.feature_interaction import Cross

from models.dcnv2 import build_dcnv2
from models.rankmixer import build_rankmixer, validate_rankmixer_config
from training.ranking_inputs import RANKING_FEATURE_COLUMNS


def build_ranker(vocab_sizes: dict, model_type: str = "dcnv2", emb_dim: int = 16,
                 hidden_units: tuple = (64, 32), dropout: float = 0.1, l2: float = 1e-6,
                 cross_layers: int = 3, numeric_dim: int = 22,
                 model_config: dict | None = None, seed: int = 42):
    if model_type == "dcnv2":
        if model_config:
            raise ValueError("DCNv2 配置需要通过隐藏层和 Cross 参数指定")
        return build_dcnv2(vocab_sizes, emb_dim, hidden_units, dropout, l2, cross_layers, numeric_dim)
    if model_type == "rankmixer":
        return build_rankmixer(vocab_sizes, emb_dim, dropout, l2, numeric_dim, validate_rankmixer_config(model_config), seed)
    raise ValueError(f"不支持的排序模型类型：{model_type}")


def initialize_common_embeddings(model, vocab_sizes: dict, emb_dim: int, seed: int) -> str:
    digest = hashlib.sha256()
    for position, name in enumerate(RANKING_FEATURE_COLUMNS):
        generator = np.random.default_rng(np.random.SeedSequence([seed, position]))
        values = generator.uniform(-0.05, 0.05, size=(vocab_sizes[name], emb_dim)).astype(np.float32)
        model.get_layer(f"{name}_embedding").set_weights([values])
        digest.update(name.encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def matrix_macs_per_candidate(model) -> int:
    total = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            repetitions = int(np.prod(layer.input_shape[1:-1]))
            total += int(np.prod(layer.kernel.shape)) * repetitions
        elif isinstance(layer, tf.keras.layers.EinsumDense):
            total += int(np.prod(layer.kernel.shape))
        elif isinstance(layer, Cross):
            if layer.get_config()["projection_dim"] is not None:
                raise ValueError("矩阵预算需要完整矩阵 Cross")
            total += int(np.prod(layer.get_weights()[0].shape))
    return total
