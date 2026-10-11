import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from training.retrieval_inputs import build_retrieval_inputs, fit_retrieval_encoders
from models.youtubednn import build_youtubednn
from data.load import load_movielens
from training.retrieval_samples import build_retrieval_samples


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_retrieval(
    samples: pd.DataFrame,
    output_dir: Path,
    max_history: int = 10,
    emb_dim: int = 16,
    num_sampled: int = 20,
    temperature: float = 0.1,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> dict:
    """完成编码、训练和保存，返回本次训练的样本数、电影数和 loss。"""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs、batch_size 和 learning_rate 必须为正数")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"请选择空的输出目录：{output_dir}")
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    encoders = fit_retrieval_encoders(samples)
    features, targets = build_retrieval_inputs(samples, encoders, max_history)
    vocab_sizes = {name: len(encoder.categories_[0]) + 2 for name, encoder in encoders.items()}
    trainer, user_tower, item_tower = build_youtubednn(
        vocab_sizes, max_history, emb_dim, num_sampled, temperature, seed
    )
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    history = trainer.fit(features, targets, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)

    # 第 i 行向量对应 movie_ids[i]，两份文件必须使用相同的词表顺序。
    known_codes = np.arange(2, vocab_sizes["movie_id"], dtype=np.int32)[:, None]
    movie_ids = encoders["movie_id"].categories_[0].astype(np.int64)
    item_vectors = item_tower({"movie_id": known_codes}, training=False).numpy()
    if not np.isfinite(item_vectors).all():
        raise ValueError("电影向量存在无效数值")
    if not np.allclose(np.linalg.norm(item_vectors, axis=1), 1, atol=1e-5):
        raise ValueError("电影向量的 L2 长度应当为 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存两个用于推荐的模型；再次读取时，需要导入 youtubednn 以注册自定义 Layer。
    user_tower.save(output_dir / "user_tower.keras")
    item_tower.save(output_dir / "item_tower.keras")
    with (output_dir / "encoders.pkl").open("wb") as destination:
        pickle.dump(encoders, destination)
    np.save(output_dir / "movie_ids.npy", movie_ids)
    np.save(output_dir / "item_embeddings.npy", item_vectors)

    report = {
        "samples": len(samples), "users": int(samples["user_id"].nunique()),
        "movies": len(movie_ids),
        "max_history": max_history, "emb_dim": emb_dim,
        "num_sampled": num_sampled, "temperature": temperature,
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "seed": seed,
        "vocab_sizes": vocab_sizes, "user_features": list(features),
        "sampled_softmax_loss": history.history["loss"],
        "tensorflow_version": tf.__version__,
    }
    with (output_dir / "training_report.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 MovieLens 训练离线 YoutubeDNN")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / ".work" / "retrieval-study")
    args = parser.parse_args()
    if args.max_users is not None and args.max_users < 1:
        raise ValueError("max_users 必须为正整数")

    data = load_movielens(PROJECT_ROOT / "data" / "processed" / "input")
    train = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "split" / "train.pkl")
    if args.max_users is not None:
        user_ids = train["user_id"].unique()[:args.max_users]
        train = train.loc[train["user_id"].isin(user_ids)]
    # 训练入口只接收已经划分好的 train；验证集和测试集用于后续离线评测。
    samples = build_retrieval_samples(
        train, data.users, data.movies, args.max_history, args.positive_threshold
    )
    report = train_retrieval(
        samples, args.output_dir, max_history=args.max_history,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
