import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from features.encode import encode_category, fit_category_encoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    split_dir = PROJECT_ROOT / "data" / "processed" / "split"
    parts = {
        name: pd.read_pickle(split_dir / f"{name}.pkl")
        for name in ("train", "validation", "test")
    }
    encoder_dir = PROJECT_ROOT / ".work" / "encoding-reference"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for column in ("user_id", "movie_id"):
        encoder = fit_category_encoder(parts["train"][column])
        vocabulary_size = len(encoder.categories_[0]) + 2
        original_categories = encoder.categories_[0].copy()
        counts = {}

        for name, frame in parts.items():
            codes = encode_category(frame[column], encoder)
            known = frame[column].astype(str).isin(original_categories)
            assert (codes[known] >= 2).all()
            assert (codes[~known] == 1).all()
            assert (codes < vocabulary_size).all()
            assert codes.index.equals(frame.index)
            assert_series_equal(codes, encode_category(frame[column], encoder))
            counts[name] = int((codes == 1).sum())

        # transform 不能增加词表类别；文件保存后仍需得到相同编码。
        np.testing.assert_array_equal(encoder.categories_[0], original_categories)
        path = encoder_dir / f"{column}.pkl"
        with path.open("wb") as destination:
            pickle.dump(encoder, destination)
        with path.open("rb") as source:
            restored = pickle.load(source)
        assert_series_equal(
            encode_category(parts["test"][column], encoder),
            encode_category(parts["test"][column], restored),
        )
        report[column] = {
            "known_categories": len(original_categories),
            "embedding_input_dim": vocabulary_size,
            "unknown_rows": counts,
        }

    output_path = PROJECT_ROOT / "outputs" / "stage-results" / "encoding_reference.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("参考代码检查通过：已知类别、未知类别、重复转换、保存和加载。")


if __name__ == "__main__":
    main()
