import pickle
from pathlib import Path

import pandas as pd
from pandas.testing import assert_series_equal

from features.encode import encode_category, fit_category_encoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "split"


def test_training_categories_receive_stable_codes() -> None:
    train = pd.read_pickle(SPLIT_DIR / "train.pkl")
    values = train["movie_id"]
    encoder = fit_category_encoder(values)
    codes = encode_category(values, encoder)

    assert codes.dtype == "int64"
    assert codes.index.equals(values.index)
    assert (codes >= 2).all()
    assert codes.nunique() == values.nunique()
    assert_series_equal(codes, encode_category(values, encoder))


def test_test_set_unknown_movies_keep_training_vocabulary() -> None:
    train = pd.read_pickle(SPLIT_DIR / "train.pkl")
    test = pd.read_pickle(SPLIT_DIR / "test.pkl")
    encoder = fit_category_encoder(train["movie_id"])
    original_categories = encoder.categories_[0].copy()
    codes = encode_category(test["movie_id"], encoder)
    known = test["movie_id"].astype(str).isin(original_categories)

    assert (~known).any()
    assert (codes[known] >= 2).all()
    assert (codes[~known] == 1).all()
    assert codes.index.equals(test.index)
    assert (encoder.categories_[0] == original_categories).all()


def test_saved_encoder_keeps_the_same_codes(tmp_path: Path) -> None:
    train = pd.read_pickle(SPLIT_DIR / "train.pkl")
    validation = pd.read_pickle(SPLIT_DIR / "validation.pkl")
    encoder = fit_category_encoder(train["user_id"])
    path = tmp_path / "user_id_encoder.pkl"

    with path.open("wb") as destination:
        pickle.dump(encoder, destination)
    with path.open("rb") as source:
        restored = pickle.load(source)

    assert_series_equal(
        encode_category(validation["user_id"], encoder),
        encode_category(validation["user_id"], restored),
    )
