import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from data.load import load_movielens
from data.prepare import prepare_movielens
from data.split import split_by_user_time


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source_dir = PROJECT_ROOT / "data" / "raw"
    input_dir = PROJECT_ROOT / "data" / "processed" / "input"
    output_dir = PROJECT_ROOT / "data" / "processed" / "split"
    report_path = PROJECT_ROOT / "outputs" / "metrics" / "data_validation.json"

    # 文件哈希用于确认本次运行使用的数据与下载记录一致。
    with (source_dir / "source.json").open(encoding="utf-8") as source:
        source_manifest = json.load(source)
    for record in source_manifest["files"]:
        with (source_dir / record["file"]).open("rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != record["sha256"]:
            raise ValueError(f"数据文件校验失败: {record['file']}")

    missing_counts = prepare_movielens(source_dir, input_dir)
    data = load_movielens(input_dir)
    print(f"完成读取: {len(data.ratings)} 条评分", flush=True)
    original = data.ratings.copy(deep=True)
    result = split_by_user_time(data.ratings)
    parts = {name: getattr(result, name) for name in (
        "train", "validation", "test", "cold_start"
    )}

    # 将四份结果组合后逐列比较，检查数量和内容完整性。
    combined = pd.concat(parts.values(), ignore_index=True)
    keys = ["user_id", "movie_id"]
    assert_frame_equal(
        combined.sort_values(keys).reset_index(drop=True),
        original.sort_values(keys).reset_index(drop=True),
    )
    assert_frame_equal(data.ratings, original)
    if combined.duplicated(keys).any():
        raise ValueError("划分结果中存在重复评分")

    # 独立计算每位用户的预期验证和测试记录，包含相同时间戳的情况。
    ordered = original.assign(event_position=range(len(original))).sort_values(
        ["user_id", "timestamp", "event_position"]
    )
    counts = ordered.groupby("user_id")["user_id"].transform("size")
    eligible = ordered[counts >= 3]
    last_two = eligible.groupby("user_id").tail(2)
    expected_validation = last_two.groupby("user_id").head(1)
    expected_test = last_two.groupby("user_id").tail(1)
    for actual, expected in (
        (result.validation, expected_validation), (result.test, expected_test)
    ):
        assert_frame_equal(
            actual.reset_index(drop=True),
            expected.drop(columns="event_position").reset_index(drop=True),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in parts.items():
        frame.to_pickle(output_dir / f"{name}.pkl")

    example_user_id = int(result.test.iloc[0]["user_id"])
    user_counts = original.groupby("user_id").size()
    report = {
        "source": source_manifest,
        "split_module": "data.split",
        "minimum_interactions": 3,
        "users": len(data.users),
        "movies": len(data.movies),
        "ratings": len(original),
        "minimum_user_interactions": int(user_counts.min()),
        "missing_metadata_replaced": missing_counts,
        "missing_category": "__MISSING__",
        "rating_conversion": "FunRec rating / 2",
        "official_reference_url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "parts": {
            name: {"rows": len(frame), "users": int(frame.user_id.nunique())}
            for name, frame in parts.items()
        },
        "validation_test_equal_timestamp_users": int(
            (result.validation.set_index("user_id").timestamp
             == result.test.set_index("user_id").timestamp).sum()
        ),
        "checks": {
            "source_hashes": True,
            "all_ratings_match_official_movielens": True,
            "all_records_preserved": True,
            "input_unchanged": True,
            "no_duplicate_ratings": True,
            "expected_holdouts": True,
        },
        "example_user_id": example_user_id,
        "example_last_records": {
            name: frame[frame.user_id == example_user_id].tail(3).to_dict("records")
            for name, frame in parts.items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report["parts"], ensure_ascii=False, indent=2))
    print(f"验证报告: {report_path}")


if __name__ == "__main__":
    main()
