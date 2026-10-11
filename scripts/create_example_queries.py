import argparse
from pathlib import Path

import pandas as pd

from features.user_context import build_user_contexts
from pipeline.artifact_store import load_bundle
from retrieval.cold_start import classify_users


def create_queries(bundle) -> pd.DataFrame:
    min_history = bundle.config["min_history"]
    if min_history < 2:
        raise ValueError("三种历史示例要求 min_history 至少为 2")
    known_users = bundle.retrieval.encoders["user_id"].categories_[0].astype("int64")
    histories = {
        user_id: events.sort_values("timestamp", kind="stable")
        for user_id, events in bundle.history.groupby("user_id")
        if user_id in set(known_users)
    }
    records = []
    used = set()
    expected_groups = ("no_history", "short_history", "warm")
    for group in expected_groups:
        for user_id in sorted(histories):
            if user_id in used:
                continue
            events = histories[user_id]
            positives = events.loc[events["rating"] >= bundle.config["positive_threshold"]]
            if group == "no_history":
                timestamp = int(events["timestamp"].min()) - 1
                if timestamp < 0:
                    continue
            elif group == "short_history":
                counts = positives.groupby("timestamp").size().cumsum()
                times = counts.loc[(counts >= 1) & (counts < min_history)].index
                if times.empty:
                    continue
                timestamp = int(times[0])
            else:
                if len(positives) < min_history:
                    continue
                timestamp = int(events["timestamp"].max())
            records.append({"user_id": int(user_id), "as_of_timestamp": timestamp})
            used.add(user_id)
            break
        else:
            raise ValueError(f"模型历史无法提供示例用户：{group}")
    queries = pd.DataFrame(records)
    contexts = build_user_contexts(
        queries, bundle.history, bundle.users, bundle.movies,
        bundle.config["max_history"], bundle.config["positive_threshold"],
    )
    if classify_users(contexts, known_users, min_history).tolist() != list(expected_groups):
        raise ValueError("示例请求的历史分组校验失败")
    return queries


def main():
    parser = argparse.ArgumentParser(description="使用模型中的真实历史生成三种推荐请求")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    queries = create_queries(load_bundle(args.bundle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as destination:
        queries.to_csv(destination, index=False)
    print(f"已生成 {len(queries)} 位用户的示例请求：{args.output}")


if __name__ == "__main__":
    main()
