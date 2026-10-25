import argparse
from pathlib import Path

from training.ranking_dataset import export_ranking_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="从已验证的模型目录导出固定排序数据")
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_ranking_dataset(args.source_bundle, args.output_dir, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
