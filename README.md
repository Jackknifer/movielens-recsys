# movielens-recsys

基于 MovieLens-1M 的电影推荐项目，使用 Python 和 TensorFlow。
包含数据准备、特征构造、多路召回、排序、重排和评估。

## 推荐流程

YoutubeDNN、物品相似、类型兴趣和热门电影提供召回候选。
Snake Merge（交替遍历合并）去除重复和已交互电影，并保留来源信息。
DCN-V2 使用类别特征及 22 个数值特征进行排序。
重排限制连续相同电影类型和年代的数量。
零历史、少历史和普通用户分别选择适合的召回通道。
评估记录各阶段的 Recall、NDCG、MRR、覆盖率和多样性。

## 运行方法

需要 Python 3.11 和 uv，依赖版本由 `uv.lock` 保存。
TensorFlow 使用 2.15，Apple Silicon 使用对应的 macOS 版本。
在项目根目录执行：

```bash
mkdir -p .work/runtime-tmp
export TMPDIR="$PWD/.work/runtime-tmp"
uv sync --locked --cache-dir .uv-cache
export PYTHONPATH="$PWD/src"

.venv/bin/python scripts/download_movielens.py
.venv/bin/python -m pipeline.prepare_data
.venv/bin/python -m pytest -q
```

下载脚本获取 FunRec 数据表及 GroupLens 官方 ZIP。
准备步骤核对全部评分，并将评分统一为 1～5 分。
每位用户的最后两条交互分别用于验证和测试，其余用于训练。
相同时间的交互按输入顺序排列。

使用 20 位真实用户完成训练、推荐和验证：

```bash
.venv/bin/python -m pipeline.run_pipeline train \
  --max-users 20 --epochs 1 --output-dir artifacts/quickstart

.venv/bin/python scripts/create_example_queries.py \
  --bundle artifacts/quickstart --output data/example-queries.csv

.venv/bin/python -m pipeline.run_pipeline recommend \
  --bundle artifacts/quickstart --queries data/example-queries.csv \
  --output-dir outputs/recommendations

.venv/bin/python -m pipeline.run_pipeline evaluate \
  --bundle artifacts/quickstart --split validation --output-dir outputs/validation
```

推荐结果保存在 `outputs/recommendations/final.csv.gz`。
完整训练省略 `--max-users` 和 `--epochs`，使用 [offline.yaml](config/offline.yaml)。
测试评估使用 `--split test`，默认仅使用训练历史。
`train-ranking` 支持复用召回产物，独立训练排序模型。
每次运行需要新的空输出目录；读取模型时会校验文件与代码摘要。

## 文件组织

| 目录 | 内容 |
| --- | --- |
| [src](src) | `data`、`features`、`models`、`training`、`retrieval`、`ranking`、`reranking`、`evaluation` 和 `pipeline`。 |
| [config](config) | 训练与推荐参数。 |
| [scripts](scripts) | 数据下载、编码验证和示例请求生成。 |
| [tests](tests) | 使用真实数据验证训练、模型保存、推荐和评估。 |

本地数据、模型、结果、环境、缓存和个人资料由 Git 忽略。

## 来源与许可

项目参考 Datawhale 的 [FunRec](https://github.com/datawhalechina/fun-rec)。
源码与文档沿用 [CC BY-NC-SA 4.0](LICENSE)，要求署名、非商业用途和相同方式共享。
MovieLens 数据遵循 [GroupLens 使用条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。
数据再分发和商业使用需要另行取得许可。
研究发表请引用 Harper 和 Konstan 的论文：
The MovieLens Datasets: History and Context（2015，DOI：10.1145/2827872）。
