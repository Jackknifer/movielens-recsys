# movielens-recsys

基于 MovieLens-1M 的电影推荐项目，使用 Python 和 TensorFlow。
包含数据准备、特征构造、多路召回、排序、重排和评估。

## 推荐流程

YoutubeDNN、物品相似、类型兴趣和热门电影提供召回候选。
Snake Merge（交替遍历合并）去除重复和已交互电影，并保留来源信息。
排序支持 DCNv2 和 RankMixer，默认使用 DCNv2。
两种模型使用相同的 9 个类别特征和 22 个数值特征。
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
每次运行需要新的空输出目录。
读取模型时会校验文件摘要、模型结构和兼容版本。

## RankMixer

[RankMixer](https://arxiv.org/html/2507.15551v3) 将特征组织为 token（特征向量单元）。
本项目使用 8 个语义分组，分别投影为 40 维向量。
2 个 Block（计算模块）执行特征交互。
Token Mixing 通过维度交换重组特征。
FFN（前馈网络）为每个 token 使用独立参数。
残差相加后执行 LayerNorm（层归一化），形成 Post-Norm 结构。
最后对 token 求平均，输出排序得分。
训练沿用列表损失，得分用于候选间排序。

使用 [rankmixer.yaml](config/rankmixer.yaml) 训练完整流程：

```bash
.venv/bin/python -m pipeline.run_pipeline train \
  --config config/rankmixer.yaml --output-dir artifacts/rankmixer
```

实验脚本固定召回模型、候选、编码器和标准化参数。
主模型包含原结构 DCNv2、参数数量接近的 DCNv2 和 RankMixer。
学习率仅通过验证集选择，随后执行 3 个种子和 3 组结构消融。
每次实验包含 24 次训练、18 次最终测试。

```bash
.venv/bin/python -m pipeline.run_pipeline train \
  --output-dir artifacts/full-dcnv2

.venv/bin/python scripts/export_ranking_dataset.py \
  --source-bundle artifacts/full-dcnv2 \
  --output-dir data/processed/ranking-comparison

.venv/bin/python scripts/run_ranking_experiment.py \
  --dataset data/processed/ranking-comparison \
  --output-dir outputs/ranking-comparison

.venv/bin/python scripts/export_experiment_summary.py \
  --dataset data/processed/ranking-comparison \
  --experiment-dir outputs/ranking-comparison \
  --output results/ranking-comparison.json
```

完成记录支持通过相同命令追加 `--resume` 继续实验。
实验目录保存逐轮日志、逐候选得分、逐用户指标和成对重采样区间。
预测计时使用同一设备，轮换模型顺序，预热 20 次并测量 200 次。

## 实验结果

2026-10-25 的[完整汇总](results/rankmixer-movielens-20261025.json)包含全部模型和消融结果。
训练使用 5,435 位用户的 543,500 条候选记录。
测试指标使用相同的 3,563 位高评分目标用户。
每位用户具有相同的 100 个候选，比较前 20 个结果。
下表为种子 42、2025、2026 的均值与标准差。

| 排序模型 | Recall@20 | NDCG@20 |
| --- | --- | --- |
| DCNv2 | 19.665% ± 0.422% | 0.08111 ± 0.00165 |
| DCNv2（相近预算） | 19.656% ± 0.155% | 0.08140 ± 0.00066 |
| RankMixer | 19.805% ± 0.335% | 0.08054 ± 0.00153 |
| RankMixer（移除 Token Mixing） | 20.189% ± 0.351% | 0.08305 ± 0.00086 |
| Shared FFN（隐藏维度 80） | 19.618% ± 0.378% | 0.07999 ± 0.00214 |
| Shared FFN（隐藏维度 640） | 19.702% ± 0.446% | 0.08058 ± 0.00229 |

RankMixer 相对 DCNv2 的 NDCG@20 差值为 -0.000568。
按用户成对重采样 10,000 次，95% 区间为［-0.002961，0.001797］。
相近预算对照的差值区间也包含 0。
本轮结果未显示 RankMixer 带来稳定提升，默认模型保留 DCNv2。

Apple M5 CPU、TensorFlow 2.15、4 个计算线程的 p95 耗时如下。
模型计时使用编译后的函数；完整排序包括特征生成、编码和排序。

| 模型 | 100 候选模型计算 | 3,200 候选模型计算 | 100 候选完整排序 |
| --- | --- | --- | --- |
| DCNv2 | 0.404 ms | 4.502 ms | 16.341 ms |
| DCNv2（相近预算） | 0.418 ms | 4.770 ms | 15.954 ms |
| RankMixer | 0.883 ms | 13.249 ms | 21.406 ms |

汇总文件还记录 MRR、AUC、重排效果、参数数量、训练成本和文件摘要。
实验使用此前公开评估的 MovieLens 划分，配置选择仅使用验证集。
结论适用于记录的数据、固定候选和模型配置。
完整测试结果为 88 项通过。

## 文件组织

| 目录 | 内容 |
| --- | --- |
| [src](src) | `data`、`features`、`models`、`training`、`retrieval`、`ranking`、`reranking`、`evaluation` 和 `pipeline`。 |
| [config](config) | 训练与推荐参数。 |
| [scripts](scripts) | 数据下载、编码验证、示例请求和排序比较实验。 |
| [tests](tests) | 使用真实数据验证训练、模型保存、推荐和评估。 |
| [results](results) | 可以公开的实验汇总指标和核验摘要。 |

本地数据、模型、逐候选结果、环境、缓存和个人资料由 Git 忽略。

## 来源与许可

项目参考 Datawhale 的 [FunRec](https://github.com/datawhalechina/fun-rec)。
源码与文档沿用 [CC BY-NC-SA 4.0](LICENSE)，要求署名、非商业用途和相同方式共享。
MovieLens 数据遵循 [GroupLens 使用条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。
数据再分发和商业使用需要另行取得许可。
研究发表请引用 Harper 和 Konstan 的论文：
The MovieLens Datasets: History and Context（2015，DOI：10.1145/2827872）。
