# movielens-recsys

基于 MovieLens-1M 的电影推荐项目，使用 Python 和 TensorFlow。
包含数据准备、特征构造、多路召回、排序、重排和评估。

## 推荐流程

YoutubeDNN、物品相似、类型兴趣和热门电影提供召回候选。
Snake Merge（交替遍历合并）去除重复和已交互电影，并保留来源信息。
DCNv2 使用 9 个类别特征和 22 个数值特征完成排序。
重排限制连续相同电影类型和年代的数量。

无历史用户使用热门召回，少历史用户增加类型兴趣召回。
历史充足且参与过召回训练的用户使用全部通道。
有历史的新用户使用热门、类型兴趣和物品相似召回。
评估记录各阶段的召回率、排名指标、覆盖率和多样性。

## 运行方法

需要 Python 3.11 和 uv。
Python 版本要求与直接依赖统一保存在 [pyproject.toml](pyproject.toml)。
[uv.lock](uv.lock) 保存完整依赖版本。
TensorFlow 使用 2.15，TensorFlow Recommenders 使用 0.7.3。
Apple Silicon 使用对应的 macOS 版本。
在项目根目录执行：

```bash
mkdir -p .work/runtime-tmp
export TMPDIR="$PWD/.work/runtime-tmp"
export PYTHONDONTWRITEBYTECODE=1
uv sync --locked --python 3.11 --cache-dir .work/uv-cache
export PYTHONPATH="$PWD/src"

.venv/bin/python scripts/download_movielens.py
.venv/bin/python -m pipeline.prepare_data
.venv/bin/python -m pytest -q
```

下载脚本获取 FunRec 数据表及 GroupLens 官方 ZIP。
脚本校验来源文件摘要，并将下载信息写入 `data/raw/source.json`。
数据准备命令转换评分、检查字段，并核对全部官方评分。
校验报告保存在 `outputs/metrics/data_validation.json`。

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
`recommendation_report.json` 记录用户分组和候选数量。
`evaluation_report.json` 保存各阶段指标和评估设置。
每次训练、推荐和评估都需要新的空输出目录。

## 训练与评估

训练参数保存在 [offline.yaml](config/offline.yaml)。
完整训练使用全部用户和配置中的训练轮次：

```bash
.venv/bin/python -m pipeline.run_pipeline train \
  --output-dir artifacts/movielens

.venv/bin/python -m pipeline.run_pipeline train-ranking \
  --source-bundle artifacts/movielens --output-dir artifacts/movielens-ranking

.venv/bin/python -m pipeline.run_pipeline evaluate \
  --bundle artifacts/movielens --split test --output-dir outputs/test
```

`train-ranking` 复用指定目录中的召回模型和索引，独立训练排序模型。
测试评估默认使用训练历史。
添加 `--include-validation-history` 可将验证交互加入测试时的用户历史。
评估报告记录所用规则。

每位用户的交互按时间排序，相同时间按输入顺序排列。
最后两条交互分别用于验证和测试，其余用于训练。
评分统一为 1～5 分，4 分及以上为正样本，2 分及以下用于负样本评估。
类别词表与召回统计只使用相应训练阶段的数据。

排序训练将训练交互划分为历史前缀和目标窗口。
默认前缀占 80%，候选训练召回模型和统计只读取前缀。
目标窗口中的高评分交互提供列表训练标签。
DCNv2 使用列表级 Softmax 损失，并按验证集排名指标选择模型。
DeepFM 和未交互电影采样提供对照实验所需的基础模块。

模型目录包含召回模型、排序模型、编码器、标准化参数和索引。
`manifest.json` 记录配置、源数据摘要、代码摘要和模型文件摘要。
读取模型时检查文件与代码；独立排序训练和评估还会检查源数据。

## 文件组织

| 目录 | 内容 |
| --- | --- |
| [src](src) | `data`、`features`、`models`、`training`、`retrieval`、`ranking`、`reranking`、`evaluation` 和 `pipeline`。 |
| [config](config) | 训练与推荐参数。 |
| [scripts](scripts) | 数据下载和示例请求生成。 |
| [tests](tests) | 使用真实数据验证训练、模型保存、推荐和评估。 |

运行时按需生成以下目录，并由 [`.gitignore`](.gitignore) 忽略：

| 目录 | 内容 | 生成方式 |
| --- | --- | --- |
| `.venv/` | Python 环境。 | `uv sync`。 |
| `.work/` | 临时文件、测试文件和依赖缓存。 | 安装、训练和测试命令。 |
| `data/` | 来源数据、处理结果和示例请求。 | 数据下载、准备和示例脚本。 |
| `artifacts/` | 模型、编码器和索引。 | `train` 或 `train-ranking`。 |
| `outputs/` | 推荐结果和评估报告。 | 数据准备、`recommend` 或 `evaluate`。 |

项目说明统一保存在本 README。
环境、数据和模型通过上述命令生成，公开仓库保留代码、测试和运行配置。

## 来源与许可

项目参考 Datawhale 的 [FunRec](https://github.com/datawhalechina/fun-rec)。
源码与文档沿用 [CC BY-NC-SA 4.0](LICENSE)，要求署名、非商业用途和相同方式共享。
MovieLens 数据遵循 [GroupLens 使用条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。
数据再分发和商业使用需要另行取得许可。
研究发表请引用 Harper 和 Konstan 的论文：
The MovieLens Datasets: History and Context（2015，DOI：10.1145/2827872）。
