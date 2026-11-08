# movielens-recsys

使用 TensorFlow 实现 MovieLens 电影推荐流程。
召回采用 YoutubeDNN 双塔，结合物品相似、类型兴趣和热门召回。
候选通过 Snake Merge 合并，由 DCNv2 排序，并按电影类型与年代重排。

## 数据集

使用 [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)，包含 6,040 位用户、3,883 部电影和 1,000,209 条评分。
评分统一为 1～5 分，4 分及以上为正样本。
每位用户按时间保留最后两条交互，分别用于验证和测试。
训练、验证和测试集分别为 988,129、6,040 和 6,040 条。

数据下载：[GroupLens 官方入口](https://grouplens.org/datasets/movielens/1m/)。

## 训练配置

| 设置 | 取值 |
| --- | --- |
| Embedding 维度／历史长度 | 16／10 |
| 排序特征 | 9 个类别特征、22 个数值特征 |
| 召回损失／采样数量／温度 | Sampled Softmax／20／0.1 |
| 召回训练轮数 | 3 |
| DCNv2 结构 | 3 层 Cross Network 与 DNN [64, 32] 并行 |
| Dropout／L2 正则化系数 | 0.1／0.000001 |
| 优化器／学习率 | Adam／0.001 |
| Batch Size | 召回 128 条样本；排序 32 位用户 |
| 排序损失／训练轮数 | 列表级 Softmax／最多 10 轮 |
| 提前终止条件 | 连续 3 轮验证指标未改善 |
| 默认随机种子 | 42 |
| 环境 | Python 3.11、TensorFlow 2.15 |

排序训练使用每位用户前 80% 的训练交互提供历史，其余交互构成目标窗口。
每位用户保留 100 个候选，最终推荐 20 部电影。
重排时，连续相同类型或年代最多展示 2 部电影。

完整配置见 [config/offline.yaml](config/offline.yaml)。
依据验证集 NDCG@20 选择模型，指标相同时比较 Recall@20。

## 结果

固定候选实验使用 5,435 位用户的 543,500 条排序训练样本。
每位用户固定 100 个候选，评估前 20 个结果。
测试指标统计 3,563 位高评分目标用户，历史仅包含训练集交互。
结果以均值与标准差表示。

| 阶段 | Recall@20 ↑ | NDCG@20 ↑ |
| --- | ---: | ---: |
| DCNv2 排序 | 19.665% ± 0.422% | 0.08111 ± 0.00165 |
| 类型与年代重排 | 19.319% ± 0.242% | 0.07939 ± 0.00144 |

## 开源说明

- 源码：[CC BY-NC-SA 4.0](LICENSE)。
- MovieLens 数据由 GroupLens 提供，使用和分发遵循 [MovieLens 使用条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。

## 参考文献

1. Paul Covington, Jay Adams, and Emre Sargin. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/). RecSys, 2016.
2. Ruoxi Wang et al. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535). WWW, 2021.
