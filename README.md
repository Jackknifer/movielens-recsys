# movielens-recsys

使用 TensorFlow 实现 MovieLens 电影推荐流程。
召回采用 YoutubeDNN 双塔，结合物品相似、类型兴趣和热门召回。
候选通过 Snake Merge 合并，由 DCNv2 排序，再使用 MMR（最大边际相关）重排。

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
| MMR 相关性权重 λ | 0.35 |
| 环境 | Python 3.11、TensorFlow 2.15 |

排序训练使用每位用户前 80% 的训练交互提供历史，其余交互构成目标窗口。
每位用户保留 100 个候选，最终推荐 20 部电影。
MMR 使用原始精排分数和电影向量点积，每轮选择综合得分最高的电影。

$$
\lambda\,\operatorname{score}(u,t)
-(1-\lambda)\max_{s\in S}E_t^\top E_s
$$

其中，S 是已选电影集合。
第一部电影采用精排分数最高的候选。
λ 通过验证集 NDCG@20 选择。

完整配置见 [config/offline.yaml](config/offline.yaml)。
依据验证集 NDCG@20 选择模型，指标相同时比较 Recall@20。

## 结果

固定候选实验使用 5,435 位用户的 543,500 条排序训练样本。
每位用户固定 100 个候选，评估前 20 个结果。
测试指标统计 3,563 位高评分目标用户，历史仅包含训练集交互。
以下为固定模型的测试结果。

| 阶段 | Recall@20 ↑ | NDCG@20 ↑ | 列表内平均余弦距离 ↑ |
| --- | ---: | ---: | ---: |
| DCNv2 排序 | 20.123% | 0.08225 | 0.15557 |
| MMR 重排 | 20.516% | 0.08304 | 0.16871 |

## 开源说明

- 源码：[CC BY-NC-SA 4.0](LICENSE)。
- MovieLens 数据由 GroupLens 提供，使用和分发遵循 [MovieLens 使用条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。

## 参考文献

1. Paul Covington, Jay Adams, and Emre Sargin. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/). RecSys, 2016.
2. Ruoxi Wang et al. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535). WWW, 2021.
3. Jaime Carbonell and Jade Goldstein. [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf). SIGIR, 1998.
