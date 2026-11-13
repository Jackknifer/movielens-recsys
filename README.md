<h1 align="center">movielens-recsys</h1>

<p align="center">
  <em>基于 MovieLens-1M 的离线电影推荐系统</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/框架-TensorFlow-E87500?style=for-the-badge" alt="框架：TensorFlow">
  <img src="https://img.shields.io/badge/数据集-MovieLens--1M-0675BB?style=for-the-badge" alt="数据集：MovieLens-1M">
  <a href="LICENSE"><img src="https://img.shields.io/badge/源码许可-CC_BY--NC--SA_4.0-318329?style=for-the-badge" alt="源码许可：CC BY-NC-SA 4.0"></a>
</p>

<p align="center">
  <a href="#项目简介">项目简介</a> ·
  <a href="#快速链接">快速链接</a> ·
  <a href="#数据集">数据集</a> ·
  <a href="#训练配置">训练配置</a> ·
  <a href="#实验结果">实验结果</a> ·
  <a href="#参考文献">参考文献</a>
</p>

## 项目简介

使用 TensorFlow 实现 MovieLens 电影推荐流程。
召回采用 YoutubeDNN 双塔，结合物品相似、类型兴趣和热门召回。
候选通过 Snake Merge 合并，由 DCNv2 排序，生成推荐列表。

## 快速链接

| 资源 | 入口 |
| --- | --- |
| 完整实验配置 | [config/offline.yaml](config/offline.yaml) |
| 推荐流程入口 | [src/pipeline/run_pipeline.py](src/pipeline/run_pipeline.py) |
| 模型实现 | [YoutubeDNN](src/models/youtubednn.py) 和 [DCNv2](src/models/dcnv2.py) |

## 数据集

| 项目 | 说明 |
| --- | --- |
| 数据来源 | GroupLens 提供的 MovieLens-1M |
| 用户数量 | 6,040 位 |
| 电影数量 | 3,883 部 |
| 评分数量 | 1,000,209 条 |
| 评分范围 | 1～5 分 |
| 正样本定义 | 4 分及以上 |
| 划分方式 | 每位用户按时间保留最后两条交互，分别用于验证和测试 |
| 训练集 | 988,129 条 |
| 验证集 | 6,040 条 |
| 测试集 | 6,040 条 |

数据下载：[GroupLens 官方入口](https://grouplens.org/datasets/movielens/1m/)。

## 训练配置

| 设置 | 取值 |
| --- | --- |
| Embedding（嵌入）维度／历史长度 | 16／10 |
| 排序特征 | 9 个类别特征、22 个数值特征 |
| 召回损失／采样数量／温度 | Sampled Softmax／20／0.1 |
| 召回训练轮数 | 3 |
| DCNv2 结构 | 3 层 Cross Network 与 DNN [64, 32] 并行 |
| Dropout／L2 正则化系数 | 0.1／0.000001 |
| 优化器／学习率 | Adam／0.001 |
| Batch Size（批次大小） | 召回 128 条样本；排序 32 位用户 |
| 排序损失／训练轮数 | 列表级 Softmax／最多 10 轮 |
| 提前终止条件 | 连续 3 轮验证指标未改善 |
| 默认随机种子 | 42 |
| 运行环境 | Python 3.11、TensorFlow 2.15 |

完整配置见 [config/offline.yaml](config/offline.yaml)。
依据验证集 NDCG@20 选择模型，指标相同时比较 Recall@20。

## 实验结果

固定候选实验使用 5,435 位用户的 543,500 条排序训练样本。
每位用户固定 100 个候选，评估前 20 个结果。

| Recall@20 | NDCG@20 |
| :---: | :---: |
| 19.534% | 0.08018 |

## 参考文献

1. Paul Covington, Jay Adams, and Emre Sargin. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/). RecSys, 2016.
2. Ruoxi Wang et al. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535). WWW, 2021.
