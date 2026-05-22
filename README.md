# DeepLOB: Deep Learning for Limit Order Book Prediction

## Introduction

This project aims at reproduction of **Deep CNN for Level-2 Limit Order Book**, commonly known as the **DeepLOB model**. 

The DeepLOB model is a deep convolutional neural network designed to predict short-term price movements using Level-2 limit order book (LOB) data. By analyzing the structure and dynamics of the order book, this model captures market microstructure patterns that drive price changes, enabling accurate prediction of mid-price movements in financial markets.

### Project Goal

Reproduce and implement the DeepLOB architecture to:
- Process Level-2 limit order book data
- Extract meaningful features from order book dynamics
- Predict future price movements
- Demonstrate deep learning applications in quantitative finance

## Dataset

The project uses intraday limit order book data from various trading days and symbols.

# DeepLOB 项目可调节超参数汇总

## 1. 数据工程超参数 (Data Engineering)
这些参数位于 `process_data.py` 和 `DeepLOB.py` 的数据处理/加载模块中，直接决定输入模型的特征质量。

* **`label_method`**: 
    * 选项: `'l1'`, `'l2'`, `'l3'`
    * 描述: 决定了标签的生成逻辑。建议优先使用 `'l3'`（双向平滑，滤噪效果更好）。
* **`alpha`**: 
    * 描述: 价格波动阈值。用于将价格变动百分比划分为 `Down (-1)`, `Neutral (0)`, `Up (1)`。
* **`window_normalize` -> `window_size`**: 
    * 描述: 宏观滑动窗口标准化时，用于计算均值 ($\mu$) 和标准差 ($\sigma$) 的回溯历史天数。
* **`window_size` (TimeSeriesDataset)**: 
    * 描述: 送入 CNN 的时间序列深度。建议保持在 100-200 之间。

## 2. 模型架构超参数 (Model Architecture)
虽然核心 DeepLOB 拓扑已复刻，但在初始化模型时需注意以下配置：

* **`num_features`**: 
    * 建议值: `20` (对应 5 档原始盘口：Price/Size 各 4 列) 或 `40` (对应 10 档)。
    * 描述: 输入数据的特征列数，必须与 `process_data.py` 输出的 DataFrame 列数严格匹配。
* **`num_classes`**: 
    * 建议值: `3` (涨、跌、平)。

## 3. 训练与优化策略 (Training & Optimization)
这些参数位于训练脚本 (`DeepLOB_demo.ipynb`) 中，是模型收敛和泛化的核心。

* **`learning_rate`**: 
    * 建议值: `0.001` (配合预热策略)。
    * 描述: 优化器步长。过大会导致 Loss 震荡。
* **`batch_size`**: 
    * 建议值: `32` 或 `64`。
    * 描述: 单次训练样本数量，影响梯度估计。
* **`gamma` (针对 FocalLoss)**: 
    * 建议值: `2.0`。
    * 描述: Focal Loss 的调节因子。增加该值可强迫模型学习难以区分的样本。

## 4. 稳定性调控组件 (Stability & Scheduling)
* **`Gradient Clipping (max_norm)`**:
    * 建议值: `1.0`。
    * 描述: 防止 LSTM 层在处理长序列时出现梯度爆炸。
* **`LR Scheduler (SequentialLR)`**:
    * **`warmup_steps`**: 预热步数（推荐占总步数的 10%）。
    * **`eta_min` (CosineAnnealingLR)**: 退火阶段的最小学习率，用于精细拟合。