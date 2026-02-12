
---

# 水下机器人多模态定位数据代码库

## 最新项目目录结构 (2026-02-12 更新)

```text
.
├── ronin-net/                   # 【新增】基于MoE架构的去噪与定位网络
│   ├── config/                  # 配置文件 (resnet_lstm.yaml, only_sns.yaml等)
│   ├── data/                    # 数据集目录
│   │   ├── all_data/            # 原始全量数据集
│   │   └── processed_data/      # 预处理后的有效数据集
│   ├── dataloader/              # PyTorch 数据加载实现 (dataset.py)
│   ├── denoise/                 # 核心去噪模块 (基于第一版Gyro-net的优化版)
│   │   ├── src/                 # 专家模型训练与调试 (debug_moe.py, check_specialists.py)
│   │   └── output/              # 训练输出与结果保存
│   ├── model/                   # 网络架构定义 (model_lstm.py, losses.py)
│   ├── utils/                   # 工具库 (指标计算 metric.py, 后处理 postprocess.py)
│   ├── weights/                 # 模型权重文件 (weights.pt)
│   ├── main_net.py              # 模块主程序入口
│   └── train.py / test.py       # 训练与测试脚本
│
├── CNN-MODEL/                   # IMU标定
│   ├── figures/                 # 结果可视化 (boxplot.jpg, rpy.jpg等)
│   └── src/                     # 核心算法 (lie_algebra.py, networks.py等)
├── MonoVision-Depth-Aided-Trajectory/  # 单目+深度约束生成参考轨迹
│   └── generate_gt_colmap.py     # 基于Colmap的真值生成脚本
├── data process/                # 数据处理基础脚本
│   └── process_rotate.py        # 坐标转换与数据清洗
└── IMU-DVL/                    # IMU-DVL融合定位
    └── Compare_Experiment/      # 多模型对比实验
        ├── Com.py               # 四模型对比实验主程序
        ├── config.json          # 实验配置文件
        ├── results_summary.csv  # 实验结果汇总表
        └── 0730data/            # 2025年7月30日实验数据集

```

---

## 2026年2月12日更新

**更新者**：姜昕彤

**更新内容**：新增 **ronin-net** 模块（基于 MoE 架构的陀螺仪去噪与定位优化）

### 核心改进：ronin-net 模块

该模块是针对早期去噪算法的重大升级，主要特点包括：

* **MoE (Mixture of Experts) 架构**：在 `denoise` 模块中采用了混合专家架构，通过 `train_specialists.py` 训练了 **4 个专家模型**，以应对不同维度的传感器噪声。
* **有效数据管理**：在 `data/` 目录下将数据严格划分为 `all-data`（全量）与 `process-data`（已处理），确保训练集的纯净度。
* **配置化设计**：支持通过 `config/` 下的 YAML 文件快速切换不同的骨干网络（如 ResNet, LSTM）及超参数。
* **调试与分析工具**：新增 `debug_moe.py` 和 `check_specialists.py`，用于可视化分析各专家在不同运动状态下的贡献权重。

---

## 2026年2月3日更新

**更新者**：李书宇

**更新内容**：IMU-DVL 多模型对比实验、0730 实验数据

### 新增主要内容

* **Com.py**：涵盖 CNN、LSTM、IONet、TCN 四种模型的对比实验主程序。
* **0730data**：包含 5 组已对齐的 `SenseINS_aligned` 实验数据集。
* **指标评估**：支持 MSE、N-SRMSE、RMSE、MaxError 及运行耗时（Time）等多维度评估。

---

## 2026年1月5日更新

**更新者**：孙超

**更新内容**：单目 + 深度约束生成参考轨迹代码

* 基于 **SOLAQUA** 数据集，通过单目视觉与深度计软约束生成高精度参考轨迹（Ground Truth）。

---

## 2025年12月24日更新

**更新者**：姜昕彤、李书宇

**更新内容**：CNN 深度学习去噪与姿态解算、对比实验复现

* **CNN-MODEL**：基于空洞卷积（Dilated CNN）与流形李代数（SO3 Exp/Log）的姿态解算工程。
* **CNN-LSTM.py**：复现相关论文的混合神经网络模型。

---

## 2025年12月23日更新

**更新者**：曹帅

**更新内容**：数据预处理核心代码

* 实现异常值补齐（5 点以内插值）、时间戳线性对齐、以及坐标系转换（IMU 对齐 DVL）等基础逻辑。

---

