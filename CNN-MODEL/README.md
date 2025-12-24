这是一个为你定制的 `README.md` 文件。

我是根据你提供的文件列表（蓝色高亮部分）、我们之前修复的逻辑（轴向对齐、数据清洗、绘图修复）以及项目的核心功能（IMU去噪、姿态解算）编写的。

你可以直接将以下内容复制到你项目根目录下的 `README.md` 文件中。

---

# Deep Learning for IMU Gyroscope Denoising & Attitude Estimation

基于深度学习（GyroNet）的 MEMS 陀螺仪去噪与开环姿态估计项目。本项目旨在利用卷积神经网络（CNN）从受噪声和零偏影响的原始 IMU 数据中提取纯净的角速度，从而实现高精度的航位推算（Dead Reckoning）。

## 📖 项目简介

本项目复现并改进了基于数据驱动的惯性导航去噪方案。核心功能包括：

1. **数据诊断与清洗**：自动识别 IMU 与真值（Ground Truth）之间的轴向错位（Axis Misalignment）和时间偏移。
2. **GyroNet 模型训练**：使用 Dilated CNN 学习陀螺仪的误差模型。
3. **实时推理支持**：训练后的模型可用于在线实时去噪。
4. **高精度评估**：提供 AOE (绝对姿态误差) 和 ROE (相对姿态误差) 评估，并解决了欧拉角可视化的相位折叠（Phase Wrapping）问题。

## 📂 项目目录结构

以下列出了项目中的核心文件及其功能：

### 根目录

* **`main_EUROC.py`** (核心入口):
* **功能**：项目的主训练脚本。
* **自动化**：内置了 `regenerate_data_from_csv` 函数，每次运行时会自动清理旧缓存、读取原始 CSV、应用轴向修正并生成 `.p` 训练数据。
* **流程**：数据生成 -> 训练 -> 初步测试。


* **`evaluate_result.py`** (核心评估):
* **功能**：加载训练好的模型 (`weights.pt`) 和测试集，计算 AOE 和 ROE 指标。
* **可视化**：生成无跳变（Unwrapped）的姿态轨迹对比图和误差分布箱线图。


* **`main_TUMVI.py`**:
* **功能**：针对 TUM-VI 数据集格式的训练入口脚本。



### `src/` 核心模块

* **`src/learning.py`**:
* 定义了 `LearningBasedProcessing` 和 `GyroLearningBasedProcessing` 类。
* 包含了训练循环 (Train Loop)、验证、测试以及**绘图函数**（已修复 `plot_orientation` 中的 180 度跳变问题）。


* **`src/networks.py`**:
* 定义了 `GyroNet` 网络架构（基于空洞卷积 Dilated CNN）。


* **`src/dataset.py`**:
* 定义了 `EUROCDataset` 类，负责加载 `.p` 数据文件，进行归一化处理，并构建 PyTorch DataLoader。


* **`src/losses.py`**:
* 定义了训练用的损失函数（如 `GyroLoss`），基于  旋转矩阵计算误差。


* **`src/lie_algebra.py`**:
* 实现了李群/李代数  的数学运算（指数映射、对数映射、四元数运算等）。


* **`src/utils.py`**:
* 通用的工具函数，包括 Pickle 文件的读写 (`pload`, `pdump`) 等。


* **`src/process.py`**:
* 数据预处理相关的辅助函数。


* **`src/data/`**:
* **`check_gyro_unit.py`**: 检查陀螺仪单位（rad/s vs deg/s）的工具。
* **`test_new_dataset.py`**:用于测试新数据集加载是否正常的调试脚本。
* **`MY_DATA/`**: (自动生成) 存放处理后的二进制 `.p` 数据文件。



## 🚀 快速开始

### 1. 环境准备

确保安装了以下依赖库：

```bash
pip install torch numpy scipy pandas matplotlib termcolor

```

### 2. 数据准备

将你的原始 CSV 数据放入指定目录（例如 `E:\水下导航资料\processed_data`）。CSV 应包含 IMU 数据 (`gryX`, `accX`...) 和真值数据 (`roll_body`, `pitch_body`...)。

### 3. 训练 (Training)

直接运行 `main_EUROC.py`。该脚本会自动执行以下步骤：

1. 清理 `src/data/MY_DATA` 下的旧文件。
2. 读取 CSV，应用轴向修正，生成新的 `.p` 文件。
3. 开始训练模型（默认 300 Epochs）。

```bash
python main_EUROC.py

```

### 4. 评估 (Evaluation)

训练完成后，运行评估脚本查看详细指标和图表：

```bash
python evaluate_result.py

```

结果将保存在 `E:\results\EUROC\<timestamp>\<seq_id>\` 目录下，包含：

* `eval_ate_error.png`: 绝对误差曲线。
* `eval_traj_unwrapped.png`: **去除了相位折叠的姿态对比图（最直观的效果图）**。
* `eval_ROE_boxplot.png`: 相对误差分布图。

## 📊 结果展示

经过训练，模型在测试集上表现优异，显著消除了陀螺仪的零偏漂移：

* **Yaw (航向角)**: 漂移几乎被完全消除，长时间运行仍能紧贴真值。
* **Pitch/Roll**: 动态跟踪性能良好。

*(在此处可以放一张你生成的 eval_traj_unwrapped.png 图片)*

## 🛠️ 部署 (Real-time Deployment)

训练生成的 `weights.pt` 可直接用于实车部署。模型仅依赖过去的 IMU 数据窗口（Causal），无需未来数据，适合实时运行。

## 📝 引用

本项目参考了 Martin Brossard 等人的论文 *"Denoising IMU Gyroscopes With Deep Learning for Open-Loop Attitude Estimation"*。