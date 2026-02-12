收到！已将更新者名字修正为 **姜昕彤**，并保持了完整的详细目录结构。你可以直接复制下面的内容覆盖你现在的 `README.md`。

---

# 水下机器人多模态定位数据代码库

## 项目目录结构

```
.
├── README.md                    # 项目说明文档
├── imu_dvl动捕数据.zip          # IMU-DVL动捕原始数据
├── processed.zip                # 处理后的数据
├── raw.zip                      # 原始数据
├── code/                        # [曹帅/李书宇] 传统处理与对比实验代码
│   ├── process_rotate.py        # 数据处理脚本
│   ├── CNN-LSTM.py              # CNN-LSTM混合神经网络模型
│   └── comparison_experiment1.py # 对比实验
└── CNN-MODEL/                   # [新增] 深度学习去噪与姿态解算项目 (姜昕彤)
    ├── figures/                 # 结果可视化图表 (boxplot, trajectories等)
    ├── src/                     # 核心算法源码
    │   ├── data/                # 数据存放目录
    │   │   └── MY_DATA/         # 自定义数据集
    │   ├── check_gyro_unit.py   # 陀螺仪数据单位检查工具
    │   ├── dataset.py           # PyTorch数据集加载类 (Dataset/Dataloader)
    │   ├── learning.py          # 模型训练与验证核心逻辑
    │   ├── lie_algebra.py       # 李群李代数工具库 (SO3 Exp/Log 运算)
    │   ├── losses.py            # 损失函数定义 (Loss Functions)
    │   ├── networks.py          # Dilated CNN 网络架构定义
    │   ├── pic.py               # 绘图与可视化工具脚本
    │   ├── process.py           # 数据预处理核心流程
    │   ├── test_new_dataset.py  # 新数据集测试与推理脚本
    │   └── utils.py             # 通用工具函数库 (加载/保存/计算指标)
    ├── check_data_quality.py    # 数据质量检查脚本
    ├── convert_my_data.py       # 自定义数据格式转换工具
    ├── evaluate_result.py       # 结果评估脚本 (计算AOE/RWE误差)
    ├── imu_to_bvh.py            # IMU数据转BVH骨骼动画工具
    ├── main_EUROC.py            # Euroc数据集训练入口
    ├── main_TUMVI.py            # TUM-VI数据集训练入口
    ├── requirements.txt         # 项目依赖库清单
    └── LICENCE                  # 许可证文件

```
---

## 2025年12月24日更新 (2)

**更新者**：姜昕彤
**更新内容**：CNN-MODEL 深度学习去噪与姿态解算模块

### 新增代码目录

**CNN-MODEL/** - 基于空洞卷积神经网络(Dilated CNN)的陀螺仪去噪与姿态解算完整工程

### 核心训练入口说明 (`root` 目录)

这两个文件是项目的**主程序入口**，直接运行即可开始训练或测试：

1. **`main_EUROC.py` (Euroc数据集主入口)**
* **功能**：针对 **EuRoC MAV Dataset** 的完整训练与测试脚本。
* **流程**：自动加载 `src/data` 下的 Euroc 格式数据，实例化 Dilated CNN 模型，定义优化器（Adam）与损失函数，执行 Training Loop，并实时在验证集上评估性能。
* **输出**：训练完成后会自动保存模型权重至 `results/` 目录，并输出初步的轨迹对比图。


2. **`main_TUMVI.py` (TUM-VI数据集主入口)**
* **功能**：针对 **TUM-VI Dataset** 的训练与测试脚本。
* **适配**：适配了 TUM-VI 数据集特有的数据结构与坐标系定义。
* **用途**：用于在更大规模数据集上验证模型的泛化能力，功能逻辑与 `main_EUROC.py` 保持一致。



### 核心算法源码说明 (`src/` 目录)

* **networks.py**: 定义了用于去噪的 Dilated CNN 网络模型结构。
* **lie_algebra.py**: 实现了流形上的李代数运算 (Exp/Log)，保证姿态更新的数学严谨性。
* **learning.py**: 封装了模型的训练循环（Training Loop）、验证 (Validation) 与模型保存逻辑。
* **dataset.py**: 处理复杂的多模态数据加载，支持 PyTorch `DataLoader` 接口。
* **process.py**: 包含数据清洗、对齐与归一化等预处理步骤。

### 工具脚本说明

* **evaluate_result.py**: **[核心评估]** 加载训练好的模型，计算绝对航向误差 (AOE) 与相对误差 (ROE/RWE)，并生成论文级别的误差分析图表。
* **check_data_quality.py**: 用于在训练前自动诊断数据异常（如单位错误、丢帧、NaN值等）。
* **imu_to_bvh.py**: 实用工具，将解算出的姿态导出为 .bvh 骨骼动画格式，便于在 Unity/Blender 中进行可视化回放。
---

## 2025年12月24日更新 (1)

**更新者**：李书宇

**更新内容**：对比实验代码

### 新增代码文件

1. **CNN-LSTM.py** - CNN-LSTM混合神经网络模型
2. **comparison_experiment1.py** - 模型对比实验脚本

### 主要功能

* 基于清洗后数据进行对比试验：CNN/LSTM/IONet/TCN
* 复现论文的CNN-LSTM混合神经网络模型

---

## 2025年12月23日更新

**更新者**：曹帅

**更新内容**：数据处理代码

### 新增代码文件

**process_rotate.py** - 数据处理脚本

### 主要功能

* 异常值处理（5个数据异常以内补齐）
* 重叠截断（从各传感器、真值同时有效的部分开始）
* 时间戳对齐（线性插值）
* IMU对齐DVL坐标系
* 航位推算
* BASE_DIR 主目录
* SUB_DIR 子目录
* FILE_GT 真值名称
* MANUAL_OFFSET 手动调整初始航向角，填0自动寻找角度
* LIMIT_COUNT 决定连续异常值补齐最大数量，默认为5