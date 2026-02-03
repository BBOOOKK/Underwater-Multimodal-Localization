
# 水下机器人多模态定位数据代码库

## 项目目录结构

```
.
├── README.md                    # 项目说明文档
├── CNN-MODEL/                   # IMU标定
│   ├── figures/                 # 结果可视化
│   │   ├── boxplot.jpg          
│   │   ├── methode.jpg         
│   │   ├── roe.jpg              
│   │   └── rpy.jpg              
│   └── src/                     # 核心算法
│       ├── dataset.py           # PyTorch数据集加载类
│       ├── learning.py          # 模型训练与验证核心逻辑
│       ├── lie_algebra.py       # 李群李代数工具库 (SO3 Exp/Log运算)
│       ├── losses.py            # 损失函数定义
│       ├── main_EUROC.py        # Euroc数据集训练入口
│       ├── main_TUMVI.py        # TUM-VI数据集训练入口
│       ├── networks.py          # Dilated CNN网络架构定义
│       ├── pic.py               # 绘图与可视化工具脚本
│       ├── process.py           # 数据预处理核心流程
│       └── utils.py             # 通用工具函数库
│
├── MonoVision-Depth-Aided-Trajectory/  # 单目+深度约束生成参考轨迹
│   └── generate_gt_colmap.py     
├── data process/                # 数据处理
│   └── process_rotate.py 
└── IMU-DVL/                    # IMU-DVL融合定位
    └── Compare_Experiment/     # 多模型对比实验
        ├── Com.py              # 四模型对比实验主程序
        ├── config.json         # 实验配置文件
        ├── results_summary.csv # 实验结果汇总表
        └── pth/                # 模型权重文件
```

---
## 2026年2月3日更新

**更新者**：李书宇

**更新内容**：IMU-DVL多模型对比实验

### 新增代码文件

1. **Com.py** - 四模型对比实验主程序（CNN/LSTM/IONet/TCN）
2. **config.json** - 实验配置文件
3. **results_summary.csv** - 实验结果汇总表

### 主要内容
* **数据预处理**：深度异常值清洗、位移增量计算
* **模型架构**：
  - Multi-branch CNN：4层1D卷积网络
  - 2-Layer LSTM：双层LSTM网络
  - IONet：双向LSTM网络
  - 2-Layer TCN：时序卷积网络
* **评估指标**：N-SRMSE、RMSE、端点误差、推理时间

---
## 2026年1月5日更新

**更新者**：孙超

**更新内容**：单目+深度约束生成参考轨迹代码

### 新增代码文件

1. **generate_gt_colmap.py** - 使用colmap生成参考轨迹脚本

### 主要内容
* 基于SOLAQUA数据集，解析bag文件
* 通过单目视觉信息+深度计软约束生成参考轨迹

---

## 2025年12月24日更新 (2)

**更新者**：姜昕彤

**更新内容**：CNN深度学习去噪与姿态解算模块

### 新增工程文件

1. **CNN-MODEL** - 基于空洞卷积神经网络(Dilated CNN)的陀螺仪去噪与姿态解算工程

### 主要内容
* **networks.py**: 定义了用于去噪的 Dilated CNN 网络模型结构
* **lie_algebra.py**: 实现了流形上的李代数运算 (Exp/Log)
* **learning.py**: 封装了模型的训练循环（Training Loop）、验证 (Validation) 与模型保存逻辑
* **dataset.py**: 处理复杂的多模态数据加载，支持 PyTorch `DataLoader` 接口

---

## 2025年12月24日更新 (1)

**更新者**：李书宇

**更新内容**：对比实验代码

### 新增代码文件

1. **CNN-LSTM.py** - CNN-LSTM混合神经网络模型
2. **comparison_experiment1.py** - CNN/LSTM/IONet/TCN对比实验脚本

### 主要内容

* 基于清洗后数据进行对比试验：CNN/LSTM/IONet/TCN
* 复现论文CNN-LSTM混合神经网络模型

---

## 2025年12月23日更新

**更新者**：曹帅

**更新内容**：数据处理代码

### 新增代码文件

1. **process_rotate.py** - 数据处理脚本

### 主要内容

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
