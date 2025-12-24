
# 水下机器人多模态定位数据代码库

## 项目目录结构

```
.
├── README.md                    # 项目说明文档
├── .gitignore                   # Git忽略文件
├── CNN-MODEL/                   # 深度学习去噪与姿态解算模块
│   ├── figures/                 # 结果可视化图表
│   │   ├── boxplot.jpg          # 箱线图
│   │   ├── methode.jpg          # 方法示意图
│   │   ├── roe.jpg              # 旋转误差图
│   │   └── rpy.jpg              # 滚转-俯仰-偏航图
│   └── src/                     # 核心算法源码
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
├── data process/                # 数据处理模块
│   └── process_rotate.py        # 数据处理脚本（异常值处理、时间戳对齐等）
└── IMU + DVL/                   # 传统处理与对比实验模块
    ├── CNN-LSTM.py              # CNN-LSTM混合神经网络模型
    └── comparison_experiment1.py # 模型对比实验脚本
```

---

## 2025年12月24日更新 (2)

**更新者**：姜昕彤

**更新内容**：CNN深度学习去噪与姿态解算模块

### 新增工程文件

**CNN-MODEL/** - 基于空洞卷积神经网络(Dilated CNN)的陀螺仪去噪与姿态解算工程

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

**process_rotate.py** - 数据处理脚本

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
