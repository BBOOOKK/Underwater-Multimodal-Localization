# 水下机器人多模态定位数据代码库

## 项目目录结构

```
.
├── README.md                    # 项目说明文档
├── imu_dvl动捕数据.zip          # IMU-DVL动捕原始数据
├── processed.zip                # 处理后的数据
├── raw.zip                      # 原始数据
└── code/
    ├── process_rotate.py        # 数据处理脚本（异常值处理、重叠截断、时间戳对齐等）
    ├── CNN-LSTM.py              # CNN-LSTM混合神经网络模型
    └── comparison_experiment1.py # 模型对比实验脚本
```


---

## 2025年12月24日更新

**更新者**：李书宇  
**更新内容**：对比实验

### 新增代码文件
1. **CNN-LSTM.py** - CNN-LSTM混合神经网络模型
2. **comparison_experiment1.py** - 模型对比实验脚本

### 主要功能
- 实现CNN-LSTM混合架构用于水下定位
- 添加模型训练与评估流程
- 进行不同模型的对比实验分析

---

## 2025年12月23日更新

**更新者**：曹帅  
**更新内容**：数据处理功能与代码

## 新增代码文件

**process_rotate.py** - 数据处理脚本

### 主要功能

- 异常值处理（5个数据异常以内补齐）
- 重叠截断（从各传感器、真值同时有效的部分开始）
- 时间戳对齐（线性插值）
- IMU对齐DVL坐标系
- 航位推算
- BASE_DIR 主目录
- SUB_DIR 子目录  
- FILE_GT 真值名称
- MANUAL_OFFSET 手动调整初始航向角，填0自动寻找角度
- LIMIT_COUNT 决定连续异常值补齐最大数量，默认为5

---
*