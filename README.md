# Underwater-Multimodal-Localization
Underwater Multimodal Localization Technology

## Version 0.0

本次数据由process脚本处理，其中实现功能如下：

### 数据处理功能

1. **异常值处理**
   - 5个数据异常以内补齐

2. **重叠截断**
   - 从各传感器、真值同时有效的部分开始

3. **时间戳对齐**
   - 线性插值

4. **IMU对齐DVL坐标系**

5. **航位推算**

### 关键参数说明

- **BASE_DIR**: 主目录
- **SUB_DIR**: 子目录  
- **FILE_GT**: 真值名称
- **MANUAL_OFFSET**: 手动调整初始航向角，填0自动寻找角度
- **LIMIT_COUNT**: 决定连续异常值补齐最大数量，默认为5

> 有问题联系 #曹帅
