"""
* This file is part of RNIN-VIO
* (Architecture Upgrade: Two-Stream Encoder + SE-Attention Mechanism)
"""

import torch
import torch.nn as nn
from torch.nn.init import orthogonal_

# =============================================================================
# 1. SE-Block (Squeeze-and-Excitation)
# =============================================================================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# =============================================================================
# 2. SE-ResBlock (带注意力机制的残差块)
# =============================================================================
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(planes * self.expansion),
        )
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

        # 🔥 [新增] SE-Block: 让每个残差块都能自动调整通道权重
        self.se = SELayer(planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.convs(x)

        # 🔥 应用注意力加权
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# =============================================================================
# 3. Two-Stream ResNet-LSTM Network
# =============================================================================
class ResNetLSTMSeqNet(nn.Module):
    def __init__(self, cfg):
        super(ResNetLSTMSeqNet, self).__init__()
        self.in_dim = cfg['model_param']['in_dim'] # 应该是 9 (6 IMU + 3 DVL)
        self.out_dim = cfg['model_param']['output_dim']
        self.c0 = cfg['model_param']['c0']
        self.dropout = cfg['model_param']['dropout']

        # ---------------------------------------------------------------------
        # 🔥 [核心升级] 双流编码器 (Two-Stream Encoder)
        # ---------------------------------------------------------------------
        # Stream 1: IMU 专用通道 (处理高频加速度/角速度)
        self.imu_dim = 6
        self.input_block_imu = nn.Sequential(
            nn.Conv1d(self.imu_dim, self.c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.c0),
            nn.ReLU(inplace=True)
        )

        # Stream 2: DVL 专用通道 (处理低频速度观测)
        self.dvl_dim = self.in_dim - self.imu_dim # 通常是 3
        self.input_block_dvl = nn.Sequential(
            nn.Conv1d(self.dvl_dim, self.c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.c0),
            nn.ReLU(inplace=True)
        )

        # Fusion Layer: 融合两个流 + SE Attention 加权
        # 输入: c0 + c0 = 2 * c0
        # 输出: c0 (恢复到 ResNet 的输入维度)
        self.fusion_se = SELayer(self.c0 * 2)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(self.c0 * 2, self.c0, kernel_size=1, bias=False), # 1x1 卷积降维
            nn.BatchNorm1d(self.c0),
            nn.ReLU(inplace=True)
        )

        # ---------------------------------------------------------------------
        # ResNet Backbone (SE-ResNet)
        # ---------------------------------------------------------------------
        self.base_plane = 64
        self.residual_groups = self._make_layer(ResBlock, self.c0, cfg['model_param']['ks'], cfg['model_param']['ds'])

        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(self.c0, self.base_plane, kernel_size=1, bias=False), # expansion
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), # GAP -> [B, 64, 1]
            nn.Dropout(self.dropout)
        )

        # ---------------------------------------------------------------------
        # LSTM & Output Heads
        # ---------------------------------------------------------------------
        self.lstm_size = 128 # 增大一点 LSTM 容量以适应丰富特征
        self.lstm = nn.LSTM(input_size=self.base_plane, hidden_size=self.lstm_size,
                            num_layers=2, batch_first=True, dropout=self.dropout)

        self.fc_out = nn.Linear(self.lstm_size, self.out_dim)
        self.fc_cov = nn.Linear(self.lstm_size, self.out_dim)

        self.init_weights()

    def _make_layer(self, block, planes, ks, ds):
        layers = []
        for kernel, dilation in zip(ks, ds):
            layers.append(block(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Orthogonal init for LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                orthogonal_(param.data)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Forget gate bias = 1
                param.data[self.lstm_size:2*self.lstm_size].fill_(1.0)

    def init_hidden(self, x, batch_size, first_batch=True):
        # 如果是序列的第一个 batch，初始化全0
        h_shape = (2, batch_size, self.lstm_size) # num_layers=2
        if first_batch:
            if torch.cuda.is_available():
                return (torch.zeros(*h_shape).to(x.device),
                        torch.zeros(*h_shape).to(x.device))
            else:
                return (torch.zeros(*h_shape), torch.zeros(*h_shape))
        else:
            # 如果不是，通常由外部传入上一时刻的 hidden (训练循环中处理)
            return None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, compute_type=None, hn=None, cn=None):
        """
        x shape: [Batch, Seq_Len, Channels(9)]
        """
        self.lstm.flatten_parameters()
        batch_size = x.size(0)
        seq_len = x.size(1)

        # ResNet 需要 [B, C, L] 格式
        # 先合并 Batch 和 Seq 维度进行并行卷积处理
        x = x.view(batch_size * seq_len, x.size(2), 1).permute(0, 2, 1) # -> [B*S, 9, 1]
        # 注意：这里的 1 是因为我们用 Conv1d 处理每个时间步的特征向量
        # 但原始代码似乎期望 [B*S, C, Win] 或者是把 Window 维度展开了？
        # 检查原始 dataset: feat shape 是 [Seq_Len, Window_Size, Channels]
        # 在 dataset.py 中: seq_feat = np.array(seq_feat) -> [Seq_Len, Channels, Window_Size] (转置过)
        # 这里的 x 输入应该是 [Batch, Seq_Len, Channels, Window_Size]

        # 修正输入 View 逻辑
        # 假设输入 x 来自 DataLoader: [Batch, Seq_Len, Channels, Window_Size]
        # 需要 view 成 [Batch * Seq_Len, Channels, Window_Size]
        x = x.view(-1, x.size(2), x.size(3))

        # -----------------------------------------------------------------
        # 🔥 [Step 1] 双流拆分 (Split)
        # -----------------------------------------------------------------
        x_imu = x[:, :self.imu_dim, :]  # IMU (6)
        x_dvl = x[:, self.imu_dim:, :]  # DVL (3)

        # -----------------------------------------------------------------
        # 🔥 [Step 2] 独立编码 (Independent Encode)
        # -----------------------------------------------------------------
        feat_imu = self.input_block_imu(x_imu) # -> [B*S, c0, W/2]
        feat_dvl = self.input_block_dvl(x_dvl) # -> [B*S, c0, W/2]

        # -----------------------------------------------------------------
        # 🔥 [Step 3] 融合与加权 (Fusion & Attention)
        # -----------------------------------------------------------------
        feat_fused = torch.cat([feat_imu, feat_dvl], dim=1) # -> [B*S, 2*c0, W/2]

        # SE-Block 自动判断 DVL 和 IMU 哪个更重要
        feat_fused = self.fusion_se(feat_fused)

        # 1x1 卷积融合回 c0
        x = self.fusion_conv(feat_fused) # -> [B*S, c0, W/2]

        # -----------------------------------------------------------------
        # [Step 4] ResNet Backbone (SE-ResNet)
        # -----------------------------------------------------------------
        x = self.residual_groups(x)
        embed = self.resnet_post_pro(x) # -> [B*S, 64, 1]

        # -----------------------------------------------------------------
        # [Step 5] LSTM 时序建模
        # -----------------------------------------------------------------
        embed = embed.view(batch_size, seq_len, -1) # -> [B, S, 64]

        if hn is None:
            # 训练时通常每个Batch独立初始化Hidden
            hidden = self.init_hidden(x, batch_size, first_batch=True)
        else:
            hidden = (hn, cn)

        out, (hn2, cn2) = self.lstm(embed, hidden) # out: [B, S, 128]

        # -----------------------------------------------------------------
        # [Step 6] Output Heads
        # -----------------------------------------------------------------
        pred = self.fc_out(out)     # [B, S, 3]
        pred_cov = self.fc_cov(out) # [B, S, 3]

        if compute_type == 'dp':
            # 只返回预测，不返回 Hidden 状态（用于 Loss 计算等）
            return pred

        return pred, pred_cov