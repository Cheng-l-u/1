# attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SecurityWeightedSpatialAttention(nn.Module):
    """
    安全权重空间注意力机制
    核心创新：根据节点类型（账号/IP/设备）和风险历史分配注意力权重
    """

    def __init__(self, in_features, hidden_dim=64):
        super().__init__()

        # 节点类型编码 (0:user, 1:ip, 2:device, 3:unknown)
        self.node_type_embed = nn.Embedding(4, 16)

        # 风险历史编码
        self.risk_proj = nn.Linear(1, 16)

        # 注意力计算网络
        self.attn_proj = nn.Sequential(
            nn.Linear(in_features + 32, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, node_features, node_types, risk_history):
        """
        Args:
            node_features: [N, F] 节点特征
            node_types: [N] 节点类型
            risk_history: [N, 1] 风险历史分数
        """
        # 类型嵌入
        type_embed = self.node_type_embed(node_types)

        # 风险嵌入
        risk_embed = self.risk_proj(risk_history)

        # 组合特征
        combined = torch.cat([node_features, type_embed, risk_embed], dim=1)

        # 计算注意力分数
        attn_scores = self.attn_proj(combined).squeeze(-1)
        attn_weights = self.softmax(attn_scores.unsqueeze(0))

        # 加权聚合
        weighted_features = attn_weights @ node_features

        return weighted_features.squeeze(0), attn_weights


class AnomalyTemporalAttention(nn.Module):
    """
    异常时间注意力机制
    核心创新：对异常事件时间点给予更高关注
    """

    def __init__(self, time_steps, hidden_dim=128):
        super().__init__()
        self.time_steps = time_steps

        # 时间位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, time_steps, hidden_dim))

        # 异常模式检测
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 时序注意力
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # 攻击模式模板
        self.bruteforce_pattern = nn.Parameter(torch.randn(1, 10, hidden_dim))
        self.credential_stuffing_pattern = nn.Parameter(torch.randn(1, 5, hidden_dim))

    def forward(self, time_series_features):
        """
        Args:
            time_series_features: [B, T, F] 时序特征
        """
        B, T, F = time_series_features.shape

        # 添加位置编码
        if T != self.time_steps:
            # 动态调整位置编码
            pos_encoding = self.pos_encoding[:, :T, :F]
        else:
            pos_encoding = self.pos_encoding[:, :T, :F]

        pos_encoded = time_series_features + pos_encoding

        # 时序自注意力
        attn_out, attn_weights = self.temporal_attn(pos_encoded, pos_encoded, pos_encoded)

        # 异常模式检测
        anomaly_scores = self.anomaly_detector(attn_out)

        # 与攻击模板匹配（暴力破解）
        pattern_match_scores = []
        if T >= 10:
            for i in range(T - 10 + 1):
                window = attn_out[:, i:i + 10, :]
                similarity = F.cosine_similarity(
                    window.mean(dim=1, keepdim=True),
                    self.bruteforce_pattern.expand(B, -1, -1).mean(dim=1, keepdim=True),
                    dim=-1
                )
                pattern_match_scores.append(similarity)

        pattern_score = torch.stack(pattern_match_scores, dim=1).max(dim=1)[0] if pattern_match_scores else torch.zeros(
            B, 1, device=time_series_features.device)

        # 加权时序特征
        temporal_weights = anomaly_scores * (1 + pattern_score.unsqueeze(-1))
        weighted_features = (attn_out * temporal_weights).sum(dim=1) / (temporal_weights.sum(dim=1) + 1e-8)

        return weighted_features, attn_weights, anomaly_scores


class SpatioTemporalFusionModule(nn.Module):
    """
    时空联合模块
    融合空间图特征和时间序列特征
    """

    def __init__(self, spatial_dim, temporal_dim, fusion_dim=256):
        super().__init__()

        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(spatial_dim, num_heads=4, batch_first=True)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, fusion_dim),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Linear(spatial_dim + temporal_dim, fusion_dim)
        self.layer_norm = nn.LayerNorm(fusion_dim)

        # 残差连接
        self.residual_proj = nn.Linear(spatial_dim + temporal_dim, fusion_dim)

    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features: [B, spatial_dim] 空间图特征
            temporal_features: [B, temporal_dim] 时序特征
        """
        # 处理维度不匹配
        if len(spatial_features.shape) == 1:
            spatial_features = spatial_features.unsqueeze(0)
        if len(temporal_features.shape) == 1:
            temporal_features = temporal_features.unsqueeze(0)

        # 确保batch维度一致
        if spatial_features.shape[0] != temporal_features.shape[0]:
            if spatial_features.shape[0] == 1:
                spatial_features = spatial_features.expand(temporal_features.shape[0], -1)
            elif temporal_features.shape[0] == 1:
                temporal_features = temporal_features.expand(spatial_features.shape[0], -1)

        # 特征拼接
        combined = torch.cat([spatial_features, temporal_features], dim=1)

        # 门控机制
        gate_value = self.gate(combined)

        # 投影
        fused = self.fusion_proj(combined)

        # 门控加权
        output = gate_value * fused + (1 - gate_value) * self.residual_proj(combined)

        return self.layer_norm(output)