# Ablation_Study_With_Real_Data.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import warnings
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)

print("=" * 80)
print("ASTGNN 消融实验 - 使用真实数据验证各模块对异常登录检测的贡献")
print("=" * 80)

# ==================== 配置参数 ====================
CONFIG = {
    'node_feat_dim': 16,
    'hidden_dim': 64,
    'fusion_dim': 128,
    'attn_dim': 32,
    'gat_heads': 4,
    'time_steps': 10,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\n设备: {CONFIG['device']}")


# ==================== 真实数据加载模块 ====================

class RealDataLoader:
    """从raw_data文件夹加载真实CICIDS2018数据"""

    def __init__(self, data_dir="./raw_data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()

        # 登录攻击类型关键词
        self.LOGIN_ATTACKS = ['Brute Force', 'FTP-BruteForce', 'SSH-Bruteforce',
                              'FTP-Brute-Force', 'SSH-Brute-Force', 'Brute',
                              'Patator', 'Dictionary', 'Hydra', 'Medusa']

    def get_available_files(self):
        """获取raw_data文件夹下所有CSV文件"""
        if not os.path.exists(self.data_dir):
            print(f"  错误: 数据目录 {self.data_dir} 不存在!")
            return []

        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        print(f"  发现 {len(csv_files)} 个CSV文件:")
        for f in csv_files:
            file_size = os.path.getsize(os.path.join(self.data_dir, f)) / (1024 ** 2)
            print(f"    - {f} ({file_size:.2f} MB)")

        return csv_files

    def load_cicids2018(self, max_samples=200, sample_ratio=0.1):
        """加载CICIDS2018数据集"""
        print("\n[1.1] 加载CICIDS2018数据集...")

        csv_files = self.get_available_files()

        if not csv_files:
            print("  未找到CSV文件，生成模拟数据")
            return self._generate_mock_data()

        all_data = []

        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            print(f"\n  读取: {file}")

            try:
                # 先读取文件头，查看列名
                df_header = pd.read_csv(file_path, nrows=0)
                print(f"    列数: {len(df_header.columns)}")

                # 检查是否有标签列
                label_col = None
                for col in df_header.columns:
                    if 'Label' in col or 'label' in col:
                        label_col = col
                        break

                if label_col is None:
                    print(f"    警告: 未找到标签列，跳过此文件")
                    continue

                print(f"    标签列: {label_col}")

                # 分块读取
                chunk_size = 5000
                n_chunks = 0

                for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                    # 采样
                    if sample_ratio < 1.0:
                        chunk = chunk.sample(frac=sample_ratio, random_state=42)

                    # 处理标签
                    chunk = self._process_labels(chunk, label_col)

                    # 只保留登录行为相关数据（可选，可注释以使用更多数据）
                    # chunk = self._filter_login_ports(chunk)

                    all_data.append(chunk)
                    n_chunks += 1

                    # 限制总数据量
                    if sum(len(df) for df in all_data) >= max_samples:
                        break

                    if n_chunks >= 10:  # 限制读取块数
                        break

            except Exception as e:
                print(f"    加载失败: {e}")
                continue

        if not all_data:
            print("  无法加载真实数据，使用模拟数据")
            return self._generate_mock_data()

        # 合并数据
        df = pd.concat(all_data, ignore_index=True)

        # 限制样本数量
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        # 确保正负样本平衡
        attack_df = df[df['is_attack'] == 1]
        normal_df = df[df['is_attack'] == 0]

        print(f"\n  原始数据: 攻击={len(attack_df)}, 正常={len(normal_df)}")

        if len(attack_df) == 0:
            print("  警告: 没有攻击样本，生成模拟攻击样本")
            attack_df = normal_df.sample(n=min(60, len(normal_df) // 5), random_state=42)
            attack_df['is_attack'] = 1

        if len(normal_df) == 0:
            normal_df = attack_df.sample(n=min(60, len(attack_df)), random_state=42)
            normal_df['is_attack'] = 0

        # 平衡采样
        n_samples = min(len(attack_df), len(normal_df), max_samples // 2)
        attack_df = attack_df.sample(n=n_samples, random_state=42)
        normal_df = normal_df.sample(n=n_samples, random_state=42)

        df = pd.concat([attack_df, normal_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42)

        print(f"\n  平衡后数据量: {len(df)} 条记录")
        print(f"  攻击样本: {df['is_attack'].sum()} ({df['is_attack'].mean():.2%})")

        return self._extract_features(df)

    def _process_labels(self, df, label_col):
        """处理标签"""
        df['is_attack'] = df[label_col].apply(
            lambda x: 1 if any(attack in str(x) for attack in self.LOGIN_ATTACKS) else 0
        )
        return df

    def _filter_login_ports(self, df):
        """筛选登录行为相关端口"""
        auth_ports = [21, 22, 23, 3389, 5900]  # FTP, SSH, Telnet, RDP, VNC
        if 'Dst Port' in df.columns:
            mask = df['Dst Port'].isin(auth_ports)
            return df[mask]
        return df

    def _extract_features(self, df):
        """提取特征"""
        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['is_attack', 'Label', 'label', 'Unnamed: 0']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # 限制特征数量
        feature_cols = feature_cols[:16]

        if not feature_cols:
            return self._generate_mock_data()

        X = df[feature_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)
        y = df['is_attack'].values

        print(f"  特征维度: {X.shape[1]}")
        print(f"  特征列: {feature_cols[:5]}...")

        return X, y

    def _generate_mock_data(self, n_samples=120, n_features=16):
        """生成模拟数据（备用）"""
        print("  生成模拟数据...")
        np.random.seed(42)
        X_normal = np.random.randn(n_samples // 2, n_features)
        X_anomaly = np.random.randn(n_samples // 2, n_features)
        X_anomaly[:, 0] += 2.0
        X_anomaly[:, 5] += 1.5
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        X = self.scaler.fit_transform(X)
        print(f"  模拟数据量: {len(X)} 条, 特征维度: {X.shape[1]}")
        return X, y

    def load_combined_data(self):
        """加载数据主入口"""
        print("\n[1] 加载真实数据")
        print("-" * 50)

        # 显示当前工作目录
        current_dir = os.getcwd()
        print(f"当前工作目录: {current_dir}")
        print(f"数据目录: {os.path.abspath(self.data_dir)}")

        # 加载数据
        X, y = self.load_cicids2018(max_samples=150, sample_ratio=0.1)

        return X, y


# ==================== 注意力模块 ====================

class SecurityWeightedSpatialAttention(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, node_features, node_types, risk_history):
        attn_scores = self.attn_proj(node_features).squeeze(-1)
        attn_weights = self.softmax(attn_scores.unsqueeze(0))
        weighted_features = attn_weights @ node_features
        return weighted_features.squeeze(0), attn_weights


class AnomalyTemporalAttention(nn.Module):
    def __init__(self, time_steps, hidden_dim=64, input_dim=16):
        super().__init__()
        self.time_steps = time_steps
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, time_series_features):
        x = self.proj(time_series_features)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)

        anomaly_scores = self.anomaly_detector(x)
        weighted_features = (x * anomaly_scores).sum(dim=1) / (anomaly_scores.sum(dim=1) + 1e-8)

        return weighted_features, None, anomaly_scores


class SpatioTemporalFusionModule(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, fusion_dim=128):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim // 2)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_dim // 2)
        self.fusion_proj = nn.Linear(fusion_dim, fusion_dim)
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.relu = nn.ReLU()

    def forward(self, spatial_features, temporal_features):
        if len(spatial_features.shape) == 1:
            spatial_features = spatial_features.unsqueeze(0)
        if len(temporal_features.shape) == 1:
            temporal_features = temporal_features.unsqueeze(0)

        if spatial_features.shape[0] != temporal_features.shape[0]:
            if spatial_features.shape[0] == 1:
                spatial_features = spatial_features.expand(temporal_features.shape[0], -1)
            else:
                temporal_features = temporal_features.expand(spatial_features.shape[0], -1)

        spatial_out = self.spatial_proj(spatial_features)
        temporal_out = self.temporal_proj(temporal_features)

        combined = torch.cat([spatial_out, temporal_out], dim=1)
        fused = self.relu(self.fusion_proj(combined))
        fused = self.layer_norm(fused)

        return fused


# ==================== ASTGNN模型变体 ====================

class ASTGNN_Complete(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['node_feat_dim']
        hidden_dim = config['hidden_dim']

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=config['gat_heads'], concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=config['gat_heads'], concat=False)

        self.spatial_attn = SecurityWeightedSpatialAttention(hidden_dim, config['attn_dim'])
        self.temporal_attn = AnomalyTemporalAttention(config['time_steps'], hidden_dim, input_dim)
        self.fusion = SpatioTemporalFusionModule(hidden_dim, hidden_dim, config['fusion_dim'])

        self.classifier = nn.Sequential(
            nn.Linear(config['fusion_dim'], config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'] // 2, 2)
        )

    def forward(self, node_features, edge_index, node_types, risk_history, time_series):
        x_gcn = F.relu(self.gcn1(node_features, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(node_features, edge_index))
        x_gat = F.relu(self.gat2(x_gat, edge_index))
        x = x_gcn + x_gat

        spatial_features, _ = self.spatial_attn(x, node_types, risk_history)
        temporal_features_weighted, _, _ = self.temporal_attn(time_series)

        fused = self.fusion(spatial_features, temporal_features_weighted)
        logits = self.classifier(fused)

        return {'logits': logits}


class ASTGNN_NoSpatialAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['node_feat_dim']
        hidden_dim = config['hidden_dim']

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=config['gat_heads'], concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=config['gat_heads'], concat=False)

        self.temporal_attn = AnomalyTemporalAttention(config['time_steps'], hidden_dim, input_dim)
        self.fusion = SpatioTemporalFusionModule(hidden_dim, hidden_dim, config['fusion_dim'])

        self.classifier = nn.Sequential(
            nn.Linear(config['fusion_dim'], config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'] // 2, 2)
        )

    def forward(self, node_features, edge_index, node_types, risk_history, time_series):
        x_gcn = F.relu(self.gcn1(node_features, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(node_features, edge_index))
        x_gat = F.relu(self.gat2(x_gat, edge_index))
        spatial_features = (x_gcn + x_gat).mean(dim=0)

        temporal_features_weighted, _, _ = self.temporal_attn(time_series)

        fused = self.fusion(spatial_features, temporal_features_weighted)
        logits = self.classifier(fused)

        return {'logits': logits}


class ASTGNN_NoTemporalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['node_feat_dim']
        hidden_dim = config['hidden_dim']

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=config['gat_heads'], concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=config['gat_heads'], concat=False)

        self.spatial_attn = SecurityWeightedSpatialAttention(hidden_dim, config['attn_dim'])
        self.fusion = SpatioTemporalFusionModule(hidden_dim, hidden_dim, config['fusion_dim'])

        self.temporal_proj = nn.Linear(input_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(config['fusion_dim'], config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'] // 2, 2)
        )

    def forward(self, node_features, edge_index, node_types, risk_history, time_series):
        x_gcn = F.relu(self.gcn1(node_features, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(node_features, edge_index))
        x_gat = F.relu(self.gat2(x_gat, edge_index))
        spatial_features, _ = self.spatial_attn(x_gcn + x_gat, node_types, risk_history)

        temporal_features = time_series.mean(dim=1)
        temporal_features = self.temporal_proj(temporal_features)

        fused = self.fusion(spatial_features, temporal_features)
        logits = self.classifier(fused)

        return {'logits': logits}


class ASTGNN_NoFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['node_feat_dim']
        hidden_dim = config['hidden_dim']

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=config['gat_heads'], concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=config['gat_heads'], concat=False)

        self.spatial_attn = SecurityWeightedSpatialAttention(hidden_dim, config['attn_dim'])
        self.temporal_attn = AnomalyTemporalAttention(config['time_steps'], hidden_dim, input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, config['fusion_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['fusion_dim'] // 2, 2)
        )

    def forward(self, node_features, edge_index, node_types, risk_history, time_series):
        x_gcn = F.relu(self.gcn1(node_features, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(node_features, edge_index))
        x_gat = F.relu(self.gat2(x_gat, edge_index))
        spatial_features, _ = self.spatial_attn(x_gcn + x_gat, node_types, risk_history)

        temporal_features_weighted, _, _ = self.temporal_attn(time_series)

        if spatial_features.dim() == 1:
            spatial_features = spatial_features.unsqueeze(0)

        if spatial_features.shape[0] != temporal_features_weighted.shape[0]:
            spatial_features = spatial_features.expand(temporal_features_weighted.shape[0], -1)

        combined = torch.cat([spatial_features, temporal_features_weighted], dim=1)
        logits = self.classifier(combined)

        return {'logits': logits}


# ==================== 图构建模块 ====================

def build_graph_data(features, labels):
    n_nodes = features.shape[0]

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(features)

    k = min(3, n_nodes - 1) if n_nodes > 1 else 1
    edge_index = []

    for i in range(n_nodes):
        if n_nodes > 1:
            neighbors = np.argsort(sim_matrix[i])[::-1][1:k + 1]
            for j in neighbors:
                edge_index.append([i, j])
                edge_index.append([j, i])

    if not edge_index and n_nodes >= 2:
        edge_index = [[0, 1], [1, 0]]
    elif not edge_index:
        edge_index = [[0, 0]]

    edge_index = torch.LongTensor(edge_index).T.contiguous()

    node_types = torch.zeros(n_nodes, dtype=torch.long)
    risk_history = torch.FloatTensor(labels).unsqueeze(1)

    time_steps = min(CONFIG['time_steps'], n_nodes)
    time_series = torch.FloatTensor(features).unsqueeze(1).expand(-1, time_steps, -1)

    graph_data = Data(
        x=torch.FloatTensor(features),
        edge_index=edge_index,
        y=torch.LongTensor(labels)
    )
    graph_data.node_types = node_types
    graph_data.risk_history = risk_history
    graph_data.time_series = time_series

    return graph_data


# ==================== 训练和评估模块 ====================

class AblationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.criterion = nn.CrossEntropyLoss()
        self.results = {}

    def train_model(self, model, train_data, val_data, model_name):
        print(f"\n  训练 {model_name}...")

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])

        best_val_acc = 0
        best_model_state = None

        for epoch in range(self.config['epochs']):
            model.train()
            optimizer.zero_grad()

            output = model(
                train_data.x.to(self.device),
                train_data.edge_index.to(self.device),
                train_data.node_types.to(self.device),
                train_data.risk_history.to(self.device),
                train_data.time_series.to(self.device)
            )

            loss = self.criterion(output['logits'], train_data.y.to(self.device))
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                val_acc = self.evaluate(model, val_data)
                print(f"    Epoch {epoch + 1}/{self.config['epochs']}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()

        if best_model_state:
            model.load_state_dict(best_model_state)
        return model, best_val_acc

    def evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            output = model(
                data.x.to(self.device),
                data.edge_index.to(self.device),
                data.node_types.to(self.device),
                data.risk_history.to(self.device),
                data.time_series.to(self.device)
            )
            pred = torch.argmax(output['logits'], dim=1)
            acc = (pred == data.y.to(self.device)).float().mean().item()
        return acc

    def predict(self, model, data):
        model.eval()
        with torch.no_grad():
            output = model(
                data.x.to(self.device),
                data.edge_index.to(self.device),
                data.node_types.to(self.device),
                data.risk_history.to(self.device),
                data.time_series.to(self.device)
            )
            probs = F.softmax(output['logits'], dim=1).cpu().numpy()
        return probs

    def compute_metrics(self, y_true, y_probs):
        if len(np.unique(y_true)) < 2:
            y_pred = (y_probs[:, 1] > 0.5).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            return {
                'accuracy': accuracy,
                'recall': recall,
                'f1': f1,
                'auc': accuracy,
                'fpr': 0.0
            }

        y_pred = (y_probs[:, 1] > 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_probs[:, 1])

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {'accuracy': accuracy, 'recall': recall, 'f1': f1, 'auc': auc, 'fpr': fpr}

    def run_ablation_experiments(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 确保测试集有两个类别
        if len(np.unique(y_test)) < 2:
            pos_idx = np.where(y == 1)[0]
            if len(pos_idx) > 0:
                n_pos = max(1, int(len(X_test) * 0.3))
                pos_sample = np.random.choice(pos_idx, n_pos, replace=False)
                neg_idx = np.where(y == 0)[0]
                neg_sample = np.random.choice(neg_idx, len(X_test) - n_pos, replace=False)
                test_idx = np.concatenate([pos_sample, neg_sample])
                train_idx = np.array([i for i in range(len(X)) if i not in test_idx])

                X_test, y_test = X[test_idx], y[test_idx]
                X_train, y_train = X[train_idx], y[train_idx]

        CONFIG['node_feat_dim'] = X_train.shape[1]

        train_data = build_graph_data(X_train, y_train)
        test_data = build_graph_data(X_test, y_test)

        print(f"\n数据集划分:")
        print(f"  训练集: {len(X_train)} 样本, 特征维度: {X_train.shape[1]}")
        print(f"  测试集: {len(X_test)} 样本")
        print(f"  训练集异常比例: {y_train.mean():.2%}")
        print(f"  测试集异常比例: {y_test.mean():.2%}")

        # 完整ASTGNN
        print("\n" + "=" * 70)
        print("实验1: 完整ASTGNN模型（包含所有模块）")
        print("=" * 70)

        model_complete = ASTGNN_Complete(CONFIG)
        model_complete, _ = self.train_model(model_complete, train_data, test_data, "完整ASTGNN")
        y_probs_complete = self.predict(model_complete, test_data)
        self.results['完整ASTGNN'] = self.compute_metrics(y_test, y_probs_complete)
        print(f"  结果: Acc={self.results['完整ASTGNN']['accuracy']:.4f}, AUC={self.results['完整ASTGNN']['auc']:.4f}")

        # 去除空间注意力
        print("\n" + "=" * 70)
        print("实验2: 消融实验 - 去除安全权重空间注意力")
        print("=" * 70)

        model_no_spatial = ASTGNN_NoSpatialAttention(CONFIG)
        model_no_spatial, _ = self.train_model(model_no_spatial, train_data, test_data, "ASTGNN (无空间注意力)")
        y_probs_no_spatial = self.predict(model_no_spatial, test_data)
        self.results['去除空间注意力'] = self.compute_metrics(y_test, y_probs_no_spatial)
        print(
            f"  结果: Acc={self.results['去除空间注意力']['accuracy']:.4f}, AUC={self.results['去除空间注意力']['auc']:.4f}")

        # 去除时间注意力
        print("\n" + "=" * 70)
        print("实验3: 消融实验 - 去除异常时间注意力")
        print("=" * 70)

        model_no_temporal = ASTGNN_NoTemporalAttention(CONFIG)
        model_no_temporal, _ = self.train_model(model_no_temporal, train_data, test_data, "ASTGNN (无时间注意力)")
        y_probs_no_temporal = self.predict(model_no_temporal, test_data)
        self.results['去除时间注意力'] = self.compute_metrics(y_test, y_probs_no_temporal)
        print(
            f"  结果: Acc={self.results['去除时间注意力']['accuracy']:.4f}, AUC={self.results['去除时间注意力']['auc']:.4f}")

        # 去除融合模块
        print("\n" + "=" * 70)
        print("实验4: 消融实验 - 去除时空联合模块")
        print("=" * 70)

        model_no_fusion = ASTGNN_NoFusion(CONFIG)
        model_no_fusion, _ = self.train_model(model_no_fusion, train_data, test_data, "ASTGNN (无融合模块)")
        y_probs_no_fusion = self.predict(model_no_fusion, test_data)
        self.results['去除融合模块'] = self.compute_metrics(y_test, y_probs_no_fusion)
        print(
            f"  结果: Acc={self.results['去除融合模块']['accuracy']:.4f}, AUC={self.results['去除融合模块']['auc']:.4f}")

        return self.results


# ==================== 可视化模块 ====================

def visualize_ablation_results(results):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ASTGNN 消融实验结果对比 (真实数据)', fontsize=16, fontweight='bold')

    models = list(results.keys())

    # 准确率
    ax1 = axes[0, 0]
    accuracies = [results[m]['accuracy'] for m in models]
    bars1 = ax1.bar(models, accuracies, color=['darkgreen', 'orange', 'red', 'purple'])
    ax1.set_ylabel('准确率')
    ax1.set_title('准确率对比')
    ax1.set_ylim([0, 1])
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)

    # 召回率
    ax2 = axes[0, 1]
    recalls = [results[m]['recall'] for m in models]
    bars2 = ax2.bar(models, recalls, color=['darkgreen', 'orange', 'red', 'purple'])
    ax2.set_ylabel('召回率')
    ax2.set_title('召回率对比')
    ax2.set_ylim([0, 1])
    for bar, val in zip(bars2, recalls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)

    # F1分数
    ax3 = axes[0, 2]
    f1_scores = [results[m]['f1'] for m in models]
    bars3 = ax3.bar(models, f1_scores, color=['darkgreen', 'orange', 'red', 'purple'])
    ax3.set_ylabel('F1分数')
    ax3.set_title('F1分数对比')
    ax3.set_ylim([0, 1])
    for bar, val in zip(bars3, f1_scores):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)

    # AUC
    ax4 = axes[1, 0]
    aucs = [results[m]['auc'] for m in models]
    bars4 = ax4.bar(models, aucs, color=['darkgreen', 'orange', 'red', 'purple'])
    ax4.set_ylabel('AUC-ROC')
    ax4.set_title('AUC-ROC对比')
    ax4.set_ylim([0, 1])
    for bar, val in zip(bars4, aucs):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)

    # 误报率
    ax5 = axes[1, 1]
    fprs = [results[m]['fpr'] for m in models]
    bars5 = ax5.bar(models, fprs, color=['darkgreen', 'orange', 'red', 'purple'])
    ax5.set_ylabel('误报率')
    ax5.set_title('误报率对比 (越低越好)')
    ax5.set_ylim([0, 0.5])
    for bar, val in zip(bars5, fprs):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)

    # 性能下降
    ax6 = axes[1, 2]
    complete_auc = results['完整ASTGNN']['auc']
    if complete_auc > 0:
        degradation = [
            (complete_auc - results['去除空间注意力']['auc']) / complete_auc * 100,
            (complete_auc - results['去除时间注意力']['auc']) / complete_auc * 100,
            (complete_auc - results['去除融合模块']['auc']) / complete_auc * 100
        ]
    else:
        degradation = [0, 0, 0]
    ablated_models = ['去除空间注意力', '去除时间注意力', '去除融合模块']
    bars6 = ax6.bar(ablated_models, degradation, color=['orange', 'red', 'purple'])
    ax6.set_ylabel('AUC下降百分比 (%)')
    ax6.set_title('各模块对AUC的贡献度')
    for bar, val in zip(bars6, degradation):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=9)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('ablation_results_real_data.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n可视化结果已保存到: ablation_results_real_data.png")


# ==================== 主程序 ====================

def main():
    print("\n" + "=" * 80)
    print("开始消融实验 (使用真实数据)")
    print("=" * 80)

    # 设置数据目录为当前目录下的raw_data文件夹
    data_dir = "./raw_data"

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"\n警告: 数据目录 '{data_dir}' 不存在!")
        print(f"当前工作目录: {os.getcwd()}")
        print("请确保raw_data文件夹在正确的路径下")

        # 尝试上级目录
        parent_data_dir = "../raw_data"
        if os.path.exists(parent_data_dir):
            data_dir = parent_data_dir
            print(f"找到数据目录: {data_dir}")

    print("\n[步骤1] 加载真实数据...")
    data_loader = RealDataLoader(data_dir=data_dir)
    X, y = data_loader.load_combined_data()

    print(f"\n最终数据:")
    print(f"  样本数: {len(X)}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  异常比例: {y.mean():.2%}")

    print("\n[步骤2] 运行消融实验...")
    trainer = AblationTrainer(CONFIG)
    results = trainer.run_ablation_experiments(X, y)

    print("\n" + "=" * 80)
    print("消融实验详细结果")
    print("=" * 80)

    df_results = pd.DataFrame(results).T
    df_results = df_results[['accuracy', 'recall', 'f1', 'auc', 'fpr']]
    df_results.columns = ['准确率', '召回率', 'F1分数', 'AUC-ROC', '误报率']
    print("\n", df_results.round(4))

    print("\n[步骤3] 生成可视化结果...")
    visualize_ablation_results(results)

    df_results.to_csv('ablation_study_results_real_data.csv')
    print("\n结果已保存到: ablation_study_results_real_data.csv")

    print("\n" + "=" * 80)
    print("消融实验完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()