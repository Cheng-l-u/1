# data/build_graph.py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.neighbors import kneighbors_graph


class SpatioTemporalGraphBuilder:
    """
    时空图构建模块

    节点类型：
    - 账号节点 (User Nodes)
    - IP节点 (IP Nodes)
    - 设备节点 (Device Nodes)

    边类型：
    - 登录边: 账号-IP (带时间戳)
    - 设备关联边: 账号-设备
    - 共现边: IP-IP (共享账号)
    """

    def __init__(self, time_window=3600, spatial_k=5):
        self.time_window = time_window  # 时间窗口（秒）
        self.spatial_k = spatial_k  # 空间近邻K值

    def build_node_features(self, df, node_type='user'):
        """构建节点特征"""
        if node_type == 'user':
            # 用户特征：历史登录模式
            node_features = df.groupby('user_id').agg({
                'login_timestamp': ['count', 'mean', 'std'],
                'hour': lambda x: x.mode()[0] if len(x) > 0 else 0,
                'is_fraud': 'mean'
            }).fillna(0)
            # 展平多层索引
            node_features.columns = ['login_count', 'avg_login_time', 'std_login_time',
                                     'peak_hour', 'fraud_ratio']

        elif node_type == 'ip':
            # IP特征：地理位置、异常历史
            node_features = df.groupby('ip_address').agg({
                'login_timestamp': 'count',
                'is_fraud': 'mean',
                'user_id': 'nunique'  # 关联账号数
            }).fillna(0)
            node_features.columns = ['login_count', 'fraud_ratio', 'unique_users']

        else:  # device
            node_features = df.groupby('device_id').agg({
                'login_timestamp': 'count',
                'is_fraud': 'mean',
                'user_id': 'nunique'
            }).fillna(0)
            node_features.columns = ['login_count', 'fraud_ratio', 'unique_users']

        return node_features.values

    def build_time_series(self, df, time_steps=48, time_granularity='hour'):
        """
        构建时序特征

        Args:
            df: 登录事件DataFrame
            time_steps: 时间步数
            time_granularity: 时间粒度 ('hour', 'minute')
        """
        df['time_bin'] = df['login_time'].dt.floor(time_granularity)

        # 按用户和时间聚合
        time_series = df.groupby(['user_id', 'time_bin']).size().unstack(fill_value=0)

        # 固定时间步长
        if len(time_series.columns) > time_steps:
            time_series = time_series.iloc[:, -time_steps:]
        else:
            # 填充
            pad_cols = time_steps - len(time_series.columns)
            for i in range(pad_cols):
                time_series.insert(0, f'pad_{i}', 0)

        return time_series.values

    def build_spatial_graph(self, node_features, node_ids, edge_df):
        """
        构建空间图（异构图）

        Returns:
            edge_index: [2, E] 边索引
            edge_attr: [E, feat_dim] 边特征
            node_mapping: 节点ID到索引的映射
        """
        # 节点映射
        all_nodes = list(set(node_ids['user']) | set(node_ids['ip']) | set(node_ids.get('device', [])))
        node_mapping = {nid: idx for idx, nid in enumerate(all_nodes)}

        # 构建边
        edge_index = []
        edge_attr = []

        # 1. 登录边（账号-IP）
        for _, row in edge_df[edge_df['edge_type'] == 'user_ip'].iterrows():
            src = node_mapping.get(row['user_id'])
            dst = node_mapping.get(row['ip_address'])
            if src is not None and dst is not None:
                edge_index.append([src, dst])
                edge_attr.append([1.0, row.get('login_count', 1)])  # 边类型权重

        # 2. 设备边
        if 'user_device' in edge_df['edge_type'].values:
            for _, row in edge_df[edge_df['edge_type'] == 'user_device'].iterrows():
                src = node_mapping.get(row['user_id'])
                dst = node_mapping.get(row['device_id'])
                if src is not None and dst is not None:
                    edge_index.append([src, dst])
                    edge_attr.append([2.0, 1.0])

        # 3. IP共现边（基于近邻图）
        ip_nodes = [node_mapping[nid] for nid in node_ids['ip'] if nid in node_mapping]
        if len(ip_nodes) > self.spatial_k:
            ip_features = node_features[ip_nodes]
            adj = kneighbors_graph(ip_features, self.spatial_k, mode='connectivity')
            for i, j in zip(*adj.nonzero()):
                edge_index.append([ip_nodes[i], ip_nodes[j]])
                edge_attr.append([3.0, 1.0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return edge_index, edge_attr, node_mapping

    def build_temporal_edges(self, login_events, time_window=3600):
        """
        构建时序边（时间窗口内的登录关联）
        """
        temporal_edges = []
        login_events = login_events.sort_values('login_timestamp')

        for i, event1 in login_events.iterrows():
            # 查找时间窗口内的其他事件
            time_mask = (login_events['login_timestamp'] - event1['login_timestamp']).abs() <= time_window
            related = login_events[time_mask]

            for _, event2 in related.iterrows():
                if event1['user_id'] != event2['user_id']:
                    temporal_edges.append([event1['user_id'], event2['user_id']])

        return torch.tensor(temporal_edges, dtype=torch.long).t().contiguous() if temporal_edges else None


class ASTGNDataset(Dataset):
    """ASTGNN数据集类"""

    def __init__(self, graph_data, time_series, labels, transform=None):
        self.graph_data = graph_data
        self.time_series = time_series
        self.labels = labels
        self.transform = transform

    def len(self):
        return len(self.labels)

    def get(self, idx):
        # 返回时空图数据
        return {
            'x': self.graph_data.x[idx],  # 节点特征
            'edge_index': self.graph_data.edge_index,
            'edge_attr': self.graph_data.edge_attr,
            'time_series': self.time_series[idx],  # 时序特征
            'y': self.labels[idx]
        }