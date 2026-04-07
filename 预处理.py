# data/preprocess_cicids.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class CICIDS2018Preprocessor:
    """
    CICIDS2018数据集预处理 - 聚焦登录行为
    提取SSH、FTP、Telnet等认证相关流量
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.label_enc = LabelEncoder()
        self.scaler = StandardScaler()

        # 登录行为相关端口
        self.AUTH_PORTS = [21, 22, 23, 3389, 5900]  # FTP, SSH, Telnet, RDP, VNC

        # 身份认证相关的流量特征列
        self.AUTH_FEATURES = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Mean', 'Bwd IAT Std',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Header Length', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
            'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]

    def load_and_filter(self, chunk_size=100000):
        """分块加载并过滤登录行为数据"""
        chunks = []
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size, low_memory=False):
            # 过滤认证相关端口的流量
            mask = (
                    chunk['Dst Port'].isin(self.AUTH_PORTS) |
                    chunk['Src Port'].isin(self.AUTH_PORTS)
            )
            filtered = chunk[mask].copy()
            if len(filtered) > 0:
                chunks.append(filtered)

        df = pd.concat(chunks, ignore_index=True)
        print(f"Filtered authentication traffic: {len(df)} samples")
        return df

    def extract_session_features(self, df):
        """提取登录会话特征"""
        # 添加时间戳特征（基于Flow IAT）
        df['session_duration'] = df['Flow Duration'].clip(upper=3600 * 1000)  # 最长1小时
        df['packet_rate'] = (df['Total Fwd Packets'] + df['Total Backward Packets']) / (df['session_duration'] + 1)
        df['byte_rate'] = (df['Fwd Packet Length Mean'] + df['Bwd Packet Length Mean']) / (df['session_duration'] + 1)

        # 失败登录尝试特征（基于标志位）
        df['reset_rate'] = (df['RST Flag Count']) / (df['Total Fwd Packets'] + df['Total Backward Packets'] + 1)
        df['syn_ratio'] = df['SYN Flag Count'] / (df['Total Fwd Packets'] + df['Total Backward Packets'] + 1)

        # 异常行为特征
        df['small_packet_ratio'] = (df['Average Packet Size'] < 100).astype(int)
        df['high_iat_std'] = (df['Flow IAT Std'] > df['Flow IAT Std'].quantile(0.95)).astype(int)

        return df

    def create_labels(self, df):
        """创建异常登录标签"""
        # CICIDS2018中，Benign为正常流量
        df['is_attack'] = (df['Label'] != 'Benign').astype(int)

        # 细化攻击类型
        attack_map = {
            'Brute Force': 1,  # 暴力破解
            'FTP-BruteForce': 1,
            'SSH-Bruteforce': 1,
            'Infiltration': 2,  # 渗透攻击
            'Bot': 3,  # 僵尸网络
            'DoS': 4  # 拒绝服务
        }

        df['attack_type'] = 0
        for attack, code in attack_map.items():
            df.loc[df['Label'].str.contains(attack, na=False), 'attack_type'] = code

        return df

    def prepare_features(self, df):
        """准备模型输入特征"""
        # 选择可用特征
        available_features = [f for f in self.AUTH_FEATURES if f in df.columns]
        X = df[available_features].copy()

        # 处理缺失值
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # 添加会话特征
        session_features = ['session_duration', 'packet_rate', 'byte_rate',
                            'reset_rate', 'syn_ratio', 'small_packet_ratio', 'high_iat_std']
        for f in session_features:
            if f in df.columns:
                X[f] = df[f]

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, df['is_attack'].values, df.get('attack_type', np.zeros(len(df))).values

    def run(self):
        """执行完整预处理流程"""
        print("Loading CICIDS2018 data...")
        df = self.load_and_filter()

        print("Extracting session features...")
        df = self.extract_session_features(df)

        print("Creating labels...")
        df = self.create_labels(df)

        print("Preparing features...")
        X, y, attack_types = self.prepare_features(df)

        # 划分数据集（按时间顺序）
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Attack ratio: {y.mean():.4f}")

        return (X_train, y_train), (X_test, y_test), attack_types