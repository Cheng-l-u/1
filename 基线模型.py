# 基线模型.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("网络安全异常登录检测 - 基线模型对比实验")
print("=" * 60)


# ==================== 数据加载模块 ====================

def load_and_clean_data(data_dir="./raw_data", sample_ratio=0.1):
    """加载并清洗真实数据"""

    print(f"\n[数据加载] 从 {data_dir} 加载数据...")

    all_data = []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"警告: 在 {data_dir} 中未找到CSV文件，使用模拟数据")
        return generate_mock_data()

    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        print(f"  读取: {file}")

        try:
            # 分块读取
            chunk_size = 50000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                # 采样
                if sample_ratio < 1.0:
                    chunk = chunk.sample(frac=sample_ratio, random_state=42)

                all_data.append(chunk)

        except Exception as e:
            print(f"  读取失败: {e}")
            continue

    if not all_data:
        print("无法读取真实数据，使用模拟数据")
        return generate_mock_data()

    # 合并数据
    df = pd.concat(all_data, ignore_index=True)
    print(f"总数据量: {len(df)} 条")

    # 处理标签
    df = process_labels(df)

    # 提取特征
    X, y = extract_features(df)

    print(f"特征维度: {X.shape[1]}")
    print(f"异常比例: {y.mean():.2%}")

    return X, y


def generate_mock_data(n_samples=5000, n_features=20):
    """生成模拟数据（备用）"""
    print("生成模拟数据...")
    np.random.seed(42)

    # 正常样本
    X_normal = np.random.randn(n_samples // 2, n_features)
    y_normal = np.zeros(n_samples // 2)

    # 异常样本
    X_anomaly = np.random.randn(n_samples // 2, n_features)
    X_anomaly[:, 0] += 2.0
    X_anomaly[:, 5] += 1.5
    y_anomaly = np.ones(n_samples // 2)

    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([y_normal, y_anomaly])

    # 打乱
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return X, y


def process_labels(df):
    """处理标签列"""
    # 查找标签列
    label_col = None
    for col in df.columns:
        if 'Label' in col or 'label' in col:
            label_col = col
            break

    if label_col is None:
        print("未找到标签列，使用默认标签")
        df['is_attack'] = 0
        return df

    # 登录攻击关键词
    login_attacks = ['Brute Force', 'FTP-BruteForce', 'SSH-Bruteforce',
                     'FTP-Brute-Force', 'SSH-Brute-Force', 'Brute']

    df['is_attack'] = df[label_col].apply(
        lambda x: 1 if any(attack in str(x) for attack in login_attacks) else 0
    )

    print(f"标签分布: 正常={len(df) - df['is_attack'].sum()}, 攻击={df['is_attack'].sum()}")

    return df


def extract_features(df):
    """提取数值特征并清洗"""
    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 排除不需要的列
    exclude_cols = ['is_attack', 'Label', 'label', 'Unnamed: 0']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    if not feature_cols:
        print("未找到数值特征，使用基础特征")
        feature_cols = numeric_cols[:20] if len(numeric_cols) >= 20 else numeric_cols

    # 提取特征
    X = df[feature_cols].values

    # 清洗数据：处理NaN和无穷值
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 检查是否有异常值
    X = np.clip(X, -1e10, 1e10)  # 限制极端值

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 再次清洗（标准化后可能产生NaN）
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df['is_attack'].values

    return X, y


# ==================== 基线模型定义 ====================

class SnortRuleEngine:
    """Snort规则引擎模拟"""

    def __init__(self):
        self.threshold = 1.0

    def fit(self, X, y):
        """学习阈值"""
        # 基于特征均值设置阈值
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-6

    def predict(self, X):
        """基于规则的预测"""
        scores = np.zeros(len(X))

        for i, sample in enumerate(X):
            # 检查特征是否偏离正常范围
            deviation = np.abs(sample - self.feature_means) / (self.feature_stds + 1e-6)
            scores[i] = np.sum(deviation > 2.0)  # 超过2个标准差的特征数

        return (scores > 3).astype(int)  # 超过3个特征异常判定为攻击

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[:, 0] = 1 - preds * 0.9
        proba[:, 1] = preds * 0.9
        return proba


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict_proba(X_scaled)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)


# ==================== 训练和评估函数 ====================

def train_temporal_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 确保数据没有NaN
            batch_x = torch.nan_to_num(batch_x, nan=0.0)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    return model


def evaluate_model(model, X_test, y_test, model_type='sklearn'):
    """评估模型性能"""

    # 确保数据没有NaN
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    if model_type == 'sklearn':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
            output = model(X_tensor)
            y_proba = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            y_pred = (y_proba > 0.5).astype(int)

    # 确保y_test是numpy数组
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'fpr': fpr
    }


# ==================== 主程序 ====================

if __name__ == "__main__":

    # 1. 加载数据
    print("\n[1] 加载数据")
    print("-" * 40)

    X, y = load_and_clean_data(data_dir="./raw_data", sample_ratio=0.05)  # 先用5%数据测试

    print(f"\n数据形状: {X.shape}")
    print(f"特征范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"是否存在NaN: {np.isnan(X).any()}")

    # 2. 划分数据集
    print("\n[2] 划分数据集")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"训练集异常比例: {y_train.mean():.2%}")
    print(f"测试集异常比例: {y_test.mean():.2%}")

    results = {}

    # 3. 传统安全检测模型
    print("\n" + "=" * 50)
    print("3. 传统安全检测模型")
    print("=" * 50)

    # Snort规则引擎
    print("\n[Snort规则引擎]")
    try:
        snort = SnortRuleEngine()
        snort.fit(X_train, y_train)
        results['Snort'] = evaluate_model(snort, X_test, y_test, 'sklearn')
        print(f"  准确率: {results['Snort']['accuracy']:.4f}")
        print(f"  召回率: {results['Snort']['recall']:.4f}")
        print(f"  F1分数: {results['Snort']['f1']:.4f}")
        print(f"  AUC: {results['Snort']['auc']:.4f}")
        print(f"  误报率: {results['Snort']['fpr']:.4f}")
    except Exception as e:
        print(f"  Snort模型失败: {e}")
        results['Snort'] = {'accuracy': 0, 'recall': 0, 'f1': 0, 'auc': 0, 'fpr': 0}

    # 逻辑回归
    print("\n[逻辑回归]")
    try:
        lr_model = LogisticRegressionModel()
        lr_model.fit(X_train, y_train)
        results['LogisticRegression'] = evaluate_model(lr_model, X_test, y_test, 'sklearn')
        print(f"  准确率: {results['LogisticRegression']['accuracy']:.4f}")
        print(f"  召回率: {results['LogisticRegression']['recall']:.4f}")
        print(f"  F1分数: {results['LogisticRegression']['f1']:.4f}")
        print(f"  AUC: {results['LogisticRegression']['auc']:.4f}")
        print(f"  误报率: {results['LogisticRegression']['fpr']:.4f}")
    except Exception as e:
        print(f"  逻辑回归失败: {e}")
        results['LogisticRegression'] = {'accuracy': 0, 'recall': 0, 'f1': 0, 'auc': 0, 'fpr': 0}

    # 4. 时序模型（需要调整维度）
    print("\n" + "=" * 50)
    print("4. 时序模型")
    print("=" * 50)

    try:
        # 调整时序维度
        time_steps = 10
        n_features = X.shape[1]
        n_features_adjusted = (n_features // time_steps) * time_steps

        if n_features_adjusted >= time_steps:
            X_adjusted = X[:, :n_features_adjusted]
            X_time = X_adjusted.reshape(-1, time_steps, n_features_adjusted // time_steps)

            X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
                X_time, y, test_size=0.3, random_state=42, stratify=y
            )

            train_dataset = TensorDataset(torch.FloatTensor(X_train_time), torch.LongTensor(y_train_time))
            test_dataset = TensorDataset(torch.FloatTensor(X_test_time), torch.LongTensor(y_test_time))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)

            # LSTM
            print("\n[LSTM模型]")
            lstm_model = LSTMModel(input_dim=X_train_time.shape[2])
            lstm_model = train_temporal_model(lstm_model, train_loader, test_loader, epochs=20)
            results['LSTM'] = evaluate_model(lstm_model, X_test_time, y_test_time, 'deep')
            print(f"  准确率: {results['LSTM']['accuracy']:.4f}")
            print(f"  召回率: {results['LSTM']['recall']:.4f}")
            print(f"  F1分数: {results['LSTM']['f1']:.4f}")
            print(f"  AUC: {results['LSTM']['auc']:.4f}")
            print(f"  误报率: {results['LSTM']['fpr']:.4f}")

            # GRU
            print("\n[GRU模型]")
            gru_model = GRUModel(input_dim=X_train_time.shape[2])
            gru_model = train_temporal_model(gru_model, train_loader, test_loader, epochs=20)
            results['GRU'] = evaluate_model(gru_model, X_test_time, y_test_time, 'deep')
            print(f"  准确率: {results['GRU']['accuracy']:.4f}")
            print(f"  召回率: {results['GRU']['recall']:.4f}")
            print(f"  F1分数: {results['GRU']['f1']:.4f}")
            print(f"  AUC: {results['GRU']['auc']:.4f}")
            print(f"  误报率: {results['GRU']['fpr']:.4f}")
        else:
            print("特征维度不足，跳随时序模型")

    except Exception as e:
        print(f"时序模型训练失败: {e}")

    # 5. 结果汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    results_df = pd.DataFrame(results).T
    results_df = results_df[['accuracy', 'recall', 'f1', 'auc', 'fpr']]
    results_df.columns = ['准确率', '召回率', 'F1分数', 'AUC-ROC', '误报率']

    print("\n", results_df.round(4))

    # 保存结果
    results_df.to_csv('baseline_results.csv')
    print("\n结果已保存到: baseline_results.csv")

    # 找出最佳模型
    if len(results_df) > 0:
        best_auc = results_df['AUC-ROC'].idxmax()
        best_f1 = results_df['F1分数'].idxmax()
        print(f"\n最佳模型（按AUC）: {best_auc}")
        print(f"最佳模型（按F1）: {best_f1}")

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)