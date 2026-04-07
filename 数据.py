# 数据.py - 改进版
import os
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

DATA_DIR = "./raw_data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def download_from_opendatalab():
    """从OpenDataLab下载（国内源，速度快）"""
    print("从 OpenDataLab 下载 CICIDS2018...")
    url = "https://opendatalab.org.cn/OpenDataLab/CICIDS2018"
    print(f"请手动访问以下链接下载：\n{url}\n")
    print("下载后请将CSV文件放置在 ./raw_data 目录下")
    input("下载完成后按Enter键继续...")
    return list(Path(DATA_DIR).glob("*CICFlowMeter*.csv"))


def download_from_kaggle_netflow():
    """下载Kaggle上的NetFlow版本（已清洗，推荐）"""
    print("\n尝试从Kaggle下载NetFlow版本...")

    # 检查kaggle是否安装
    try:
        import kaggle
        print("使用Kaggle API下载...")
        # 注意：需要先配置Kaggle API密钥
        kaggle.api.dataset_download_files(
            'dhoogla/nfcsecicids2018v2',
            path=DATA_DIR,
            unzip=True
        )
        print("✓ Kaggle数据集下载完成")
        return True
    except ImportError:
        print("请先安装kaggle: pip install kaggle")
        print("并从 https://www.kaggle.com/account 创建API密钥")
        return False
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def generate_mock_data_for_testing():
    """生成测试用的模拟数据（确保代码能立即运行）"""
    print("\n生成模拟登录数据用于测试...")

    # 模拟一周的登录事件
    dates = pd.date_range('2024-01-01', periods=10000, freq='5min')

    # 正常用户行为模式
    np.random.seed(42)
    n_users = 50
    user_ids = [f'user_{i}' for i in range(n_users)]

    data = []
    for i, ts in enumerate(dates):
        user = user_ids[i % n_users]
        hour = ts.hour

        # 正常登录时间分布（工作日白天为主）
        is_night = hour < 6 or hour > 22
        is_weekend = ts.dayofweek >= 5

        # 异常注入：特定用户凌晨大量登录
        is_anomaly = False
        if user == 'user_10' and is_night:
            is_anomaly = np.random.random() < 0.3  # 30%异常率
        elif user == 'user_25' and ts.day >= 15:
            is_anomaly = np.random.random() < 0.2

        data.append({
            'timestamp': ts,
            'user_id': user,
            'ip_address': f'192.168.1.{hash(user) % 50}',
            'device_id': f'device_{hash(str(ts)) % 20}',
            'login_success': 0 if is_anomaly else 1,
            'is_fraud': 1 if is_anomaly else 0
        })

    df = pd.DataFrame(data)
    df.to_csv(Path(DATA_DIR) / 'login_data.csv', index=False)
    print(f"✓ 生成了 {len(df)} 条登录记录，异常比例: {df['is_fraud'].mean():.2%}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("网络安全时空图神经网络 - 数据准备")
    print("=" * 60)

    # 1. 尝试从国内源下载
    cic_files = download_from_opendatalab()

    # 2. 如果没有下载，生成测试数据
    if not cic_files:
        print("\n未检测到CICIDS2018数据，生成测试数据...")
        mock_data = generate_mock_data_for_testing()
        print(f"测试数据已保存到: {Path(DATA_DIR).absolute()}")

    print("\n" + "=" * 60)
    print("数据准备阶段完成！")
    print("=" * 60)
