import pandas as pd
import numpy as np
import os
from glob import glob

# ==========================
# 固定配置（统一Schema）
# ==========================
OUTPUT_COLS = [
    "timestamp",
    "user_id",
    "ip",
    "device",
    "is_attack",
    "success",
] + [f"f{i}" for i in range(64)]

# ==========================
# 时间转换（无警告、稳定）
# ==========================
def safe_convert_timestamp(series):
    series = series.astype(str).str.strip()
    series = series.replace("Timestamp", np.nan)
    series = pd.to_datetime(series, format="mixed", errors="coerce")
    return series.view('int64') // 10**9

# ==========================
# 标准化（完全匹配你给的列名 + 不过滤任何数据）
# ==========================
def standardize_single_file(df):
    # 1. 过滤脏行
    df = df[df["Timestamp"] != "Timestamp"].copy()
    df = df.dropna(subset=["Timestamp", "Label"])

    # 2. 时间戳
    df["timestamp"] = safe_convert_timestamp(df["Timestamp"])
    df = df.dropna(subset=["timestamp"])

    # 3. 构造统一ID
    df["ip"] = df["Dst Port"].astype(str)
    df["user_id"] = df["Dst Port"].astype(str) + "_" + df["Protocol"].astype(str)
    df["device"] = df["ip"]

    # 4. 标签：只要不是 BENIGN 都算攻击（兼容所有标签！）
    df["is_attack"] = df["Label"].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)
    df["success"] = 1

    # 5. 严格按你给的列提取64维特征
    feature_cols = [
        "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
        "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max",
        "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
        "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
        "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean",
        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
        "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s",
        "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max",
        "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
        "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
        "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
        "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio",
        "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
        "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
        "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
        "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts"
    ][:64]

    for i, c in enumerate(feature_cols):
        df[f"f{i}"] = pd.to_numeric(df[c], errors="coerce")

    # 6. 只保留标准Schema
    df = df[OUTPUT_COLS].copy()
    return df

# ==========================
# 分块读取（低内存）
# ==========================
def load_and_process_single_file(file_path, chunk_size=200000):
    print(f"正在处理：{os.path.basename(file_path)}")
    chunks = []
    for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunk_size):
        chunk_std = standardize_single_file(chunk)
        chunks.append(chunk_std)
        print(f"  → 有效样本：{len(chunk_std)}")
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

# ==========================
# 主函数（防空数据崩溃）
# ==========================
def preprocess_login_data(data_root="data/CSE-CIC-IDS2018"):
    print("=== 开始预处理 CSE-CIC-IDS2018 ===")
    os.makedirs("processed", exist_ok=True)
    csv_files = glob(os.path.join(data_root, "*.csv"))
    print(f"发现 {len(csv_files)} 个文件")

    all_dfs = []
    for f in csv_files:
        df = load_and_process_single_file(f)
        if not df.empty:
            all_dfs.append(df)

    # 【防报错】如果没有数据，直接退出
    if len(all_dfs) == 0:
        print("❌ 未读取到任何有效数据")
        return

    # 合并并保存
    final_df = pd.concat(all_dfs).sort_values("timestamp").reset_index(drop=True)
    final_df.to_csv("processed/login_events.csv", index=False)

    print("\n✅ 预处理全部完成！")
    print(f"总样本：{len(final_df)}")
    print(f"异常登录：{final_df.is_attack.sum()}")
    print(f"正常登录：{(final_df.is_attack==0).sum()}")
    print("输出文件：processed/login_events.csv")

if __name__ == "__main__":
    preprocess_login_data()