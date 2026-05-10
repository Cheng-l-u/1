import pandas as pd
import torch
import torch_geometric as pyg

def build_spatiotemporal_graph(df, time_window=3600):
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].values
    n = len(df)

    # === 空间边：user - ip - device ===
    edge_index_spatial = [[], []]
    node2id = {u:i for i,u in enumerate(df["user_id"].unique())}
    ip2id = {ip:i+n for i,ip in enumerate(df["ip"].unique())}
    dev2id = {d:i+n+len(ip2id) for i,d in enumerate(df["device"].unique())}

    for i, row in df.iterrows():
        u = node2id[row["user_id"]]
        ip = ip2id[row["ip"]]
        dev = dev2id[row["device"]]
        edge_index_spatial[0] += [u, ip, dev, u]
        edge_index_spatial[1] += [ip, u, dev, dev]

    # === 时序边：双指针 O(n) 构建 ===
    edge_index_temporal = [[], []]
    l = 0
    for r in range(n):
        while timestamps[r] - timestamps[l] > time_window:
            l += 1
        for k in range(l, r):
            edge_index_temporal[0].append(k)
            edge_index_temporal[1].append(r)

    edge_index = torch.tensor(edge_index_spatial + edge_index_temporal, dtype=torch.long)
    x = torch.tensor(df[[f"f{i}" for i in range(64)]].values, dtype=torch.float)
    y = torch.tensor(df["is_attack"].values, dtype=torch.long)
    g = pyg.data.Data(x=x, edge_index=edge_index, y=y)
    return g