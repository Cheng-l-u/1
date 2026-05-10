import torch
import random
import numpy as np
import pandas as pd
from graph_builder import build_spatiotemporal_graph
from models import (
    ASTGNN,
    LSTMModel,
    GRUModel,
    GCNModel,
    GATModel,
    STGCNModel,
    ASTGNN_NoSpatial,
    ASTGNN_NoTemporal,
    ASTGNN_NoSTAtt
)
from evaluate import evaluate

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 加载数据
df = pd.read_csv("processed/login_events.csv").fillna(0)
data = build_spatiotemporal_graph(df)
device = torch.device("cpu")
data = data.to(device)

# 训练函数
def run_exp(model, name):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(15):
        model.train()
        opt.zero_grad()
        logits = model(data)
        loss = ce(logits, data.y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(1)
            score = torch.softmax(model(data), dim=1)[:, 1]

        metrics = evaluate(data.y.cpu(), pred.cpu(), score.cpu())
        print(f"[{name}] Epoch {epoch} | {metrics}")

# ==============================
# 运行所有毕设需要的模型
# ==============================
print("=" * 50)
print("🚀 开始训练 ASTGNN 主模型")
print("=" * 50)
run_exp(ASTGNN(), "ASTGNN")

print("\n" + "=" * 50)
print("📊 训练基线模型")
print("=" * 50)
run_exp(LSTMModel(), "LSTM")
run_exp(GRUModel(), "GRU")
run_exp(GCNModel(), "GCN")
run_exp(GATModel(), "GAT")
run_exp(STGCNModel(), "STGCN")

print("\n" + "=" * 50)
print("🧪 消融实验")
print("=" * 50)
run_exp(ASTGNN_NoSpatial(), "ASTGNN-无空间注意力")
run_exp(ASTGNN_NoTemporal(), "ASTGNN-无时间注意力")
run_exp(ASTGNN_NoSTAtt(), "ASTGNN-无时空注意力")

print("\n✅ 全部训练完成！")