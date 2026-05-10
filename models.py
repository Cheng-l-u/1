import torch
import torch.nn as nn
import torch_geometric.nn as gnn

# ==========================
# 注意力模块
# ==========================
class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x, edge_index):
        weight = torch.sigmoid(self.att(x))
        return gnn.GCNConv(x.size(1), dim)(x, edge_index, weight)


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        score = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1))), dim=-1)
        return torch.matmul(score, v)


class SpatiotemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s_att = SpatialAttention(dim)
        self.t_att = TemporalAttention(dim)

    def forward(self, x, edge_index):
        return self.s_att(x, edge_index) + self.t_att(x)


# ==========================
# ASTGNN 主模型
# ==========================
class ASTGNN(nn.Module):
    def __init__(self, in_dim=64, hid=64):
        super().__init__()
        self.lin = nn.Linear(in_dim, hid)
        self.st_att = SpatiotemporalAttention(hid)
        self.out = nn.Linear(hid, 2)

    def forward(self, data):
        x = torch.relu(self.lin(data.x))
        x = self.st_att(x, data.edge_index)
        return self.out(x)


# ==========================
# 消融实验模型
# ==========================
class ASTGNN_NoSpatial(nn.Module):
    def __init__(self, in_dim=64, hid=64):
        super().__init__()
        self.lin = nn.Linear(in_dim, hid)
        self.t_att = TemporalAttention(hid)
        self.out = nn.Linear(hid, 2)

    def forward(self, d):
        x = torch.relu(self.lin(d.x))
        x = self.t_att(x)
        return self.out(x)


class ASTGNN_NoTemporal(nn.Module):
    def __init__(self, in_dim=64, hid=64):
        super().__init__()
        self.lin = nn.Linear(in_dim, hid)
        self.s_att = SpatialAttention(hid)
        self.out = nn.Linear(hid, 2)

    def forward(self, d):
        x = torch.relu(self.lin(d.x))
        x = self.s_att(x, d.edge_index)
        return self.out(x)


class ASTGNN_NoSTAtt(nn.Module):
    def __init__(self, in_dim=64, hid=64):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU())
        self.out = nn.Linear(hid, 2)

    def forward(self, d):
        return self.out(self.lin(d.x))


# ==========================
# 基线模型 LSTM / GRU / GCN / GAT / STGCN
# ==========================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 64, 2, batch_first=True)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        return self.out(self.lstm(x)[0][:, -1])


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(64, 64, 2, batch_first=True)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        return self.out(self.gru(x)[0][:, -1])


class GCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = gnn.GCNConv(64, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, d):
        return self.out(torch.relu(self.gcn(d.x, d.edge_index)))


class GATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = gnn.GATConv(64, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, d):
        return self.out(torch.relu(self.gat(d.x, d.edge_index)))


class STGCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = gnn.GCNConv(64, 64)
        self.gru = nn.GRU(64, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, d):
        x = torch.relu(self.gcn(d.x, d.edge_index))
        x, _ = self.gru(x.unsqueeze(0))
        return self.out(x.squeeze(0))