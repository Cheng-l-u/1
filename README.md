# ASTGNN 账号异常登录欺诈检测

# 环境
torch torch_geometric pandas scikit-learn

# 数据准备
CSE-CIC-IDS2018 → data/CSE-CIC-IDS2018.csv
微软异常数据集 → data/microsoft_anomaly.csv

##训练
python preprocessing.py
python train.py

##推理
python predict.py
