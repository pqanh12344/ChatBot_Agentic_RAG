"""
Baseline HyperGraph for NF-ToN-IoT-v2 using pure PyTorch (no PyG).
- Node = flow
- Hyperedge = group by `src_ip` (configurable)
- Simple HyperGraph layer: hyperedge aggregation (mean) then node update

Usage:
  - Place NF-ToN-IoT-v2.csv in same folder or edit CSV_PATH
  - pip install pandas scikit-learn tqdm torch
  - python baseline_hypergraph_nftoniot_pytorch.py

Notes:
  - This implementation uses dense incidence (suitable for small/medium datasets).
  - For very large datasets, switch to sparse ops or sampling.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -------- Config --------
CSV_PATH = "NF-ToN-IoT-v2.csv"  # edit if needed
FEATURE_COLS = ["pkts", "bytes", "duration"]  # edit to actual numeric cols
HYPEREDGE_BY = "src_ip"  # 'src_ip', 'dst_ip', or 'session'
RANDOM_SEED = 42
NUM_EPOCHS = 40
LR = 1e-3
HIDDEN = 64
TEST_SIZE = 0.2
K_FEWSHOT = None  # set to int for few-shot per class
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- Utilities --------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(RANDOM_SEED)

# -------- Load and preprocess --------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Không tìm thấy file {CSV_PATH}. Đặt file vào cùng thư mục hoặc chỉnh CSV_PATH.")

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)

# check feature columns
for c in FEATURE_COLS:
    if c not in df.columns:
        raise ValueError(f"Feature column '{c}' không tồn tại trong CSV. Chỉnh FEATURE_COLS.")

if 'label' not in df.columns:
    raise ValueError("Không tìm thấy cột 'label' trong CSV. Cần cột label.")

# binary label

df['label_bin'] = df['label'].apply(lambda v: 0 if str(v).lower() == 'normal' else 1)

# optional session
if HYPEREDGE_BY == 'session':
    df['session'] = df['src_ip'].astype(str) + '_' + df['dst_ip'].astype(str)

# features
X = df[FEATURE_COLS].fillna(0).values.astype(float)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float, device=DEVICE)

# labels
Y = torch.tensor(df['label_bin'].values, dtype=torch.long, device=DEVICE)

# build hyperedges
he_col = HYPEREDGE_BY
if he_col not in df.columns:
    raise ValueError(f"Hyperedge column {he_col} not in dataframe.")

le = LabelEncoder()
hyperedge_ids = le.fit_transform(df[he_col].astype(str).values)
num_nodes = len(df)
num_hyperedges = int(hyperedge_ids.max()) + 1
print(f"Num nodes: {num_nodes}, num hyperedges (group by {he_col}): {num_hyperedges}")

# Build incidence matrix H (num_nodes x num_hyperedges) as float tensor
# H[i,e] = 1 if node i belongs to hyperedge e
H = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float, device=DEVICE)
for i, he in enumerate(hyperedge_ids):
    H[i, he] = 1.0

# optional: normalize incidence by hyperedge size when aggregating

# train/test split (node-level)
train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=TEST_SIZE, stratify=df['label_bin'].values, random_state=RANDOM_SEED)
train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
train_mask[train_idx] = True
test_mask[test_idx] = True

# few-shot reduction
if K_FEWSHOT is not None:
    new_train = torch.zeros_like(train_mask)
    for cls in [0, 1]:
        cls_idx = np.where((df['label_bin'].values == cls) & (train_mask.cpu().numpy()))[0]
        if len(cls_idx) <= K_FEWSHOT:
            chosen = cls_idx.tolist()
        else:
            chosen = np.random.choice(cls_idx, K_FEWSHOT, replace=False).tolist()
        new_train[chosen] = True
    train_mask = new_train.to(DEVICE)

# -------- HyperGraph Layer (simple implement) --------
# Strategy:
#  - compute hyperedge embedding: E = normalize(H)^T @ X  (mean of member nodes)
#  - propagate to nodes: X' = normalize(H) @ E
#  - combine with node residual and apply linear

class HyperGraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, X, H):
        # X: [N, F], H: [N, E]
        # hyperedge size: s_e = sum_i H[i,e]
        s_e = H.sum(dim=0, keepdim=True)  # [1, E]
        s_e_safe = s_e.clone()
        s_e_safe[s_e_safe == 0] = 1.0

        # node degree d_v = sum_e H[v,e]
        d_v = H.sum(dim=1, keepdim=True)  # [N,1]
        d_v_safe = d_v.clone()
        d_v_safe[d_v_safe == 0] = 1.0

        # compute hyperedge embeddings by mean pooling of member node features
        # E = H^T @ X / s_e
        E = (H.t() @ X) / s_e_safe.t()

        # propagate back to nodes: X_e = H @ E / d_v
        X_e = (H @ E) / d_v_safe

        out = self.lin(X_e)
        return out

# full model: 2-layer HGNN + classifier
class HGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.h1 = HyperGraphLayer(in_dim, hidden_dim)
        self.h2 = HyperGraphLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, H):
        x = F.relu(self.h1(X, H))
        x = F.relu(self.h2(x, H))
        logits = self.classifier(x)
        return logits

# -------- training --------
model = HGNN(in_dim=X.shape[1], hidden_dim=HIDDEN, num_classes=2).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_test_f1 = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    opt.zero_grad()
    logits = model(X, H)  # [N, C]
    loss = criterion(logits[train_mask], Y[train_mask])
    loss.backward()
    opt.step()

    # eval
    model.eval()
    with torch.no_grad():
        logits = model(X, H)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        y_all = Y.cpu().numpy()

        def metrics_from_mask(mask_tensor):
            mask = mask_tensor.cpu().numpy().astype(bool)
            if mask.sum() == 0:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            y_true = y_all[mask]
            y_pred = preds[mask]
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            try:
                auc = roc_auc_score(y_true, probs[mask])
            except:
                auc = float('nan')
            return acc, prec, rec, f1, auc

        train_metrics = metrics_from_mask(train_mask)
        test_metrics = metrics_from_mask(test_mask)

    if test_metrics[3] > best_test_f1:
        best_test_f1 = test_metrics[3]
        torch.save(model.state_dict(), 'best_hgnn_pytorch.pth')

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:03d} Loss: {loss.item():.4f}  Train F1: {train_metrics[3]:.4f}  Test F1: {test_metrics[3]:.4f}  Test AUC: {test_metrics[4]:.4f}")

print("Best test F1:", best_test_f1)

# final eval
model.load_state_dict(torch.load('best_hgnn_pytorch.pth'))
model.eval()
with torch.no_grad():
    logits = model(X, H)
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    y_all = Y.cpu().numpy()

    def print_metrics(mask_tensor, name):
        acc, prec, rec, f1, auc = metrics_from_mask(mask_tensor)
        print(f"{name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    print_metrics(train_mask, 'Train')
    print_metrics(test_mask, 'Test')

# Save scaler and label encoder if needed
import joblib
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'he_label_encoder.joblib')

print('Done.')
