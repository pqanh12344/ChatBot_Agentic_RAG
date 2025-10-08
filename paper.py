import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================
# 1️⃣ ĐỌC DỮ LIỆU
# =====================================================
DATA_PATH = "NF-ToN-IoT-v2-train.csv"   # hoặc NF-ToN-IoT-v2-train.csv
df = pd.read_csv(DATA_PATH, nrows=50000)  # đọc demo 50k dòng

# Xác định cột nhãn
label_col = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower()]
if len(label_col) == 0:
    raise ValueError("❌ Không tìm thấy cột nhãn. Hãy kiểm tra tên cột trong file.")
df = df.rename(columns={label_col[0]: "label"})

# Encode label
le = LabelEncoder()
df["y"] = le.fit_transform(df["label"].astype(str))
print(f"🟢 Đã mã hóa {len(le.classes_)} lớp:", le.classes_)

# =====================================================
# 2️⃣ LÀM SẠCH DỮ LIỆU SỐ
# =====================================================
feature_cols = [c for c in df.columns if c not in ["label", "y"]]
X = df[feature_cols].copy()

# Ép toàn bộ về dạng số
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')

# Thay thế NaN, Inf, giá trị cực lớn
X = X.replace([np.inf, -np.inf, np.nan], 0)
X = np.clip(X, -1e9, 1e9)
X = X.astype(np.float32)

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tensor hóa
X_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(df["y"].values, dtype=torch.long)

num_nodes = X_tensor.shape[0]
num_features = X_tensor.shape[1]
num_classes = len(le.classes_)
print(f"✅ Node: {num_nodes} | Feature: {num_features} | Lớp: {num_classes}")

# =====================================================
# 3️⃣ TẠO EDGE GIẢ LẬP DỰA TRÊN SIMILARITY
# =====================================================
from sklearn.neighbors import NearestNeighbors

k = 5  # số láng giềng
nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

edge_index = []
for i in range(num_nodes):
    for j in indices[i][1:]:  # bỏ chính nó
        edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# =====================================================
# 4️⃣ TẠO GRAPH DATA
# =====================================================
data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)

# Train/Test Split
train_mask, test_mask = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_mask] = True
data.test_mask[test_mask] = True

print(f"🧩 Train nodes: {data.train_mask.sum()} | Test nodes: {data.test_mask.sum()}")

# =====================================================
# 5️⃣ MÔ HÌNH GRAPH TRANSFORMER
# =====================================================
class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.lin = nn.Linear(hidden_channels * heads, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin(x)
        return x

# =====================================================
# 6️⃣ TRAIN LOOP
# =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphTransformer(num_features, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

data = data.to(device)

for epoch in range(1, 11):  # train 10 epoch demo
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Evaluate
    model.eval()
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()

    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

# =====================================================
# 7️⃣ ĐÁNH GIÁ CUỐI CÙNG
# =====================================================
model.eval()
pred = out.argmax(dim=1).cpu().numpy()
true = data.y.cpu().numpy()

print("\n📊 Classification Report:")
print(classification_report(true[data.test_mask.cpu().numpy()],
                            pred[data.test_mask.cpu().numpy()],
                            target_names=le.classes_))





#####################################
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors

# =====================================================
# 1️⃣ ĐỌC VÀ XỬ LÝ DỮ LIỆU
# =====================================================
DATA_PATH = "NF-ToN-IoT-v2.csv"
df = pd.read_csv(DATA_PATH, nrows=50000)

label_col = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower()]
if len(label_col) == 0:
    raise ValueError("❌ Không tìm thấy cột nhãn.")
df = df.rename(columns={label_col[0]: "label"})

le = LabelEncoder()
df["y"] = le.fit_transform(df["label"].astype(str))

feature_cols = [c for c in df.columns if c not in ["label", "y"]]
X = df[feature_cols].copy()

# Làm sạch dữ liệu
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X = X.replace([np.inf, -np.inf, np.nan], 0)
X = np.clip(X, -1e9, 1e9).astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(df["y"].values, dtype=torch.long)

num_nodes = X_tensor.shape[0]
num_features = X_tensor.shape[1]
num_classes = len(le.classes_)
print(f"✅ Node: {num_nodes} | Feature: {num_features} | Lớp: {num_classes}")

# =====================================================
# 2️⃣ TẠO HYPEREDGE (mỗi hyperedge nối k node gần nhau)
# =====================================================
k = 5
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_scaled)
_, indices = nbrs.kneighbors(X_scaled)

# edge_index_hyper: (2, num_hyperedges*k)
# Mỗi node i sẽ có 1 hyperedge id = i
hyperedges = []
for i in range(num_nodes):
    for j in indices[i][1:]:
        hyperedges.append([i, j])
hyperedges = torch.tensor(hyperedges, dtype=torch.long).t().contiguous()

# =====================================================
# 3️⃣ ĐỊNH NGHĨA MÔ HÌNH HYPERGRAPH
# =====================================================
class HyperGraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.3):
        super().__init__()
        self.hconv1 = HypergraphConv(in_channels, hidden_channels)
        self.hconv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = torch.relu(self.hconv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.hconv2(x, edge_index))
        x = self.dropout(x)
        x = self.lin(x)
        return x

# =====================================================
# 4️⃣ TRAIN / TEST SPLIT
# =====================================================
train_mask, test_mask = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)
data = Data(x=X_tensor, edge_index=hyperedges, y=y_tensor)
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_mask] = True
data.test_mask[test_mask] = True

# =====================================================
# 5️⃣ TRAINING LOOP
# =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HyperGraphNet(num_features, 64, num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 11):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    pred = out.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

# =====================================================
# 6️⃣ BÁO CÁO KẾT QUẢ
# =====================================================
model.eval()
pred = out.argmax(dim=1).cpu().numpy()
true = data.y.cpu().numpy()
print("\n📊 Classification Report:")
print(classification_report(true[data.test_mask.cpu().numpy()],
                            pred[data.test_mask.cpu().numpy()],
                            target_names=le.classes_))
