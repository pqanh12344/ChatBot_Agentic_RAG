# # # hyper_fed_fecograph_baseline.py
# # import argparse
# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import random
# # from sklearn.preprocessing import StandardScaler, LabelEncoder
# # from sklearn.model_selection import train_test_split
# # from torch_geometric.data import Data
# # from torch_geometric.nn import SAGEConv
# # from tqdm import tqdm
# # from collections import defaultdict
# # import math

# # # -------------------------
# # # Utils / preprocessing
# # # -------------------------
# # def seed_everything(seed=42):
# #     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
# # seed_everything(42)

# # def load_nf_toniot(csv_path, nrows=None):
# #     """
# #     Simplified loader for NF-ToN-IoT-v2 CSV.
# #     We expect a label column named 'Label' or 'label' (adjust if different).
# #     Returns DataFrame.
# #     """
# #     df = pd.read_csv(csv_path, nrows=nrows)
# #     # find label column
# #     label_cols = [c for c in df.columns if c.lower()=='label' or c.lower()=='class' or 'attack' in c.lower()]
# #     if not label_cols:
# #         raise ValueError("Can't find label column. Columns: " + ",".join(df.columns[:50]))
# #     label_col = label_cols[0]
# #     df = df.rename(columns={label_col: 'label'})
# #     return df

# # def simple_preprocess(df, feature_cols=None, max_rows=None):
# #     """
# #     Keep numeric features only for baseline. Encode label.
# #     """
# #     if max_rows:
# #         df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
# #     # Keep numeric columns (drop strings like IP if necessary but we may keep srcIP for hyperedges)
# #     num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# #     # fallback: if no numeric, try some known NetFlow fields
# #     if feature_cols is None:
# #         feature_cols = num_cols
# #     else:
# #         feature_cols = [c for c in feature_cols if c in df.columns]
# #     # Fill NaN
# #     df[feature_cols] = df[feature_cols].fillna(0)
# #     # label encode
# #     le = LabelEncoder()
# #     df['y'] = le.fit_transform(df['label'].astype(str))
# #     return df, feature_cols, le

# # # -------------------------
# # # Hyperedge construction
# # # -------------------------
# # def build_hyperedges_from_flows(df, group_rules=None, time_col=None, time_window=1.0):
# #     """
# #     Construct hyperedges. Strategy:
# #       - group by srcIP in time windows
# #       - group by dstPort
# #       - group by protocol
# #     We'll create hyperedges as lists of flow indices.
# #     """
# #     n = len(df)
# #     hyperedges = []
# #     # rule 1: same srcIP in sliding time windows (if time_col exists)
# #     if time_col and time_col in df.columns:
# #         # sort by time
# #         df_sorted = df.sort_values(time_col).reset_index()
# #         # bucket by time_window (seconds)
# #         buckets = defaultdict(list)
# #         times = pd.to_datetime(df_sorted[time_col], errors='coerce')
# #         # fallback if parsing fails: use numeric
# #         if times.isna().all():
# #             times = pd.to_numeric(df_sorted[time_col], errors='coerce').fillna(0)
# #         else:
# #             times = (times.astype('int64') // 1_000_000_000)
# #         for i, (orig_idx, r) in enumerate(df_sorted[['index']].itertuples(index=False)):
# #             key = (df.loc[orig_idx, 'srcIP'] if 'srcIP' in df.columns else None, int(times.iloc[i] // time_window))
# #             buckets[key].append(orig_idx)
# #         for k, v in buckets.items():
# #             if len(v) > 1:
# #                 hyperedges.append(v)
# #     # rule 2: same dstPort
# #     if 'dstPort' in df.columns:
# #         groups = df.groupby('dstPort').indices
# #         for k, v in groups.items():
# #             if len(v) > 1:
# #                 hyperedges.append(list(v))
# #     # rule 3: same protocol
# #     if 'protocol' in df.columns:
# #         groups = df.groupby('protocol').indices
# #         for k, v in groups.items():
# #             if len(v) > 1:
# #                 hyperedges.append(list(v))
# #     # deduplicate hyperedges (as sets)
# #     uniq = {}
# #     for he in hyperedges:
# #         key = tuple(sorted(set(he)))
# #         uniq[key] = he
# #     hyperedges = list(uniq.values())
# #     return hyperedges

# # # -------------------------
# # # Convert hypergraph -> bipartite graph for PyG
# # # -------------------------
# # def hypergraph_to_bipartite(hyperedges, num_nodes):
# #     """
# #     Create bipartite edges: node_idx <-> hyperedge_idx (as new nodes offset by num_nodes)
# #     Returns edge_index (2 x E) for PyG (undirected represented twice).
# #     """
# #     edge_list = []
# #     H = len(hyperedges)
# #     for he_idx, he in enumerate(hyperedges):
# #         he_node = num_nodes + he_idx
# #         for v in he:
# #             edge_list.append((v, he_node))
# #             edge_list.append((he_node, v))
# #     if not edge_list:
# #         return torch.empty((2,0), dtype=torch.long)
# #     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
# #     return edge_index

# # # -------------------------
# # # Simple Hypergraph GNN (applied on bipartite graph)
# # # -------------------------
# # class SimpleHyperGNN(nn.Module):
# #     def __init__(self, in_channels, hidden, out_dim, num_layers=2):
# #         super().__init__()
# #         self.convs = nn.ModuleList()
# #         self.convs.append(SAGEConv(in_channels, hidden))
# #         for _ in range(num_layers-2):
# #             self.convs.append(SAGEConv(hidden, hidden))
# #         self.convs.append(SAGEConv(hidden, out_dim))
# #         self.projector = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
# #     def forward(self, x, edge_index):
# #         # x: (N_total, F) where N_total = nodes + hyperedges
# #         for conv in self.convs:
# #             x = conv(x, edge_index)
# #             x = F.relu(x)
# #         z = self.projector(x)
# #         # return only node embeddings (exclude hyperedge nodes)
# #         return z

# # # -------------------------
# # # Contrastive loss (NT-Xent)
# # # -------------------------
# # def nt_xent_loss(z1, z2, temperature=0.5, positive_mask=None):
# #     """
# #     z1,z2: (N, D)
# #     positive_mask: optional (N,N) boolean where True marks positive pair between i and j (symmetric)
# #     Basic implementation: use cosine similarity and NT-Xent where matching i in view2 is positive.
# #     We'll implement both instance-wise positives and extra positives via positive_mask.
# #     """
# #     z1 = F.normalize(z1, dim=1)
# #     z2 = F.normalize(z2, dim=1)
# #     batch_size = z1.size(0)
# #     sims_aa = torch.mm(z1, z1.t()) / temperature
# #     sims_bb = torch.mm(z2, z2.t()) / temperature
# #     sims_ab = torch.mm(z1, z2.t()) / temperature

# #     # labels: positive for i with i
# #     positives = torch.exp(torch.diag(sims_ab))
# #     # denominators: sum over all except self? We'll follow standard symmetric NT-Xent
# #     exp_sims_ab = torch.exp(sims_ab)
# #     exp_sims_aa = torch.exp(sims_aa)
# #     exp_sims_bb = torch.exp(sims_bb)

# #     # mask self
# #     diag_mask = torch.eye(batch_size, device=z1.device).bool()
# #     # denominator for view1->view2: sum over all exp_sims_ab row
# #     denom1 = exp_sims_ab.sum(dim=1)  # includes self
# #     loss1 = -torch.log(positives / denom1)

# #     # view2->view1
# #     positives2 = torch.exp(torch.diag(sims_ab.t()))
# #     denom2 = exp_sims_ab.sum(dim=0)
# #     loss2 = -torch.log(positives2 / denom2)

# #     loss = (loss1 + loss2) * 0.5
# #     loss = loss.mean()

# #     # incorporate positive_mask (label-aware): we add extra positive pairs by encouraging similarity
# #     if positive_mask is not None:
# #         # compute average negative log-sim loss for positive_mask pairs
# #         pm = positive_mask.float()
# #         # remove diagonal
# #         pm = pm - torch.diag(torch.diag(pm))
# #         if pm.sum() > 0:
# #             # encourage z1 and z2 of positive pairs to be similar via cosine
# #             pos_pairs = pm.nonzero(as_tuple=False)
# #             sim_pairs = (F.cosine_similarity(z1[pos_pairs[:,0]], z2[pos_pairs[:,1]]) + 1.0) / 2.0
# #             # treat as additional loss (we want sim_pairs near 1 -> (1 - sim))
# #             loss += (1.0 - sim_pairs).mean() * 0.5
# #     return loss

# # # -------------------------
# # # Augmentations (feature masking / node dropout)
# # # -------------------------
# # def augment_features(x, mask_ratio=0.1):
# #     x = x.clone()
# #     n, f = x.size()
# #     mask = torch.rand((n,f), device=x.device) < mask_ratio
# #     x[mask] = 0.0
# #     return x

# # def drop_edges(edge_index, drop_ratio=0.1):
# #     # edge_index: 2 x E (long)
# #     E = edge_index.size(1)
# #     keep = torch.rand(E) > drop_ratio
# #     return edge_index[:, keep]

# # # -------------------------
# # # Federated simulation (FedAvg)
# # # -------------------------
# # def split_clients(n_nodes, n_clients=5, iid=True):
# #     idx = np.arange(n_nodes)
# #     if iid:
# #         np.random.shuffle(idx)
# #         splits = np.array_split(idx, n_clients)
# #     else:
# #         # non-iid: split by label cluster would be better, but for baseline random uneven split
# #         np.random.shuffle(idx)
# #         splits = np.array_split(idx, n_clients)
# #     return [list(s) for s in splits]

# # def fed_avg(models_state_dicts):
# #     # average param tensors
# #     avg = {}
# #     n = len(models_state_dicts)
# #     for k in models_state_dicts[0].keys():
# #         avg[k] = sum([sd[k].float() for sd in models_state_dicts]) / n
# #     return avg

# # # -------------------------
# # # Main training pipeline
# # # -------------------------
# # def main(args):
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print("Device:", device)

# #     print("Loading DATA...")
# #     df = load_nf_toniot(args.data_path, nrows=args.max_rows)
# #     # keep potential columns used for hyperedges: srcIP, dstPort, protocol, flowStart? adapt as dataset
# #     df_cols = df.columns.tolist()
# #     # try common names
# #     time_col = None
# #     for cand in ['Timestamp','ts','time','start_time','FlowStartMilliseconds','StartTime']:
# #         if cand in df_cols:
# #             time_col = cand; break
# #     print("Detected time col:", time_col)
# #     # preprocess numeric features
# #     df_proc, feature_cols, label_encoder = simple_preprocess(df, feature_cols=None, max_rows=args.max_rows)
# #     print("Feature cols count:", len(feature_cols))
# #     # Build hyperedges
# #     print("Building hyperedges...")
# #     hyperedges = build_hyperedges_from_flows(df_proc, time_col=time_col, time_window=args.time_window)
# #     print("Num hyperedges:", len(hyperedges))
# #     num_nodes = len(df_proc)
# #     edge_index = hypergraph_to_bipartite(hyperedges, num_nodes)  # node + hyperedge nodes
# #     N_total = num_nodes + len(hyperedges)
# #     # Build node features matrix: for hyperedge nodes we init as zero vectors
# #     scaler = StandardScaler()
# #     X_nodes = scaler.fit_transform(df_proc[feature_cols].values)
# #     X_nodes = torch.tensor(X_nodes, dtype=torch.float)
# #     if len(hyperedges) > 0:
# #         X_hyper = torch.zeros((len(hyperedges), X_nodes.size(1)), dtype=torch.float)
# #         X = torch.cat([X_nodes, X_hyper], dim=0)
# #     else:
# #         X = X_nodes
# #     # labels for first num_nodes only
# #     y = torch.tensor(df_proc['y'].values, dtype=torch.long)
# #     num_classes = len(np.unique(df_proc['y']))
# #     print("Num nodes, classes:", num_nodes, num_classes)

# #     # create PyG Data (we mainly feed edge_index)
# #     data = Data(x=X, edge_index=edge_index)
# #     data = data.to(device)

# #     # create model
# #     model = SimpleHyperGNN(in_channels=X.size(1), hidden=args.hidden, out_dim=args.rep_dim, num_layers=args.num_layers).to(device)

# #     # federated split of node indices
# #     clients = split_clients(num_nodes, n_clients=args.n_clients, iid=args.iid)
# #     print("Simulated clients:", [len(c) for c in clients])

# #     # federated rounds
# #     global_state = None
# #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# #     for round in range(args.global_rounds):
# #         local_states = []
# #         local_losses = []
# #         print(f"=== Global round {round+1}/{args.global_rounds} ===")
# #         # send global model to clients
# #         if global_state is not None:
# #             model.load_state_dict(global_state)
# #         model.train()
# #         for c_idx, client_nodes in enumerate(clients):
# #             # local training
# #             # create local mask
# #             local_mask = torch.zeros(num_nodes, dtype=torch.bool)
# #             local_mask[client_nodes] = True
# #             # local epochs
# #             for ep in range(args.local_epochs):
# #                 # augmentation: two views
# #                 # compute embeddings for node subset only: we compute full forward but loss on nodes only
# #                 # view 1
# #                 x1 = data.x.clone().to(device)
# #                 x1[:num_nodes] = augment_features(x1[:num_nodes], mask_ratio=args.mask_ratio)
# #                 edge1 = drop_edges(data.edge_index, drop_ratio=args.drop_edge)
# #                 z1 = model(x1, edge1)
# #                 # view 2
# #                 x2 = data.x.clone().to(device)
# #                 x2[:num_nodes] = augment_features(x2[:num_nodes], mask_ratio=args.mask_ratio*1.2)
# #                 edge2 = drop_edges(data.edge_index, drop_ratio=args.drop_edge*1.2)
# #                 z2 = model(x2, edge2)
# #                 # take node embeddings
# #                 z1_nodes = z1[:num_nodes]
# #                 z2_nodes = z2[:num_nodes]
# #                 # positive mask from labels (label-aware)
# #                 label_tensor = y.to(device)
# #                 pos_mask = (label_tensor.unsqueeze(1) == label_tensor.unsqueeze(0)).to(device)
# #                 # but restrict to local nodes for efficiency
# #                 loc_idx = torch.tensor(client_nodes, dtype=torch.long, device=device)
# #                 z1_loc = z1_nodes[loc_idx]
# #                 z2_loc = z2_nodes[loc_idx]
# #                 pos_mask_loc = pos_mask[loc_idx][:, loc_idx]
# #                 loss = nt_xent_loss(z1_loc, z2_loc, temperature=args.temperature, positive_mask=pos_mask_loc)
# #                 optimizer.zero_grad()
# #                 loss.backward()
# #                 optimizer.step()
# #             local_states.append({k: v.cpu() for k,v in model.state_dict().items()})
# #             local_losses.append(loss.item())
# #             print(f" Client {c_idx} loss {loss.item():.4f}")
# #         # aggregate
# #         global_state = fed_avg(local_states)
# #         model.load_state_dict(global_state)
# #         print("Round avg loss:", np.mean(local_losses))

# #     # After federated pretraining, we can freeze encoder and train a linear probe for node classification
# #     # Linear probe (on node embeddings)
# #     model.eval()
# #     with torch.no_grad():
# #         z = model(data.x.to(device), data.edge_index.to(device))[:num_nodes].cpu().numpy()
# #     # train/test split for classification
# #     train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42, stratify=df_proc['y'])
# #     from sklearn.linear_model import LogisticRegression
# #     clf = LogisticRegression(max_iter=200, multi_class='auto')
# #     clf.fit(z[train_idx], y.numpy()[train_idx])
# #     preds = clf.predict(z[test_idx])
# #     from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# #     acc = accuracy_score(y.numpy()[test_idx], preds)
# #     f1 = f1_score(y.numpy()[test_idx], preds, average='macro')
# #     print("Linear probe results -- Acc:", acc, "F1(macro):", f1)

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--data-path", type=str, default="./data/NF-ToN-IoT-v2-train.csv")
# #     parser.add_argument("--max-rows", type=int, default=200000)  # for speed/test; set None to use full
# #     parser.add_argument("--time-window", type=float, default=1.0)
# #     parser.add_argument("--hidden", type=int, default=128)
# #     parser.add_argument("--rep-dim", type=int, default=64)
# #     parser.add_argument("--num-layers", type=int, default=3)
# #     parser.add_argument("--n-clients", type=int, default=5)
# #     parser.add_argument("--iid", action='store_true')
# #     parser.add_argument("--global-rounds", type=int, default=3)
# #     parser.add_argument("--local-epochs", type=int, default=2)
# #     parser.add_argument("--lr", type=float, default=1e-3)
# #     parser.add_argument("--mask-ratio", type=float, default=0.1)
# #     parser.add_argument("--drop-edge", type=float, default=0.1)
# #     parser.add_argument("--temperature", type=float, default=0.5)
# #     args = parser.parse_args()
# #     main(args)

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch_geometric.nn import SAGEConv
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score

# # ==============================
# # 1. Load dữ liệu
# # ==============================
# DATA_PATH = "NF-ToN-IoT-v2-train.csv"   # đổi thành file của bạn
# df = pd.read_csv(DATA_PATH, nrows=50000)  # đọc 50k dòng demo

# # Tìm cột nhãn
# label_col = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower()][0]
# df = df.rename(columns={label_col: "label"})

# # Encode label
# le = LabelEncoder()
# df["y"] = le.fit_transform(df["label"].astype(str))

# # ==============================
# # 2. Xử lý đặc trưng (fillna + scale)
# # ==============================
# num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# X_num = df[num_cols].fillna(0).values   # xử lý NaN
# scaler = StandardScaler()
# X_num = scaler.fit_transform(X_num)
# X_num = torch.tensor(X_num, dtype=torch.float)

# y = torch.tensor(df["y"].values, dtype=torch.long)
# num_nodes = len(df)
# print("Số node:", num_nodes, " Số class:", len(le.classes_))

# # ==============================
# # 3. Tạo hyperedges
# # ==============================
# hyperedges = []

# # Group theo L4_DST_PORT
# if "L4_DST_PORT" in df.columns:
#     for _, idx in df.groupby("L4_DST_PORT").indices.items():
#         if len(idx) > 1:
#             hyperedges.append(list(idx))

# # Group theo PROTOCOL
# if "PROTOCOL" in df.columns:
#     for _, idx in df.groupby("PROTOCOL").indices.items():
#         if len(idx) > 1:
#             hyperedges.append(list(idx))

# print("Số hyperedge:", len(hyperedges))

# # ==============================
# # 4. Tạo bipartite graph node <-> hyperedge
# # ==============================
# edge_list = []
# for he_id, he in enumerate(hyperedges):
#     he_node = num_nodes + he_id   # node id mới cho hyperedge
#     for v in he:
#         edge_list.append((v, he_node))
#         edge_list.append((he_node, v))

# edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# # Thêm feature cho hyperedge (zero vector)
# X_hyper = torch.zeros((len(hyperedges), X_num.size(1)))
# X_total = torch.cat([X_num, X_hyper], dim=0)

# print("Shape feature:", X_total.shape)
# print("Edge_index shape:", edge_index.shape)

# # ==============================
# # 5. Model HyperSAGE
# # ==============================
# class HyperSAGE(nn.Module):
#     def __init__(self, in_dim, hidden, out_dim):
#         super().__init__()
#         self.conv1 = SAGEConv(in_dim, hidden)
#         self.conv2 = SAGEConv(hidden, out_dim)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x

# model = HyperSAGE(in_dim=X_total.size(1), hidden=64, out_dim=32)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# X_total = X_total.to(device)
# edge_index = edge_index.to(device)
# y = y.to(device)

# # ==============================
# # 6. Contrastive Loss (mini-batch)
# # ==============================
# def contrastive_loss_batch(z1, z2, batch_size=1024, tau=0.5):
#     """
#     SimCLR-style contrastive loss cho mini-batch
#     z1, z2: (N, d) hai view embeddings
#     """
#     N = z1.size(0)
#     losses = []

#     for start in range(0, N, batch_size):
#         end = min(start + batch_size, N)

#         z1_batch = z1[start:end]  # (B, d)
#         z2_batch = z2[start:end]  # (B, d)
#         B = z1_batch.size(0)

#         # Chuẩn hóa
#         z1_batch = F.normalize(z1_batch, dim=1)
#         z2_batch = F.normalize(z2_batch, dim=1)

#         # Tạo cặp (positive)
#         reps = torch.cat([z1_batch, z2_batch], dim=0)  # (2B, d)
#         sim = torch.mm(reps, reps.t()) / tau  # (2B, 2B)

#         # Mask loại bỏ self-similarity
#         mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
#         sim = sim.masked_fill(mask, -9e15)

#         # Label: mỗi index i ghép với i+B (và ngược lại)
#         labels = torch.arange(B, device=z1.device)
#         labels = torch.cat([labels + B, labels], dim=0)

#         loss = F.cross_entropy(sim, labels)
#         losses.append(loss)

#     return torch.stack(losses).mean()

# # ==============================
# # 7. Training loop
# # ==============================
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# for epoch in range(5):
#     model.train()
#     z1 = model(X_total, edge_index)
#     z2 = model(X_total, edge_index)  # view khác (ở đây tạm clone)

#     z1 = z1[:num_nodes]
#     z2 = z2[:num_nodes]

#     loss = contrastive_loss_batch(z1, z2, batch_size=512, tau=0.5)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# # ==============================
# # 8. Linear probe
# # ==============================
# model.eval()
# with torch.no_grad():
#     z_final = model(X_total, edge_index)[:num_nodes].cpu().numpy()

# # Xử lý NaN trước logistic regression
# z_final = np.nan_to_num(z_final, nan=0.0, posinf=0.0, neginf=0.0)

# train_idx, test_idx = train_test_split(
#     np.arange(num_nodes),
#     test_size=0.2,
#     stratify=y.cpu().numpy(),
#     random_state=42
# )

# clf = LogisticRegression(max_iter=200)
# clf.fit(z_final[train_idx], y.cpu().numpy()[train_idx])
# preds = clf.predict(z_final[test_idx])

# print("Accuracy:", accuracy_score(y.cpu().numpy()[test_idx], preds))
# print("F1:", f1_score(y.cpu().numpy()[test_idx], preds, average="macro"))

# !pip install torch_geometric

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ==============================
# 1. Load dữ liệu
# ==============================
DATA_PATH = "NF-ToN-IoT-v2-train.csv"   # đổi thành file của bạn
df = pd.read_csv(DATA_PATH, nrows=50000)  # đọc 50k dòng demo

# Tìm cột nhãn
label_col = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower()][0]
df = df.rename(columns={label_col: "label"})

# Encode label
le = LabelEncoder()
df["y"] = le.fit_transform(df["label"].astype(str))

# ==============================
# 2. Xử lý đặc trưng (fillna + scale)
# ==============================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_num = df[num_cols].fillna(0).values   # xử lý NaN
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)
X_num = torch.tensor(X_num, dtype=torch.float)

y = torch.tensor(df["y"].values, dtype=torch.long)
num_nodes = len(df)
print("Số node:", num_nodes, " Số class:", len(le.classes_))

# ==============================
# 3. Tạo hyperedges
# ==============================
hyperedges = []

# Group theo L4_DST_PORT
if "L4_DST_PORT" in df.columns:
    for _, idx in df.groupby("L4_DST_PORT").indices.items():
        if len(idx) > 1:
            hyperedges.append(list(idx))

# Group theo PROTOCOL
if "PROTOCOL" in df.columns:
    for _, idx in df.groupby("PROTOCOL").indices.items():
        if len(idx) > 1:
            hyperedges.append(list(idx))

print("Số hyperedge:", len(hyperedges))

# ==============================
# 4. Tạo bipartite graph node <-> hyperedge
# ==============================
edge_list = []
for he_id, he in enumerate(hyperedges):
    he_node = num_nodes + he_id   # node id mới cho hyperedge
    for v in he:
        edge_list.append((v, he_node))
        edge_list.append((he_node, v))

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Thêm feature cho hyperedge (zero vector)
X_hyper = torch.zeros((len(hyperedges), X_num.size(1)))
X_total = torch.cat([X_num, X_hyper], dim=0)

print("Shape feature:", X_total.shape)
print("Edge_index shape:", edge_index.shape)

# ==============================
# 5. Model HyperSAGE
# ==============================
class HyperSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = HyperSAGE(in_dim=X_total.size(1), hidden=64, out_dim=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_total = X_total.to(device)
edge_index = edge_index.to(device)
y = y.to(device)

# ==============================
# 6. Contrastive Loss (mini-batch)
# ==============================
def contrastive_loss_batch(z1, z2, batch_size=1024, tau=0.5):
    """
    SimCLR-style contrastive loss cho mini-batch
    z1, z2: (N, d) hai view embeddings
    """
    N = z1.size(0)
    losses = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        z1_batch = z1[start:end]  # (B, d)
        z2_batch = z2[start:end]  # (B, d)
        B = z1_batch.size(0)

        # Chuẩn hóa
        z1_batch = F.normalize(z1_batch, dim=1)
        z2_batch = F.normalize(z2_batch, dim=1)

        # Tạo cặp (positive)
        reps = torch.cat([z1_batch, z2_batch], dim=0)  # (2B, d)
        sim = torch.mm(reps, reps.t()) / tau  # (2B, 2B)

        # Mask loại bỏ self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
        sim = sim.masked_fill(mask, -9e15)

        # Label: mỗi index i ghép với i+B (và ngược lại)
        labels = torch.arange(B, device=z1.device)
        labels = torch.cat([labels + B, labels], dim=0)

        loss = F.cross_entropy(sim, labels)
        losses.append(loss)

    return torch.stack(losses).mean()

# ==============================
# 7. Training loop
# ==============================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    z1 = model(X_total, edge_index)
    z2 = model(X_total, edge_index)  # view khác (ở đây tạm clone)

    z1 = z1[:num_nodes]
    z2 = z2[:num_nodes]

    loss = contrastive_loss_batch(z1, z2, batch_size=512, tau=0.5)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==============================
# 8. Linear probe
# ==============================
model.eval()
with torch.no_grad():
    z_final = model(X_total, edge_index)[:num_nodes].cpu().numpy()

# Xử lý NaN trước logistic regression
z_final = np.nan_to_num(z_final, nan=0.0, posinf=0.0, neginf=0.0)

train_idx, test_idx = train_test_split(
    np.arange(num_nodes),
    test_size=0.2,
    stratify=y.cpu().numpy(),
    random_state=42
)

clf = LogisticRegression(max_iter=200)
clf.fit(z_final[train_idx], y.cpu().numpy()[train_idx])
preds = clf.predict(z_final[test_idx])

print("Accuracy:", accuracy_score(y.cpu().numpy()[test_idx], preds))
print("F1:", f1_score(y.cpu().numpy()[test_idx], preds, average="macro"))
