import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# ============================================================
# 1️⃣ Tiền xử lý text tiếng Việt
# ============================================================
def clean_vi_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"_", " ", text)
    text = re.sub(r"colonsmile", ":)", text)
    text = re.sub(r"colonsad", ":(", text)
    text = re.sub(r"colonlove", "<3", text)
    text = re.sub(r"colonbigsmile", ":v", text)
    text = re.sub(r"dotdotdot", "...", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# 2️⃣ Dataset class cho UIT-VSFC
# ============================================================
class VSFCDataset(Dataset):
    def __init__(self, sents_path, labels_path, tokenizer, max_len=128):
        with open(sents_path, encoding='utf-8') as f:
            self.sents = [clean_vi_text(line.strip()) for line in f]
        with open(labels_path, encoding='utf-8') as f:
            self.labels = [int(line.strip()) for line in f]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        text = self.sents[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

# ============================================================
# 3️⃣ Mamba for Sequence Classification
# ============================================================
class MambaForSequenceClassification(nn.Module):
    def __init__(self, pretrained_name="state-spaces/mamba-130m-hf", num_labels=3):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state   # [B, L, H]
        pooled = last_hidden_state.mean(dim=1)          # mean pooling
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# ============================================================
# 4️⃣ Training + Evaluation loop
# ============================================================
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            logits = out["logits"]

            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(preds)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return avg_loss, acc, prec, rec, f1

# ============================================================
# 5️⃣ Main training
# ============================================================
def train_mamba_vsfc(base_path="UIT-VSFC", model_name="state-spaces/mamba-130m-hf", epochs=5, batch_size=8, lr=2e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForSequenceClassification(model_name, num_labels=3).to(device)

    # Load data
    train_dataset = VSFCDataset(os.path.join(base_path, "train/sents.txt"),
                                os.path.join(base_path, "train/sentiments.txt"), tokenizer)
    dev_dataset = VSFCDataset(os.path.join(base_path, "dev/sents.txt"),
                              os.path.join(base_path, "dev/sentiments.txt"), tokenizer)
    test_dataset = VSFCDataset(os.path.join(base_path, "test/sents.txt"),
                               os.path.join(base_path, "test/sentiments.txt"), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0
    patience, patience_counter = 2, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, acc, prec, rec, f1 = evaluate(model, dev_loader, device)

        print(f"\nEpoch {epoch+1}: TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
              f"Acc={acc:.4f} | F1={f1:.4f}")

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_mamba.pt")
            print("✅ Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    # === Test ===
    print("\n=== Testing best model ===")
    model.load_state_dict(torch.load("checkpoints/best_mamba.pt"))
    test_loss, acc, prec, rec, f1 = evaluate(model, test_loader, device)
    print(f"\n📊 Test Results: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/mamba_vsfc_metrics.txt", "w") as f:
        f.write(f"Acc={acc:.4f}\nPrec={prec:.4f}\nRec={rec:.4f}\nF1={f1:.4f}\n")

    print("✅ Results saved to results/mamba_vsfc_metrics.txt")

============================================================
6️⃣ Run
============================================================
if __name__ == "__main__":
    train_mamba_vsfc()



import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

sentence = "I love AI"
tokens = tokenizer.tokenize(sentence)
ids = tokenizer.encode(sentence, return_tensors="pt")

print("Tokens:", tokens)
print("Token IDs:", ids)
print("Vocabulary size:", tokenizer.vocab_size)

d_model = 4
embedding = nn.Embedding(tokenizer.vocab_size, d_model)

hidden_states = embedding(ids)  # (B=1, L, d_model)
print("Hidden states shape:", hidden_states.shape)

expand = 2
d_inner = d_model * expand  # 8
d_conv = 2
dt_rank = 2
d_state = 1

in_proj = nn.Linear(d_model, d_inner*2, bias=False)
conv1d = nn.Conv1d(
    in_channels=d_inner,
    out_channels=d_inner,
    bias=False,
    kernel_size=d_conv,
    groups=d_inner,
    padding=d_conv - 1
)
act = nn.SiLU()
x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

B, L, _ = hidden_states.shape

xz = in_proj(hidden_states)
x, z = torch.chunk(xz, 2, dim=2)
print("\nIn_proj -> x shape:", x.shape, "| z shape:", z.shape)

# (b) Conv1d causal
x_perm = x.permute(0, 2, 1)           # (B, d_inner, L)
x_conv = conv1d(x_perm)[..., :L]      # causal conv
x_conv_act = act(x_conv)
x = x_conv_act.permute(0, 2, 1)       # (B, L, d_inner)
print("After conv+SiLU -> x shape:", x.shape)

x_flat = x.reshape(B * L, d_inner)
x_db = x_proj(x_flat)
dt_flat = x_db[:, :dt_rank]
B_flat = x_db[:, dt_rank:dt_rank + d_state]
C_flat = x_db[:, dt_rank + d_state:dt_rank + 2 * d_state]

print("\nRaw dt, B, C:")
print("dt_flat:", dt_flat[:3])
print("B_flat:", B_flat[:3])
print("C_flat:", C_flat[:3])

dt_scaled = dt_proj(dt_flat)
print("\nScaled dt:", dt_scaled[:3])

A_log = torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1)
A = -torch.exp(A_log)
D = torch.ones(d_inner)

print("\nA:", A[:3])
print("D:", D[:3])

print("\n✅ Ta đã có đủ đầu vào cho SSM scan:")
print(" - dt_scaled (Δt):", dt_scaled.shape)
print(" - B_flat:", B_flat.shape)
print(" - C_flat:", C_flat.shape)
print(" - A:", A.shape)
print(" - D:", D.shape)


# import torch
# import torch.nn.functional as F

# # Giả sử ta có các giá trị từ phần trước
# torch.manual_seed(42)

# B = 1          # batch size
# L = 3          # "I love AI"
# d_inner = 8
# d_state = 1

# # Tạo x (đầu ra sau conv) ngẫu nhiên cho minh họa
# x = torch.randn(B, L, d_inner)

# # Giả lập tham số A, B, C, D, dt
# A = -torch.exp(torch.randn(d_inner, d_state))    # hệ số suy giảm
# B_param = torch.randn(B, L, d_state)             # input-to-state
# C_param = torch.randn(B, L, d_state)             # state-to-output
# D = torch.ones(d_inner)                          # residual
# dt = F.softplus(torch.randn(B, L, d_inner))      # thời gian trễ dương

# # Khởi tạo hidden state ban đầu
# h = torch.zeros(B, d_inner, d_state)

# # Danh sách lưu output
# y_list = []

# print("=== Bắt đầu SSM scan ===\n")
# for t in range(L):
#     # Lấy từng token
#     x_t = x[:, t, :]             # (B, d_inner)
#     dt_t = dt[:, t, :]           # (B, d_inner)
#     B_t = B_param[:, t, :]       # (B, d_state)
#     C_t = C_param[:, t, :]       # (B, d_state)

#     # Tính ma trận rời rạc hóa
#     dA = torch.exp(dt_t.unsqueeze(-1) * A)       # (B, d_inner, d_state)
#     dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)   # (B, d_inner, d_state)

#     # Cập nhật hidden state
#     h = h * dA + x_t.unsqueeze(-1) * dB          # (B, d_inner, d_state)

#     # Tính output
#     y_t = torch.einsum("bdn,bn->bd", h, C_t) + D * x_t
#     y_list.append(y_t)

#     print(f"Step {t+1}:")
#     print(f"  x_t shape: {x_t.shape}")
#     print(f"  h_t[0, :3]: {h[0, :3, 0].detach().numpy()}")
#     print(f"  y_t[0, :3]: {y_t[0, :3].detach().numpy()}")
#     print()

# # Gộp lại output toàn chuỗi
# y = torch.stack(y_list, dim=1)
# print("Output y shape:", y.shape)
