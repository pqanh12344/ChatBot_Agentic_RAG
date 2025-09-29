import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load file gốc
data = pd.read_csv("NF-ToN-IoT-V2.csv", encoding="utf-8")

# Tìm cột nhãn (tự động)
label_col = [c for c in data.columns if "label" in c.lower() or "attack" in c.lower()][0]
print("Label column:", label_col)

# Loại bỏ NaN trong cột nhãn
data = data.dropna(subset=[label_col])

# Chuẩn hóa nhãn về string
data[label_col] = data[label_col].astype(str).str.strip()

# Split dữ liệu: 70% train, 15% val, 15% test
train_data, temp_data = train_test_split(
    data,
    test_size=0.3,
    random_state=42,
    stratify=data[label_col]
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_data[label_col]
)

# Lưu ra file
train_data.to_csv("NF-ToN-IoT-v2-train.csv", index=False)
val_data.to_csv("NF-ToN-IoT-v2-val.csv", index=False)
test_data.to_csv("NF-ToN-IoT-v2-test.csv", index=False)

print("✅ Split done!")
print("Train:", train_data.shape, " Val:", val_data.shape, " Test:", test_data.shape)
print("Train label distribution:\n", train_data[label_col].value_counts(normalize=True))
