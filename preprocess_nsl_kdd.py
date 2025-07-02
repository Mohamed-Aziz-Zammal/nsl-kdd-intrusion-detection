import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

# 📂 تحميل البيانات
train_df = pd.read_csv("KDDTrain+.txt", header=None)
test_df = pd.read_csv("KDDTest+.txt", header=None)

# تحميل أسماء الأعمدة
columns_df = pd.read_csv("Field Names.csv", header=None)
features = columns_df[0].tolist()
columns = features + ["label", "difficulty"]
train_df.columns = columns
test_df.columns = columns

# 🔄 تحويل label إلى 0 (normal) و 1 (attack)
train_df["binary_label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
test_df["binary_label"] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

# 🧠 الأعمدة الرمزية فقط
categorical_cols = ["protocol_type", "service", "flag"]

# باقي الأعمدة (features فقط، بدون label, difficulty, binary)
X_train_raw = train_df.drop(["label", "difficulty", "binary_label"], axis=1)
X_test_raw = test_df.drop(["label", "difficulty", "binary_label"], axis=1)
y_train = train_df["binary_label"].values
y_test = test_df["binary_label"].values

# 🔁 One-Hot Encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_encoded = encoder.fit_transform(X_train_raw[categorical_cols])
X_test_encoded = encoder.transform(X_test_raw[categorical_cols])

# ✨ ندمج الأعمدة الغير رمزية + one-hot
X_train_final = np.hstack([X_train_raw.drop(columns=categorical_cols).values, X_train_encoded])
X_test_final = np.hstack([X_test_raw.drop(columns=categorical_cols).values, X_test_encoded])

# 💾 نحفظ كل شيء
np.save("X_train.npy", X_train_final)
np.save("X_test.npy", X_test_final)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
joblib.dump(encoder, "encoder.joblib")

print("✅ البيانات جهزناها وحفظناها بنجاح 🎉")
