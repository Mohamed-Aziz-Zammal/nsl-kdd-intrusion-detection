import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 📥 تحميل البيانات الجاهزة
df_train = pd.read_csv("train_processed.csv")
df_test = pd.read_csv("test_processed.csv")

# 🏷️ استخراج الخصائص والهدف
X_train = df_train.drop(columns=["label", "binary_label"])
y_train = df_train["binary_label"]

X_test = df_test.drop(columns=["label", "binary_label"])
y_test = df_test["binary_label"]

# 🔄 التطبيع
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 💾 حفظ البيانات المُعالجة كنماذج
import numpy as np

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

print("✅ البيانات تجهزت بنجاح وحُفظت 🧪")
