import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# 📂 المسارات
train_path = "KDDTrain+.txt"
test_path = "KDDTest+.txt"
columns_path = "Field Names.csv"

# 🧾 تحميل أسماء الأعمدة
columns_df = pd.read_csv(columns_path)
features = columns_df.iloc[:, 0].tolist()
columns = features + ["label", "difficulty"]

# 🧠 تحميل البيانات
df_train = pd.read_csv(train_path, names=columns)
df_test = pd.read_csv(test_path, names=columns)

# 🔄 تحويل labels
df_train['binary_label'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['binary_label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# 🧩 إعداد OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train = encoder.fit_transform(df_train.drop(["label", "difficulty", "binary_label"], axis=1))
X_test = encoder.transform(df_test.drop(["label", "difficulty", "binary_label"], axis=1))

# 💾 حفظ البيانات
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", df_train["binary_label"].values)
np.save("y_test.npy", df_test["binary_label"].values)

# 💾 حفظ الـ encoder
joblib.dump(encoder, "encoder.joblib")

print("✅ OneHotEncoder و البيانات تحفظو بنجاح!")
