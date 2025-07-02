import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# تحميل test data
test_df = pd.read_csv("KDDTest+.txt", header=None)
columns_df = pd.read_csv("Field Names.csv", header=None)
features = columns_df[0].tolist()
columns = features + ["label", "difficulty"]
test_df.columns = columns

# معالجة البيانات
test_df["binary_label"] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

categorical_cols = ["protocol_type", "service", "flag"]
X_test_raw = test_df.drop(["label", "difficulty", "binary_label"], axis=1)
y_test = test_df["binary_label"]

# تحميل encoder
encoder = joblib.load("encoder.joblib")
X_test_encoded = encoder.transform(X_test_raw[categorical_cols])
X_test_final = np.hstack([X_test_raw.drop(columns=categorical_cols).values, X_test_encoded])

# تحميل الموديل
model = joblib.load("rf_model.joblib")
y_pred = model.predict(X_test_final)

# رسم Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
