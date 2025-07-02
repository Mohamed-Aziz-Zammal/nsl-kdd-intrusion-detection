import pandas as pd

# تحميل البيانات
train_path = "KDDTrain+.txt"
test_path = "KDDTest+.txt"

# تعريف أسماء الأعمدة من الملف المحلي
field_names_path = "Field Names.csv"
fields_df = pd.read_csv(field_names_path, header=None)
columns = fields_df[0].tolist() + ['label', 'difficulty']  # 41 ميزة + 2 أعمدة إضافية

# قراءة البيانات
df_train = pd.read_csv(train_path, names=columns)
df_test = pd.read_csv(test_path, names=columns)

# عرض أول 5 صفوف
print("🧠 Training Sample:")
print(df_train.head())

# معلومات عامة
print("\n📊 Shape:", df_train.shape)
print("🧷 Unique labels:", df_train['label'].unique())
