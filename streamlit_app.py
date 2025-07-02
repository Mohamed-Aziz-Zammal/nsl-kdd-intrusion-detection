import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🔐 NSL-KDD Intrusion Detection Web App")
st.title("🔐 NSL-KDD Intrusion Detection Web App")
st.markdown("📁 حمّل ملف KDD بصيغة CSV")

uploaded_file = st.file_uploader("اختر ملف .csv", type="csv")

if uploaded_file is not None:
    try:
        # 📥 تحميل الملف
        df = pd.read_csv(uploaded_file)

        st.write("📋 عدد الأعمدة:", df.shape[1])
        st.write("👀 بعض الأعمدة:", df.columns.tolist()[:5])

        # ✅ الأعمدة الرمزية اللي درّبنا عليها
        categorical_cols = ["protocol_type", "service", "flag"]

        # نحذف الأعمدة الغير لازمة
        X_raw = df.drop(["label", "difficulty"], axis=1, errors="ignore")

        # نجهزو المعطيات بنفس طريقة التدريب
        encoder = joblib.load("encoder.joblib")
        X_encoded = encoder.transform(X_raw[categorical_cols])
        X_numeric = X_raw.drop(columns=categorical_cols).values
        X_final = np.hstack([X_numeric, X_encoded])

        # نحمل النموذج المدرب
        model = joblib.load("rf_model.joblib")
        y_pred = model.predict(X_final)

        # نضيف النتائج
        df["Prediction"] = y_pred
        df["Prediction_Label"] = df["Prediction"].map({0: "Normal", 1: "Attack"})

        st.success("✅ تمت عملية التنبؤ بنجاح!")
        st.dataframe(df["Prediction_Label"].value_counts().reset_index(name="Count"))

        st.markdown("📊 بعض النتائج:")
        st.dataframe(df[["Prediction_Label"]].head(10))

        # تحميل النتائج بصيغة CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ تحميل النتائج", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ خطأ أثناء التنبؤ: {e}")
