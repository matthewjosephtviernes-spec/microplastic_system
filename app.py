import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

st.title("📊 Risk Score ML Modeling System")

# ------------------------------------------------------------------
# 1. Upload CSV
# ------------------------------------------------------------------
uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ------------------------------------------------------------------
    # 2. Column Selection
    # ------------------------------------------------------------------
    st.subheader("🔧 Select Columns")

    target_col = st.selectbox("Select Target Column (Risk_Type)", df.columns)
    num_cols = st.multiselect("Select Numerical Columns", df.select_dtypes(include=[np.number]).columns)
    cat_cols = st.multiselect("Select Categorical Columns", df.select_dtypes(exclude=[np.number]).columns)

    if st.button("Run Pipeline"):
        
        # ------------------------------------------------------------------
        # 3. Outlier Treatment
        # ------------------------------------------------------------------
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)

        # ------------------------------------------------------------------
        # 4. Transform Skewed Numerical Columns
        # ------------------------------------------------------------------
        for col in num_cols:
            df[col] = np.log1p(df[col] - df[col].min() + 1)

        # ------------------------------------------------------------------
        # 5. Encode Categorical Columns
        # ------------------------------------------------------------------
        df_encoded = pd.get_dummies(df, columns=cat_cols)

        # ------------------------------------------------------------------
        # 6. Feature Scaling
        # ------------------------------------------------------------------
        scaler = StandardScaler()
        df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

        # ------------------------------------------------------------------
        # 7. Feature Importance (Mutual Info)
        # ------------------------------------------------------------------
        X = df_encoded.drop(target_col, axis=1)
        y = df_encoded[target_col]

        st.subheader("📌 Feature Importance (Mutual Information)")

        mi = mutual_info_classif(X, y)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascendin
g=False)

        st.bar_chart(mi_series)

        # ------------------------------------------------------------------
        # 8. Address Class Imbalance (SMOTE)
        # ------------------------------------------------------------------
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)

        # ------------------------------------------------------------------
        # 9. Split Data
        # ------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        # ------------------------------------------------------------------
        # 10. Train Models
        # ------------------------------------------------------------------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric="logloss")
        }

        model_reports = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            model_reports[name] = report

            st.subheader(f"📈 {name} Performance")
            st.text(classification_report(y_test, preds))

        # ------------------------------------------------------------------
        # 11. Compare Model Accuracy
        # ------------------------------------------------------------------
        st.subheader("🏆 Model Accuracy Comparison")

        accuracy_data = {m: model_reports[m]["accuracy"] for m in model_reports}
        st.bar_chart(accuracy_data)

        best_model = max(accuracy_data, key=accuracy_data.get)
        st.success(f"🏆 BEST MODEL: **{best_model}**")

