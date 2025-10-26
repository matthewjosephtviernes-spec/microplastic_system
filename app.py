# streamlit_microplastic_app.py
# Microplastic Risk Analysis Dashboard — Styled Version
# Usage: streamlit run streamlit_microplastic_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Microplastic Risk Analysis Dashboard", page_icon="🧪")
st.title("🧪 Microplastic Risk Analysis Dashboard")
st.caption("Interactive defense demo — explore dataset, clustering, classification, validation, regression, and summary.")

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.info("Use buttons to manually trigger each stage for your thesis defense.")

st.sidebar.header("1️⃣ Upload or Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your .csv dataset", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset (uploaded earlier)", value=True)

def load_default(paths):
    for p in paths:
        try:
            return pd.read_csv(p)
        except Exception:
            continue
    return None

default_paths = ["/mnt/data/Cleaned_Microplastic_Dataset.csv", "/mnt/data/Encoded_Microplastic_Dataset.csv"]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default(default_paths) if use_default else None

if df is None:
    st.warning("⚠️ Please upload a dataset or enable 'Use default dataset'.")
    st.stop()

st.success(f"✅ Dataset loaded successfully — {df.shape[0]} rows × {df.shape[1]} columns")

# --- Utility functions ---
@st.cache_data
def preprocess_dataframe(in_df):
    df = in_df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    for c in cat_cols:
        try:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
        except Exception:
            pass

    return df, num_cols, cat_cols

@st.cache_data
def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)
    return Xp, pca

# --- 2️⃣ Preprocessing ---
with st.expander("⚙️ Step 2: Preprocessing", expanded=True):
    if st.button("Run Preprocessing", key="preprocess"):
        with st.spinner("Cleaning and encoding dataset..."):
            cleaned_df, num_cols, cat_cols = preprocess_dataframe(df)
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['num_cols'] = num_cols
            st.session_state['cat_cols'] = cat_cols
            st.success("✅ Preprocessing complete!")
            st.dataframe(cleaned_df.head())
    else:
        cleaned_df = st.session_state.get('cleaned_df', None)

if cleaned_df is None:
    st.info("Run preprocessing first.")
    st.stop()

# --- 3️⃣ Clustering ---
with st.expander("🔹 Step 3: K-Means Clustering", expanded=False):
    cluster_cols = st.multiselect("Select features for clustering", options=cleaned_df.columns.tolist(), default=st.session_state['num_cols'])
    n_clusters = st.slider("Number of clusters", 2, 10, 3)

    if st.button("Run K-Means", key="cluster"):
        X = cleaned_df[cluster_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        X_pca, _ = compute_pca(X_scaled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        cleaned_df['Cluster'] = labels

        st.session_state['cleaned_df'] = cleaned_df
        st.success(f"✅ K-Means complete — {n_clusters} clusters identified!")

        fig = px.scatter(
            x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="K-Means Clusters (PCA 2D Visualization)", labels={'x':'PC1','y':'PC2'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Cluster Counts:**")
        st.dataframe(cleaned_df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count'))

# --- 4️⃣ Classification ---
with st.expander("🧠 Step 4: Random Forest Classification", expanded=False):
    target_candidates = [c for c in cleaned_df.columns if c.lower() in ('risk_level','risk','label','target','risk_category')]
    target_col = st.selectbox("Select target column (optional, defaults to Cluster)", [None]+cleaned_df.columns.tolist(), index=0)

    if st.button("Train Classifier", key="classifier"):
        target = cleaned_df[target_col] if target_col else cleaned_df['Cluster']
        X = cleaned_df.drop(columns=[target_col]) if target_col else cleaned_df.drop(columns=['Cluster'])
        X = X.select_dtypes(include=[np.number])
        X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.75, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.success(f"✅ Model trained — Accuracy: {acc:.3f}")
        cm = confusion_matrix(y_test, preds)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)

        st.session_state['clf_acc'] = acc

# --- 5️⃣ K-Fold Validation ---
with st.expander("📊 Step 5: K-Fold Validation", expanded=False):
    n_splits = st.slider("Number of folds", 2, 10, 5)
    if st.button("Run Validation", key="validation"):
        X = cleaned_df.select_dtypes(include=[np.number])
        y = cleaned_df['Cluster']
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, X, y, cv=kf)

        st.success(f"✅ Mean Accuracy: {scores.mean():.3f}")
        fig = px.line(y=scores, markers=True, title='K-Fold Accuracy Scores', color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
        st.session_state['cv_scores'] = scores

# --- 6️⃣ Regression ---
with st.expander("📈 Step 6: Regression — Predict Risk Severity", expanded=False):
    reg_target = st.selectbox("Select numeric target for regression", [None]+cleaned_df.columns.tolist())
    if st.button("Run Regression", key="regression"):
        y = cleaned_df[reg_target] if reg_target else cleaned_df['Cluster']
        X = cleaned_df.select_dtypes(include=[np.number]).drop(columns=[reg_target]) if reg_target else cleaned_df.select_dtypes(include=[np.number]).drop(columns=['Cluster'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        regr = RandomForestRegressor(n_estimators=100, random_state=42)
        regr.fit(X_train, y_train)
        preds = regr.predict(X_test)
        r2 = r2_score(y_test, preds)

        st.success(f"✅ Regression complete — R²: {r2:.3f}")
        fig = px.scatter(x=y_test, y=preds, color_discrete_sequence=['#EF553B'], title='Actual vs Predicted Risk Severity', labels={'x':'Actual','y':'Predicted'})
        st.plotly_chart(fig, use_container_width=True)
        st.session_state['reg_r2'] = r2

# --- 7️⃣ Final Summary ---
with st.expander("📜 Step 7: Final Summary", expanded=False):
    st.subheader("Performance Summary")
    metrics = {}
    if 'clf_acc' in st.session_state:
        metrics['Classification Accuracy'] = st.session_state['clf_acc']
    if 'cv_scores' in st.session_state:
        metrics['Mean K-Fold Accuracy'] = st.session_state['cv_scores'].mean()
    if 'reg_r2' in st.session_state:
        metrics['Regression R²'] = st.session_state['reg_r2']

    if metrics:
        summary_df = pd.DataFrame(metrics.items(), columns=['Metric','Value'])
        fig = px.bar(summary_df, x='Metric', y='Value', text='Value', color='Metric', color_discrete_sequence=px.colors.qualitative.Bold, title='Model Performance Summary')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics available yet. Run previous steps first.")

# --- Export ---
with st.expander("💾 Export Cleaned Dataset", expanded=False):
    if st.button("Generate Download Link"):
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", csv, "cleaned_microplastic_dataset.csv", "text/csv")

st.markdown("---")
st.caption("App built for thesis defense — use section expanders for interactive presentation.")
