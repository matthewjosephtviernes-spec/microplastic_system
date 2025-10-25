import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report
)
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Microplastic Risk Prediction System", layout="wide")
st.title("Microplastic Risk Prediction System (Upload Dataset)")

# -------------------------
# Helper functions
# -------------------------
def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a .csv or .xlsx file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def basic_preprocess(df, drop_na=True, impute_strategy="mean"):
    # Copy to avoid modifying original
    df = df.copy()
    # If drop_na chosen, drop rows with any NA
    if drop_na:
        df = df.dropna()
    else:
        # Impute numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy=impute_strategy)
        if len(num_cols) > 0:
            df[num_cols] = imputer.fit_transform(df[num_cols])
        # For object cols, fill with mode
        obj_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in obj_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")
    return df

def encode_features_and_target(df, target_col=None, task="classification"):
    X = df.copy()
    y = None
    if target_col:
        y = X.pop(target_col)
    # Encode target if classification and non-numeric
    le = None
    if y is not None and task == "classification":
        if y.dtype == object or str(y.dtype).startswith("category"):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    # Standardize numeric features for some models/clustering
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, y, le

# -------------------------
# Upload dataset
# -------------------------
st.sidebar.header("1) Upload dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"])

if uploaded_file is None:
    st.info("No file uploaded yet. You can upload a CSV or Excel dataset here. The rest of the app will use the uploaded dataset for preprocessing, modeling, validation, and cross-validation.")
    st.stop()

# Read file
df = read_uploaded_file(uploaded_file)
if df is None:
    st.stop()

st.subheader("Dataset Preview")
st.write(f"Filename: {uploaded_file.name} — {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(50))

# -------------------------
# Preprocessing options
# -------------------------
st.sidebar.header("2) Preprocessing options")
drop_na = st.sidebar.checkbox("Drop rows with missing values (otherwise will impute)", value=True)
impute_strategy = st.sidebar.selectbox("Impute numeric missing values strategy", ["mean", "median", "most_frequent"], index=0)

# Option to create Risk_Level from a numeric column (e.g., MP_Count)
st.sidebar.markdown("Create Risk Level from a numeric column (optional)")
auto_create_risk = st.sidebar.checkbox("Create Risk_Level from numeric column (e.g., MP_Count)", value=False)

risk_source_col = None
if auto_create_risk:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.sidebar.error("No numeric columns available in dataset to derive risk.")
        auto_create_risk = False
    else:
        risk_source_col = st.sidebar.selectbox("Select numeric column to derive risk (e.g., MP_Count)", numeric_columns)
        st.sidebar.write("Adjust thresholds for Low / Medium / High")
        thr_low = st.sidebar.number_input("Low <= ", value=1.0, step=0.1)
        thr_medium = st.sidebar.number_input("Medium <= ", value=3.0, step=0.1)

        def assign_risk_level(count, l=thr_low, m=thr_medium):
            try:
                if count <= l:
                    return "Low"
                elif count <= m:
                    return "Medium"
                else:
                    return "High"
            except Exception:
                return "Unknown"

        df["Risk_Level"] = df[risk_source_col].apply(assign_risk_level)

# If user does not auto-create risk, they can choose an existing column as target
st.sidebar.markdown("Target/Label selection")
target_col = st.sidebar.selectbox("Select target column (leave blank to perform unsupervised/clustering)", [""] + df.columns.tolist())

# Preprocess dataset
df_clean = basic_preprocess(df, drop_na=drop_na, impute_strategy=impute_strategy)

st.write("✅ Preprocessing complete")
st.write(f"Cleaned dataset shape: {df_clean.shape}")
st.dataframe(df_clean.head(20))

# -------------------------
# Modeling choices
# -------------------------
st.sidebar.header("3) Modeling choices")
task = st.sidebar.radio("Select task", ("classification", "regression", "clustering"))

# Validate that a target is selected for supervised tasks
if task in ("classification", "regression") and (target_col == "" or target_col is None):
    st.sidebar.error("Please select a target column for supervised tasks (classification or regression).")
    st.stop()

# If user selected a clustering task, ignore target_col
if task == "clustering":
    target_col = None

# Allow user to select features (default: all except target)
all_columns = df_clean.columns.tolist()
if target_col:
    default_features = [c for c in all_columns if c != target_col]
else:
    default_features = all_columns

selected_features = st.sidebar.multiselect("Select feature columns to use", all_columns, default=default_features)

if len(selected_features) == 0:
    st.sidebar.error("Please select at least one feature column.")
    st.stop()

# -------------------------
# Prepare X, y
# -------------------------
df_model = df_clean[selected_features + ([target_col] if target_col else [])].copy()

X, y, label_encoder = encode_features_and_target(df_model, target_col=target_col, task=task)

st.write("Feature matrix and target prepared.")
st.write(f"X shape: {X.shape}")
if y is not None:
    st.write(f"y shape: {y.shape} | unique values (if classification): {np.unique(y)[:20]}")

# -------------------------
# Train/Test split and model selection
# -------------------------
st.sidebar.header("4) Training and validation")
test_size = st.sidebar.slider("Test size (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random state (seed)", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y if (task=="classification" and y is not None and len(np.unique(y))>1) else None)

st.header("⚙️ Model Training and Evaluation")

if task == "classification":
    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=int(random_state)),
        "Decision Tree": DecisionTreeClassifier(random_state=int(random_state)),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }
    results = {}
    for name, model in classifiers.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    st.subheader("Model evaluation on test set (classification)")
    st.table(pd.DataFrame(results).T)

    # Show confusion matrix and classification report for best model by F1
    try:
        best_model_name = max(results.items(), key=lambda kv: kv[1].get("F1 Score", -1))[0]
        st.write(f"Best model by F1 score: {best_model_name}")
        best_model = classifiers[best_model_name]
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix — {best_model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred, zero_division=0))
    except Exception as e:
        st.warning(f"Could not produce confusion matrix/classification report: {e}")

elif task == "regression":
    regressors = {
        "Random Forest Regressor": RandomForestRegressor(random_state=int(random_state)),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=int(random_state)),
        "Linear Regression": LinearRegression()
    }
    results = {}
    for name, model in regressors.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred)
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    st.subheader("Model evaluation on test set (regression)")
    st.table(pd.DataFrame(results).T)

elif task == "clustering":
    st.subheader("Unsupervised clustering")
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    km = KMeans(n_clusters=n_clusters, random_state=int(random_state))
    cluster_labels = km.fit_predict(X)
    st.write("Cluster counts:")
    st.write(pd.Series(cluster_labels).value_counts())
    if X.shape[1] >= 2:
        try:
            sil = silhouette_score(X, cluster_labels)
            st.write(f"Silhouette Score: {sil:.4f}")
        except Exception as e:
            st.warning(f"Could not compute silhouette score: {e}")

    # Scatter plot of first two principal features (if available)
    fig, ax = plt.subplots()
    if X.shape[1] >= 2:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_title("Cluster scatter (first two features)")
    else:
        ax.text(0.1, 0.5, "Not enough numeric features for scatter plot", transform=ax.transAxes)
    st.pyplot(fig)

# -------------------------
# Cross-validation
# -------------------------
st.header("🔁 Cross-validation (K-Fold)")

n_splits = st.sidebar.number_input("K-Fold splits", min_value=2, max_value=20, value=10, step=1)
kfold = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))

cv_results = {}
if task == "classification":
    for name, model in classifiers.items():
        try:
            scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
            cv_results[name] = {"mean_accuracy": float(scores.mean()), "std": float(scores.std())}
        except Exception as e:
            cv_results[name] = {"error": str(e)}
elif task == "regression":
    for name, model in regressors.items():
        try:
            scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
            cv_results[name] = {"mean_r2": float(scores.mean()), "std": float(scores.std())}
        except Exception as e:
            cv_results[name] = {"error": str(e)}
elif task == "clustering":
    st.write("Cross-validation is not directly applicable to unsupervised clustering in the same way; consider using silhouette score, stability analysis or hold-out/resampling strategies instead.")
    cv_results = {"note": "See above"}

st.write(cv_results)

# -------------------------
# Visualizations
# -------------------------
st.header("📊 Visualizations")

# Risk level distribution if exists
if "Risk_Level" in df_clean.columns:
    fig, ax = plt.subplots()
    df_clean["Risk_Level"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Risk Levels")
    st.pyplot(fig)

# Pairplot for small number of features
if X.shape[1] <= 6:
    try:
        sns_plot = sns.pairplot(pd.concat([X, pd.Series(y, name="target")] if y is not None else X, axis=1))
        st.pyplot(sns_plot.fig)
    except Exception as e:
        st.warning(f"Could not generate pairplot: {e}")

st.success("Processing complete. You uploaded a dataset, it was preprocessed, a model was trained and evaluated, and cross-validation was performed.")

st.markdown("---")
st.markdown("Notes / Assumptions:")
st.markdown("""
- This app expects the uploaded dataset to contain the features needed for predicting microplastic risk.
- You can optionally create a derived Risk_Level column from a numeric column (e.g., MP_Count) using configurable thresholds.
- For classification tasks, non-numeric targets will be label-encoded automatically.
- Preprocessing offers simple options (drop rows with NA or impute numeric values). For production, consider more robust cleaning and domain-specific feature engineering, especially when extracting risk from textual sources.
- Extracting risk from text (articles, papers) would require NLP pipelines (not implemented here). If you want that, I can add a text-extraction + NLP labeling flow (entity extraction, keyword matching or supervised text classification).
""")
