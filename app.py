# Redesigned Streamlit app for Microplastic Risk Prediction
# - Cleaner structure, robust preprocessing, safer modeling fallbacks
# - Improved visuals: aggregated category bars, rotated/annotated labels, cap pairplots
# - Professional look via simple CSS and clearer sidebar steps

import io
import re
from typing import Optional, Tuple, List, Dict

import streamlit as st
import pandas as pd
import numpy as np
@@ -20,17 +27,46 @@
)
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Microplastic Risk Prediction System (robust)", layout="wide")
st.title("Microplastic Risk Prediction System")
# -------------------------
# App configuration & style
# -------------------------
st.set_page_config(page_title="Microplastic Risk Prediction (Professional)", layout="wide")
st.markdown(
    """
    <style>
      .stApp { font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial; }
      .big-header { font-size:22px; font-weight:600; margin-bottom:6px; }
      .muted { color: #6c757d; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-header">Microplastic Risk Prediction System</div>', unsafe_allow_html=True)
st.write("A robust, professional Streamlit app for exploring microplastic datasets, preprocessing, modeling, and visualizations.")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def safe_read_file(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", on_bad_lines="skip")
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip().replace("/", "_").replace(" ", "_"))
    df = df.rename(columns=lambda c: re.sub(r"[^\w]", "_", str(c).strip()))
    return df

def parse_mp_count(value):
def parse_mp_count(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
@@ -39,6 +75,7 @@ def parse_mp_count(value):
    if s.lower() in ("n/a", "na", "-", "", "nd", "no", "no (nd)", "not detected"):
        return np.nan
    s = s.replace("–", "-").replace("—", "-")
    # handle "12 ± 2" etc
    if "±" in s:
        m = re.search(r"([-+]?\d*\.\d+|\d+)\s*±", s)
        if m:
@@ -55,150 +92,162 @@ def parse_mp_count(value):
    except:
        return np.nan

def safe_read(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", on_bad_lines="skip")
        elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def summarize_cardinality(df):
    cat_info = []
def summarize_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        cat_info.append((c, df[c].dtype, df[c].nunique(dropna=False)))
    return pd.DataFrame(cat_info, columns=["column", "dtype", "nunique"]).sort_values("nunique")

def reduce_high_cardinality(df, cat_cols, threshold=50):
    out = df.copy()
    for c in cat_cols:
        if out[c].nunique() > threshold:
            freqs = out[c].value_counts(normalize=True)
            out[c] = out[c].map(freqs).fillna(0.0)
    return out

def prepare_features(df, selected_features, target_col, task, impute_strategy="mean"):
    # Subset
        rows.append({"column": c, "dtype": str(df[c].dtype), "nunique": int(df[c].nunique(dropna=False))})
    return pd.DataFrame(rows).sort_values("nunique")

def aggregate_small_categories(series: pd.Series, top_n: int = 20, other_name: str = "Other") -> pd.Series:
    vc = series.fillna("NaN").astype(str).value_counts()
    if len(vc) <= top_n:
        return vc
    top = vc.iloc[:top_n]
    other_sum = vc.iloc[top_n:].sum()
    top[other_name] = other_sum
    return top

def prepare_features(
    df: pd.DataFrame,
    selected_features: List[str],
    target_col: Optional[str],
    task: str,
    impute_strategy: str = "mean",
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[LabelEncoder]]:
    df_sub = df[selected_features + ([target_col] if target_col else [])].copy()

    # Drop columns with single unique value
    drop_cols = [c for c in df_sub.columns if df_sub[c].nunique() <= 1]
    # drop columns with single unique value
    drop_cols = [c for c in df_sub.columns if df_sub[c].nunique(dropna=False) <= 1]
    if drop_cols:
        df_sub = df_sub.drop(columns=drop_cols)
        df_sub.drop(columns=drop_cols, inplace=True)
        st.info(f"Dropped non-informative columns: {drop_cols}")

    # Separate
    if target_col and target_col not in df_sub.columns:
        st.error("Selected target column not present after preprocessing.")
        return pd.DataFrame(), None, None

    X = df_sub.drop(columns=[target_col]) if target_col else df_sub.copy()
    y = df_sub[target_col] if target_col else None

    # Convert MP_Presence to binary if present
    if "MP_Presence" in X.columns:
        X["MP_Presence"] = X["MP_Presence"].astype(str).str.lower().map(lambda v: 1 if "yes" in v else (0 if "no" in v or "nd" in v or v.strip()=="" else np.nan))
    # special handling for MP presence-like columns
    for c in X.columns:
        if "mp_presence" in c.lower() or c.lower() == "mp_presence".lower():
            X[c] = X[c].astype(str).str.lower().map(lambda v: 1 if "yes" in v else (0 if "no" in v or "nd" in v or v.strip() == "" else np.nan))

    # Identify object columns
    # reduce very high cardinality for object columns by frequency mapping
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in obj_cols:
        if X[c].nunique() > 40:
            freqs = X[c].value_counts(normalize=True)
            X[c] = X[c].map(freqs).fillna(0.0)

    # Reduce cardinality where needed
    X = reduce_high_cardinality(X, obj_cols, threshold=40)

    # One-hot encode remaining object columns
    # one-hot encode categoricals (safe)
    X = pd.get_dummies(X, drop_first=True)

    # Encode target for classification
    # Encode target if classification
    label_enc = None
    if y is not None and task == "classification":
        if y.dtype == object or str(y.dtype).startswith("category"):
            label_enc = LabelEncoder()
            y = label_enc.fit_transform(y.astype(str))
        else:
            # if numeric but discrete, keep as-is
            y = pd.to_numeric(y, errors="coerce")
    elif y is not None and task == "regression":
        y = pd.to_numeric(y, errors="coerce")

    # Final: handle missing values in X and y
    # Place y as Series and drop rows with NaN target
    if y is not None:
        y = pd.Series(y, index=df_sub.index)
        mask_y_na = y.isna()
        if mask_y_na.any():
            st.warning(f"Dropping {mask_y_na.sum()} rows because target contains NaN.")
            X = X.loc[~mask_y_na].copy()
            y = y.loc[~mask_y_na].copy()

    # Impute numeric features in X (after one-hot)
        if y.isna().any():
            n_drop = y.isna().sum()
            st.warning(f"Dropping {n_drop} rows with missing target values.")
            mask = ~y.isna()
            X = X.loc[mask].copy()
            y = y.loc[mask].copy()

    # Impute numeric cols
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy=impute_strategy)
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # Drop any remaining non-numeric (should be none after get_dummies)
    # Warn & drop any remaining non-numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        st.warning(f"Dropping non-numeric columns after encoding: {non_numeric}")
        X = X.drop(columns=non_numeric)
        X.drop(columns=non_numeric, inplace=True)

    # Scale numeric cols
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
    # Scale numeric features
    num_cols = X.columns.tolist()
    if num_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Remove rows with NaN in features (should be none)
    # Final drop rows with NaNs (should be none)
    mask_x_na = X.isna().any(axis=1)
    if mask_x_na.any():
        st.warning(f"Dropping {mask_x_na.sum()} rows because features contain NaN after imputation.")
        st.warning(f"Dropping {mask_x_na.sum()} rows with NaN after imputation.")
        X = X.loc[~mask_x_na].copy()
        if y is not None:
            y = y.loc[X.index].copy()

    if y is not None:
        X = X.loc[y.index]
        return X, np.asarray(y), label_enc
    return X, None, label_enc

    if y is not None:
        y = np.asarray(y)
def safe_train_test_split(X, y, test_size: float, random_state: int, task: str):
    if y is None:
        return None, None, None, None
    stratify = None
    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) >= 2 and np.min(counts) >= 2:
            stratify = y
        else:
            st.warning("Stratify disabled: some classes have fewer than 2 samples.")
            stratify = None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except Exception as e:
        st.warning(f"train_test_split with stratify failed: {e}. Retrying without stratify.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

    return X, y, label_enc
def has_nan_or_inf(arr) -> bool:
    return np.isnan(arr).any() or np.isinf(arr).any()

# -------------------------
# Upload / load dataset
# Sidebar: upload & preprocessing
# -------------------------
st.sidebar.header("1) Upload dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel (Data1_Microplastic.csv example)", type=["csv", "xls", "xlsx"])
st.sidebar.header("1) Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel (e.g., Data1_Microplastic.csv)", type=["csv", "xls", "xlsx"])
if not uploaded_file:
    st.info("Please upload your dataset file.")
    st.info("Please upload your dataset file to proceed.")
    st.stop()

df_raw = safe_read(uploaded_file)
df_raw = safe_read_file(uploaded_file)
if df_raw is None:
    st.stop()

st.subheader("Raw dataset preview (first 50 rows)")
st.write(f"Filename: {uploaded_file.name} — shape: {df_raw.shape}")
st.dataframe(df_raw.head(50))

st.sidebar.markdown(f"**File:** {uploaded_file.name} — rows: {df_raw.shape[0]} cols: {df_raw.shape[1]}")
df_raw = clean_column_names(df_raw)
st.subheader("Data preview (first 50 rows)")
st.dataframe(df_raw.head(50))

# -------------------------
# Special cleaning for MP_Count-like columns
# -------------------------
# Parse MP count candidate
mp_count_candidates = [c for c in df_raw.columns if "mp_count" in c.lower() or "count" in c.lower() or "items" in c.lower()]
mp_count_col = None
mp_count_candidates = [c for c in df_raw.columns if "MP_Count" in c or "count" in c.lower() or "items" in c.lower()]
if mp_count_candidates:
    mp_count_col = st.sidebar.selectbox("Select column to parse as numeric MP count (optional)", [""] + mp_count_candidates)
    mp_count_col = st.sidebar.selectbox("Column to parse as numeric MP count (optional)", [""] + mp_count_candidates)
else:
    mp_count_col = st.sidebar.selectbox("Select column to parse as numeric MP count (optional)", [""] + df_raw.columns.tolist())

    mp_count_col = st.sidebar.selectbox("Column to parse as numeric MP count (optional)", [""] + df_raw.columns.tolist())
if mp_count_col:
    parsed = df_raw[mp_count_col].apply(parse_mp_count)
    df_raw["_MP_Count_parsed"] = parsed
    st.write(f"Parsed numeric MP_Count saved to column _MP_Count_parsed (n_non-null = {parsed.notna().sum()})")
    st.dataframe(df_raw[[mp_count_col, "_MP_Count_parsed"]].head(20))
    df_raw["_MP_Count_parsed"] = df_raw[mp_count_col].apply(parse_mp_count)
    st.write(f"Parsed MP count saved to '_MP_Count_parsed' — non-null: {df_raw['_MP_Count_parsed'].notna().sum()}")

# -------------------------
# Preprocessing options
# -------------------------
st.sidebar.header("2) Preprocessing options")
drop_na = st.sidebar.checkbox("Drop rows with any missing values (otherwise numeric imputation applied)", value=False)
impute_strategy = st.sidebar.selectbox("Imputation strategy for numeric columns", ["mean", "median", "most_frequent"], index=0)
# Preprocessing controls
st.sidebar.header("2) Preprocessing")
drop_na = st.sidebar.checkbox("Drop rows with any missing values", value=False)
impute_strategy = st.sidebar.selectbox("Numeric imputation strategy", ["mean", "median", "most_frequent"], index=0)

if drop_na:
    before = len(df_raw)
@@ -211,123 +260,98 @@ def prepare_features(df, selected_features, target_col, task, impute_strategy="m
        imputer = SimpleImputer(strategy=impute_strategy)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if df[c].isna().sum():
        if df[c].isna().any():
            mode = df[c].mode()
            fill = mode.iloc[0] if not mode.empty else ""
            df[c] = df[c].fillna(fill)

st.write("✅ Preprocessing complete")
st.write(f"Cleaned dataset shape: {df.shape}")
st.dataframe(df.head(20))
st.write("✅ Preprocessing finished")
st.write(f"Dataset shape after cleaning: {df.shape}")

# -------------------------
# Modeling task selection
# Modeling selection
# -------------------------
st.sidebar.header("3) Modeling")
st.sidebar.header("3) Modeling & Features")
task = st.sidebar.selectbox("Task", ("classification", "regression", "clustering"))

st.sidebar.markdown("Target column (for classification/regression)")
target_col = st.sidebar.selectbox("Pick target column (leave blank for clustering/unsupervised)", [""] + df.columns.tolist())
if task in ("classification", "regression") and (target_col == "" or target_col is None):
    st.sidebar.warning("Please select a target column for supervised tasks (e.g., _MP_Count_parsed for regression or Risk_Level for classification).")
    st.stop()
target_col = ""
if task != "clustering":
    target_col = st.sidebar.selectbox("Target column (for supervised tasks)", [""] + df.columns.tolist())
    if target_col == "" or target_col is None:
        st.sidebar.warning("Please select a target column for supervised tasks.")
        st.stop()

all_cols = df.columns.tolist()
default_features = [c for c in all_cols if c != target_col]
selected_features = st.sidebar.multiselect("Select features (columns) to use", all_cols, default=default_features)
selected_features = st.sidebar.multiselect("Select feature columns (at least one)", all_cols, default=default_features)
if not selected_features:
    st.sidebar.error("Select at least one feature column.")
    st.stop()

# Prepare X, y with robust imputation and NaN handling
X, y, label_enc = prepare_features(df, selected_features, target_col if target_col else None, task, impute_strategy=impute_strategy)
st.write("Prepared features and target")
st.write(f"X shape: {X.shape}")
if y is not None:
    st.write(f"y shape: {y.shape} | unique values: {np.unique(y)[:20]}")
st.write(f"Prepared features: X shape {X.shape}" + (f", y shape {y.shape}" if y is not None else ""))

# Guard: if X has zero columns after preprocessing
if X.shape[1] == 0:
    st.error("No usable features after preprocessing (all dropped or encoded away). Please select different features.")
    st.error("No usable features after preprocessing. Please revise your feature selection.")
    st.stop()

# -------------------------
# Safe train/test split with stratify checks
# -------------------------
st.sidebar.markdown("Train/test split params")
test_size = st.sidebar.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
# Train/test split options
st.sidebar.header("4) Train/Test split")
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

def safe_train_test_split(X, y, test_size, random_state, task):
    if y is None:
        return None, None, None, None
    stratify = None
    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) >= 2 and np.min(counts) >= 2:
            stratify = y
        else:
            st.warning("Stratify disabled: target has too few classes or some classes have <2 samples.")
            stratify = None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError as e:
        st.warning(f"train_test_split with stratify failed: {e}. Retrying without stratify.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

if task in ("classification", "regression"):
    if y is None or len(y) == 0:
        st.error("No target values available after preprocessing — cannot train.")
        st.stop()
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size, random_state, task)
    st.write(f"Train shape: {X_train.shape} - Test shape: {X_test.shape}")
    st.write(f"Train shape: {X_train.shape} — Test shape: {X_test.shape}")
else:
    X_train = X_test = y_train = y_test = None

# -------------------------
# Models and evaluation
# Modeling & evaluation
# -------------------------
st.header("Model training & evaluation")

def has_nan_or_inf(arr):
    return np.isnan(arr).any() or np.isinf(arr).any()

if task == "classification":
    if y_train is None or len(np.unique(y_train)) < 2:
    unique_classes = np.unique(y_train) if y_train is not None else []
    if y_train is None or len(unique_classes) < 2:
        st.error("Classification requires at least 2 classes with enough samples.")
    else:
        # Define classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(max_iter=500)
            "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs")
        }
        results = {}
        metrics = {}
        for name, clf in classifiers.items():
            try:
                if has_nan_or_inf(X_train.values) or has_nan_or_inf(y_train):
                    raise ValueError("Training data contains NaN/inf. Ensure imputation before fitting.")
                if has_nan_or_inf(X_train.values) or has_nan_or_inf(np.asarray(y_train)):
                    raise ValueError("Training data contains NaN/inf. Ensure imputation.")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                results[name] = {
                metrics[name] = {
                    "Accuracy": float(accuracy_score(y_test, y_pred)),
                    "Precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                    "F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                }
            except Exception as e:
                results[name] = {"error": str(e)}
                metrics[name] = {"error": str(e)}
        st.subheader("Classification results (test set)")
        st.table(pd.DataFrame(results).T)
        st.table(pd.DataFrame(metrics).T)

        good_models = {k: v for k, v in results.items() if "F1" in v}
        if good_models:
            best = max(good_models.items(), key=lambda t: t[1]["F1"])[0]
        # show confusion matrix for best model
        valid = {k: v for k, v in metrics.items() if "F1" in v}
        if valid:
            best = max(valid.items(), key=lambda t: t[1]["F1"])[0]
            st.write(f"Best model by F1: {best}")
            try:
                model = classifiers[best]
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix — {best}")
                ax.set_xlabel("Predicted")
@@ -336,19 +360,20 @@ def has_nan_or_inf(arr):
                st.text("Classification report:")
                st.text(classification_report(y_test, y_pred, zero_division=0))
            except Exception as e:
                st.warning(f"Could not show confusion matrix: {e}")
                st.warning(f"Could not display confusion matrix: {e}")

elif task == "regression":
    if y_train is None:
        st.error("Regression requires a numeric target column.")
        st.error("Regression requires a numeric target.")
    else:
        # Impute if necessary
        if has_nan_or_inf(X_train.values):
            st.warning("Found NaNs or Infs in feature matrix before training — applying numeric imputation.")
            st.warning("Imputing NaNs in features before training.")
            imputer = SimpleImputer(strategy=impute_strategy)
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        if np.isnan(y_train).any():
            st.warning("Found NaNs in target vector — dropping rows with NaN target.")
            st.warning("Dropping rows with NaN in target for training.")
            mask = ~np.isnan(y_train)
            X_train = X_train.loc[mask]
            y_train = y_train[mask]
@@ -358,50 +383,49 @@ def has_nan_or_inf(arr):
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
            "Linear Regression": LinearRegression()
        }
        results = {}
        metrics = {}
        for name, reg in regressors.items():
            try:
                if X_train.shape[0] < 2:
                    raise ValueError("Not enough training samples for regression.")
                if has_nan_or_inf(X_train.values) or np.isnan(y_train).any():
                    raise ValueError("Training data contains NaN/inf. Impute or drop missing values.")
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                results[name] = {
                metrics[name] = {
                    "R2": float(r2_score(y_test, y_pred)),
                    "MAE": float(mean_absolute_error(y_test, y_pred)),
                    "MSE": float(mean_squared_error(y_test, y_pred))
                }
            except Exception as e:
                results[name] = {"error": str(e)}
                metrics[name] = {"error": str(e)}
        st.subheader("Regression results (test set)")
        st.table(pd.DataFrame(results).T)
        st.table(pd.DataFrame(metrics).T)

elif task == "clustering":
    st.subheader("Clustering (unsupervised)")
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, min(10, max(2, X.shape[0])), 3)
    max_k = max(2, min(10, X.shape[0] - 1)) if X.shape[0] > 2 else 2
    n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=max_k, value=min(3, max_k))
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
        if X.shape[1] >= 2:
        if X.shape[1] >= 2 and X.shape[0] >= n_clusters:
            try:
                sil = silhouette_score(X, labels)
                st.write(f"Silhouette score: {sil:.4f}")
            except Exception as e:
                st.warning(f"Could not compute silhouette score: {e}")
            fig, ax = plt.subplots()
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="tab10", alpha=0.7)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="tab10", alpha=0.8)
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            st.pyplot(fig)
        else:
            st.info("Not enough numeric features for scatter plot.")
            st.info("Not enough dimensions or samples for scatter plot.")
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# -------------------------
# Cross-validation (K-Fold) with protections and smarter fallbacks
# Cross-validation
# -------------------------
st.header("Cross-validation")
requested_splits = int(st.sidebar.number_input("K-Fold splits", min_value=2, max_value=20, value=5))
@@ -410,7 +434,7 @@ def has_nan_or_inf(arr):
if task in ("classification", "regression") and y is not None:
    # Prepare X_cv, y_cv: impute X if needed, drop NaN targets
    if has_nan_or_inf(X.values) or np.isnan(y).any():
        st.warning("Imputing any remaining NaNs in features and dropping NaNs in target before cross-validation.")
        st.warning("Imputing remaining NaNs in features and dropping NaNs in target before CV.")
        imputer = SimpleImputer(strategy=impute_strategy)
        X_cv = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        mask = ~np.isnan(y)
@@ -420,13 +444,10 @@ def has_nan_or_inf(arr):
        X_cv = X.copy()
        y_cv = np.asarray(y)

    # Guard: ensure there are enough samples
    n_samples = X_cv.shape[0]
    if n_samples < 2:
        cv_results = {"error": f"Not enough samples ({n_samples}) for cross-validation."}
        st.write(cv_results)
    else:
        # Adjust n_splits to dataset size
        n_splits = min(requested_splits, n_samples)
        if n_splits < 2:
            n_splits = 2
@@ -436,34 +457,26 @@ def has_nan_or_inf(arr):
            n_classes = len(unique)
            min_count = int(np.min(counts)) if len(counts) > 0 else 0

            # Decide strategy
            if n_classes < 2:
                cv_results = {"error": "Cross-validation requires at least 2 classes."}
            else:
                # Prefer StratifiedKFold when every class has >= n_splits samples
                if min_count >= n_splits:
                    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    st.info(f"Using StratifiedKFold with n_splits={n_splits} (min samples per class = {min_count}).")
                else:
                    # Fallbacks:
                    # 1) Try reducing n_splits to min_count (if min_count >= 2)
                    # try reduce n_splits to min_count if possible
                    if min_count >= 2:
                        n_splits_reduced = min(n_splits, min_count)
                        cv_strategy = StratifiedKFold(n_splits=n_splits_reduced, shuffle=True, random_state=random_state)
                        st.warning(f"Reduced n_splits to {n_splits_reduced} to satisfy stratification (min samples/class = {min_count}).")
                        st.warning(f"Reduced n_splits to {n_splits_reduced} for stratification.")
                    else:
                        # 2) As last resort use regular KFold (no stratify) with n_splits adjusted to sample size
                        n_splits_kfold = min(n_splits, max(2, n_samples // 2))
                        cv_strategy = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state)
                        st.warning(
                            "Stratified cross-validation not possible: at least one class has fewer than 2 samples. "
                            f"Falling back to KFold (no stratification) with n_splits={n_splits_kfold}."
                        )
                        st.warning("Using KFold (no stratification) due to very small class counts.")

                models_for_cv = {
                    "Random Forest": RandomForestClassifier(random_state=random_state),
                    "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                    "Logistic Regression": LogisticRegression(max_iter=500)
                    "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs")
                }
                for name, model in models_for_cv.items():
                    try:
@@ -473,7 +486,6 @@ def has_nan_or_inf(arr):
                        cv_results[name] = {"error": str(e)}

        else:  # regression
            # For regression, KFold is fine; adjust n_splits to sample size
            n_splits_reg = min(n_splits, max(2, n_samples // 2))
            if n_splits_reg < 2:
                n_splits_reg = 2
@@ -495,24 +507,53 @@ def has_nan_or_inf(arr):
st.write(cv_results)

# -------------------------
# Visualizations
# Visualizations (improved readability)
# -------------------------
st.header("Visualizations")

# 1) Distribution of a selected categorical column with aggregation for readability
st.subheader("Categorical distribution (top categories aggregated)")
cat_col = st.selectbox("Select a categorical column to display", options=[c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name.startswith("category")] + [""])
if cat_col:
    top_n = st.slider("Top N categories to show (others -> Other)", min_value=5, max_value=50, value=15)
    series = df[cat_col].fillna("NaN").astype(str)
    agg = aggregate_small_categories(series, top_n=top_n)
    # horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, min(6, 0.25 * len(agg))))
    agg.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("tab10", n_colors=len(agg)))
    ax.set_xlabel("Count")
    ax.set_ylabel(cat_col)
    ax.set_title(f"{cat_col} distribution (top {top_n})")
    for i, v in enumerate(agg.sort_values()):
        ax.text(v + max(agg.max()*0.01, 1e-6), i, str(int(v)), va="center")
    st.pyplot(fig)

# 2) Risk_Level if present (improved)
if "Risk_Level" in df.columns:
    fig, ax = plt.subplots()
    df["Risk_Level"].value_counts().plot(kind="bar", ax=ax)
    st.subheader("Risk_Level distribution (improved)")
    rl = df["Risk_Level"].fillna("NaN").astype(str)
    agg_rl = aggregate_small_categories(rl, top_n=20)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(agg_rl))))
    agg_rl.sort_values().plot(kind="barh", ax=ax, color="salmon")
    ax.set_xlabel("Count")
    ax.set_ylabel("Risk_Level")
    ax.set_title("Risk_Level distribution")
    st.pyplot(fig)

if X.shape[1] <= 6:
# 3) Pairplot (numeric features only) capped to avoid clutter
st.subheader("Pairplot (numeric features subset)")
numeric_cols = [c for c in X.columns]
if numeric_cols:
    max_features = st.number_input("Max numeric features to include in pairplot", min_value=2, max_value=8, value=min(6, len(numeric_cols)))
    sel = numeric_cols[:int(max_features)]
    try:
        temp = X.copy()
        if y is not None:
            temp["_target"] = y
        sns_plot = sns.pairplot(temp.sample(n=min(len(temp), 200)))
        sample_n = min(len(X), 300)
        df_pair = pd.DataFrame(X[sel]).sample(n=sample_n, random_state=42)
        sns_plot = sns.pairplot(df_pair, corner=True, plot_kws={"s": 20, "alpha": 0.6})
        st.pyplot(sns_plot.fig)
    except Exception as e:
        st.warning(f"Could not create pairplot: {e}")
else:
    st.info("No numeric features available for pairplot.")

st.success("Processing finished. Cross-validation selection now adapts to small/imbalanced class counts and falls back safely when stratified CV isn't possible.")

st.success("Finished. Visualizations are aggregated and rotated to improve readability for long/crowded labels.")
