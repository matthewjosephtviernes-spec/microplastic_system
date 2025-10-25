import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
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

st.set_page_config(page_title="Microplastic Risk Prediction System (robust)", layout="wide")
st.title("Microplastic Risk Prediction System")

# -------------------------
# Helpers
# -------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip().replace("/", "_").replace(" ", "_"))
    return df

def parse_mp_count(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.lower() in ("n/a", "na", "-", "", "nd", "no", "no (nd)", "not detected"):
        return np.nan
    s = s.replace("–", "-").replace("—", "-")
    if "±" in s:
        m = re.search(r"([-+]?\d*\.\d+|\d+)\s*±", s)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if not nums:
        return np.nan
    try:
        nums_f = [float(n) for n in nums]
        return float(np.mean(nums_f))
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
    df_sub = df[selected_features + ([target_col] if target_col else [])].copy()

    # Drop columns with single unique value
    drop_cols = [c for c in df_sub.columns if df_sub[c].nunique() <= 1]
    if drop_cols:
        df_sub = df_sub.drop(columns=drop_cols)
        st.info(f"Dropped non-informative columns: {drop_cols}")

    # Separate
    X = df_sub.drop(columns=[target_col]) if target_col else df_sub.copy()
    y = df_sub[target_col] if target_col else None

    # Convert MP_Presence to binary if present
    if "MP_Presence" in X.columns:
        X["MP_Presence"] = X["MP_Presence"].astype(str).str.lower().map(lambda v: 1 if "yes" in v else (0 if "no" in v or "nd" in v or v.strip()=="" else np.nan))

    # Identify object columns
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Reduce cardinality where needed
    X = reduce_high_cardinality(X, obj_cols, threshold=40)

    # One-hot encode remaining object columns
    X = pd.get_dummies(X, drop_first=True)

    # Encode target for classification
    label_enc = None
    if y is not None and task == "classification":
        if y.dtype == object or str(y.dtype).startswith("category"):
            label_enc = LabelEncoder()
            y = label_enc.fit_transform(y.astype(str))
    elif y is not None and task == "regression":
        y = pd.to_numeric(y, errors="coerce")

    # Final: handle missing values in X and y
    if y is not None:
        y = pd.Series(y, index=df_sub.index)
        mask_y_na = y.isna()
        if mask_y_na.any():
            st.warning(f"Dropping {mask_y_na.sum()} rows because target contains NaN.")
            X = X.loc[~mask_y_na].copy()
            y = y.loc[~mask_y_na].copy()

    # Impute numeric features in X (after one-hot)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy=impute_strategy)
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # Drop any remaining non-numeric (should be none after get_dummies)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        st.warning(f"Dropping non-numeric columns after encoding: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Scale numeric cols
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Remove rows with NaN in features (should be none)
    mask_x_na = X.isna().any(axis=1)
    if mask_x_na.any():
        st.warning(f"Dropping {mask_x_na.sum()} rows because features contain NaN after imputation.")
        X = X.loc[~mask_x_na].copy()
        if y is not None:
            y = y.loc[X.index].copy()

    if y is not None:
        X = X.loc[y.index]

    if y is not None:
        y = np.asarray(y)

    return X, y, label_enc

# -------------------------
# Upload / load dataset
# -------------------------
st.sidebar.header("1) Upload dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel (Data1_Microplastic.csv example)", type=["csv", "xls", "xlsx"])
if not uploaded_file:
    st.info("Please upload your dataset file.")
    st.stop()

df_raw = safe_read(uploaded_file)
if df_raw is None:
    st.stop()

st.subheader("Raw dataset preview (first 50 rows)")
st.write(f"Filename: {uploaded_file.name} — shape: {df_raw.shape}")
st.dataframe(df_raw.head(50))

df_raw = clean_column_names(df_raw)

# -------------------------
# Special cleaning for MP_Count-like columns
# -------------------------
mp_count_col = None
mp_count_candidates = [c for c in df_raw.columns if "MP_Count" in c or "count" in c.lower() or "items" in c.lower()]
if mp_count_candidates:
    mp_count_col = st.sidebar.selectbox("Select column to parse as numeric MP count (optional)", [""] + mp_count_candidates)
else:
    mp_count_col = st.sidebar.selectbox("Select column to parse as numeric MP count (optional)", [""] + df_raw.columns.tolist())

if mp_count_col:
    parsed = df_raw[mp_count_col].apply(parse_mp_count)
    df_raw["_MP_Count_parsed"] = parsed
    st.write(f"Parsed numeric MP_Count saved to column _MP_Count_parsed (n_non-null = {parsed.notna().sum()})")
    st.dataframe(df_raw[[mp_count_col, "_MP_Count_parsed"]].head(20))

# -------------------------
# Preprocessing options
# -------------------------
st.sidebar.header("2) Preprocessing options")
drop_na = st.sidebar.checkbox("Drop rows with any missing values (otherwise numeric imputation applied)", value=False)
impute_strategy = st.sidebar.selectbox("Imputation strategy for numeric columns", ["mean", "median", "most_frequent"], index=0)

if drop_na:
    before = len(df_raw)
    df = df_raw.dropna()
    st.write(f"Dropped rows with missing values: {before} -> {len(df)}")
else:
    df = df_raw.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        imputer = SimpleImputer(strategy=impute_strategy)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if df[c].isna().sum():
            mode = df[c].mode()
            fill = mode.iloc[0] if not mode.empty else ""
            df[c] = df[c].fillna(fill)

st.write("✅ Preprocessing complete")
st.write(f"Cleaned dataset shape: {df.shape}")
st.dataframe(df.head(20))

# -------------------------
# Modeling task selection
# -------------------------
st.sidebar.header("3) Modeling")
task = st.sidebar.selectbox("Task", ("classification", "regression", "clustering"))

st.sidebar.markdown("Target column (for classification/regression)")
target_col = st.sidebar.selectbox("Pick target column (leave blank for clustering/unsupervised)", [""] + df.columns.tolist())
if task in ("classification", "regression") and (target_col == "" or target_col is None):
    st.sidebar.warning("Please select a target column for supervised tasks (e.g., _MP_Count_parsed for regression or Risk_Level for classification).")
    st.stop()

all_cols = df.columns.tolist()
default_features = [c for c in all_cols if c != target_col]
selected_features = st.sidebar.multiselect("Select features (columns) to use", all_cols, default=default_features)
if not selected_features:
    st.sidebar.error("Select at least one feature column.")
    st.stop()

# Prepare X, y with robust imputation and NaN handling
X, y, label_enc = prepare_features(df, selected_features, target_col if target_col else None, task, impute_strategy=impute_strategy)
st.write("Prepared features and target")
st.write(f"X shape: {X.shape}")
if y is not None:
    st.write(f"y shape: {y.shape} | unique values: {np.unique(y)[:20]}")

# Guard: if X has zero columns after preprocessing
if X.shape[1] == 0:
    st.error("No usable features after preprocessing (all dropped or encoded away). Please select different features.")
    st.stop()

# -------------------------
# Safe train/test split with stratify checks
# -------------------------
st.sidebar.markdown("Train/test split params")
test_size = st.sidebar.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
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
else:
    X_train = X_test = y_train = y_test = None

# -------------------------
# Models and evaluation
# -------------------------
st.header("Model training & evaluation")

def has_nan_or_inf(arr):
    return np.isnan(arr).any() or np.isinf(arr).any()

if task == "classification":
    if y_train is None or len(np.unique(y_train)) < 2:
        st.error("Classification requires at least 2 classes with enough samples.")
    else:
        classifiers = {
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(max_iter=500)
        }
        results = {}
        for name, clf in classifiers.items():
            try:
                if has_nan_or_inf(X_train.values) or has_nan_or_inf(y_train):
                    raise ValueError("Training data contains NaN/inf. Ensure imputation before fitting.")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                results[name] = {
                    "Accuracy": float(accuracy_score(y_test, y_pred)),
                    "Precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                    "F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        st.subheader("Classification results (test set)")
        st.table(pd.DataFrame(results).T)

        good_models = {k: v for k, v in results.items() if "F1" in v}
        if good_models:
            best = max(good_models.items(), key=lambda t: t[1]["F1"])[0]
            st.write(f"Best model by F1: {best}")
            try:
                model = classifiers[best]
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix — {best}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                st.text("Classification report:")
                st.text(classification_report(y_test, y_pred, zero_division=0))
            except Exception as e:
                st.warning(f"Could not show confusion matrix: {e}")

elif task == "regression":
    if y_train is None:
        st.error("Regression requires a numeric target column.")
    else:
        if has_nan_or_inf(X_train.values):
            st.warning("Found NaNs or Infs in feature matrix before training — applying numeric imputation.")
            imputer = SimpleImputer(strategy=impute_strategy)
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        if np.isnan(y_train).any():
            st.warning("Found NaNs in target vector — dropping rows with NaN target.")
            mask = ~np.isnan(y_train)
            X_train = X_train.loc[mask]
            y_train = y_train[mask]

        regressors = {
            "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
            "Linear Regression": LinearRegression()
        }
        results = {}
        for name, reg in regressors.items():
            try:
                if X_train.shape[0] < 2:
                    raise ValueError("Not enough training samples for regression.")
                if has_nan_or_inf(X_train.values) or np.isnan(y_train).any():
                    raise ValueError("Training data contains NaN/inf. Impute or drop missing values.")
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                results[name] = {
                    "R2": float(r2_score(y_test, y_pred)),
                    "MAE": float(mean_absolute_error(y_test, y_pred)),
                    "MSE": float(mean_squared_error(y_test, y_pred))
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        st.subheader("Regression results (test set)")
        st.table(pd.DataFrame(results).T)

elif task == "clustering":
    st.subheader("Clustering (unsupervised)")
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, min(10, max(2, X.shape[0])), 3)
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
        if X.shape[1] >= 2:
            try:
                sil = silhouette_score(X, labels)
                st.write(f"Silhouette score: {sil:.4f}")
            except Exception as e:
                st.warning(f"Could not compute silhouette score: {e}")
            fig, ax = plt.subplots()
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="tab10", alpha=0.7)
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            st.pyplot(fig)
        else:
            st.info("Not enough numeric features for scatter plot.")
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# -------------------------
# Cross-validation (K-Fold) with protections and smarter fallbacks
# -------------------------
st.header("Cross-validation")
requested_splits = int(st.sidebar.number_input("K-Fold splits", min_value=2, max_value=20, value=5))
cv_results = {}

if task in ("classification", "regression") and y is not None:
    # Prepare X_cv, y_cv: impute X if needed, drop NaN targets
    if has_nan_or_inf(X.values) or np.isnan(y).any():
        st.warning("Imputing any remaining NaNs in features and dropping NaNs in target before cross-validation.")
        imputer = SimpleImputer(strategy=impute_strategy)
        X_cv = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        mask = ~np.isnan(y)
        X_cv = X_cv.loc[mask]
        y_cv = np.asarray(y)[mask]
    else:
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

        if task == "classification":
            unique, counts = np.unique(y_cv, return_counts=True)
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
                    if min_count >= 2:
                        n_splits_reduced = min(n_splits, min_count)
                        cv_strategy = StratifiedKFold(n_splits=n_splits_reduced, shuffle=True, random_state=random_state)
                        st.warning(f"Reduced n_splits to {n_splits_reduced} to satisfy stratification (min samples/class = {min_count}).")
                    else:
                        # 2) As last resort use regular KFold (no stratify) with n_splits adjusted to sample size
                        n_splits_kfold = min(n_splits, max(2, n_samples // 2))
                        cv_strategy = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state)
                        st.warning(
                            "Stratified cross-validation not possible: at least one class has fewer than 2 samples. "
                            f"Falling back to KFold (no stratification) with n_splits={n_splits_kfold}."
                        )

                models_for_cv = {
                    "Random Forest": RandomForestClassifier(random_state=random_state),
                    "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                    "Logistic Regression": LogisticRegression(max_iter=500)
                }
                for name, model in models_for_cv.items():
                    try:
                        scores = cross_val_score(model, X_cv, y_cv, cv=cv_strategy, scoring="accuracy")
                        cv_results[name] = {"mean_accuracy": float(scores.mean()), "std": float(scores.std())}
                    except Exception as e:
                        cv_results[name] = {"error": str(e)}

        else:  # regression
            # For regression, KFold is fine; adjust n_splits to sample size
            n_splits_reg = min(n_splits, max(2, n_samples // 2))
            if n_splits_reg < 2:
                n_splits_reg = 2
            kfold = KFold(n_splits=n_splits_reg, shuffle=True, random_state=random_state)
            models_for_cv = {
                "RF Regressor": RandomForestRegressor(random_state=random_state),
                "DT Regressor": DecisionTreeRegressor(random_state=random_state),
                "LinearReg": LinearRegression()
            }
            for name, model in models_for_cv.items():
                try:
                    scores = cross_val_score(model, X_cv, y_cv, cv=kfold, scoring="r2")
                    cv_results[name] = {"mean_r2": float(scores.mean()), "std": float(scores.std())}
                except Exception as e:
                    cv_results[name] = {"error": str(e)}
else:
    cv_results = {"note": "Cross-validation not applicable for unsupervised task or missing target."}

st.write(cv_results)

# -------------------------
# Visualizations
# -------------------------
st.header("Visualizations")
if "Risk_Level" in df.columns:
    fig, ax = plt.subplots()
    df["Risk_Level"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Risk_Level distribution")
    st.pyplot(fig)

if X.shape[1] <= 6:
    try:
        temp = X.copy()
        if y is not None:
            temp["_target"] = y
        sns_plot = sns.pairplot(temp.sample(n=min(len(temp), 200)))
        st.pyplot(sns_plot.fig)
    except Exception as e:
        st.warning(f"Could not create pairplot: {e}")

st.success("Processing finished. Cross-validation selection now adapts to small/imbalanced class counts and falls back safely when stratified CV isn't possible.")

