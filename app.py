#!/usr/bin/env python3
"""
Microplastic Risk Prediction System - Streamlit app (full app.py)

This version includes:
- File upload (CSV / Excel)
- Robust parsing of MP count text values into numeric (_MP_Count_parsed)
- Optional Risk_Level creation from parsed counts (Low/Medium/High)
- Defensive preprocessing: coercion of numeric-like strings, imputation, one-hot encoding,
  frequency encoding for very high cardinality columns, dropping all-NaN numeric columns
- Safe train/test splitting with stratify checks & fallbacks
- Classification, Regression and Clustering flows with multiple models
- Adaptive K-Fold cross-validation with StratifiedKFold if possible, else fallback to KFold or reduced splits
- Extensive warnings and user controls in sidebar
"""

import io
import re
import sys
import traceback
from typing import Optional, Tuple, List

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

# Streamlit page configuration
st.set_page_config(page_title="Microplastic Risk Prediction System", layout="wide")
st.title("Microplastic Risk Prediction System")

# -------------------------
# Helper functions
# -------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip().replace("/", "_").replace(" ", "_"))

def parse_mp_count(value) -> float:
    """Parse textual MP count values into a numeric (mean of numbers found).
    Handles ranges "0.3–2.5", ± notation "27.3 ± 6.5", single numbers, "~32.5", etc.
    Returns np.nan when parsing fails or value indicates missing/ND.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    lowec = s.lower()
    if any(token in lowec for token in ("n/a", "na", "nd", "not detected", "no", "no (nd)")):
        return np.nan
    # normalize dashes
    s = s.replace("–", "-").replace("—", "-")
    # ± pattern -> take the main number before ±
    if "±" in s:
        m = re.search(r"([-+]?\d*\.\d+|\d+)\s*±", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    # extract numbers
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if not nums:
        return np.nan
    try:
        nums_f = [float(n) for n in nums]
        return float(np.mean(nums_f))
    except Exception:
        return np.nan

def safe_read_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Read uploaded CSV or Excel robustly."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            # Use python engine to be tolerant of irregularities
            return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", on_bad_lines="skip")
        elif filename.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Upload .csv or .xls/.xlsx")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.exception(e)
        return None

def reduce_high_cardinality(df: pd.DataFrame, cat_cols: List[str], threshold: int = 50) -> pd.DataFrame:
    """Frequency-encode columns with cardinality above threshold."""
    out = df.copy()
    for c in cat_cols:
        if out[c].nunique(dropna=False) > threshold:
            freqs = out[c].value_counts(normalize=True)
            out[c] = out[c].map(freqs).fillna(0.0)
    return out

def has_nan_or_inf(arr: np.ndarray) -> bool:
    return np.isnan(arr).any() or np.isinf(arr).any()

def prepare_features(
    df: pd.DataFrame,
    selected_features: List[str],
    target_col: Optional[str],
    task: str,
    impute_strategy: str = "mean",
    high_card_threshold: int = 40,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[LabelEncoder]]:
    """
    Prepares feature matrix X and target y:
      - Subset selected features (+ target)
      - Convert MP_Presence to binary if present
      - Frequency-encode very high-cardinality categoricals
      - One-hot encode remaining categoricals
      - Try to coerce object-like numeric columns to numeric
      - Drop numeric columns that are all NaN (imputer can't fit them)
      - Impute numeric columns
      - Scale numeric columns
      - Align and return X (DataFrame), y (numpy array or None), label_encoder if used
    """
    df_sub = df[selected_features + ([target_col] if target_col else [])].copy()

    # Drop non-informative columns (single unique value)
    drop_cols = [c for c in df_sub.columns if df_sub[c].nunique(dropna=False) <= 1]
    if drop_cols:
        df_sub = df_sub.drop(columns=drop_cols)
        st.info(f"Dropped non-informative columns: {drop_cols}")

    X = df_sub.drop(columns=[target_col]) if target_col else df_sub.copy()
    y = df_sub[target_col] if target_col else None

    # Convert MP_Presence to binary 1/0/NaN
    if "MP_Presence" in X.columns:
        X["MP_Presence"] = X["MP_Presence"].astype(str).str.lower().map(
            lambda v: 1 if "yes" in v else (0 if "no" in v or "nd" in v or v.strip() == "" else np.nan)
        )

    # Frequency-encode high-cardinality categorical columns
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = reduce_high_cardinality(X, obj_cols, threshold=high_card_threshold)

    # One-hot encode remaining categorical/object columns
    X = pd.get_dummies(X, drop_first=True)

    # Attempt to coerce object-like columns to numeric when possible (strip commas, units)
    coerced = []
    for col in X.columns:
        if X[col].dtype == object:
            cleaned = X[col].astype(str).str.replace(r"[,\(\)%\s]+", "", regex=True)
            coerced_series = pd.to_numeric(cleaned.replace("", np.nan), errors="coerce")
            if coerced_series.notna().sum() > 0:
                X[col] = coerced_series
                coerced.append(col)
    if coerced:
        st.info(f"Coerced object columns to numeric where possible: {coerced}")

    # Encode target if classification
    label_enc = None
    if y is not None and task == "classification":
        if y.dtype == object or str(y.dtype).startswith("category"):
            label_enc = LabelEncoder()
            y = label_enc.fit_transform(y.astype(str))
    elif y is not None and task == "regression":
        y = pd.to_numeric(y, errors="coerce")

    # If target exists remove rows where y is NaN
    if y is not None:
        y = pd.Series(y, index=df_sub.index)
        mask_y_na = y.isna()
        if mask_y_na.any():
            count_drop = int(mask_y_na.sum())
            st.warning(f"Dropping {count_drop} rows because target contains NaN.")
            X = X.loc[~mask_y_na].copy()
            y = y.loc[~mask_y_na].copy()

    # Numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Drop numeric columns that are all NaN (imputer can't fit)
    allnan_numeric = [c for c in numeric_cols if X[c].isna().all()]
    if allnan_numeric:
        st.warning(f"Dropping numeric columns that are entirely NaN: {allnan_numeric}")
        X = X.drop(columns=allnan_numeric)
        numeric_cols = [c for c in numeric_cols if c not in allnan_numeric]

    # Impute numeric columns if present
    if numeric_cols:
        imputer = SimpleImputer(strategy=impute_strategy)
        try:
            X_numeric = X[numeric_cols]
            X[numeric_cols] = imputer.fit_transform(X_numeric)
        except Exception as exc:
            st.error(f"Numeric imputation failed: {exc}")
            raise

    # Drop any remaining non-numeric columns (should be none)
    non_numeric_remaining = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_remaining:
        st.warning(f"Dropping non-numeric columns after encoding/coercion: {non_numeric_remaining}")
        X = X.drop(columns=non_numeric_remaining)

    # Scale numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Drop rows with NaNs in features (protection)
    mask_x_na = X.isna().any(axis=1)
    if mask_x_na.any():
        st.warning(f"Dropping {int(mask_x_na.sum())} rows because features still contain NaN after imputation.")
        X = X.loc[~mask_x_na].copy()
        if y is not None:
            y = y.loc[X.index].copy()

    # Ensure X, y align and return
    if y is not None:
        X = X.loc[y.index]
        y = np.asarray(y)

    return X, y, label_enc

# -------------------------
# UI: Upload dataset
# -------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"])

if uploaded_file is None:
    st.info("Upload a dataset to begin. This app expects environmental / microplastic measurement data.")
    st.stop()

df_original = safe_read_file(uploaded_file)
if df_original is None:
    st.stop()

st.subheader("Raw dataset preview")
st.write(f"Filename: {uploaded_file.name} — {df_original.shape[0]} rows × {df_original.shape[1]} columns")
st.dataframe(df_original.head(50))

# Clean column names
df_original = clean_column_names(df_original)

# -------------------------
# Detect candidate MP count columns and parse
# -------------------------
mp_candidates = [c for c in df_original.columns if "MP_Count" in c or "count" in c.lower() or "items" in c.lower()]
st.sidebar.header("MP Count / Target helpers")
st.sidebar.write("Detected possible MP count columns (auto):")
for c in mp_candidates[:10]:
    st.sidebar.write(f"- {c}")

mp_count_col_choice = st.sidebar.selectbox("Select column to parse as numeric MP count (optional)", [""] + df_original.columns.tolist())
if mp_count_col_choice:
    df_original["_MP_Count_parsed"] = df_original[mp_count_col_choice].apply(parse_mp_count)
    st.write("Parsed MP count preview:")
    st.dataframe(df_original[[mp_count_col_choice, "_MP_Count_parsed"]].head(20))

# -------------------------
# Preprocessing options
# -------------------------
st.sidebar.header("Preprocessing Options")
drop_na = st.sidebar.checkbox("Drop rows with any missing values (otherwise impute numeric values)", value=False)
impute_strategy = st.sidebar.selectbox("Numeric imputation strategy", ["mean", "median", "most_frequent"], index=0)
high_card_threshold = st.sidebar.number_input("High-cardinality threshold for frequency encoding", value=40, min_value=5, max_value=1000, step=1)

# Apply basic imputation or drop at dataframe level
df = df_original.copy()
if drop_na:
    before = df.shape[0]
    df = df.dropna()
    st.write(f"Dropped rows with missing values: {before} -> {df.shape[0]}")
else:
    # impute numeric columns at dataframe-level as preliminary step to help parsing
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = SimpleImputer(strategy=impute_strategy).fit_transform(df[num_cols])
    # Fill object columns with mode where missing
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if df[c].isna().sum():
            mode = df[c].mode()
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
            else:
                df[c] = df[c].fillna("")

st.write("✅ Basic preprocessing done")
st.write(f"Working dataset shape: {df.shape}")
st.dataframe(df.head(20))

# -------------------------
# Risk_Level creation (optional)
# -------------------------
st.sidebar.header("Risk Level (optional)")
auto_risk = st.sidebar.checkbox("Create Risk_Level from _MP_Count_parsed if available", value=("_MP_Count_parsed" in df.columns))
if auto_risk and "_MP_Count_parsed" in df.columns:
    use_quantiles = st.sidebar.checkbox("Use tertile (quantile) thresholds", value=True)
    if use_quantiles:
        valid_counts = df["_MP_Count_parsed"].dropna()
        if len(valid_counts) >= 3:
            thr_low = float(valid_counts.quantile(1/3))
            thr_medium = float(valid_counts.quantile(2/3))
        else:
            thr_low, thr_medium = 1.0, 3.0
    else:
        thr_low = st.sidebar.number_input("Low threshold (<=)", value=1.0, step=0.1)
        thr_medium = st.sidebar.number_input("Medium threshold (<=)", value=3.0, step=0.1)

    def assign_risk(count):
        try:
            if pd.isna(count):
                return np.nan
            c = float(count)
            if c <= thr_low:
                return "Low"
            elif c <= thr_medium:
                return "Medium"
            else:
                return "High"
        except Exception:
            return np.nan

    df["Risk_Level"] = df["_MP_Count_parsed"].apply(assign_risk)
    st.write("Risk_Level sample distribution:")
    st.write(df["Risk_Level"].value_counts(dropna=False))

# -------------------------
# Modeling selections
# -------------------------
st.sidebar.header("Modeling & Evaluation")
task = st.sidebar.radio("Select task", ("classification", "regression", "clustering"))

st.sidebar.markdown("Select target column (for classification/regression)")
target_col = st.sidebar.selectbox("Target column (leave blank for clustering)", [""] + df.columns.tolist())
if task in ("classification", "regression") and (target_col == "" or target_col is None):
    st.sidebar.warning("Please select a target for supervised tasks (e.g., Risk_Level for classification, _MP_Count_parsed for regression).")
    st.stop()

# Features selection
all_columns = df.columns.tolist()
default_features = [c for c in all_columns if c != target_col]
selected_features = st.sidebar.multiselect("Select feature columns (default: all except target)", all_columns, default=default_features)
if not selected_features:
    st.sidebar.error("Select at least one feature column.")
    st.stop()

# Train/test split params
test_size = st.sidebar.slider("Test size (fraction)", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

# -------------------------
# Prepare X and y
# -------------------------
try:
    X, y, label_enc = prepare_features(df, selected_features, target_col if target_col else None, task,
                                      impute_strategy=impute_strategy, high_card_threshold=high_card_threshold)
except Exception as exc:
    st.error("Feature preparation failed. See details in the error below.")
    st.exception(exc)
    st.stop()

st.write("Feature matrix and target prepared.")
st.write(f"X shape: {X.shape}")
if y is not None:
    st.write(f"y shape: {y.shape} | unique values (if classification): {np.unique(y)[:20]}")

if X.shape[1] == 0:
    st.error("No usable features after preprocessing. Please adjust selected features.")
    st.stop()

# -------------------------
# Safe train/test split with stratify/fallbacks
# -------------------------
def safe_train_test_split(X_df: pd.DataFrame, y_arr: Optional[np.ndarray], test_size: float, random_state: int, task: str):
    if y_arr is None:
        return None, None, None, None
    stratify_param = None
    if task == "classification":
        unique, counts = np.unique(y_arr, return_counts=True)
        if len(unique) > 1 and np.min(counts) >= 2:
            stratify_param = y_arr
        else:
            st.warning("Stratified split skipped: target has a single class or at least one class has fewer than 2 samples.")
            stratify_param = None
    try:
        return train_test_split(X_df, y_arr, test_size=test_size, random_state=random_state, stratify=stratify_param)
    except ValueError as e:
        st.warning(f"train_test_split with stratify failed: {e}. Retrying without stratify.")
        return train_test_split(X_df, y_arr, test_size=test_size, random_state=random_state, stratify=None)

if task in ("classification", "regression"):
    if y is None or len(y) == 0:
        st.error("No target values available after preprocessing — cannot train.")
        st.stop()
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size, random_state, task)
    st.write(f"Train shape: {X_train.shape} - Test shape: {X_test.shape}")
else:
    X_train = X_test = y_train = y_test = None

# -------------------------
# Modeling - Train & Evaluate
# -------------------------
st.header("Model Training and Evaluation")

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
        for name, model in classifiers.items():
            try:
                if has_nan_or_inf(X_train.values) or has_nan_or_inf(y_train):
                    raise ValueError("Training data contains NaN/Inf. Please impute or remove missing values.")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, average="macro", zero_division=0)
                }
            except Exception as exc:
                results[name] = {"error": str(exc)}
        st.table(pd.DataFrame(results).T)

        # Best model by F1 (if available)
        scored = {k: v for k, v in results.items() if "F1 Score" in v}
        if scored:
            best = max(scored.items(), key=lambda kv: kv[1]["F1 Score"])[0]
            st.write(f"Best model by F1 score: {best}")
            try:
                best_model = classifiers[best]
                y_pred = best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix — {best}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                st.text("Classification report:")
                st.text(classification_report(y_test, y_pred, zero_division=0))
            except Exception as exc:
                st.warning(f"Could not show confusion matrix / report: {exc}")

elif task == "regression":
    if y_train is None:
        st.error("Regression requires a numeric target column.")
    else:
        # Impute/clean if necessary
        if has_nan_or_inf(X_train.values):
            st.warning("Detected NaNs/Infs in features. Applying imputation.")
            imp = SimpleImputer(strategy=impute_strategy)
            X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)
        if np.isnan(y_train).any():
            st.warning("Dropping rows with NaN target values in training set.")
            mask = ~np.isnan(y_train)
            X_train = X_train.loc[mask]
            y_train = y_train[mask]

        regressors = {
            "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
            "Linear Regression": LinearRegression()
        }
        results = {}
        for name, model in regressors.items():
            try:
                if X_train.shape[0] < 2:
                    raise ValueError("Not enough training samples.")
                if has_nan_or_inf(X_train.values) or np.isnan(y_train).any():
                    raise ValueError("Training data contains NaN/Inf.")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    "R2": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred)
                }
            except Exception as exc:
                results[name] = {"error": str(exc)}
        st.table(pd.DataFrame(results).T)

elif task == "clustering":
    st.subheader("Clustering (unsupervised)")
    k_default = 3 if X.shape[0] >= 3 else max(2, X.shape[0])
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, max(2, min(10, X.shape[0])), value=k_default, step=1)
    try:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
        if X.shape[1] >= 2:
            try:
                sil = silhouette_score(X, labels)
                st.write(f"Silhouette Score: {sil:.4f}")
            except Exception:
                st.info("Could not compute silhouette score (maybe insufficient data).")
            fig, ax = plt.subplots()
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="tab10", alpha=0.7)
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            st.pyplot(fig)
    except Exception as exc:
        st.error(f"Clustering failed: {exc}")
        st.exception(exc)

# -------------------------
# Cross-validation
# -------------------------
st.header("Cross-validation (K-Fold)")
requested_splits = int(st.sidebar.number_input("K-Fold splits", min_value=2, max_value=20, value=5, step=1))
cv_results = {}

if task in ("classification", "regression") and y is not None:
    # Prepare X_cv, y_cv: impute features if necessary and drop NaN targets
    X_cv = X.copy()
    y_cv = np.asarray(y)
    if has_nan_or_inf(X_cv.values) or np.isnan(y_cv).any():
        st.warning("Imputing remaining NaNs in features and dropping NaN targets before CV.")
        imp = SimpleImputer(strategy=impute_strategy)
        X_cv = pd.DataFrame(imp.fit_transform(X_cv), columns=X_cv.columns, index=X_cv.index)
        mask = ~np.isnan(y_cv)
        X_cv = X_cv.loc[mask]
        y_cv = y_cv[mask]

    n_samples = X_cv.shape[0]
    if n_samples < 2:
        cv_results = {"error": f"Not enough samples ({n_samples}) for cross-validation."}
    else:
        n_splits = min(requested_splits, n_samples)
        if n_splits < 2:
            n_splits = 2

        if task == "classification":
            unique, counts = np.unique(y_cv, return_counts=True)
            n_classes = len(unique)
            min_count = int(np.min(counts)) if len(counts) > 0 else 0

            if n_classes < 2:
                cv_results = {"error": "Cross-validation requires at least 2 classes."}
            else:
                # Prefer StratifiedKFold when possible
                if min_count >= n_splits:
                    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    st.info(f"Using StratifiedKFold with n_splits={n_splits}.")
                else:
                    if min_count >= 2:
                        n_splits_reduced = min(n_splits, min_count)
                        cv_strategy = StratifiedKFold(n_splits=n_splits_reduced, shuffle=True, random_state=random_state)
                        st.warning(f"Reduced n_splits to {n_splits_reduced} for stratification (min samples/class = {min_count}).")
                    else:
                        n_splits_kf = min(n_splits, max(2, n_samples // 2))
                        cv_strategy = KFold(n_splits=n_splits_kf, shuffle=True, random_state=random_state)
                        st.warning(f"Stratified CV not possible (min samples/class < 2). Falling back to KFold with n_splits={n_splits_kf}.")

                models_for_cv = {
                    "Random Forest": RandomForestClassifier(random_state=random_state),
                    "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                    "Logistic Regression": LogisticRegression(max_iter=500)
                }
                for name, model in models_for_cv.items():
                    try:
                        scores = cross_val_score(model, X_cv, y_cv, cv=cv_strategy, scoring="accuracy")
                        cv_results[name] = {"mean_accuracy": float(scores.mean()), "std": float(scores.std())}
                    except Exception as exc:
                        cv_results[name] = {"error": str(exc)}

        else:  # regression
            n_splits_reg = min(n_splits, max(2, n_samples // 2))
            if n_splits_reg < 2:
                n_splits_reg = 2
            kf = KFold(n_splits=n_splits_reg, shuffle=True, random_state=random_state)
            models_for_cv = {
                "RF Regressor": RandomForestRegressor(random_state=random_state),
                "DT Regressor": DecisionTreeRegressor(random_state=random_state),
                "LinearReg": LinearRegression()
            }
            for name, model in models_for_cv.items():
                try:
                    scores = cross_val_score(model, X_cv, y_cv, cv=kf, scoring="r2")
                    cv_results[name] = {"mean_r2": float(scores.mean()), "std": float(scores.std())}
                except Exception as exc:
                    cv_results[name] = {"error": str(exc)}
else:
    cv_results = {"note": "Cross-validation not applicable for unsupervised task or missing target."}

st.write("Cross-validation results:")
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
    except Exception as exc:
        st.warning(f"Could not generate pairplot: {exc}")

st.success("Processing complete. See sidebar for options to re-run with different settings.")
