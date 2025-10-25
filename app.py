# Updated Streamlit app (app.py)
# Fix: robust handling before SimpleImputer.fit_transform to avoid ValueError from sklearn
# - Coerce object-like numeric columns when possible
# - Drop numeric columns that are all-NaN before imputation
# - Skip imputation when there are no numeric columns
# - Add clearer warnings and try/except around imputation

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
st.title("Microplastic Risk Prediction System — Robust data handling")

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

    # Identify object columns and reduce cardinality
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = reduce_high_cardinality(X, obj_cols, threshold=40)

    # One-hot encode remaining categorical/object columns
    X = pd.get_dummies(X, drop_first=True)

    # === NEW: attempt to coerce object-like numeric columns that survived encoding ===
    # Some columns may still be object due to mixed content; try to coerce columns that look numeric
    coerced_cols = []
    for col in X.columns:
        if X[col].dtype == object:
            # remove commas and common non-digit chars except minus and dot and exponent
            cleaned = X[col].astype(str).str.replace(r"[,\(\)%\s]+", "", regex=True)
            coerced = pd.to_numeric(cleaned.replace("", np.nan), errors="coerce")
            # if coercion yields at least one non-NaN numeric value, accept the coercion
            if coerced.notna().sum() > 0:
                X[col] = coerced
                coerced_cols.append(col)
    if coerced_cols:
        st.info(f"Coerced these object columns to numeric where possible: {coerced_cols}")

    # Encode target for classification OR coerce numeric for regression
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

    # Identify numeric columns after coercion/encoding
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Drop numeric columns that are entirely NaN (imputer can't handle columns with no values)
    allnan_numeric = [c for c in numeric_cols if X[c].isna().all()]
    if allnan_numeric:
        st.warning(f"Dropping numeric columns that are all NaN and would break imputation: {allnan_numeric}")
        X = X.drop(columns=allnan_numeric)
        numeric_cols = [c for c in numeric_cols if c not in allnan_numeric]

    # If there are numeric columns, impute; otherwise skip imputation step
    if numeric_cols:
        imputer = SimpleImputer(strategy=impute_strategy)
        try:
            # Ensure we pass a valid 2D array to the imputer
            X_numeric = X[numeric_cols]
            # Fit/transform
            X[numeric_cols] = imputer.fit_transform(X_numeric)
        except ValueError as ve:
            st.error(f"Imputation failed: {ve}. Inspect numeric columns: {numeric_cols}")
            # Re-raise for logs / debugging in Streamlit environment
            raise

    # Drop any remaining non-numeric (should be none after get_dummies + coercion)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        st.warning(f"Dropping non-numeric columns after encoding/coercion: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # Scale numeric cols
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Remove rows with NaN in features (should be none after impute; but keep protection)
    mask_x_na = X.isna().any(axis=1)
    if mask_x_na.any():
        st.warning(f"Dropping {mask_x_na.sum()} rows because features contain NaN after imputation.")
        X = X.loc[~mask_x_na].copy()
        if y is not None:
            y = y.loc[X.index].copy()

    # Align indices and convert y to numpy array
    if y is not None:
        X = X.loc[y.index]
        y = np.asarray(y)

    return X, y, label_enc

# (rest of the app continues unchanged...) - below is the same logic as before for upload, preprocessing options, train/test split,
# modeling selection, training, cross-validation and visualizations. The only functional changes are inside prepare_features()
# shown above so regression and cross-validation do not fail due to imputer/check_array errors.

# -------------------------
# The remaining code (upload, UI, modeling and CV) is omitted here for brevity in this snippet,
# but in your local file keep the existing app flow unchanged below this point.
# -------------------------
