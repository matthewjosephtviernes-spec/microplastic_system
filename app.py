import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import EmptyDataError, ParserError

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._encoders")

# Optional: SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_OK = True
except Exception:
    IMBLEARN_OK = False


# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
TARGET_RISK_TYPE = "Risk_Type"
TARGET_RISK_LEVEL = "Risk_Level"

NUMERIC_COLS = [
    "MP_Count_per_L",
    "Risk_Score",
    "Microplastic_Size_mm",
    "Density",
    "Latitude",
    "Longitude",
]

CATEGORICAL_COLS = [
    "Location",
    "Shape",
    "Polymer_Type",
    "pH",
    "Salinity",
    "Industrial_Activity",
    "Population_Density",
    "Author",
    "Source",
]

DEFAULT_MODEL_DROP_COLS = ["Location", "Author"]


# -------------------------------------------------------
# LOADING + CLEANING
# -------------------------------------------------------
def load_data(uploaded_file=None):
    """Robust CSV reader with encoding fallbacks."""
    if uploaded_file is None:
        path = "Microplastic.csv"
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, sep=None, engine="python")
    else:
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=None, engine="python")


def handle_missing_values(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df


def encode_categorical_columns(df: pd.DataFrame, categorical_cols: list, drop_cols_for_model: tuple = ()):
    """Encode categorical columns with proper reporting."""
    df_encoded = df.copy()
    encoding_report = []
    
    for col in categorical_cols:
        if col not in df_encoded.columns or col in drop_cols_for_model:
            continue
        
        if col in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL]:
            unique_vals = df_encoded[col].dropna().unique()
            label_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            df_encoded[col] = df_encoded[col].map(label_map)
            
            encoding_report.append({
                "Column": col,
                "Encoding_Type": "Label Encoding",
                "Unique_Values": len(unique_vals),
                "Values": str(list(unique_vals)[:5]) + ("..." if len(unique_vals) > 5 else ""),
                "Result_Columns": col,
            })
        else:
            unique_vals = df_encoded[col].dropna().unique()
            encoding_report.append({
                "Column": col,
                "Encoding_Type": "One-Hot Encoding",
                "Unique_Values": len(unique_vals),
                "Values": str(list(unique_vals)[:5]) + ("..." if len(unique_vals) > 5 else ""),
                "Result_Columns": f"{len(unique_vals)} binary columns",
            })
    
    return df_encoded, pd.DataFrame(encoding_report)


def detect_and_handle_outliers(df: pd.DataFrame, numeric_cols: list):
    """Detect and handle outliers using IQR method."""
    df_outliers_handled = df.copy()
    outlier_report = []
    
    for col in numeric_cols:
        if col not in df_outliers_handled.columns:
            continue
        
        s = pd.to_numeric(df_outliers_handled[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_count = ((s < lower_bound) | (s > upper_bound)).sum()
        
        outlier_report.append({
            "Column": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower_Bound": round(lower_bound, 4),
            "Upper_Bound": round(upper_bound, 4),
            "Outliers_Detected": int(outliers_count),
            "Outlier_Percentage": round((outliers_count / s.notna().sum() * 100), 2),
        })
        
        df_outliers_handled[col] = s.clip(lower_bound, upper_bound)
    
    return df_outliers_handled, pd.DataFrame(outlier_report)


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = s.clip(lower, upper)
    return df


def transform_skewed(df: pd.DataFrame, numeric_cols, threshold=0.5):
    """Transform skewed numerical columns and return analysis."""
    df = df.copy()
    present = [c for c in numeric_cols if c in df.columns]
    
    skewness_before = df[present].apply(lambda x: pd.to_numeric(x, errors="coerce")).skew(numeric_only=True)
    skewed_cols = skewness_before[skewness_before.abs() > threshold].index.tolist()
    
    transformation_report = []
    df_transformed = df.copy()
    
    for col in skewed_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        
        shift = -s.min() if s.min() < 0 else 0
        s_log = np.log1p(s + shift)
        skewness_after = s_log.skew()
        
        transformation_report.append({
            "Column": col,
            "Skewness_Before": round(skewness_before[col], 4),
            "Skewness_After": round(skewness_after, 4),
            "Skewness_Reduced": round(abs(skewness_before[col]) - abs(skewness_after), 4),
            "Transformation": "Log1p",
            "Shift_Applied": shift,
        })
        
        df_transformed[col] = s_log
    
    return df_transformed, skewness_before, skewed_cols, pd.DataFrame(transformation_report)


def scale_numeric_with_report(df: pd.DataFrame, numeric_cols):
    """Apply StandardScaler and return scaling report."""
    df = df.copy()
    scaler = StandardScaler()
    present = [c for c in numeric_cols if c in df.columns]
    
    scaling_report = []
    
    if present:
        vals = df[present].apply(pd.to_numeric, errors="coerce")
        original_stats = vals.describe().T
        
        scaled_vals = scaler.fit_transform(vals.fillna(vals.median()))
        df[present] = scaled_vals
        
        scaled_stats = pd.DataFrame(scaled_vals, columns=present).describe().T
        
        for col in present:
            orig_mean = original_stats.loc[col, "mean"]
            orig_std = original_stats.loc[col, "std"]
            scaled_mean = scaled_stats.loc[col, "mean"]
            scaled_std = scaled_stats.loc[col, "std"]
            
            scaling_report.append({
                "Column": col,
                "Original_Mean": round(orig_mean, 4),
                "Original_Std": round(orig_std, 4),
                "Scaled_Mean": round(scaled_mean, 6),
                "Scaled_Std": round(scaled_std, 4),
                "Scaling_Method": "StandardScaler (z-score)",
            })
    
    return df, scaler, pd.DataFrame(scaling_report)


def coerce_numeric_like(df: pd.DataFrame, columns):
    df = df.copy()
    for c in columns:
        if c in df.columns:
            s = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


# -------------------------------------------------------
# ENHANCED FEATURE SELECTION METHODS
# -------------------------------------------------------
def perform_enhanced_feature_selection(df: pd.DataFrame, target_col: str, n_top_features: int = 20):
    """
    Perform comprehensive feature selection with one-hot encoding.
    Returns MI scores, Chi2 scores, and RF importances.
    """
    
    # Create a copy for encoding
    df_work = df.copy()
    
    # Remove rows with missing target
    df_work = df_work[df_work[target_col].notna()]
    
    if len(df_work) == 0:
        st.error(f"No valid data for {target_col}")
        return None, None, None
    
    # Identify categorical columns to encode
    cat_cols = [c for c in CATEGORICAL_COLS if c in df_work.columns and c != target_col]
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_work, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    y = df_encoded[target_col].copy()
    
    # Identify one-hot encoded columns (new columns from encoding)
    original_cols = df_work.columns.tolist()
    if target_col in original_cols:
        original_cols.remove(target_col)
    
    ohe_cols = [col for col in df_encoded.columns if col not in original_cols]
    
    # Add numeric columns that weren't in the original list
    numeric_features = [c for c in NUMERIC_COLS if c in df_work.columns]
    
    X_cols = numeric_features + ohe_cols
    X = df_encoded[X_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    
    if len(X) == 0 or X.shape[1] == 0:
        st.error("No features available after encoding")
        return None, None, None
    
    results = {}
    
    # 1. Mutual Information
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.Series(mi_scores, index=X.columns, name="MI_Score").sort_values(ascending=False)
        results["Mutual Information"] = mi_df.head(n_top_features)
    except Exception as e:
        st.warning(f"MI failed: {e}")
        results["Mutual Information"] = pd.Series(dtype=float)
    
    # 2. Chi-squared Test (only works with non-negative values)
    try:
        # Shift values to be non-negative if needed
        X_nonneg = X - X.min() + 1e-10
        chi2_scores, p_values = chi2(X_nonneg, y)
        chi2_df = pd.Series(chi2_scores, index=X.columns, name="Chi2_Score").sort_values(ascending=False)
        results["Chi-squared"] = chi2_df.head(n_top_features)
    except Exception as e:
        st.warning(f"Chi-squared failed: {e}")
        results["Chi-squared"] = pd.Series(dtype=float)
    
    # 3. Random Forest Feature Importance
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_df = pd.Series(rf.feature_importances_, index=X.columns, name="RF_Importance").sort_values(ascending=False)
        results["Random Forest"] = rf_df.head(n_top_features)
    except Exception as e:
        st.warning(f"Random Forest failed: {e}")
        results["Random Forest"] = pd.Series(dtype=float)
    
    return results, X, y


# -------------------------------------------------------
# SPLIT HELPERS
# -------------------------------------------------------
def merge_rare_classes(y: pd.Series, min_count: int = 2, other_label: str = "Other"):
    y = pd.Series(y).copy()
    counts = y.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    y = y.where(~y.isin(rare), other_label)
    return y


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target.")

    counts = y.value_counts()
    min_class = int(counts.min())
    n = len(y)
    k = y.nunique()

    if min_class < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        return (X_train, X_test, y_train, y_test), False, float(test_size)

    min_test_size = k / n
    max_test_size = 1 - (k / n)

    ts = float(test_size)
    ts = max(ts, min_test_size)
    if max_test_size > 0:
        ts = min(ts, max_test_size)

    for ts_try in [ts, 0.2, 0.15, 0.1, 0.05]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts_try, random_state=random_state, stratify=y
            )
            return (X_train, X_test, y_train, y_test), True, float(ts_try)
        except ValueError:
            continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    return (X_train, X_test, y_train, y_test), False, float(test_size)


# -------------------------------------------------------
# SAFE CV PICKER
# -------------------------------------------------------
def pick_safe_cv(y: pd.Series, requested_splits: int, stratified: bool):
    y = pd.Series(y).dropna()
    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes for cross-validation.")

    counts = y.value_counts()
    min_count = int(counts.min())

    if stratified:
        safe_splits = min(requested_splits, min_count)
        if safe_splits < 2:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            note = "⚠️ StratifiedKFold not possible (classes too small). Using KFold(n_splits=3)."
            return cv, note

        if safe_splits != requested_splits:
            cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
            note = f"⚠️ Reduced folds from {requested_splits} to {safe_splits} due to small class counts."
            return cv, note

        cv = StratifiedKFold(n_splits=requested_splits, shuffle=True, random_state=42)
        return cv, "✅ Using StratifiedKFold."
    else:
        n = len(y)
        safe_splits = min(requested_splits, n)
        safe_splits = max(2, safe_splits)
        if safe_splits != requested_splits:
            cv = KFold(n_splits=safe_splits, shuffle=True, random_state=42)
            return cv, f"⚠️ Reduced folds from {requested_splits} to {safe_splits} based on sample size."
        cv = KFold(n_splits=requested_splits, shuffle=True, random_state=42)
        return cv, "✅ Using KFold."


# -------------------------------------------------------
# PIPELINES
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_preprocess_pipeline_cached(df_raw: pd.DataFrame, drop_cols_for_model: tuple):
    """Build preprocessing pipeline with robust OneHotEncoder."""
    numeric_features = [c for c in NUMERIC_COLS if c in df_raw.columns]
    numeric_features = [c for c in numeric_features if df_raw[c].notna().any()]

    categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns and c not in drop_cols_for_model]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
            dtype=np.float64
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def get_Xy_for_target(df_raw: pd.DataFrame, target_col: str, drop_cols_for_model: tuple):
    """Extract features and target."""
    df = df_raw.copy()
    df = coerce_numeric_like(df, NUMERIC_COLS)

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found in dataset.")

    drop_targets = [TARGET_RISK_TYPE, TARGET_RISK_LEVEL]
    feature_cols = [c for c in df.columns if c not in drop_targets and c not in drop_cols_for_model]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    y = y.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    y = merge_rare_classes(y, min_count=2, other_label="Other")
    
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("missing")
        else:
            X[col] = X[col].fillna(X[col].median())
    
    return X, y


def build_models_fast(fast_mode: bool):
    rf_estimators = 150 if fast_mode else 400
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=rf_estimators, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


@st.cache_data(show_spinner=False)
def train_holdout_models_cached(
    df_raw: pd.DataFrame,
    target_col: str,
    test_size: float,
    drop_cols_for_model: tuple,
    fast_mode: bool,
    use_smote: bool = False,
):
    """Train holdout models."""
    X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)

    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)
    models = build_models_fast(fast_mode)

    metrics_list = []
    fitted_pipes = {}

    for name, model in models.items():
        if use_smote:
            if not IMBLEARN_OK:
                raise RuntimeError("imbalanced-learn is required for SMOTE. Install: pip install imbalanced-learn")
            pipe = ImbPipeline(steps=[
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ])
        else:
            pipe = Pipeline(steps=[
                ("prep", preprocessor),
                ("model", model),
            ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        fitted_pipes[name] = pipe
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")

    split_note = (
        f"✅ Stratified split used (test_size={final_test_size:.2f})."
        if used_stratify
        else f"⚠️ Non-stratified split used (test_size={final_test_size:.2f}) because some classes are too small."
    )

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
        "final_test_size": final_test_size,
    }

    return fitted_pipes, metrics_df, split_info, split_note


def smote_and_tune_logreg_pipeline(
    df_raw: pd.DataFrame,
    target_col: str,
    test_size: float,
    drop_cols_for_model: tuple,
    fast_mode: bool,
):
    """SMOTE and hyperparameter tuning."""
    if not IMBLEARN_OK:
        raise RuntimeError("imbalanced-learn is required for SMOTE. Install: pip install imbalanced-learn")

    X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)

    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)

    class_counts = y_train.value_counts()
    min_count = int(class_counts.min())

    if min_count <= 1:
        use_smote = False
        k_neighbors = None
    else:
        use_smote = True
        k_neighbors = max(1, min(5, min_count - 1))

    if use_smote:
        base_pipe = ImbPipeline(steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
    else:
        base_pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])

    param_grid = {"model__C": [0.01, 0.1, 1, 10]}
    cv_folds = 3 if fast_mode else 5
    cv_folds = min(cv_folds, max(2, min_count))

    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv_folds,
        n_jobs=-1,
        error_score="raise"
    )

    tuning_note = None
    try:
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        best_params = grid.best_params_
        tuned_label = "Logistic Regression (Tuned + SMOTE)" if use_smote else "Logistic Regression (Tuned)"
    except ValueError:
        fallback_pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        grid2 = GridSearchCV(
            estimator=fallback_pipe,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=min(3 if fast_mode else 5, max(2, min_count)),
            n_jobs=-1,
            error_score="raise"
        )
        grid2.fit(X_train, y_train)
        best_pipe = grid2.best_estimator_
        best_params = grid2.best_params_
        tuned_label = "Logistic Regression (Tuned)"
        tuning_note = (
            "⚠️ SMOTE tuning failed due to very small class counts in CV folds. "
            "Fell back to tuning Logistic Regression WITHOUT SMOTE."
        )

    y_pred = best_pipe.predict(X_test)

    tuned_metrics = pd.DataFrame([{
        "Model": tuned_label,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }]).set_index("Model")

    split_note = (
        f"✅ Stratified split used (test_size={final_test_size:.2f})."
        if used_stratify
        else f"⚠️ Non-stratified split used (test_size={final_test_size:.2f}) because some classes are too small."
    )
    split_note += f" CV folds={cv_folds}."
    if use_smote and tuning_note is None:
        split_note += f" SMOTE k_neighbors={k_neighbors}."
    if tuning_note:
        split_note += " " + tuning_note

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
        "final_test_size": final_test_size,
    }

    return best_pipe, tuned_metrics, best_params, split_info, split_note


def run_cv(
    df_raw: pd.DataFrame,
    target_col: str,
    model_name: str,
    n_splits: int,
    stratified: bool,
    drop_cols_for_model: tuple,
    fast_mode: bool,
):
    """Manual cross validation."""
    X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)

    models = build_models_fast(fast_mode)
    if model_name not in models:
        raise ValueError("Unknown model selected.")
    model = models[model_name]

    cv, cv_note = pick_safe_cv(y, n_splits, stratified)
    preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)

    fold_scores = {
        "accuracy": [],
        "precision_w": [],
        "recall_w": [],
        "f1_w": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2:
            for k in fold_scores.keys():
                fold_scores[k].append(np.nan)
            continue

        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])

        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            fold_scores["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_scores["precision_w"].append(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            fold_scores["recall_w"].append(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            fold_scores["f1_w"].append(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
        except Exception:
            for k in fold_scores.keys():
                fold_scores[k].append(np.nan)

    summary = {}
    for metric_key in fold_scores:
        arr = np.array(fold_scores[metric_key], dtype=float)
        summary[metric_key] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
        }
    summary_df = pd.DataFrame(summary).T
    summary_df = summary_df.rename(index={
        "accuracy": "Accuracy",
        "precision_w": "Precision (weighted)",
        "recall_w": "Recall (weighted)",
        "f1_w": "F1-score (weighted)",
    })

    return summary_df, fold_scores, cv_note


# -------------------------------------------------------
# VISUALS
# -------------------------------------------------------
def plot_hist_box(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if len(s) == 0:
        axes[0].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
        axes[1].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
    else:
        sns.histplot(s, kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=s, ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()

    fig, ax = plt.subplots(figsize=(6, 4))
    if mask.sum() == 0:
        ax.text(0.5, 0.5, f"No numeric data for {x_col} and {y_col}", ha="center", va="center")
    else:
        ax.scatter(x[mask], y[mask], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics_df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df[["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title(f"Model Performance {title_suffix}")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def plot_box_by_category_readable(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    top_n: int = 10,
    other_label: str = "Other",
    figsize=(10, 5),
    horizontal=True,
):
    d = df.copy()
    d[category_col] = d[category_col].astype(str).str.strip()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    vc = d[category_col].value_counts()
    top = vc.head(top_n).index
    d[category_col] = d[category_col].where(d[category_col].isin(top), other_label)

    order = d.groupby(category_col)[value_col].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        sns.boxplot(data=d, y=category_col, x=value_col, order=order, ax=ax)
    else:
        sns.boxplot(data=d, x=category_col, y=value_col, order=order, ax=ax)
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    ax.set_title(f"{value_col} by {category_col} (Top {top_n} + {other_label})")
    plt.tight_layout()
    return fig


def plot_categorical_topn_bar(
    series: pd.Series,
    title: str,
    top_n: int = 15,
    other_label: str = "Other",
    figsize=(10, 6),
):
    s = series.dropna().astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
    counts = s.value_counts()

    if counts.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No category data available", ha="center", va="center")
        plt.tight_layout()
        return fig, counts

    top = counts.head(top_n)
    remainder = counts.iloc[top_n:].sum()
    if remainder > 0:
        top = pd.concat([top, pd.Series({other_label: remainder})])

    fig, ax = plt.subplots(figsize=figsize)
    top.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel(series.name if series.name else "Category")
    plt.tight_layout()
    return fig, counts


def plot_feature_ranking_comparison(ranking_results: dict, target_col: str):
    """Visualize feature ranking."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ["Mutual Information", "Chi-squared", "Random Forest"]
    for idx, (ax, method) in enumerate(zip(axes, methods)):
        if method in ranking_results and not ranking_results[method].empty:
            data = ranking_results[method]
            if len(data) > 0:
                data_sorted = data.sort_values()
                ax.barh(data_sorted.index, data_sorted.values, color="steelblue")
                ax.set_title(f"{method}\n{target_col}")
                ax.set_xlabel("Score")
            else:
                ax.text(0.5, 0.5, f"No data for {method}", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, f"No data for {method}", ha="center", va="center")
    
    plt.tight_layout()
    return fig


def plot_model_comparison_across_targets(metrics_rt: pd.DataFrame, metrics_rl: pd.DataFrame):
    """Compare model performance across targets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(metrics_rt.index))
        width = 0.35
        
        ax.bar(x - width/2, metrics_rt[metric], width, label="Risk_Type", alpha=0.8)
        ax.bar(x + width/2, metrics_rl[metric], width, label="Risk_Level", alpha=0.8)
        
        ax.set_ylabel("Score")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_rt.index, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# APP MAIN
# -------------------------------------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        """
        This app demonstrates the analysis and modeling workflow for predicting **Risk_Type**
        and **Risk_Level** using microplastic and environmental features.

        ✅ Modeling + CV are leakage-safe (Pipeline does preprocessing inside train/CV folds).  
        ✅ OneHotEncoder safely handles unknown categories with `handle_unknown='ignore'`.  
        ✅ Enhanced feature selection using MI, Chi-squared, and Random Forest.
        """
    )

    # =====================================================
    # Sidebar Navigation
    # =====================================================
    st.sidebar.header("Navigation")

    NAV = {
        "🏠 Home": [
            "Data Overview & Task 1",
            "Polymer Type Distribution",
        ],
        "🧼 Data Preparation": [
            "Preprocessing (Task 2)",
            "Feature Selection & Relevance (Task 3 & 6)",
        ],
        "🧠 Modeling": [
            "Classification Modeling (Tasks 4, 5 & 7)",
            "Cross Validation (K-Fold)",
        ],
        "⚙��� Optimization": [
            "SMOTE & Hyperparameter Tuning (Risk_Type)",
        ],
    }

    if "nav_category" not in st.session_state:
        st.session_state["nav_category"] = "🏠 Home"
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = NAV[st.session_state["nav_category"]][0]

    category = st.sidebar.selectbox(
        "Category",
        list(NAV.keys()),
        index=list(NAV.keys()).index(st.session_state["nav_category"])
    )
    st.session_state["nav_category"] = category

    pages_in_cat = NAV[category]
    if st.session_state["nav_page"] not in pages_in_cat:
        st.session_state["nav_page"] = pages_in_cat[0]

    page = st.sidebar.radio(
        "Go to",
        pages_in_cat,
        index=pages_in_cat.index(st.session_state["nav_page"])
    )
    st.session_state["nav_page"] = page

    st.sidebar.subheader("Performance")
    fast_mode = st.sidebar.toggle("Fast Mode (recommended)", value=True)
    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.sidebar.subheader("Model Features")
    drop_location_author = st.sidebar.checkbox(
        "Drop Location & Author for modeling/CV (speeds up a lot)",
        value=True,
        help="These columns have many unique values and cause huge one-hot matrices."
    )
    drop_cols_for_model = tuple(DEFAULT_MODEL_DROP_COLS) if drop_location_author else tuple()

    st.sidebar.subheader("Data source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Microplastic CSV",
        type=["csv"],
        help="If you don't upload anything, the app will try to use 'Microplastic.csv' from the app folder."
    )

    try:
        df_raw = load_data(uploaded_file=uploaded_file)
    except UnicodeDecodeError:
        st.error("⚠️ Unable to decode the file. Please upload a proper CSV (text).")
        st.stop()
    except EmptyDataError:
        st.error("⚠️ The uploaded file appears empty/unreadable as CSV.")
        st.stop()
    except ParserError:
        st.error("⚠️ The file is not a valid CSV format. Re-export as CSV and try again.")
        st.stop()
    except FileNotFoundError:
        df_raw = None

    if df_raw is None:
        st.error("❌ No dataset found. Upload a CSV or add 'Microplastic.csv' beside app.py.")
        st.stop()

    # -------------------- PAGE 1 --------------------
    if page == "Data Overview & Task 1":
        st.header("Data Overview & Task 1: Risk_Score Analysis")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Raw Data",
            "Risk_Score Distribution",
            "MP_Count vs Risk_Score",
            "Risk_Score by Risk_Level",
        ])

        with tab1:
            st.subheader("Raw Dataset (first 10 rows)")
            st.dataframe(df_raw.head(10))
            st.markdown(f"**Shape:** `{df_raw.shape[0]}` rows × `{df_raw.shape[1]}` columns")

        with tab2:
            if "Risk_Score" in df_raw.columns:
                st.subheader("Distribution of Risk_Score")
                st.pyplot(plot_hist_box(df_raw, "Risk_Score"))
            else:
                st.info("Column 'Risk_Score' not found.")

        with tab3:
            if "MP_Count_per_L" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("MP_Count vs Risk_Score")
                st.pyplot(plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score"))
            else:
                st.info("Columns not found.")

        with tab4:
            if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Risk_Score by Risk_Level")
                st.pyplot(
                    plot_box_by_category_readable(
                        df_raw,
                        value_col="Risk_Score",
                        category_col="Risk_Level",
                        top_n=8,
                        figsize=(12, 5),
                        horizontal=True,
                    )
                )
            else:
                st.info("Columns not found.")

    # -------------------- PAGE 2 - PREPROCESSING --------------------
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Preprocessing & Feature Engineering")
        
        st.subheader("Step 1: Categorical Encoding")
        categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns]
        
        df_encoded, encoding_report = encode_categorical_columns(df_raw, categorical_features, drop_cols_for_model)
        
        if not encoding_report.empty:
            st.write("**Encoding Report:**")
            st.dataframe(encoding_report, use_container_width=True)
        
        st.subheader("Step 2: Outlier Detection")
        numeric_cols_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
        
        df_outliers_handled, outlier_report = detect_and_handle_outliers(df_raw, numeric_cols_present)
        
        if not outlier_report.empty:
            st.dataframe(outlier_report, use_container_width=True)
        
        st.subheader("Step 3: Feature Scaling")
        df_scaled, _, scaling_report = scale_numeric_with_report(df_outliers_handled, numeric_cols_present)
        
        if not scaling_report.empty:
            st.dataframe(scaling_report, use_container_width=True)
        
        st.subheader("Step 4: Skewness Analysis")
        df_transformed, skewness_before, skewed_cols, transformation_report = transform_skewed(
            df_scaled, numeric_cols_present, threshold=0.5
        )
        
        if not transformation_report.empty:
            st.dataframe(transformation_report, use_container_width=True)

    # -------------------- PAGE 3 - ENHANCED FEATURE SELECTION --------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Enhanced Feature Selection & Relevance")
        
        st.markdown("""
        **Feature Selection Methods:**
        - Mutual Information (MI)
        - Chi-squared Test
        - Random Forest Importance
        
        **Classification Models:**
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        """)
        
        tab_rt, tab_rl = st.tabs([TARGET_RISK_TYPE, TARGET_RISK_LEVEL])

        def run_enhanced_feature_analysis(target_col: str):
            st.write(f"### Analyzing {target_col}")
            
            with st.spinner(f"Performing feature selection for {target_col}..."):
                ranking_results, X, y = perform_enhanced_feature_selection(df_raw, target_col, n_top_features=20)
            
            if ranking_results is None:
                return
            
            st.write(f"**Data:** {X.shape[0]} samples × {X.shape[1]} features")
            st.write(f"**Classes:** {dict(y.value_counts())}")

            st.subheader("Feature Ranking Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Mutual Information**")
                if "Mutual Information" in ranking_results and not ranking_results["Mutual Information"].empty:
                    mi_df = ranking_results["Mutual Information"].to_frame("Score")
                    st.dataframe(mi_df, height=400)
            
            with col2:
                st.write("**Chi-squared**")
                if "Chi-squared" in ranking_results and not ranking_results["Chi-squared"].empty:
                    chi2_df = ranking_results["Chi-squared"].to_frame("Score")
                    st.dataframe(chi2_df, height=400)
            
            with col3:
                st.write("**Random Forest**")
                if "Random Forest" in ranking_results and not ranking_results["Random Forest"].empty:
                    rf_df = ranking_results["Random Forest"].to_frame("Score")
                    st.dataframe(rf_df, height=400)
            
            if ranking_results:
                st.pyplot(plot_feature_ranking_comparison(ranking_results, target_col))
            
            st.subheader("Classification Model Performance")
            
            (X_train, X_test, y_train, y_test), used_stratify, _ = safe_train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            with st.spinner("Training models..."):
                preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)
                models = build_models_fast(fast_mode)
                
                metrics_list = []
                for model_name, model in models.items():
                    try:
                        pipe = Pipeline(steps=[
                            ("prep", preprocessor),
                            ("model", model),
                        ])
                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_test)
                        
                        metrics_list.append({
                            "Model": model_name,
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                            "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                        })
                    except Exception as e:
                        st.warning(f"{model_name} failed: {e}")
                
                if metrics_list:
                    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
                    st.dataframe(metrics_df.round(4))
                    st.pyplot(plot_metrics_bar(metrics_df, f"({target_col})"))

        with tab_rt:
            if TARGET_RISK_TYPE in df_raw.columns:
                run_enhanced_feature_analysis(TARGET_RISK_TYPE)
            else:
                st.warning(f"{TARGET_RISK_TYPE} not found")

        with tab_rl:
            if TARGET_RISK_LEVEL in df_raw.columns:
                run_enhanced_feature_analysis(TARGET_RISK_LEVEL)
            else:
                st.warning(f"{TARGET_RISK_LEVEL} not found")

    # -------------------- PAGE 4 --------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Classification Modeling")
        tab1, tab2 = st.tabs(["Risk_Type", "Risk_Level"])

        with tab1:
            if TARGET_RISK_TYPE not in df_raw.columns:
                st.warning("Risk_Type not found")
            else:
                st.subheader("Models for Risk-Type")
                with st.spinner("Training..."):
                    _, metrics_rt, _, split_note_rt = train_holdout_models_cached(
                        df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )
                st.dataframe(metrics_rt.round(3))
                st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))
                st.info(split_note_rt)

        with tab2:
            if TARGET_RISK_LEVEL not in df_raw.columns:
                st.warning("Risk_Level not found")
            else:
                st.subheader("Models for Risk-Level")
                with st.spinner("Training..."):
                    _, metrics_rl, _, split_note_rl = train_holdout_models_cached(
                        df_raw, TARGET_RISK_LEVEL, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )
                st.dataframe(metrics_rl.round(3))
                st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))
                st.info(split_note_rl)

    # -------------------- PAGE 5 --------------------
    elif page == "Cross Validation (K-Fold)":
        st.header("Cross Validation (K-Fold)")

        target = TARGET_RISK_TYPE
        model_name = "Logistic Regression"
        st.info(f"Target: **{target}** | Model: **{model_name}**")
        
        n_splits = st.slider("Number of folds", min_value=3, max_value=5, value=3)
        stratified = st.checkbox("Stratified K-Fold", value=True)

        if len(df_raw) > 500:
            st.warning(f"Sampling 500 rows for efficient CV")
            df_cv = df_raw.sample(500, random_state=42).reset_index(drop=True)
        else:
            df_cv = df_raw.copy()

        if st.button("Run CV", type="primary"):
            with st.spinner(f"Running {n_splits}-Fold CV..."):
                try:
                    summary_df, _, cv_note = run_cv(
                        df_raw=df_cv,
                        target_col=target,
                        model_name=model_name,
                        n_splits=n_splits,
                        stratified=stratified,
                        drop_cols_for_model=drop_cols_for_model,
                        fast_mode=fast_mode,
                    )
                    st.info(cv_note)
                    st.subheader("CV Summary (mean ± std)")
                    st.dataframe(summary_df.round(4))
                except Exception as e:
                    st.error(f"CV failed: {e}")

    # -------------------- PAGE 6 --------------------
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution")
        df = handle_missing_values(df_raw)

        if "Polymer_Type" in df.columns:
            polymer = df["Polymer_Type"].astype(str).str.strip().replace({"": np.nan})
            polymer = polymer.dropna()
            
            tabA, tabB = st.tabs(["Counts", "Plot"])

            with tabA:
                st.subheader("Polymer_Type Counts")
                st.dataframe(polymer.value_counts())

            with tabB:
                top_n = st.slider("Top N types", 5, 30, 15)
                fig, _ = plot_categorical_topn_bar(
                    polymer,
                    title=f"Polymer_Type Distribution (Top {top_n})",
                    top_n=top_n,
                )
                st.pyplot(fig)
        else:
            st.warning("Polymer_Type not found")

    # -------------------- PAGE 7 --------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("SMOTE & Tuning (Risk_Type)")

        if TARGET_RISK_TYPE not in df_raw.columns:
            st.warning("Risk_Type not found")
            return

        if not IMBLEARN_OK:
            st.error("Install imbalanced-learn: pip install imbalanced-learn")
            st.stop()

        tab1, tab2 = st.tabs(["Base Models", "SMOTE + Tuning"])

        with tab1:
            st.subheader("Base Models")
            with st.spinner("Training..."):
                _, base_metrics, _, note = train_holdout_models_cached(
                    df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode, use_smote=False
                )
            st.dataframe(base_metrics.round(3))
            st.pyplot(plot_metrics_bar(base_metrics, "(Base)"))
            st.info(note)

        with tab2:
            st.subheader("SMOTE + Tuning")
            with st.spinner("Training..."):
                _, tuned_metrics, best_params, _, note = smote_and_tune_logreg_pipeline(
                    df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode
                )
            st.write("**Best Parameters:**")
            st.json(best_params)
            st.dataframe(tuned_metrics.round(3))
            st.info(note)


if __name__ == "__main__":
    main()
