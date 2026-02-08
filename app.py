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
from sklearn.feature_selection import mutual_info_classif, chi2

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
    """
    Robust CSV reader with encoding fallbacks.
    If uploaded_file is None, tries to read Microplastic.csv beside app.py.
    """
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
    """
    Encode categorical columns using One-Hot Encoding for features and Label Encoding for targets.
    Returns the encoded dataframe and encoding report.
    """
    df_encoded = df.copy()
    encoding_report = []
    
    for col in categorical_cols:
        if col not in df_encoded.columns or col in drop_cols_for_model:
            continue
        
        # For target variables, use label encoding
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
            # For features, use one-hot encoding (show counts)
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
    """
    Detect outliers using IQR method and return detailed report.
    """
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
        
        # Cap outliers
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
    """
    Transform skewed numerical columns and return detailed analysis.
    """
    df = df.copy()
    present = [c for c in numeric_cols if c in df.columns]
    
    # Calculate skewness before transformation
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
    """
    Apply StandardScaler and return scaling report.
    """
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
# FEATURE SELECTION METHODS
# -------------------------------------------------------
def compute_mutual_information(X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> pd.DataFrame:
    """
    Compute Mutual Information scores for each feature with the target.
    Works with both numeric and categorical features.
    """
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            "Feature": X.columns,
            "MI_Score": mi_scores,
        }).sort_values("MI_Score", ascending=False)
        
        return mi_df.head(n_top)
    except Exception as e:
        st.error(f"Mutual Information computation failed: {e}")
        return pd.DataFrame()


def compute_chi_squared(X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> pd.DataFrame:
    """
    Compute Chi-squared scores for categorical features.
    Note: Chi-squared requires non-negative features.
    """
    try:
        # Shift features to be non-negative (chi2 requirement)
        X_shifted = X - X.min() + 1e-10
        chi2_scores = chi2(X_shifted, y)[0]
        
        chi2_df = pd.DataFrame({
            "Feature": X.columns,
            "Chi2_Score": chi2_scores,
        }).sort_values("Chi2_Score", ascending=False)
        
        return chi2_df.head(n_top)
    except Exception as e:
        st.warning(f"Chi-squared computation failed (may require non-negative features): {e}")
        return pd.DataFrame()


def compute_rf_importance(X: pd.DataFrame, y: pd.Series, n_top: int = 20, n_estimators: int = 200) -> pd.DataFrame:
    """
    Compute feature importance using Random Forest.
    """
    try:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        rf_df = pd.DataFrame({
            "Feature": X.columns,
            "RF_Importance": rf.feature_importances_,
        }).sort_values("RF_Importance", ascending=False)
        
        return rf_df.head(n_top)
    except Exception as e:
        st.error(f"Random Forest importance computation failed: {e}")
        return pd.DataFrame()


def rank_features_multi_method(X: pd.DataFrame, y: pd.Series, n_top: int = 15) -> dict:
    """
    Rank features using multiple methods and return comprehensive ranking report.
    """
    results = {}
    
    # Mutual Information
    st.info("Computing Mutual Information scores...")
    mi_scores = compute_mutual_information(X, y, n_top=n_top)
    results["Mutual Information"] = mi_scores
    
    # Chi-squared
    st.info("Computing Chi-squared scores...")
    chi2_scores = compute_chi_squared(X, y, n_top=n_top)
    results["Chi-squared"] = chi2_scores
    
    # Random Forest Importance
    st.info("Computing Random Forest feature importance...")
    rf_scores = compute_rf_importance(X, y, n_top=n_top, n_estimators=200)
    results["Random Forest"] = rf_scores
    
    return results


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
# PIPELINES - FIXED WITH ROBUST ONEHOTENCODER
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_preprocess_pipeline_cached(df_raw: pd.DataFrame, drop_cols_for_model: tuple):
    """
    Build preprocessing pipeline with robust handling of unknown categories.
    Uses handle_unknown='ignore' and sparse_output=False to prevent warnings.
    """
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
            handle_unknown="ignore",  # Ignore unknown categories instead of error
            drop="first",              # Avoid multicollinearity
            sparse_output=False,       # Return dense array
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
    """
    Extract features and target with proper handling of missing values and rare classes.
    """
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
    
    # Handle missing values in features
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
    """
    Train holdout models with proper preprocessing pipeline.
    """
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
    """
    SMOTE and hyperparameter tuning with proper preprocessing.
    """
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
    """
    Manual, crash-safe cross validation with robust preprocessing.
    """
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
    """
    Visualize feature ranking from multiple methods side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ["Mutual Information", "Chi-squared", "Random Forest"]
    for idx, (ax, method) in enumerate(zip(axes, methods)):
        if method in ranking_results and not ranking_results[method].empty:
            df = ranking_results[method].iloc[:10]  # Top 10
            score_col = [c for c in df.columns if c != "Feature"][0]
            df_sorted = df.sort_values(score_col)
            ax.barh(df_sorted["Feature"], df_sorted[score_col], color="steelblue")
            ax.set_title(f"{method}\n{target_col}")
            ax.set_xlabel("Score")
        else:
            ax.text(0.5, 0.5, f"No data for {method}", ha="center", va="center")
    
    plt.tight_layout()
    return fig


def plot_model_comparison_across_targets(metrics_rt: pd.DataFrame, metrics_rl: pd.DataFrame):
    """
    Compare model performance across Risk_Type and Risk_Level.
    """
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
# APP
# -------------------------------------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        """
        This app demonstrates the analysis and modeling workflow for predicting **Risk_Type**
        and **Risk_Level** using microplastic and environmental features.

        ✅ Modeling + CV are leakage-safe (Pipeline does preprocessing inside train/CV folds).  
        ✅ Numeric coercion prevents SimpleImputer fit errors.  
        ✅ OneHotEncoder safely handles unknown categories with `handle_unknown='ignore'`.
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
        "⚙️ Optimization": [
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
        help="These columns have many unique values and cause huge one-hot matrices. Keep them for EDA, drop for modeling."
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
                st.subheader("Distribution of Risk_Score (Histogram & Boxplot)")
                st.pyplot(plot_hist_box(df_raw, "Risk_Score"))
            else:
                st.info("Column 'Risk_Score' not found in the dataset.")

        with tab3:
            if "MP_Count_per_L" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Relationship between Risk_Score and MP_Count_per_L")
                st.pyplot(plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score"))
            else:
                st.info("Columns 'MP_Count_per_L' and/or 'Risk_Score' not found.")

        with tab4:
            if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Difference in Risk_Score by Risk_Level (Boxplot)")
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
                st.info("Columns 'Risk_Level' and/or 'Risk_Score' not found.")

    # -------------------- PAGE 2 - PREPROCESSING --------------------
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Comprehensive Preprocessing & Feature Engineering")
        
        st.markdown("""
        This page shows detailed preprocessing steps with tables for each transformation:
        1. **Categorical Encoding** - One-hot encoding for features, label encoding for targets
        2. **Outlier Detection & Handling** - IQR method for numerical columns
        3. **Feature Scaling** - Standardization (z-score normalization)
        4. **Skewness Analysis** - Log transformation for skewed columns
        5. **Final Preprocessed Data** - Complete dataset ready for modeling
        """)
        
        # Step 1: Categorical Encoding
        st.subheader("Step 1: Categorical Encoding")
        categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns]
        
        df_encoded, encoding_report = encode_categorical_columns(df_raw, categorical_features, drop_cols_for_model)
        
        if not encoding_report.empty:
            st.write("**Encoding Report:**")
            st.dataframe(encoding_report, use_container_width=True)
            st.info("✅ One-hot encoding applied to categorical features, label encoding for target variables")
        
        # Step 2: Outlier Detection and Handling
        st.subheader("Step 2: Outlier Detection & Handling (IQR Method)")
        numeric_cols_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
        
        df_outliers_handled, outlier_report = detect_and_handle_outliers(df_raw, numeric_cols_present)
        
        if not outlier_report.empty:
            st.write("**Outlier Detection Report:**")
            st.dataframe(outlier_report, use_container_width=True)
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_outliers = outlier_report["Outliers_Detected"].sum()
                st.metric("Total Outliers Found", int(total_outliers))
            with col2:
                avg_outlier_pct = outlier_report["Outlier_Percentage"].mean()
                st.metric("Avg Outlier %", f"{avg_outlier_pct:.2f}%")
            with col3:
                cols_with_outliers = len(outlier_report[outlier_report["Outliers_Detected"] > 0])
                st.metric("Cols with Outliers", int(cols_with_outliers))
            
            st.info("✅ Outliers have been capped at IQR bounds (Q1 - 1.5×IQR, Q3 + 1.5×IQR)")
        
        # Step 3: Feature Scaling
        st.subheader("Step 3: Feature Scaling (StandardScaler)")
        df_scaled, _, scaling_report = scale_numeric_with_report(df_outliers_handled, numeric_cols_present)
        
        if not scaling_report.empty:
            st.write("**Scaling Report (Before → After):**")
            st.dataframe(scaling_report, use_container_width=True)
            st.info("✅ StandardScaler (z-score) applied: (x - mean) / std → scaled data has mean ≈ 0, std ≈ 1")
        
        # Step 4: Skewness Analysis & Transformation
        st.subheader("Step 4: Skewness Analysis & Log Transformation")
        df_transformed, skewness_before, skewed_cols, transformation_report = transform_skewed(
            df_scaled, numeric_cols_present, threshold=0.5
        )
        
        if not transformation_report.empty:
            st.write("**Skewness Transformation Report:**")
            st.dataframe(transformation_report, use_container_width=True)
            st.success(f"✅ {len(skewed_cols)} highly skewed column(s) transformed using log1p")
        else:
            st.info("ℹ️ No highly skewed columns detected (|skewness| ≤ 0.5)")
        
        # Summary visualization
        st.write("**Skewness Before & After:**")
        skewness_summary = pd.DataFrame({
            "Column": numeric_cols_present,
            "Skewness_Before": skewness_before[numeric_cols_present].values,
            "Transformed": [col in skewed_cols for col in numeric_cols_present]
        })
        st.dataframe(skewness_summary, use_container_width=True)
        
        # Step 5: Final Preprocessed Data Summary
        st.subheader("Step 5: Final Preprocessed Data (Ready for Modeling)")
        
        # Display sample of final preprocessed data
        st.write("**Final Preprocessed Dataset (First 10 rows):**")
        display_cols = numeric_cols_present + [c for c in categorical_features if c not in drop_cols_for_model]
        if display_cols:
            st.dataframe(df_transformed[display_cols].head(10), use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df_transformed))
        with col2:
            st.metric("Total Features", len(display_cols))
        with col3:
            st.metric("Numeric Features", len(numeric_cols_present))
        with col4:
            st.metric("Categorical Features", len([c for c in categorical_features if c not in drop_cols_for_model]))
        
        # Final descriptive statistics
        st.write("**Descriptive Statistics (Processed Data):**")
        if numeric_cols_present:
            st.dataframe(df_transformed[numeric_cols_present].describe(), use_container_width=True)
        
        st.info(
            "💡 **Note:** For modeling, preprocessing is performed inside a Pipeline to ensure "
            "leakage-safe validation. This page is for EDA and interpretation only."
        )

    # -------------------- PAGE 3 - FEATURE SELECTION --------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance (Model-based)")
        
        st.markdown("""
        This section performs comprehensive feature selection and ranking using:
        - **Mutual Information** - Dependency measurement
        - **Chi-squared** - Statistical independence test
        - **Random Forest Importance** - Model-based ranking
        """)
        
        tab_rt, tab_rl = st.tabs([TARGET_RISK_TYPE, TARGET_RISK_LEVEL])

        def run_feature_analysis(target_col: str):
            try:
                X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)
            except Exception as e:
                st.error(f"Error: {e}")
                return
            
            if y.nunique() < 2:
                st.warning(f"Not enough classes in {target_col} (need ≥ 2).")
                return

            st.write(f"**Data prepared:** {X.shape[0]} samples × {X.shape[1]} features")
            st.write(f"**Class distribution:** {dict(y.value_counts())}")

            # Feature Ranking
            st.subheader("Feature Ranking Results")
            
            with st.spinner("Computing feature rankings..."):
                # Prepare numeric data
                X_numeric = X.copy()
                for col in X_numeric.columns:
                    if X_numeric[col].dtype == "object":
                        le = LabelEncoder()
                        X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
                
                ranking_results = rank_features_multi_method(X_numeric, y, n_top=15)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Mutual Information:**")
                if "Mutual Information" in ranking_results and not ranking_results["Mutual Information"].empty:
                    st.dataframe(ranking_results["Mutual Information"], use_container_width=True, height=400)
            
            with col2:
                st.write("**Chi-squared:**")
                if "Chi-squared" in ranking_results and not ranking_results["Chi-squared"].empty:
                    st.dataframe(ranking_results["Chi-squared"], use_container_width=True, height=400)
            
            with col3:
                st.write("**Random Forest:**")
                if "Random Forest" in ranking_results and not ranking_results["Random Forest"].empty:
                    st.dataframe(ranking_results["Random Forest"], use_container_width=True, height=400)
            
            # Visualization
            if ranking_results:
                st.pyplot(plot_feature_ranking_comparison(ranking_results, target_col))
            
            # Model Training
            st.subheader("Classification Model Performance")
            
            (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
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
                        st.warning(f"Model {model_name} failed: {e}")
                
                if metrics_list:
                    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
                    st.dataframe(metrics_df.round(4), use_container_width=True)
                    st.pyplot(plot_metrics_bar(metrics_df, f"({target_col})"))

        with tab_rt:
            if TARGET_RISK_TYPE in df_raw.columns:
                run_feature_analysis(TARGET_RISK_TYPE)
            else:
                st.warning(f"{TARGET_RISK_TYPE} not found in dataset.")

        with tab_rl:
            if TARGET_RISK_LEVEL in df_raw.columns:
                run_feature_analysis(TARGET_RISK_LEVEL)
            else:
                st.warning(f"{TARGET_RISK_LEVEL} not found in dataset.")

    # -------------------- PAGE 4 --------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Classification Modeling (Tasks 4, 5 & 7)")
        tab1, tab2 = st.tabs(["Risk_Type", "Risk_Level"])

        with tab1:
            if TARGET_RISK_TYPE not in df_raw.columns:
                st.warning("Risk_Type column not found.")
            else:
                st.subheader("Models for Risk-Type (Holdout split)")
                with st.spinner("Training models (cached)."):
                    _, metrics_rt, split_info_rt, split_note_rt = train_holdout_models_cached(
                        df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )
                st.dataframe(metrics_rt.round(3))
                st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))
                st.info(split_note_rt)

        with tab2:
            if TARGET_RISK_LEVEL not in df_raw.columns:
                st.warning("Risk_Level column not found.")
            else:
                st.subheader("Models for Risk-Level (Holdout split)")
                with st.spinner("Training models (cached)."):
                    _, metrics_rl, split_info_rl, split_note_rl = train_holdout_models_cached(
                        df_raw, TARGET_RISK_LEVEL, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )
                st.dataframe(metrics_rl.round(3))
                st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))
                st.info(split_note_rl)

    # -------------------- PAGE 5 --------------------
    elif page == "Cross Validation (K-Fold)":
        st.header("Cross Validation (K-Fold) for Classification Model")

        target = TARGET_RISK_TYPE
        model_name = "Logistic Regression"
        
        n_splits = st.slider("Number of folds (k)", min_value=3, max_value=5, value=3, step=1)
        stratified = st.checkbox("Use Stratified K-Fold", value=True)

        if len(df_raw) > 500:
            st.warning(f"Dataset has {len(df_raw)} rows. Sampling 500 rows for efficient CV.")
            df_cv = df_raw.sample(500, random_state=42).reset_index(drop=True)
        else:
