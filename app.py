import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import EmptyDataError, ParserError

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    df = df.copy()
    present = [c for c in numeric_cols if c in df.columns]
    skewness = df[present].apply(lambda x: pd.to_numeric(x, errors="coerce")).skew(numeric_only=True)
    skewed_cols = skewness[skewness.abs() > threshold].index.tolist()

    for col in skewed_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        shift = -s.min() if s.min() < 0 else 0
        df[col] = np.log1p(s + shift)
    return df, skewness, skewed_cols


def scale_numeric(df: pd.DataFrame, numeric_cols):
    df = df.copy()
    scaler = StandardScaler()
    present = [c for c in numeric_cols if c in df.columns]
    if present:
        vals = df[present].apply(pd.to_numeric, errors="coerce")
        df[present] = scaler.fit_transform(vals.fillna(vals.median()))
    return df, scaler


def coerce_numeric_like(df: pd.DataFrame, columns):
    df = df.copy()
    for c in columns:
        if c in df.columns:
            s = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


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
    numeric_features = [c for c in NUMERIC_COLS if c in df_raw.columns]
    numeric_features = [c for c in numeric_features if df_raw[c].notna().any()]

    categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns and c not in drop_cols_for_model]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
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
    Manual, crash-safe cross validation:
    - uses pick_safe_cv() to choose cv object
    - loops over folds manually
    - skips folds where y_train has < 2 classes
    - catches exceptions per fold, sets metrics to NaN
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
        ✅ Model Validation is simplified & lightweight (Logistic Regression, optional row sampling).
        """
    )

    # 🔼 DATA UPLOAD SECTION SA TAAS (MAIN PAGE)
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload Microplastic CSV",
        type=["csv"],
        help="If you don't upload anything, the app will try to use 'Microplastic.csv' from the app folder."
    )

    if uploaded_file is not None:
        st.success("✅ CSV uploaded successfully. All pages will use this dataset.")
    else:
        st.info("Using default 'Microplastic.csv' in the app folder (if available).")

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
        "📊 Visualization": [
            "Visualization Dashboard",
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

    # load data using uploaded_file from top
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

    # -------------------- PAGE 2 --------------------
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Preprocessing (EDA view)")
        df_clean = handle_missing_values(df_raw)
        df_clean = cap_outliers_iqr(df_clean, NUMERIC_COLS)
        df_clean, skewness, skewed_cols = transform_skewed(df_clean, NUMERIC_COLS)
        df_clean, _ = scale_numeric(df_clean, NUMERIC_COLS)

        tab1, tab2 = st.tabs(["Descriptive Stats", "Skewness & Notes"])

        with tab1:
            numeric_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
            if numeric_present:
                st.subheader("Descriptive Stats (Raw)")
                st.write(df_raw[numeric_present].describe())
                st.subheader("Descriptive Stats (Cleaned)")
                st.write(df_clean[numeric_present].describe())
            else:
                st.info("No numeric columns found for descriptive stats.")

        with tab2:
            st.subheader("Skewness (Before Transform)")
            st.write(skewness)
            if len(skewed_cols) > 0:
                st.write("Skewed columns transformed (log1p):")
                st.write(skewed_cols)

            st.info(
                "Note: For modeling/CV, preprocessing is done inside a Pipeline (leakage-safe). "
                "This page is for EDA/interpretation only."
            )

    # -------------------- PAGE 3 --------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance (Model-based)")
        tab_rt, tab_rl = st.tabs(["Risk_Type (RF importance)", "Risk_Level (RF importance)"])

        def rf_importance(target_col: str):
            X, y = get_Xy_for_target(df_raw, target_col, drop_cols_for_model)
            if y.nunique() < 2:
                st.warning(f"Not enough classes in {target_col} after cleaning (need at least 2).")
                return

            preprocessor = build_preprocess_pipeline_cached(df_raw, drop_cols_for_model)
            rf = RandomForestClassifier(n_estimators=200 if fast_mode else 400, random_state=42, n_jobs=-1)
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", rf)])
            pipe.fit(X, y)

            try:
                feat_names = pipe.named_steps["prep"].get_feature_names_out()
            except Exception:
                feat_names = np.array([f"f{i}" for i in range(rf.feature_importances_.shape[0])])

            importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
            top = importances.head(25)

            st.subheader("Top 25 feature importances")
            st.dataframe(top.rename("importance"))

            fig, ax = plt.subplots(figsize=(10, 6))
            top.sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"RandomForest Feature Importance — {target_col}")
            plt.tight_layout()
            st.pyplot(fig)

        with tab_rt:
            if TARGET_RISK_TYPE not in df_raw.columns:
                st.warning("Risk_Type column not found; cannot compute importance.")
            else:
                rf_importance(TARGET_RISK_TYPE)

        with tab_rl:
            if TARGET_RISK_LEVEL not in df_raw.columns:
                st.warning("Risk_Level column not found; cannot compute importance.")
            else:
                rf_importance(TARGET_RISK_LEVEL)

    # -------------------- PAGE 4 --------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Classification Modeling (Tasks 4, 5 & 7)")
        tab1, tab2 = st.tabs(["Risk_Type", "Risk_Level"])

        with tab1:
            if TARGET_RISK_TYPE not in df_raw.columns:
                st.warning("Risk_Type column not found; cannot train models for Risk-Type.")
            else:
                st.subheader("Models for Risk-Type (Holdout split)")
                with st.spinner("Training models (cached)."):
                    _, metrics_rt, split_info_rt, split_note_rt = train_holdout_models_cached(
                        df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )
                st.dataframe(metrics_rt.round(3))
                st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))
                st.info(split_note_rt)
                st.write("Class distribution in training set:")
                st.write(split_info_rt["y_train_counts"])
                st.write("Class distribution in test set:")
                st.write(split_info_rt["y_test_counts"])

        with tab2:
            if TARGET_RISK_LEVEL not in df_raw.columns:
                st.warning("Risk_Level column not found; cannot train models for Risk-Level.")
            else:
                st.subheader("Models for Risk-Level (Holdout split)")
                with st.spinner("Training models (cached)."):
                    _, metrics_rl, split_info_rl, split_note_rl = train_holdout_models_cached(
                        df_raw, TARGET_RISK_LEVEL, test_size, drop_cols_for_model, fast_mode, use_smote=False
                    )
                st.dataframe(metrics_rl.round(3))
                st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))
                st.info(split_note_rl)
                st.write("Class distribution in training set:")
                st.write(split_info_rl["y_train_counts"])
                st.write("Class distribution in test set:")
                st.write(split_info_rl["y_test_counts"])

        st.subheader("Overall Notes (Speed)")
        st.markdown(
            f"""
            - Current modeling drop columns: **{', '.join(drop_cols_for_model) if drop_cols_for_model else 'None'}**  
            - If the page is slow, keep **Drop Location & Author** ON and keep **Fast Mode** ON.
            """
        )

    # -------------------- PAGE 5 --------------------
    elif page == "Cross Validation (K-Fold)":
        st.header("Cross Validation (K-Fold) for Classification Model")

        st.markdown(
            """
            This section validates a **classification model** using K-Fold Cross Validation.

            - Target variable: **Risk_Type**  
            - Model: **Logistic Regression** (lightweight, stable, good baseline)  
            - Preprocessing (imputation, scaling, encoding) is included inside the pipeline → leakage-safe.
            """
        )

        target = TARGET_RISK_TYPE
        model_name = "Logistic Regression"
        st.info(f"Target fixed to **{target}** and model fixed to **{model_name}** for validation.")

        n_splits = st.slider("Number of folds (k)", min_value=3, max_value=5, value=3, step=1)
        stratified = st.checkbox("Use Stratified K-Fold (recommended for classification)", value=True)

        max_rows = 500
        if len(df_raw) > max_rows:
            st.warning(
                f"Dataset has {len(df_raw)} rows. For stable CV in limited resources, "
                f"we sample {max_rows} rows for cross-validation."
            )
            df_cv = df_raw.sample(max_rows, random_state=42).reset_index(drop=True)
        else:
            df_cv = df_raw.copy()

        st.divider()

        colA, colB = st.columns([1, 2])
        with colA:
            st.subheader("Risk_Type distribution (after rare-class merge)")
            if target in df_cv.columns:
                y_preview = merge_rare_classes(df_cv[target].dropna(), min_count=2, other_label="Other")
                st.write(y_preview.value_counts())
            else:
                st.warning(f"Column '{target}' not found in the dataset.")

        with colB:
            if st.button("Run Cross-Validation", type="primary"):
                with st.spinner("Running K-Fold CV on Logistic Regression (Risk_Type)..."):
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

                        st.markdown(
                            """
                            **Interpretation hint (for defense):**  
                            - *Accuracy* shows overall correct predictions across folds.  
                            - *Precision (weighted)* and *Recall (weighted)* account for class imbalance.  
                            - *F1-score (weighted)* balances precision and recall and is a good summary metric.  
                            """
                        )
                    except Exception as e:
                        st.error(f"CV failed: {e}")

    # -------------------- PAGE 6 --------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("Address Class Imbalance & Tune Logistic Regression (Risk-Type)")

        if TARGET_RISK_TYPE not in df_raw.columns:
            st.warning("Risk_Type column not found; cannot run SMOTE or tuning.")
            return

        tab1, tab2 = st.tabs(["Original Distribution & Base Models", "SMOTE + Tuning & Comparison"])

        with tab1:
            st.subheader("Class Distribution of Risk-Type (after rare-class merge)")
            y_preview = merge_rare_classes(df_raw[TARGET_RISK_TYPE].dropna(), min_count=2, other_label="Other")
            st.write(y_preview.value_counts())

            with st.spinner("Training base models (cached)."):
                _, base_metrics_rt, _, split_note_base = train_holdout_models_cached(
                    df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode, use_smote=False
                )

            st.subheader("Base Models Performance (Risk-Type)")
            st.dataframe(base_metrics_rt.round(3))
            st.pyplot(plot_metrics_bar(base_metrics_rt, "(Risk-Type – Base)"))
            st.info(split_note_base)

        with tab2:
            st.subheader("SMOTE + Hyperparameter Tuning (LogReg)")
            if not IMBLEARN_OK:
                st.error("imbalanced-learn is required. Install: pip install imbalanced-learn")
                st.stop()

            with st.spinner("Running GridSearchCV on training split (leakage-safe)."):
                _, tuned_metrics, best_params, _, split_note_smote = smote_and_tune_logreg_pipeline(
                    df_raw, TARGET_RISK_TYPE, test_size, drop_cols_for_model, fast_mode
                )

            st.write("Best Hyperparameters:")
            st.json(best_params)
            st.info(split_note_smote)

            st.subheader("Tuned Logistic Regression Performance")
            st.dataframe(tuned_metrics.round(3))

            combined = pd.concat([base_metrics_rt, tuned_metrics])
            st.subheader("Comparison: Tuned Logistic Regression vs Base Models")
            st.dataframe(combined.round(3))
            st.pyplot(plot_metrics_bar(combined, "(Risk-Type – Base vs Tuned)"))

    # -------------------- PAGE 7 --------------------
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution")
        df = handle_missing_values(df_raw)

        if "Polymer_Type" in df.columns:
            polymer = df["Polymer_Type"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
            polymer = polymer.dropna()
            vc = polymer.value_counts()

            tabA, tabB = st.tabs(["Counts Table", "Readable Plot (Top N + Other)"])

            with tabA:
                st.subheader("Value Counts of Polymer_Type")
                st.dataframe(vc.rename("count"))

            with tabB:
                st.subheader("Bar Plot of Polymer_Type Distribution (Readable)")
                top_n = st.slider("Show Top N polymer types", min_value=5, max_value=30, value=15, step=1)
                fig, _ = plot_categorical_topn_bar(
                    polymer,
                    title=f"Distribution of Polymer_Type (Top {top_n} + Other)",
                    top_n=top_n,
                    other_label="Other",
                    figsize=(10, 7),
                )
                st.pyplot(fig)
        else:
            st.warning("Column 'Polymer_Type' not found in the dataset.")

    # -------------------- PAGE 8 (LAST) --------------------
    elif page == "Visualization Dashboard":
        st.header("Visualization Dashboard")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Risk Class Distribution",
            "Correlations",
            "Spatial Patterns",
            "Risk vs Factors",
        ])

        # ---- Tab 1: Risk Class Distribution ----
        with tab1:
            st.subheader("Risk_Type and Risk_Level Distribution")

            col1, col2 = st.columns(2)

            if TARGET_RISK_TYPE in df_raw.columns:
                with col1:
                    st.markdown("**Risk_Type Counts**")
                    counts_rt = df_raw[TARGET_RISK_TYPE].dropna()
                    st.dataframe(counts_rt.value_counts().rename("count"))

                    fig, ax = plt.subplots(figsize=(5, 4))
                    counts_rt.value_counts().plot(kind="bar", ax=ax)
                    ax.set_title("Risk_Type Distribution")
                    ax.set_xlabel("Risk_Type")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Risk_Type column not found in dataset.")

            if TARGET_RISK_LEVEL in df_raw.columns:
                with col2:
                    st.markdown("**Risk_Level Counts**")
                    counts_rl = df_raw[TARGET_RISK_LEVEL].dropna()
                    st.dataframe(counts_rl.value_counts().rename("count"))

                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    counts_rl.value_counts().plot(kind="bar", ax=ax2)
                    ax2.set_title("Risk_Level Distribution")
                    ax2.set_xlabel("Risk_Level")
                    ax2.set_ylabel("Count")
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig2)
            else:
                st.info("Risk_Level column not found in dataset.")

            st.markdown(
                """
                ✅ This distribution view helps justify:
                - Class imbalance for **Risk_Type / Risk_Level**
                - Why SMOTE or weighted metrics are important in modeling.
                """
            )

        # ---- Tab 2: Correlations ----
        with tab2:
            st.subheader("Correlation Heatmap (Numeric Features)")
            numeric_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
            if numeric_present:
                num_df = df_raw[numeric_present].apply(pd.to_numeric, errors="coerce")
                corr = num_df.corr()

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No numeric columns found to compute correlations.")

            st.markdown("### Optional Pairwise Relationship")
            if len(numeric_present) >= 2:
                col_x, col_y = st.columns(2)
                with col_x:
                    x_col = st.selectbox("X-axis", numeric_present, index=0)
                with col_y:
                    y_col = st.selectbox("Y-axis", numeric_present, index=1)
                st.pyplot(plot_scatter(df_raw, x_col, y_col))
            else:
                st.info("Need at least two numeric features for scatter plot.")

        # ---- Tab 3: Spatial Patterns ----
        with tab3:
            st.subheader("Spatial Distribution of Sampling Sites")

            if "Latitude" in df_raw.columns and "Longitude" in df_raw.columns:
                lat = pd.to_numeric(df_raw["Latitude"], errors="coerce")
                lon = pd.to_numeric(df_raw["Longitude"], errors="coerce")
                mask = lat.notna() & lon.notna()

                if mask.sum() == 0:
                    st.info("Latitude/Longitude columns are present but contain no valid numeric data.")
                else:
                    color_label = None
                    if TARGET_RISK_LEVEL in df_raw.columns:
                        color_label = TARGET_RISK_LEVEL
                    elif TARGET_RISK_TYPE in df_raw.columns:
                        color_label = TARGET_RISK_TYPE

                    fig, ax = plt.subplots(figsize=(7, 5))
                    if color_label is not None:
                        labels = df_raw.loc[mask, color_label].astype(str)
                        uniq = labels.unique()
                        palette = sns.color_palette("tab10", n_colors=len(uniq))
                        for lab, colr in zip(uniq, palette):
                            sel = labels == lab
                            ax.scatter(
                                lon[mask][sel],
                                lat[mask][sel],
                                label=str(lab),
                                alpha=0.7,
                                s=40
                            )
                        ax.legend(title=color_label, bbox_to_anchor=(1.05, 1), loc="upper left")
                    else:
                        ax.scatter(lon[mask], lat[mask], alpha=0.7, s=40)

                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.set_title("Sampling Locations colored by Risk (if available)")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Latitude/Longitude columns not found. Spatial visualization is skipped.")

        # ---- Tab 4: Risk vs Factors ----
        with tab4:
            st.subheader("Risk vs Microplastic and Environmental Factors")

            if "MP_Count_per_L" in df_raw.columns and TARGET_RISK_LEVEL in df_raw.columns:
                st.markdown("#### MP_Count_per_L by Risk_Level")
                st.pyplot(
                    plot_box_by_category_readable(
                        df_raw,
                        value_col="MP_Count_per_L",
                        category_col=TARGET_RISK_LEVEL,
                        top_n=10,
                        figsize=(10, 5),
                        horizontal=True,
                    )
                )
            else:
                st.info("Need columns 'MP_Count_per_L' and 'Risk_Level' for this plot.")

            st.divider()

            if "Microplastic_Size_mm" in df_raw.columns and TARGET_RISK_TYPE in df_raw.columns:
                st.markdown("#### Microplastic_Size_mm by Risk_Type")
                st.pyplot(
                    plot_box_by_category_readable(
                        df_raw,
                        value_col="Microplastic_Size_mm",
                        category_col=TARGET_RISK_TYPE,
                        top_n=10,
                        figsize=(10, 5),
                        horizontal=True,
                    )
                )
            else:
                st.info("Need columns 'Microplastic_Size_mm' and 'Risk_Type' for this plot.")

            st.divider()

            if "Risk_Score" in df_raw.columns and "Industrial_Activity" in df_raw.columns:
                st.markdown("#### Risk_Score by Industrial_Activity")
                st.pyplot(
                    plot_box_by_category_readable(
                        df_raw,
                        value_col="Risk_Score",
                        category_col="Industrial_Activity",
                        top_n=8,
                        figsize=(10, 5),
                        horizontal=True,
                    )
                )
            else:
                st.info("Need columns 'Risk_Score' and 'Industrial_Activity' for this plot.")

            st.markdown(
                """
                ✅ These visualizations support your discussion on:
                - How microplastic **count/size** relates to risk.  
                - How **industrial activity** and other factors may drive higher risk levels.
                """
            )


if __name__ == "__main__":
    main()
