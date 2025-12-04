import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from pandas.errors import EmptyDataError, ParserError


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Microplastic Risk Analysis", layout="wide")

NUMERIC_COLS = [
    "MP_Count_per_L",
    "Risk_Score",
    "Microplastic_Size_mm_midpoint",
    "Density_midpoint",
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
]

TARGET_RISK_TYPE = "Risk_Type"
TARGET_RISK_LEVEL = "Risk_Level"


# -------------------------------------------------------
# PERFORMANCE / SPEED CONTROLS
# -------------------------------------------------------
def get_perf_settings(mode: str):
    if mode == "Fast":
        return {"rf_estimators": 80, "grid_cv": 3, "cv_n_jobs": -1, "hist_kde": False}
    if mode == "Accurate":
        return {"rf_estimators": 300, "grid_cv": 5, "cv_n_jobs": -1, "hist_kde": True}
    return {"rf_estimators": 150, "grid_cv": 4, "cv_n_jobs": -1, "hist_kde": True}


def make_models(speed_mode: str, rf_estimators: int):
    """
    Speed-optimized model presets (Quick is MUCH faster on wide one-hot data).
    """
    if speed_mode == "Quick":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=600, solver="saga", n_jobs=-1, multi_class="auto"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=min(rf_estimators, 80),
                max_depth=12,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            ),
        }

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=rf_estimators, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None, path: str = "Microplastic.csv"):
    src = uploaded_file if uploaded_file is not None else path
    encodings_to_try = ["latin1", "utf-8", "cp1252"]

    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(src, encoding=enc)
            return df
        except (UnicodeDecodeError, EmptyDataError, ParserError) as e:
            last_err = e
            continue
        except FileNotFoundError:
            if uploaded_file is None:
                raise

    if last_err is not None:
        raise last_err
    return None


# -------------------------------------------------------
# PREPROCESSING (EDA-style)
# -------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.fillna(s.median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])

    return df


def cap_outliers_iqr(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        clipped = np.where(s < low, low, s)
        clipped = np.where(clipped > high, high, clipped)
        df[col] = clipped
    return df


def transform_skewed(df: pd.DataFrame, cols):
    df = df.copy()
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return df, pd.Series(dtype=float), []

    for col in cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    skewness = df[cols_present].skew(numeric_only=True)
    skewed_cols = skewness[skewness.abs() > 1].index.tolist()

    for col in skewed_cols:
        min_val = df[col].min()
        if pd.isna(min_val):
            continue
        shift = (abs(min_val) + 1e-6) if min_val <= 0 else 0
        df[col] = np.log1p(df[col] + shift)

    return df, skewness, skewed_cols


def scale_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    scaler = StandardScaler()
    cols_present = [c for c in cols if c in df.columns]
    if cols_present:
        df[cols_present] = scaler.fit_transform(df[cols_present])
    return df, scaler


def limit_cardinality_column(series: pd.Series, top_n=30, other_label="Other"):
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    vc = s.value_counts(dropna=True)
    keep = vc.head(top_n).index
    return np.where(s.isin(keep), s, other_label)


def preprocess_for_model(df: pd.DataFrame, limit_high_cardinality: bool = True, top_n_card: int = 30):
    df = df.copy()

    if TARGET_RISK_TYPE in df.columns and TARGET_RISK_LEVEL in df.columns:
        df = df.dropna(subset=[TARGET_RISK_TYPE, TARGET_RISK_LEVEL])

    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, NUMERIC_COLS)
    df, skewness, skewed_cols = transform_skewed(df, NUMERIC_COLS)
    df, _ = scale_numeric(df, NUMERIC_COLS)

    y_type = df[TARGET_RISK_TYPE] if TARGET_RISK_TYPE in df.columns else None
    y_level = df[TARGET_RISK_LEVEL] if TARGET_RISK_LEVEL in df.columns else None

    drop_cols = [c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    # HUGE speed win: collapse high-cardinality categories BEFORE get_dummies
    if limit_high_cardinality:
        for c in ["Location", "Author"]:
            if c in feature_df.columns:
                feature_df[c] = limit_cardinality_column(feature_df[c], top_n=top_n_card, other_label="Other")

    existing_cat_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns]
    X = pd.get_dummies(feature_df, columns=existing_cat_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df, X, y_type, y_level, skewness, skewed_cols


@st.cache_data(show_spinner=False)
def preprocess_for_model_cached(df_raw: pd.DataFrame, limit_high_cardinality: bool, top_n_card: int):
    return preprocess_for_model(df_raw, limit_high_cardinality=limit_high_cardinality, top_n_card=top_n_card)


# -------------------------------------------------------
# SPLIT HELPERS (STRATIFY FIX)
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
# MODEL TRAINING (cached)
# -------------------------------------------------------
def train_models(X, y, test_size=0.2, rf_estimators: int = 200, speed_mode: str = "Quick", selected_models=None):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(str)

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to train models.")

    before_counts = y.value_counts()
    y_merged = merge_rare_classes(y, min_count=2, other_label="Other").astype(str)
    after_counts = y_merged.value_counts()

    merge_note = None
    if not before_counts.equals(after_counts):
        merge_note = {"before": before_counts, "after": after_counts}

    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y_merged, test_size=test_size, random_state=42
    )

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

    models_all = make_models(speed_mode=speed_mode, rf_estimators=rf_estimators)
    if selected_models:
        models_all = {k: v for k, v in models_all.items() if k in selected_models}

    metrics_list = []
    for name, model in models_all.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    return metrics_df, split_info, split_note, merge_note


@st.cache_data(show_spinner=False)
def cached_train_models(X: pd.DataFrame, y: pd.Series, test_size: float, rf_estimators: int, speed_mode: str, selected_models_tuple):
    metrics_df, split_info, split_note, merge_note = train_models(
        X, y, test_size=test_size, rf_estimators=rf_estimators, speed_mode=speed_mode,
        selected_models=list(selected_models_tuple) if selected_models_tuple else None
    )

    split_info_slim = {
        "X_train_shape": split_info["X_train_shape"],
        "X_test_shape": split_info["X_test_shape"],
        "y_train_counts": split_info["y_train_counts"].to_dict(),
        "y_test_counts": split_info["y_test_counts"].to_dict(),
        "used_stratify": split_info["used_stratify"],
        "final_test_size": split_info["final_test_size"],
    }

    merge_note_slim = None
    if merge_note is not None:
        merge_note_slim = {"before": merge_note["before"].to_dict(), "after": merge_note["after"].to_dict()}

    return metrics_df, split_info_slim, split_note, merge_note_slim


def smote_and_tune_logreg(X, y, test_size=0.2, grid_cv: int = 5):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(str)

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to run SMOTE and tuning.")

    before_counts = y.value_counts()
    y_merged = merge_rare_classes(y, min_count=2, other_label="Other").astype(str)
    after_counts = y_merged.value_counts()

    merge_note = None
    if not before_counts.equals(after_counts):
        merge_note = {"before": before_counts, "after": after_counts}

    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y_merged, test_size=test_size, random_state=42
    )

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

    smote_used = True
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except ValueError:
        smote_used = False
        X_res, y_res = X_train, y_train

    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, multi_class="auto"),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=grid_cv,
        n_jobs=-1,
    )
    grid.fit(X_res, y_res)

    best_lr = grid.best_estimator_
    y_pred = best_lr.predict(X_test)

    tuned_metrics = pd.DataFrame([{
        "Model": "LogReg (tuned + SMOTE)" if smote_used else "LogReg (tuned, no SMOTE)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }]).set_index("Model")

    return best_lr, tuned_metrics, grid.best_params_, split_info, split_note, merge_note, smote_used


# -------------------------------------------------------
# LEAKAGE-SAFE CV HELPERS
# -------------------------------------------------------
def build_preprocess_pipeline(df_raw: pd.DataFrame):
    numeric_features = [c for c in NUMERIC_COLS if c in df_raw.columns]
    categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns]

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


@st.cache_data(show_spinner=False)
def cached_cv(df_raw: pd.DataFrame, target_col: str, model_name: str,
              n_splits: int, stratified: bool, use_smote: bool, rf_estimators: int, n_jobs: int):
    summary_df, scores = run_cross_validation(
        df_raw=df_raw,
        target_col=target_col,
        model_name=model_name,
        n_splits=n_splits,
        stratified=stratified,
        use_smote=use_smote,
        rf_estimators=rf_estimators,
        n_jobs=n_jobs
    )
    # keep scores small-ish
    slim = {
        "test_accuracy": scores["test_accuracy"].tolist(),
        "test_precision_w": scores["test_precision_w"].tolist(),
        "test_recall_w": scores["test_recall_w"].tolist(),
        "test_f1_w": scores["test_f1_w"].tolist(),
    }
    return summary_df, slim


def run_cross_validation(df_raw: pd.DataFrame, target_col: str, model_name: str,
                         n_splits: int = 5, stratified: bool = True, use_smote: bool = False,
                         rf_estimators: int = 150, n_jobs: int = -1):
    if target_col not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df_raw.dropna(subset=[target_col]).copy()
    y = df[target_col]
    X = df.drop(columns=[c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns])

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target for cross-validation.")

    y = merge_rare_classes(y, min_count=2, other_label="Other").astype(str)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=rf_estimators, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    if model_name not in models:
        raise ValueError("Unknown model selected.")
    model = models[model_name]

    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    preprocessor = build_preprocess_pipeline(df_raw)

    if use_smote:
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

    scoring = {
        "accuracy": "accuracy",
        "precision_w": "precision_weighted",
        "recall_w": "recall_weighted",
        "f1_w": "f1_weighted",
    }

    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, error_score="raise")

    summary = {}
    for k in scoring.keys():
        arr = scores[f"test_{k}"]
        summary[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    summary_df = pd.DataFrame(summary).T.rename(index={
        "accuracy": "Accuracy",
        "precision_w": "Precision (weighted)",
        "recall_w": "Recall (weighted)",
        "f1_w": "F1-score (weighted)",
    })

    return summary_df, scores


# -------------------------------------------------------
# VISUALS
# -------------------------------------------------------
def plot_hist_box(df, col, kde=True):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if len(s) == 0:
        axes[0].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
        axes[1].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
    else:
        sns.histplot(s, kde=kde, ax=axes[0])
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
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_box_by_category_readable(
    df,
    value_col,
    category_col,
    top_n=8,
    other_label="Other",
    figsize=(12, 5),
    horizontal=True,
):
    val = pd.to_numeric(df[value_col], errors="coerce")
    cat = (
        df[category_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )

    data = pd.DataFrame({value_col: val, category_col: cat}).dropna(subset=[value_col, category_col])

    fig, ax = plt.subplots(figsize=figsize)
    if data.empty:
        ax.text(0.5, 0.5, f"No usable data for {value_col} by {category_col}", ha="center", va="center")
        plt.tight_layout()
        return fig

    counts = data[category_col].value_counts()
    keep = counts.head(top_n).index
    data[category_col] = np.where(data[category_col].isin(keep), data[category_col], other_label)

    order = data.groupby(category_col)[value_col].median().sort_values().index.tolist()

    if horizontal:
        sns.boxplot(data=data, y=category_col, x=value_col, order=order, ax=ax)
    else:
        sns.boxplot(data=data, x=category_col, y=value_col, order=order, ax=ax)
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

    st.sidebar.header("Performance")
    perf_mode = st.sidebar.selectbox("Performance mode", ["Fast", "Balanced", "Accurate"], index=0)
    perf = get_perf_settings(perf_mode)

    st.sidebar.subheader("Model speed mode")
    model_speed_mode = st.sidebar.selectbox("Model speed", ["Quick", "Balanced"], index=0)

    st.sidebar.subheader("High-cardinality control (Speed Boost)")
    limit_high_cardinality = st.sidebar.checkbox("Limit Location/Author categories", value=True)
    top_n_card = st.sidebar.slider("Keep top N categories (Location/Author)", 10, 100, 30, 5)

    st.sidebar.divider()
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Data Overview & Task 1",
            "Preprocessing (Task 2)",
            "Feature Selection & Relevance (Task 3 & 6)",
            "Classification Modeling (Tasks 4, 5 & 7)",
            "Polymer Type Distribution",
            "SMOTE & Hyperparameter Tuning (Risk_Type)",
            "Cross Validation (K-Fold)",
        ],
    )

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

    # Cached preprocessing with high-cardinality controls
    df_clean, X, y_type, y_level, skewness, skewed_cols = preprocess_for_model_cached(
        df_raw, limit_high_cardinality=limit_high_cardinality, top_n_card=top_n_card
    )

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
                st.pyplot(plot_hist_box(df_raw, "Risk_Score", kde=perf["hist_kde"]))
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
        st.header("Task 2: Preprocessing")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Before Preprocessing",
            "After Preprocessing",
            "Skewness",
            "Encoded Features",
        ])

        with tab1:
            numeric_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
            if numeric_present:
                st.subheader("Before Preprocessing – Descriptive Stats")
                st.write(df_raw[numeric_present].describe())
            else:
                st.info("No numeric columns found for descriptive stats.")

        with tab2:
            numeric_present_clean = [c for c in NUMERIC_COLS if c in df_clean.columns]
            if numeric_present_clean:
                st.subheader("After Preprocessing – Descriptive Stats")
                st.write(df_clean[numeric_present_clean].describe())
            else:
                st.info("No cleaned numeric columns found.")

        with tab3:
            st.subheader("Skewness (Before Transform)")
            st.write(skewness)
            if len(skewed_cols) > 0:
                st.write("Skewed columns transformed (log1p):")
                st.write(skewed_cols)
            else:
                st.write("No columns exceeded skewness threshold.")

        with tab4:
            st.subheader("Encoded Feature Matrix (X) – First 10 Rows")
            st.dataframe(X.head(10))
            st.write("Shape of X:", X.shape)

    # -------------------- PAGE 3 --------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance")

        tab_rt, tab_rl = st.tabs(["Risk_Type Feature Importance", "Risk_Level Feature Importance"])

        with tab_rt:
            if y_type is None:
                st.warning("Risk_Type column not found.")
            else:
                if st.button("Compute Feature Importance (Risk_Type)"):
                    rf = RandomForestClassifier(n_estimators=perf["rf_estimators"], random_state=42, n_jobs=-1)
                    rf.fit(X, y_type.astype(str))
                    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.session_state["rt_featimp"] = imp

                if "rt_featimp" in st.session_state:
                    imp = st.session_state["rt_featimp"]
                    st.dataframe(imp.head(10))
                    fig = plt.figure(figsize=(8, 4))
                    imp.head(10).sort_values().plot(kind="barh")
                    plt.title("Top 10 Feature Importances (Risk_Type)")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Click the button to compute feature importance.")

        with tab_rl:
            if y_level is None:
                st.warning("Risk_Level column not found.")
            else:
                if st.button("Compute Feature Importance (Risk_Level)"):
                    rf = RandomForestClassifier(n_estimators=perf["rf_estimators"], random_state=42, n_jobs=-1)
                    rf.fit(X, y_level.astype(str))
                    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.session_state["rl_featimp"] = imp

                if "rl_featimp" in st.session_state:
                    imp = st.session_state["rl_featimp"]
                    st.dataframe(imp.head(10))
                    fig = plt.figure(figsize=(8, 4))
                    imp.head(10).sort_values().plot(kind="barh")
                    plt.title("Top 10 Feature Importances (Risk_Level)")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Click the button to compute feature importance.")

    # -------------------- PAGE 4 (KEY FIXES HERE) --------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Tasks 4, 5 & 7: Classification Modeling")

        st.info(
            "Speed tips applied: ✅ cached training results ✅ optional model selection ✅ quick model presets ✅ "
            "high-cardinality category limiting (sidebar)."
        )

        selected_models = st.multiselect(
            "Choose models to run (running fewer = faster)",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"],
            default=["Logistic Regression", "Random Forest"],
        )
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

        tab1, tab2 = st.tabs(["Risk-Type Models", "Risk-Level Models"])

        with tab1:
            if y_type is None:
                st.warning("Risk_Type column not found; cannot train models for Risk-Type.")
            else:
                if st.button("Run Risk-Type Models", type="primary"):
                    with st.spinner("Training (cached after first run)..."):
                        metrics_rt, split_info_rt, split_note_rt, merge_note_rt = cached_train_models(
                            X, y_type, test_size, perf["rf_estimators"], model_speed_mode, tuple(selected_models)
                        )

                    st.dataframe(metrics_rt.round(3))  # faster than style.format
                    st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))
                    st.info(split_note_rt)

                    if merge_note_rt is not None:
                        with st.expander("Rare-class merging details (small classes → 'Other')"):
                            st.write("Before:")
                            st.write(pd.Series(merge_note_rt["before"]))
                            st.write("After:")
                            st.write(pd.Series(merge_note_rt["after"]))

                    st.write("Train distribution:")
                    st.write(pd.Series(split_info_rt["y_train_counts"]))
                    st.write("Test distribution:")
                    st.write(pd.Series(split_info_rt["y_test_counts"]))
                else:
                    st.caption("Click the button to run training. Results are cached, so subsequent visits are fast.")

        with tab2:
            if y_level is None:
                st.warning("Risk_Level column not found; cannot train models for Risk-Level.")
            else:
                if st.button("Run Risk-Level Models", type="primary"):
                    with st.spinner("Training (cached after first run)..."):
                        metrics_rl, split_info_rl, split_note_rl, merge_note_rl = cached_train_models(
                            X, y_level, test_size, perf["rf_estimators"], model_speed_mode, tuple(selected_models)
                        )

                    st.dataframe(metrics_rl.round(3))  # faster than style.format
                    st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))
                    st.info(split_note_rl)

                    if merge_note_rl is not None:
                        with st.expander("Rare-class merging details (small classes → 'Other')"):
                            st.write("Before:")
                            st.write(pd.Series(merge_note_rl["before"]))
                            st.write("After:")
                            st.write(pd.Series(merge_note_rl["after"]))

                    st.write("Train distribution:")
                    st.write(pd.Series(split_info_rl["y_train_counts"]))
                    st.write("Test distribution:")
                    st.write(pd.Series(split_info_rl["y_test_counts"]))
                else:
                    st.caption("Click the button to run training. Results are cached, so subsequent visits are fast.")

    # -------------------- PAGE 5 --------------------
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution")

        if "Polymer_Type" in df_raw.columns:
            polymer = df_raw["Polymer_Type"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
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

    # -------------------- PAGE 6 --------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("Address Class Imbalance & Tune Logistic Regression (Risk-Type)")

        if y_type is None:
            st.warning("Risk_Type column not found; cannot run SMOTE or tuning.")
            return

        tab1, tab2 = st.tabs(["Original Distribution & Base Models", "SMOTE + Tuning"])

        with tab1:
            st.subheader("Class Distribution of Risk-Type (Original)")
            st.write(pd.Series(y_type).value_counts())

            st.info("Use the Classification Modeling page for faster base-model runs (cached + selectable models).")

        with tab2:
            st.subheader("SMOTE + Hyperparameter Tuning")

            if st.button("Run SMOTE + GridSearchCV", type="primary"):
                with st.spinner("Running SMOTE + GridSearchCV..."):
                    best_lr, tuned_metrics, best_params, split_info_smote, split_note_smote, merge_note_smote, smote_used = (
                        smote_and_tune_logreg(X, y_type, grid_cv=perf["grid_cv"])
                    )

                st.write("Best Hyperparameters:")
                st.json(best_params)
                st.info(split_note_smote)

                if not smote_used:
                    st.warning("SMOTE could not be applied due to very small minority classes; tuning continued without SMOTE.")

                st.subheader("Tuned Logistic Regression Performance")
                st.dataframe(tuned_metrics.round(3))

    # -------------------- PAGE 7 --------------------
    elif page == "Cross Validation (K-Fold)":
        st.header("Cross Validation (K-Fold / Stratified K-Fold)")

        st.markdown(
            """
            This page runs **leakage-safe cross-validation** using a preprocessing **Pipeline**.
            - For classification, **Stratified K-Fold** is recommended.
            - If enabled, **SMOTE is applied inside each fold**.
            """
        )

        target = st.selectbox("Select target", [TARGET_RISK_TYPE, TARGET_RISK_LEVEL])
        model_name = st.selectbox("Select model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
        n_splits = st.slider("Number of folds (k)", min_value=3, max_value=10, value=5, step=1)
        stratified = st.checkbox("Use Stratified K-Fold (recommended for classification)", value=True)

        use_smote = st.checkbox("Use SMOTE", value=False)

        if st.button("Run Cross-Validation", type="primary"):
            try:
                with st.spinner("Running CV (cached after first run)..."):
                    summary_df, slim_scores = cached_cv(
                        df_raw=df_raw,
                        target_col=target,
                        model_name=model_name,
                        n_splits=n_splits,
                        stratified=stratified,
                        use_smote=use_smote,
                        rf_estimators=max(perf["rf_estimators"], 150),
                        n_jobs=perf["cv_n_jobs"],
                    )

                st.success("Done!")
                show = summary_df.copy()
                show["mean±std"] = show.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
                st.dataframe(show[["mean±std"]])

                with st.expander("Show per-fold scores"):
                    fold_df = pd.DataFrame({
                        "Accuracy": slim_scores["test_accuracy"],
                        "Precision (weighted)": slim_scores["test_precision_w"],
                        "Recall (weighted)": slim_scores["test_recall_w"],
                        "F1-score (weighted)": slim_scores["test_f1_w"],
                    })
                    fold_df.index = [f"Fold {i+1}" for i in range(len(fold_df))]
                    st.dataframe(fold_df.round(3))

            except Exception as e:
                st.error(f"Cross-validation failed: {e}")


if __name__ == "__main__":
    main()
