# -*- coding: utf-8 -*-
"""
Microplastic Risk Modeling Dashboard (Streamlit)

Sequence (as requested):
1) Data Upload & Description
2) EDA (Polymer Type + Risk Score analyses)
3) Preprocessing (Outliers -> Skew transform -> Encoding -> Scaling)
4) Feature Selection
5) Modeling (Objective #1 generic classification)
6) Risk_Type Modeling (Objective #2 with optional SMOTE)
7) Hyperparameter Tuning & Best Model
8) Feature Relevance & Summary

This app is designed to be robust on Streamlit Cloud:
- Handles CSV encodings (utf-8, utf-8-sig, cp1252, latin1) and Excel files.
- SMOTE is optional; app won't crash if imbalanced-learn is missing.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance

# Optional SMOTE (won't crash if not installed)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False


warnings.filterwarnings("ignore")


# =========================
# UI Theme (readable, high-contrast)
# =========================
st.set_page_config(
    page_title="Microplastic Risk Modeling Dashboard",
    page_icon="🧪",
    layout="wide",
)

THEME_CSS = """
<style>
:root{
  --bg: #0b1220;
  --panel: #111a2e;
  --panel2: #0f172a;
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.72);
  --hint: rgba(255,255,255,0.60);
  --accent: #60a5fa;
  --accent2: #34d399;
  --danger: #fb7185;
  --warn: #fbbf24;
  --shadow: 0 10px 30px rgba(0,0,0,0.30);
}

html, body, [class*="css"]{
  color: var(--text) !important;
  background: var(--bg) !important;
}

section.main > div{
  padding-top: 1.25rem;
}

.stApp{
  background: radial-gradient(1200px 800px at 15% 10%, rgba(96,165,250,0.18), transparent 60%),
              radial-gradient(1000px 700px at 90% 15%, rgba(52,211,153,0.12), transparent 55%),
              var(--bg) !important;
}

.block-container{
  max-width: 1250px;
}

[data-testid="stSidebar"]{
  background: rgba(15,23,42,0.85) !important;
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] *{
  color: var(--text) !important;
}

h1, h2, h3, h4{
  color: var(--text) !important;
  letter-spacing: 0.2px;
}

p, li{ color: var(--text) !important; }
label, span{ color: var(--text) !important; }

a{
  color: var(--accent) !important;
}

hr{
  border: none;
  border-top: 1px solid var(--border);
  margin: 1rem 0;
}

.card{
  background: linear-gradient(180deg, rgba(17,26,46,0.92), rgba(15,23,42,0.92));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}

.badge{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid var(--border);
  color: var(--text) !important;
  background: rgba(255,255,255,0.06);
}

.kpi{
  font-size: 28px;
  font-weight: 700;
  margin: 0;
  color: var(--text) !important;
}

.kpi-sub{
  font-size: 12.5px;
  color: var(--hint) !important;
  margin-top: -6px;
}

[data-testid="stMetricValue"]{
  color: var(--text);
}

div.stButton > button{
  background: linear-gradient(135deg, rgba(96,165,250,0.95), rgba(52,211,153,0.85)) !important;
  color: #06111f !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.55rem 1rem !important;
  font-weight: 700 !important;
}

div.stButton > button:hover{
  filter: brightness(1.03);
  transform: translateY(-1px);
}

div.stDownloadButton > button{
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 0.55rem 1rem !important;
}

[data-testid="stDataFrame"]{
  border: 1px solid var(--border) !important;
  border-radius: 14px;
  overflow: hidden;
}

div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--border) !important;
}

input, textarea{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}

.stAlert{
  border-radius: 14px;
  border: 1px solid var(--border);
}


/* Make Streamlit widget labels and inputs readable */
label, .stMarkdown, .stText, .stCaption, .stCheckbox, .stRadio, .stSelectbox, .stMultiSelect, .stSlider {
  color: var(--text) !important;
}

/* Inputs text */
input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea{
  color: var(--text) !important;
}

/* Select/Dropdown text */
div[data-baseweb="select"] span, div[data-baseweb="select"] div{
  color: var(--text) !important;
}

/* Dataframe header text */
[data-testid="stDataFrame"] *{
  color: var(--text) !important;
}


/* File uploader: make dropzone text readable even before hover/click */
[data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid var(--border) !important;
}

[data-testid="stFileUploaderDropzone"] *{
  color: var(--text) !important;
  fill: var(--text) !important;
  opacity: 1 !important;
}

/* Specifically target the helper text lines */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span{
  color: var(--text) !important;
  opacity: 1 !important;
}

/* Ensure the icon is visible */
[data-testid="stFileUploaderDropzone"] svg{
  color: var(--text) !important;
  fill: var(--text) !important;
  opacity: 1 !important;
}

/* The right-side 'Browse files' button text */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button *{
  color: #06111f !important;
}


/* BaseWeb components (Selectbox/Multiselect) dropdown menu + options */
div[data-baseweb="popover"], div[data-baseweb="menu"]{
  background: rgba(15,23,42,0.98) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

div[data-baseweb="menu"] *{
  color: var(--text) !important;
}

/* Options hover/active */
div[data-baseweb="menu"] [role="option"]:hover{
  background: rgba(96,165,250,0.18) !important;
}
div[data-baseweb="menu"] [aria-selected="true"]{
  background: rgba(52,211,153,0.18) !important;
}

/* Selectbox input area */
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--border) !important;
}
div[data-baseweb="select"] *{
  color: var(--text) !important;
}

/* Placeholder text */
div[data-baseweb="select"] [data-baseweb="placeholder"]{
  color: rgba(255,255,255,0.72) !important;
}

/* Dropdown arrow icon */
div[data-baseweb="select"] svg{
  fill: var(--text) !important;
}

/* Fix list items that looked white-on-white */
div[data-baseweb="menu"] ul, 
div[data-baseweb="menu"] li{
  background: transparent !important;
}

footer{visibility: hidden;}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# =========================
# Utilities
# =========================
@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load CSV (with encoding fallbacks) or Excel into a DataFrame."""
    name = (getattr(uploaded_file, "name", "") or "").lower()

    # Excel
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    # CSV bytes
    raw = uploaded_file.getvalue()
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e

    # Try python engine with replacement errors
    try:
        return pd.read_csv(io.BytesIO(raw), engine="python", encoding_errors="replace")
    except Exception as e:
        raise RuntimeError(f"Could not read file as CSV/Excel. Last error: {e}") from last_err


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def get_default_targets(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    # Try common names
    risk_type = "Risk_Type" if "Risk_Type" in df.columns else None
    risk_level = "Risk_Level" if "Risk_Level" in df.columns else None
    # Also allow "RiskLevel" style
    if risk_level is None:
        for c in df.columns:
            if c.lower().replace(" ", "_") == "risk_level":
                risk_level = c
                break
    if risk_type is None:
        for c in df.columns:
            if c.lower().replace(" ", "_") == "risk_type":
                risk_type = c
                break
    return risk_type, risk_level


def ensure_columns_exist(df: pd.DataFrame, needed: List[str]) -> List[str]:
    return [c for c in needed if c in df.columns]


def describe_card(title: str, body: str, badge: Optional[str] = None):
    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
            <h3 style="margin:0;">{title}</h3>
            {f'<span class="badge">{badge}</span>' if badge else ''}
          </div>
          <div style="margin-top:8px; color: var(--muted); line-height: 1.6;">
            {body}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: List[Tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (value, label) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="card">
                  <p class="kpi">{value}</p>
                  <p class="kpi-sub">{label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def plot_hist(series: pd.Series, title: str, bins: int = 30):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def plot_bar_counts(series: pd.Series, title: str, top_n: int = 30):
    import matplotlib.pyplot as plt

    counts = series.astype("string").value_counts().head(top_n)
    fig = plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def plot_scatter(x: pd.Series, y: pd.Series, title: str):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    st.pyplot(fig, clear_figure=True)


def normalize_colname(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


# =========================
# Preprocessing logic aligned with your sequence
# =========================
@dataclass
class PreprocessConfig:
    risk_score_col: str = "Risk_Score"
    mp_col: str = "MP_Count_per_L"
    polymer_col: str = "Polymer_Type"


def guess_cols(df: pd.DataFrame) -> PreprocessConfig:
    cfg = PreprocessConfig()

    # Try to find risk score / mp columns by fuzzy matching
    for c in df.columns:
        n = normalize_colname(c)
        if cfg.risk_score_col not in df.columns and n in ("risk_score", "riskscore", "risk_score_value"):
            cfg.risk_score_col = c
        if cfg.mp_col not in df.columns and n in ("mp_count_per_l", "mp_count_per_liter", "mp_per_l", "mp_count"):
            cfg.mp_col = c
        if cfg.polymer_col not in df.columns and n in ("polymer_type", "polymer"):
            cfg.polymer_col = c

    return cfg


def address_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], factor: float = 1.5) -> pd.DataFrame:
    """Winsorize outliers using IQR fences."""
    out = df.copy()
    for c in numeric_cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        out[c] = s.clip(low, high)
    return out


def transform_skew_log1p(df: pd.DataFrame, numeric_cols: List[str], skew_thresh: float = 1.0) -> pd.DataFrame:
    """Apply log1p to positively skewed columns."""
    out = df.copy()
    for c in numeric_cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        # Skip non-positive columns for log1p safety
        if (s.dropna() < 0).any():
            continue
        sk = s.skew()
        if pd.notna(sk) and sk > skew_thresh:
            out[c] = np.log1p(s)
    return out


def split_feature_types(df: pd.DataFrame, target_cols: List[str]) -> Tuple[List[str], List[str]]:
    feature_df = df.drop(columns=[c for c in target_cols if c in df.columns], errors="ignore")
    num_cols = feature_df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    cat_cols = [c for c in feature_df.columns if c not in num_cols]
    return num_cols, cat_cols


def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


# =========================
# Modeling helpers
# =========================
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def render_metrics_table(results: List[Tuple[str, Dict[str, float]]]):
    dfm = pd.DataFrame([{**{"model": name}, **metrics} for name, metrics in results]).set_index("model")
    st.dataframe(dfm.style.format("{:.4f}"), use_container_width=True)


def train_and_eval_models(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Tuple[str, Dict[str, float]]], Dict[str, Pipeline], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # --- Clean y to avoid pandas <NA> issues ---
    y_clean = pd.Series(y).astype("string")
    mask = y_clean.notna()
    X = X.loc[mask].copy()
    y = y_clean.loc[mask].astype(str)

    # --- SAFE TRAIN/TEST SPLIT ---
    stratify_arg = None
    if y.nunique() > 1:
        vc = y.value_counts()
        if vc.min() >= 2:
            stratify_arg = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
        )

    preprocessor = make_preprocessor(num_cols, cat_cols)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True),
    }

    fitted: Dict[str, Pipeline] = {}
    results: List[Tuple[str, Dict[str, float]]] = []

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        fitted[name] = pipe
        results.append((name, compute_metrics(y_test, y_pred)))

    return results, fitted, (X_train, X_test, y_train.to_numpy(), y_test.to_numpy())


def train_risktype_with_optional_smote(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    use_smote: bool,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # --- Clean y to avoid pandas <NA> issues ---
    y_clean = pd.Series(y).astype("string")
    mask = y_clean.notna()
    X = X.loc[mask].copy()
    y = y_clean.loc[mask].astype(str)

    # --- SAFE TRAIN/TEST SPLIT ---
    stratify_arg = None
    if y.nunique() > 1:
        vc = y.value_counts()
        if vc.min() >= 2:
            stratify_arg = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
        )

    preprocessor = make_preprocessor(num_cols, cat_cols)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=random_state),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        if use_smote and IMBLEARN_AVAILABLE:
            # SMOTE must come after preprocessing because it expects numeric matrix
            pipe = ImbPipeline(steps=[("prep", preprocessor), ("smote", SMOTE(random_state=random_state)), ("model", model)])
        else:
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        fitted[name] = pipe
        results.append((name, compute_metrics(y_test, preds)))

    return results, fitted, (X_train, X_test, y_train.to_numpy(), y_test.to_numpy())


def tune_logreg(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    use_smote: bool,
    random_state: int = 42,
):
    preprocessor = make_preprocessor(num_cols, cat_cols)

    base = LogisticRegression(max_iter=4000)

    if use_smote and IMBLEARN_AVAILABLE:
        pipe = ImbPipeline(steps=[("prep", preprocessor), ("smote", SMOTE(random_state=random_state)), ("model", base)])
    else:
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", base)])

    param_grid = {
        "model__C": [0.1, 1.0, 5.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X, y)
    return gs


def get_feature_names_from_preprocessor(prep: ColumnTransformer) -> List[str]:
    """Recover post-transform feature names from ColumnTransformer (num + onehot)."""
    names: List[str] = []
    # numeric
    try:
        num_features = prep.named_transformers_["num"].named_steps["imputer"].feature_names_in_
        names.extend(list(num_features))
    except Exception:
        pass

    # categorical onehot
    try:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        cat_in = prep.named_transformers_["cat"].named_steps["imputer"].feature_names_in_
        ohe_names = ohe.get_feature_names_out(cat_in)
        names.extend(list(ohe_names))
    except Exception:
        pass

    return names


def compute_feature_relevance(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on the full pipeline, robust across models."""
    r = permutation_importance(
        pipeline,
        X_train,
        y_train,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
        scoring="f1_weighted" if y_train.nunique() > 2 else "f1",
    )
    importances = r.importances_mean

    # We need feature names from the preprocessor; permutation importance returns per-column of transformed matrix?
    # sklearn returns importances for input features of estimator in pipeline (after preprocessing) only if estimator sees them.
    # With pipelines, permutation_importance operates on original X by permuting columns in X, so importances align with original columns.
    # Thus use X columns.
    df_imp = pd.DataFrame({"feature": X_train.columns.astype(str), "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_k)
    return df_imp


# =========================
# Sidebar Navigation
# =========================
NAV_ITEMS = [
    "Overview / About the Study",
    "1. Data Upload & Description",
    "2. Exploratory Data Analysis (EDA)",
    "3. Data Preprocessing",
    "4. Feature Selection",
    "5. Modeling (Objective #1)",
    "6. Risk_Type Modeling (Objective #2)",
    "7. Hyperparameter Tuning & Best Model",
    "8. Feature Relevance & Summary",
]
page = st.sidebar.radio("Navigate", NAV_ITEMS)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <strong>SMOTE availability</strong>
        <span class="badge">{'Available' if IMBLEARN_AVAILABLE else 'Not installed'}</span>
      </div>
      <div style="margin-top:8px; color: var(--hint); line-height: 1.5;">
        If SMOTE is not installed on Streamlit Cloud, the app will automatically skip it.
        To enable SMOTE, add <code>imbalanced-learn</code> to requirements.txt.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Session State
# =========================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_work" not in st.session_state:
    st.session_state.df_work = None
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "targets" not in st.session_state:
    st.session_state.targets = (None, None)

if "feature_selection" not in st.session_state:
    st.session_state.feature_selection = {"enabled": False, "k": 20, "method": "Mutual Information"}

if "obj1_models" not in st.session_state:
    st.session_state.obj1_models = None
if "obj1_results" not in st.session_state:
    st.session_state.obj1_results = None

if "obj2_models" not in st.session_state:
    st.session_state.obj2_models = None
if "obj2_results" not in st.session_state:
    st.session_state.obj2_results = None
if "obj2_use_smote" not in st.session_state:
    st.session_state.obj2_use_smote = False

if "tuned_logreg" not in st.session_state:
    st.session_state.tuned_logreg = None

if "best_pipeline" not in st.session_state:
    st.session_state.best_pipeline = None
if "best_context" not in st.session_state:
    st.session_state.best_context = None  # (X_train, X_test, y_train, y_test, target_name)


# =========================
# Helper: check df
# =========================
def require_df() -> pd.DataFrame:
    if st.session_state.df_work is None:
        st.info("Upload a dataset first in **1. Data Upload & Description**.")
        st.stop()
    return st.session_state.df_work


# =========================
# Pages
# =========================
if page == "Overview / About the Study":
    st.title("🧪 Microplastic Risk Modeling Dashboard")
    describe_card(
        "What this app does",
        """
        This dashboard organizes your workflow into a clear, thesis-friendly sequence:

        <ol>
          <li><b>Upload data</b> and view data description</li>
          <li><b>EDA</b>: Polymer Type distribution + Risk Score analyses</li>
          <li><b>Preprocessing</b>: Outliers → Skew transform → Encoding → Scaling</li>
          <li><b>Feature Selection</b>: Choose and apply a method</li>
          <li><b>Objective #1 Modeling</b>: Compare baseline classifiers</li>
          <li><b>Objective #2 Modeling</b>: Risk_Type modeling with optional SMOTE</li>
          <li><b>Hyperparameter tuning</b>: Tune Logistic Regression and compare</li>
          <li><b>Feature relevance</b>: Interpret the best model and summarize findings</li>
        </ol>
        """,
        badge="v2.0",
    )

    st.markdown("---")
    st.subheader("Quick tips")
    st.markdown(
        """
- If your CSV fails to load, try uploading the **Excel (.xlsx)** version, or export CSV with **UTF-8** encoding.
- If SMOTE shows “Not installed”, add `imbalanced-learn` to `requirements.txt` on Streamlit Cloud.
- For modeling pages, ensure your dataset has a target like **Risk_Type** or **Risk_Level**.
        """
    )

elif page == "1. Data Upload & Description":
    st.title("1) Data Upload & Description")

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        describe_card(
            "Upload your dataset",
            """
            Supported formats:
            <ul>
              <li><b>CSV</b> (multiple encodings supported)</li>
              <li><b>Excel</b> (.xlsx, .xls)</li>
            </ul>
            After upload, you'll see a preview, column types, and target selection.
            """,
        )
    with c2:
        kpi_row(
            [
                ("CSV / Excel", "Supported formats"),
                ("UTF-8 / cp1252", "Encoding fallbacks"),
            ]
        )

    uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = load_dataframe(uploaded)
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            st.stop()

        st.session_state.df_raw = df.copy()
        st.session_state.df_work = df.copy()
        st.session_state.cfg = guess_cols(df)
        st.session_state.targets = get_default_targets(df)

        st.success("Dataset loaded successfully.")
        st.markdown("### Preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("### Dataset summary")
        n_rows, n_cols = df.shape
        kpi_row(
            [
                (f"{n_rows:,}", "Rows"),
                (f"{n_cols:,}", "Columns"),
                (f"{df.isna().sum().sum():,}", "Missing cells"),
            ]
        )

        st.markdown("### Column dtypes")
        dtype_df = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}).set_index("column")
        st.dataframe(dtype_df, use_container_width=True)

        st.markdown("### Select targets (if present)")
        risk_type_guess, risk_level_guess = st.session_state.targets

        with st.container():
            colA, colB = st.columns(2)
            with colA:
                risk_type = st.selectbox(
                    "Risk_Type target column",
                    options=[None] + df.columns.tolist(),
                    index=(df.columns.tolist().index(risk_type_guess) + 1) if risk_type_guess in df.columns else 0,
                )
            with colB:
                risk_level = st.selectbox(
                    "Risk_Level target column",
                    options=[None] + df.columns.tolist(),
                    index=(df.columns.tolist().index(risk_level_guess) + 1) if risk_level_guess in df.columns else 0,
                )

        st.session_state.targets = (risk_type, risk_level)

        st.markdown("---")
        if st.button("Reset dataset (clear session)"):
            for k in list(st.session_state.keys()):
                st.session_state[k] = None if k in ("df_raw", "df_work", "cfg") else st.session_state.get(k)

elif page == "2. Exploratory Data Analysis (EDA)":
    st.title("2) Exploratory Data Analysis (EDA)")
    df = require_df()
    cfg: PreprocessConfig = st.session_state.cfg or guess_cols(df)

    describe_card(
        "EDA Goals",
        """
        This section follows your guide:
        <ul>
          <li>Load & visualize <b>Polymer Type</b> distribution</li>
          <li>Analyze the <b>distribution of Risk Score</b></li>
          <li>Investigate <b>difference in Risk Score by Risk Level</b> (if available)</li>
          <li>Explore relationship between <b>Risk Score</b> and <b>MP count per L</b></li>
        </ul>
        """,
    )

    st.markdown("---")
    polymer_col = st.selectbox("Polymer Type column", options=df.columns.tolist(), index=df.columns.tolist().index(cfg.polymer_col) if cfg.polymer_col in df.columns else 0)
    risk_score_col = st.selectbox("Risk Score column", options=df.columns.tolist(), index=df.columns.tolist().index(cfg.risk_score_col) if cfg.risk_score_col in df.columns else 0)
    mp_col = st.selectbox("MP Count per L column", options=df.columns.tolist(), index=df.columns.tolist().index(cfg.mp_col) if cfg.mp_col in df.columns else 0)

    risk_type_col, risk_level_col = st.session_state.targets

    st.markdown("### 2.1 Polymer Type distribution")
    if polymer_col in df.columns:
        plot_bar_counts(df[polymer_col], f"Distribution of {polymer_col}")
    else:
        st.warning("Polymer Type column not found.")

    st.markdown("### 2.2 Risk Score distribution")
    if risk_score_col in df.columns:
        df_num = safe_numeric(df, [risk_score_col])
        plot_hist(df_num[risk_score_col], f"Distribution of {risk_score_col}")
    else:
        st.warning("Risk Score column not found.")

    st.markdown("### 2.3 Risk Score by Risk Level")
    if risk_level_col and risk_level_col in df.columns and risk_score_col in df.columns:
        import matplotlib.pyplot as plt

        tmp = safe_numeric(df[[risk_level_col, risk_score_col]].copy(), [risk_score_col]).dropna()
        groups = tmp.groupby(risk_level_col)[risk_score_col]
        fig = plt.figure()
        plt.boxplot([g.values for _, g in groups], labels=[str(k) for k in groups.groups.keys()])
        plt.title(f"{risk_score_col} by {risk_level_col}")
        plt.xlabel(risk_level_col)
        plt.ylabel(risk_score_col)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Select a valid Risk_Level target column in **1. Data Upload & Description** to enable this plot.")

    st.markdown("### 2.4 Risk Score vs MP Count per L")
    if risk_score_col in df.columns and mp_col in df.columns:
        tmp = safe_numeric(df[[risk_score_col, mp_col]].copy(), [risk_score_col, mp_col]).dropna()
        plot_scatter(tmp[mp_col], tmp[risk_score_col], f"{risk_score_col} vs {mp_col}")
    else:
        st.warning("Required columns not found for scatter plot.")

elif page == "3. Data Preprocessing":
    st.title("3) Data Preprocessing")
    df = require_df()
    cfg: PreprocessConfig = st.session_state.cfg or guess_cols(df)

    describe_card(
        "Preprocessing sequence (your guide)",
        """
        <ol>
          <li><b>Address outliers</b></li>
          <li><b>Transform skewed numerical columns</b></li>
          <li><b>Encode categorical variables</b></li>
          <li><b>Perform feature scaling</b></li>
        </ol>
        Note: Encoding + scaling are handled inside the modeling pipeline to avoid data leakage,
        but we demonstrate the conceptual steps here.
        """,
    )

    risk_type_col, risk_level_col = st.session_state.targets
    target_cols = [c for c in [risk_type_col, risk_level_col] if c]

    num_cols, cat_cols = split_feature_types(df, target_cols=target_cols)

    st.markdown("---")
    st.subheader("Choose preprocessing parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        outlier_factor = st.slider("Outlier IQR factor (winsorize)", 1.0, 3.0, 1.5, 0.1)
    with col2:
        skew_thresh = st.slider("Skew threshold for log1p transform", 0.5, 3.0, 1.0, 0.1)
    with col3:
        apply_now = st.checkbox("Apply preprocessing to working dataset", value=True)

    st.markdown("### Preview: feature types")
    st.write(f"Numeric columns: **{len(num_cols)}**")
    st.write(f"Categorical columns: **{len(cat_cols)}**")

    # Apply steps
    df_work = df.copy()
    if len(num_cols) > 0:
        st.markdown("### Step 3.1 Address outliers")
        df_work = safe_numeric(df_work, num_cols)
        df_work = address_outliers_iqr(df_work, numeric_cols=num_cols, factor=outlier_factor)
        st.success("Outliers addressed (winsorized via IQR fences).")

        st.markdown("### Step 3.2 Transform skewed numerical columns")
        df_work = transform_skew_log1p(df_work, numeric_cols=num_cols, skew_thresh=skew_thresh)
        st.success("Skew transformation applied (log1p on positively skewed columns).")
    else:
        st.info("No numeric columns detected for outlier/skew handling.")

    st.markdown("### Step 3.3 Encode categorical variables")
    st.info("Encoding is applied inside the modeling pipeline using OneHotEncoder(handle_unknown='ignore').")

    st.markdown("### Step 3.4 Perform feature scaling")
    st.info("Scaling is applied inside the modeling pipeline using StandardScaler() within the numeric pipeline.")

    st.markdown("---")
    if apply_now:
        st.session_state.df_work = df_work
        st.success("Working dataset updated.")
        st.dataframe(df_work.head(50), use_container_width=True)

    st.download_button(
        "Download preprocessed dataset (CSV)",
        data=df_work.to_csv(index=False).encode("utf-8"),
        file_name="preprocessed_dataset.csv",
        mime="text/csv",
    )

elif page == "4. Feature Selection":
    st.title("4) Feature Selection")
    df = require_df()

    describe_card(
        "Feature Selection Goals",
        """
        <ol>
          <li><b>Understand the goal</b>: choose your target (Risk_Type or Risk_Level)</li>
          <li><b>Explore methods</b>: Mutual Information is implemented here</li>
          <li><b>Implement method</b> and <b>evaluate selected features</b></li>
        </ol>
        """,
    )

    risk_type_col, risk_level_col = st.session_state.targets
    target = st.selectbox("Select target for feature selection", options=[None, risk_type_col, risk_level_col], index=0)

    if not target or target not in df.columns:
        st.info("Select a valid target column to run feature selection.")
        st.stop()

    target_cols = [target]
    num_cols, cat_cols = split_feature_types(df, target_cols=target_cols)
    X = df.drop(columns=target_cols, errors="ignore")
    y = df[target].astype("string")

    st.markdown("---")
    k = st.slider("Number of features (k) for SelectKBest", 5, min(100, max(10, X.shape[1])), 20, 1)

    # Build preprocessing (impute + encode + scale)
    preprocessor = make_preprocessor(num_cols, cat_cols)

    # Feature selection works on transformed matrix
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, max(1, X.shape[1])))

    model = LogisticRegression(max_iter=2000)
    pipe = Pipeline(steps=[("prep", preprocessor), ("select", selector), ("model", model)])

    if st.button("Run Feature Selection"):
        with st.spinner("Fitting feature selection pipeline..."):
            pipe.fit(X, y)

        # Recover feature names after preprocessing
        prep: ColumnTransformer = pipe.named_steps["prep"]
        feature_names = get_feature_names_from_preprocessor(prep)

        # selector scores align to transformed features, not original columns
        scores = pipe.named_steps["select"].scores_
        if scores is None:
            st.error("Could not compute scores.")
            st.stop()

        fs = pd.DataFrame({"feature": feature_names[: len(scores)], "score": scores}).sort_values("score", ascending=False)
        st.session_state.feature_selection = {"enabled": True, "k": k, "method": "Mutual Information", "selected": fs.head(k)}

        st.success("Feature selection completed.")
        st.dataframe(fs.head(50), use_container_width=True)

        st.download_button(
            "Download selected features (CSV)",
            data=fs.head(k).to_csv(index=False).encode("utf-8"),
            file_name="selected_features.csv",
            mime="text/csv",
        )

elif page == "5. Modeling (Objective #1)":
    st.title("5) Modeling (Objective #1) — Baseline Model Comparison")
    df = require_df()

    describe_card(
        "Objective #1: Modeling Tasks",
        """
        <ol>
          <li><b>Prepare the data</b></li>
          <li><b>Choose classification models</b></li>
          <li><b>Train the models</b></li>
          <li><b>Evaluate the models</b></li>
          <li><b>Compare model performance</b></li>
        </ol>
        """,
    )

    risk_type_col, risk_level_col = st.session_state.targets
    target = st.selectbox("Choose target for Objective #1", options=[None, risk_level_col, risk_type_col], index=0)
    if not target or target not in df.columns:
        st.info("Select a valid target column.")
        st.stop()

    X = df.drop(columns=[target], errors="ignore")
    y = df[target].astype("string")

    num_cols, cat_cols = split_feature_types(df, target_cols=[target])

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    if st.button("Train & Compare Models"):
        with st.spinner("Training models..."):
            results, fitted, ctx = train_and_eval_models(X, y, num_cols, cat_cols, test_size=test_size)
        st.session_state.obj1_results = results
        st.session_state.obj1_models = fitted
        st.session_state.best_pipeline = max(results, key=lambda x: x[1]["f1"])[0]
        st.session_state.best_context = (*ctx, target)

        st.success("Training complete.")
        st.markdown("### Performance Comparison")
        render_metrics_table(results)

        best_name = max(results, key=lambda x: x[1]["f1"])[0]
        st.info(f"Best baseline model by F1: **{best_name}**")

elif page == "6. Risk_Type Modeling (Objective #2)":
    st.title("6) Risk_Type Modeling (Objective #2)")
    df = require_df()

    describe_card(
        "Objective #2: Risk_Type Modeling Tasks",
        """
        <ol>
          <li>Prepare data for <b>Risk_Type</b> modeling</li>
          <li>Check class distribution</li>
          <li>(Optional) Address imbalance with <b>SMOTE</b></li>
          <li>Train models</li>
          <li>Evaluate & compare performance</li>
          <li>Visualize model performance</li>
        </ol>
        """,
    )

    risk_type_col, _ = st.session_state.targets
    if not risk_type_col or risk_type_col not in df.columns:
        st.info("Select a valid **Risk_Type** column in **1. Data Upload & Description**.")
        st.stop()

    X = df.drop(columns=[risk_type_col], errors="ignore")
    y = df[risk_type_col].astype("string")
    num_cols, cat_cols = split_feature_types(df, target_cols=[risk_type_col])

    st.markdown("### Class distribution")
    class_counts = y.value_counts()
    st.dataframe(class_counts.to_frame("count"), use_container_width=True)

    use_smote = st.checkbox("Use SMOTE (if available)", value=False)
    if use_smote and not IMBLEARN_AVAILABLE:
        st.warning("SMOTE requested but 'imbalanced-learn' is not installed. SMOTE will be skipped automatically.")
    st.session_state.obj2_use_smote = use_smote

    test_size = st.slider("Test size (Objective #2)", 0.1, 0.4, 0.2, 0.05, key="obj2_test")

    if st.button("Train Risk_Type Models"):
        with st.spinner("Training Risk_Type models..."):
            results, fitted, ctx = train_risktype_with_optional_smote(
                X, y, num_cols, cat_cols, use_smote=use_smote, test_size=test_size
            )
        st.session_state.obj2_results = results
        st.session_state.obj2_models = fitted
        st.session_state.best_context = (*ctx, risk_type_col)

        st.success("Training complete.")
        st.markdown("### Performance Comparison (Risk_Type)")
        render_metrics_table(results)

        # crude visualization as bar chart
        import matplotlib.pyplot as plt

        fig = plt.figure()
        names = [n for n, _ in results]
        f1s = [m["f1"] for _, m in results]
        plt.bar(names, f1s)
        plt.title("F1 Score by Model (Risk_Type)")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("F1 (weighted)")
        st.pyplot(fig, clear_figure=True)

elif page == "7. Hyperparameter Tuning & Best Model":
    st.title("7) Hyperparameter Tuning & Best Model")
    df = require_df()

    describe_card(
        "Tuning tasks (from your guide)",
        """
        <ul>
          <li>Check imports for hyperparameter tuning</li>
          <li>Perform hyperparameter tuning</li>
          <li>Evaluate the best model</li>
          <li>Compare performance of tuned Logistic Regression with other models</li>
        </ul>
        """,
    )

    risk_type_col, risk_level_col = st.session_state.targets
    target = st.selectbox("Choose target to tune (recommended: Risk_Type)", options=[None, risk_type_col, risk_level_col], index=0)
    if not target or target not in df.columns:
        st.info("Select a valid target column.")
        st.stop()

    X = df.drop(columns=[target], errors="ignore")
    y = df[target].astype("string")
    num_cols, cat_cols = split_feature_types(df, target_cols=[target])

    use_smote = st.checkbox("Use SMOTE in tuning (if available)", value=st.session_state.obj2_use_smote)

    if st.button("Run GridSearchCV (Logistic Regression)"):
        with st.spinner("Running hyperparameter tuning..."):
            gs = tune_logreg(X, y, num_cols, cat_cols, use_smote=use_smote)
        st.session_state.tuned_logreg = gs
        st.success("Tuning complete.")

        st.markdown("### Best parameters")
        st.code(str(gs.best_params_))

        st.markdown("### Best CV score (F1 weighted)")
        st.write(gs.best_score_)

        st.session_state.best_pipeline = gs.best_estimator_
        st.session_state.best_context = (None, None, None, None, target)

    # Compare tuned LR with baseline models quickly
    if st.session_state.tuned_logreg is not None:
        st.markdown("---")
        st.subheader("Compare tuned Logistic Regression with baseline models")
        gs = st.session_state.tuned_logreg

        # baseline comparisons
        with st.spinner("Training baseline models for comparison..."):
            results, fitted, _ = train_and_eval_models(X, y, num_cols, cat_cols, test_size=0.2)
        # Evaluate tuned LR on a holdout
        # Safe holdout split (handles rare classes / missing labels)
        y_clean = pd.Series(y).astype("string")
        mask = y_clean.notna()
        X2 = X.loc[mask].copy()
        y2 = y_clean.loc[mask].astype(str)
        stratify_arg = None
        if y2.nunique() > 1 and y2.value_counts().min() >= 2:
            stratify_arg = y2
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.2, stratify=stratify_arg, random_state=42)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.2, stratify=None, random_state=42)
        tuned = gs.best_estimator_
        tuned.fit(X_tr, y_tr)
        pred = tuned.predict(X_te)
        tuned_metrics = compute_metrics(y_te, pred)

        compare = results + [("Tuned Logistic Regression", tuned_metrics)]
        render_metrics_table(compare)

        best_name = max(compare, key=lambda x: x[1]["f1"])[0]
        st.info(f"Best model by holdout F1: **{best_name}**")

        if best_name == "Tuned Logistic Regression":
            st.session_state.best_pipeline = tuned
            st.session_state.best_context = (X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), target)
        else:
            st.session_state.best_pipeline = fitted[best_name]
            st.session_state.best_context = (X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), target)

elif page == "8. Feature Relevance & Summary":
    st.title("8) Feature Relevance & Summary")
    df = require_df()

    describe_card(
        "Interpretability tasks",
        """
        <ol>
          <li>Extract feature relevance</li>
          <li>Analyze feature relevance</li>
          <li>Visualize feature relevance</li>
          <li>Summarize findings</li>
        </ol>
        Permutation importance is used here because it works across many model types.
        """,
    )

    # Ensure we have a best pipeline
    if st.session_state.best_pipeline is None or st.session_state.best_context is None:
        st.info("Train models first (Objective #1 / Objective #2 / Tuning) to set a 'best model'.")
        st.stop()

    best_pipe = st.session_state.best_pipeline
    ctx = st.session_state.best_context
    X_train, X_test, y_train, y_test, target = ctx

    st.markdown("### Selected best model")
    st.code(str(best_pipe))

    # If context missing because of some flow, rebuild a split quickly
    if X_train is None:
        risk_type_col, risk_level_col = st.session_state.targets
        if target is None:
            target = risk_type_col or risk_level_col
        X = df.drop(columns=[target], errors="ignore")
        y = df[target].astype("string")
        y_clean = pd.Series(y).astype("string")
        mask = y_clean.notna()
        X2 = X.loc[mask].copy()
        y2 = y_clean.loc[mask].astype(str)
        stratify_arg = None
        if y2.nunique() > 1 and y2.value_counts().min() >= 2:
            stratify_arg = y2
        try:
            X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, stratify=stratify_arg, random_state=42)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, stratify=None, random_state=42)

    top_k = st.slider("Top K features (importance)", 5, 50, 20, 1)

    if st.button("Compute Feature Relevance"):
        with st.spinner("Computing permutation importance..."):
            imp = compute_feature_relevance(best_pipe, X_train, pd.Series(y_train), top_k=top_k)
        st.success("Feature relevance computed.")
        st.dataframe(imp, use_container_width=True)

        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.barh(imp["feature"][::-1], imp["importance"][::-1])
        plt.title("Top Feature Importances (Permutation)")
        plt.xlabel("Importance")
        st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.subheader("Summary checklist (matches your guide)")
    st.markdown(
        """
- ✅ EDA: risk score distribution; risk score by risk level; risk score vs MP count/L; polymer distribution  
- ✅ Preprocessing: outliers → skew transform → encoding → scaling  
- ✅ Feature selection: implemented (Mutual Information / SelectKBest)  
- ✅ Objective #1: baseline model comparison  
- ✅ Objective #2: Risk_Type modeling with SMOTE (optional)  
- ✅ Hyperparameter tuning: GridSearchCV Logistic Regression  
- ✅ Feature relevance: permutation importance + visualization  
        """
    )

st.markdown("---")
st.caption("© Microplastic Risk Modeling Dashboard — Streamlit")
