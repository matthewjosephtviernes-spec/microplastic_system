# file: streamlit_app.py
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")  # save to buffer, not GUI

# ----------------- helpers -----------------
@dataclass
class ColumnConfig:
    target: Optional[str]
    risk_score: Optional[str]
    risk_level: Optional[str]
    mp_count: Optional[str]
    polymer: Optional[str]
    id_cols: List[str]
    date_cols: List[str]

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

def cap_outliers_iqr(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in numeric_cols:
        s = X[c].dropna()
        if s.empty:
            continue
        q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X[c] = X[c].clip(lower, upper)
    return X

def build_preprocessor(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[ColumnTransformer, List[str], List[str]]:
    drop_cols = set((cfg.id_cols or []) + (cfg.date_cols or []))
    kept = [c for c in df.columns if c not in drop_cols and c != (cfg.target or "")]
    num_cols = [c for c in kept if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in kept if c not in num_cols]
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", RobustScaler(with_centering=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return pre, num_cols, cat_cols

def summarize_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "precision_w": p, "recall_w": r, "f1_w": f1}
    try:
        if y_proba is not None:
            if y_proba.ndim == 1:
                out["roc_auc"] = roc_auc_score(y_true, y_proba)
            else:
                out["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except Exception:
        pass
    return out

def is_binary(y: pd.Series) -> bool:
    return y.nunique(dropna=True) == 2

# ----------------- streamlit UI -----------------
st.set_page_config(page_title="Risk Analytics System", layout="wide")
st.title("Risk Analytics & Modeling")

with st.sidebar:
    st.header("1) Load data")
    up = st.file_uploader("Upload CSV (or Parquet)", type=["csv", "parquet"])
    st.caption("Tip: CSV is simplest. If Parquet, ensure pyarrow is in requirements.")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", 0, 9999, 42, 1)
    st.header("2) Run")
    run_btn = st.button("Run Pipeline", type="primary")

@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith(".parquet"):
        return sanitize_columns(pd.read_parquet(file))
    return sanitize_columns(pd.read_csv(file))

if up:
    df = load_df(up)
    st.success(f"Loaded: {up.name}  •  Rows: {len(df)}  •  Cols: {df.shape[1]}")
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.info("Upload a dataset to begin.")
    st.stop()

# column pickers
st.subheader("Column mapping")
cols = df.columns.tolist()
c1, c2, c3 = st.columns(3)
with c1:
    target = st.selectbox("Target (classification)", options=[""] + cols, index=(cols.index("Risk_Type")+1) if "Risk_Type" in cols else 0)
    risk_score = st.selectbox("Risk Score (numeric)", options=[""] + cols, index=(cols.index("Risk_Score")+1) if "Risk_Score" in cols else 0)
with c2:
    risk_level = st.selectbox("Risk Level (categorical)", options=[""] + cols, index=(cols.index("Risk_Level")+1) if "Risk_Level" in cols else 0)
    mp_count = st.selectbox("mp_count_per_l (numeric)", options=[""] + cols, index=(cols.index("mp_count_per_l")+1) if "mp_count_per_l" in cols else 0)
with c3:
    polymer = st.selectbox("Polymer Type (optional)", options=[""] + cols, index=(cols.index("Polymer_Type")+1) if "Polymer_Type" in cols else 0)
    id_cols = st.multiselect("ID columns to exclude", options=cols)
date_cols = st.multiselect("Date/time columns to exclude", options=cols)

cfg = ColumnConfig(
    target=target or None,
    risk_score=risk_score or None,
    risk_level=risk_level or None,
    mp_count=mp_count or None,
    polymer=polymer or None,
    id_cols=id_cols,
    date_cols=date_cols,
)

if not cfg.target:
    st.warning("Select a **Target** column to continue.")
    st.stop()

# ----------------- EDA -----------------
st.subheader("Exploratory Analysis")
g1, g2, g3 = st.columns(3)
with g1:
    if cfg.risk_score and cfg.risk_score in df:
        fig = plt.figure()
        plt.hist(df[cfg.risk_score].dropna(), bins=30)
        plt.title("Risk Score Distribution"); plt.xlabel(cfg.risk_score); plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)
with g2:
    if cfg.risk_score and cfg.risk_level and cfg.risk_score in df and cfg.risk_level in df:
        levels = df[cfg.risk_level].dropna().unique()
        data = [df.loc[df[cfg.risk_level]==lvl, cfg.risk_score].dropna().values for lvl in levels]
        if len(data):
            fig = plt.figure()
            plt.boxplot(data, labels=[str(x) for x in levels], showfliers=True)
            plt.title("Risk Score by Risk Level"); plt.xlabel("Risk Level"); plt.ylabel(cfg.risk_score)
            st.pyplot(fig, clear_figure=True)
with g3:
    if cfg.risk_score and cfg.mp_count and cfg.risk_score in df and cfg.mp_count in df:
        m = df[[cfg.mp_count, cfg.risk_score]].dropna()
        fig = plt.figure()
        plt.scatter(m[cfg.mp_count], m[cfg.risk_score], alpha=0.6)
        plt.title("Risk Score vs mp_count_per_l"); plt.xlabel(cfg.mp_count); plt.ylabel(cfg.risk_score)
        st.pyplot(fig, clear_figure=True)

# ----------------- RUN PIPELINE -----------------
if not run_btn:
    st.stop()

with st.spinner("Preparing data..."):
    drop_cols = list(set(cfg.id_cols + cfg.date_cols + [cfg.target]))
    X = df.drop(columns=[c for c in drop_cols if c in df], errors="ignore")
    y = df[cfg.target].astype("category")
    num_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = cap_outliers_iqr(X, num_all)
    pre, _, _ = build_preprocessor(df.drop(columns=[cfg.target]), cfg)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=int(random_state))

with st.spinner("Training models..."):
    models = {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))]),
        "rf": Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=int(random_state), n_jobs=-1))]),
    }
    # quick tuning for LR
    gs = GridSearchCV(models["logreg"], {"clf__C": [0.1, 1.0, 3.0, 10.0]}, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1_weighted", n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)
    models["logreg_tuned"] = gs.best_estimator_

    results = []
    best_name, best_score, best_pipe = None, -1.0, None
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        proba = None
        try:
            proba = pipe.predict_proba(Xte)
        except Exception:
            pass
        metrics = summarize_metrics(yte, yhat, proba[:,1] if (proba is not None and proba.ndim==2 and proba.shape[1]==2) else proba)
        metrics["model"] = name
        results.append(metrics)
        if metrics["f1_w"] > best_score:
            best_name, best_score, best_pipe = name, metrics["f1_w"], pipe

leader = pd.DataFrame(results).sort_values("f1_w", ascending=False)
st.subheader("Model leaderboard (F1-weighted)")
st.dataframe(leader, use_container_width=True)

# comparison bar
fig = plt.figure()
plt.bar(leader["model"], leader["f1_w"])
plt.title("Model Comparison (F1-weighted)"); plt.xlabel("Model"); plt.ylabel("F1_weighted")
st.pyplot(fig, clear_figure=True)

# Confusion matrix & ROC
st.subheader(f"Best model: {best_name}")
yhat_best = best_pipe.predict(Xte)
labels = np.unique(yte)
cm = confusion_matrix(yte, yhat_best, labels=labels)
fig = plt.figure()
plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
for (i, j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
st.pyplot(fig, clear_figure=True)

if is_binary(yte):
    try:
        proba = best_pipe.predict_proba(Xte)
        fig = plt.figure()
        RocCurveDisplay.from_predictions(yte, proba[:,1] if proba.ndim==2 else proba)
        plt.title("ROC Curve")
        st.pyplot(fig, clear_figure=True)
    except Exception:
        pass

# Feature relevance (simple: RF importances or LR coefs)
st.subheader("Top features")
try:
    pre: ColumnTransformer = best_pipe.named_steps["pre"]
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feat_names = list(num_cols) + cat_out

    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
    elif hasattr(clf, "coef_"):
        coefs = np.mean(np.abs(clf.coef_), axis=0) if getattr(clf.coef_, "ndim", 1) > 1 else np.abs(clf.coef_)
        imp = pd.DataFrame({"feature": feat_names, "importance": coefs})
    else:
        imp = None

    if imp is not None:
        imp = imp.sort_values("importance", ascending=False).head(20)
        fig = plt.figure()
        plt.barh(imp["feature"][::-1], imp["importance"][::-1])
        plt.title("Top Feature Importance"); plt.xlabel("Importance")
        st.pyplot(fig, clear_figure=True)
        st.dataframe(imp, use_container_width=True)
except Exception as e:
    st.info(f"Feature importance not available: {e}")

st.success("Done.")
