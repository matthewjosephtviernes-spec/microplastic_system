# file: risk_system.py
"""
End-to-end risk analytics + modeling pipeline.

Usage:
  python risk_system.py --input data.csv --target Risk_Type --risk-score Risk_Score \
    --risk-level Risk_Level --mp-count mp_count_per_l --polymer Polymer_Type \
    --id-cols ID --date-cols Date

Outputs:
  outputs/plots/*.png
  outputs/summary.md
  outputs/best_model.joblib
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Optional libs
try:
    import shap  # noqa
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline   # type: ignore
    from imblearn.over_sampling import SMOTE                # type: ignore
    HAS_IMB = True
except Exception:
    HAS_IMB = False

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")  # headless

# -------------------------- Utility --------------------------

def ensure_dirs() -> Dict[str, str]:
    """Create output folders and return their paths."""
    base = "outputs"
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    return {"base": base, "plots": plots}

def savefig(path: str) -> None:
    """Save current figure with tight layout."""
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def is_binary(y: pd.Series) -> bool:
    return y.nunique(dropna=True) == 2

def summarize_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Compact metrics dict; roc_auc if available."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_w": precision,
        "recall_w": recall,
        "f1_w": f1,
    }
    # Only compute ROC-AUC if binary and proba present
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            else:
                metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            pass
    return metrics

# -------------------------- Data Loading --------------------------

def load_table(path: str) -> pd.DataFrame:
    """Load CSV or Parquet with dtype inference."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names."""
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

# -------------------------- Column Detection --------------------------

@dataclass
class ColumnConfig:
    id_cols: List[str]
    date_cols: List[str]
    target: Optional[str]
    risk_score: Optional[str]
    risk_level: Optional[str]
    mp_count: Optional[str]
    polymer: Optional[str]

def detect_columns(df: pd.DataFrame, args: argparse.Namespace) -> ColumnConfig:
    """Resolve columns from args or sensible defaults."""
    cols = set(df.columns)
    def pick(name: Optional[str], candidates: List[str]) -> Optional[str]:
        if name and name in cols:
            return name
        for c in candidates:
            if c in cols:
                return c
        return None

    target = pick(args.target, ["Risk_Type", "risk_type", "RISK_TYPE"])
    risk_score = pick(args.risk_score, ["Risk_Score", "risk_score", "score"])
    risk_level = pick(args.risk_level, ["Risk_Level", "risk_level", "level"])
    mp_count = pick(args.mp_count, ["mp_count_per_l", "mp_count", "MP_Count", "mp"])
    polymer = pick(args.polymer, ["Polymer_Type", "polymer_type", "polymer"])

    id_cols = [c for c in (args.id_cols or "").split(",") if c in cols] if args.id_cols else []
    date_cols = [c for c in (args.date_cols or "").split(",") if c in cols] if args.date_cols else []
    return ColumnConfig(id_cols=id_cols, date_cols=date_cols, target=target,
                        risk_score=risk_score, risk_level=risk_level, mp_count=mp_count, polymer=polymer)

# -------------------------- EDA --------------------------

def plot_risk_score_distribution(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not cfg.risk_score or cfg.risk_score not in df.columns:
        return None
    s = df[cfg.risk_score].dropna()
    plt.figure()
    plt.hist(s, bins=30)
    plt.title("Risk Score Distribution")
    plt.xlabel(cfg.risk_score)
    plt.ylabel("Count")
    path = os.path.join(outdir, "risk_score_distribution.png")
    savefig(path)
    return path

def plot_risk_score_by_level(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not cfg.risk_score or not cfg.risk_level:
        return None
    if cfg.risk_score not in df.columns or cfg.risk_level not in df.columns:
        return None
    grouped = [df.loc[df[cfg.risk_level] == lvl, cfg.risk_score].dropna().values for lvl in df[cfg.risk_level].dropna().unique()]
    labels = [str(lvl) for lvl in df[cfg.risk_level].dropna().unique()]
    if len(grouped) == 0:
        return None
    plt.figure()
    plt.boxplot(grouped, labels=labels, showfliers=True)
    plt.title("Risk Score by Risk Level")
    plt.xlabel("Risk Level")
    plt.ylabel(cfg.risk_score)
    path = os.path.join(outdir, "risk_score_by_level.png")
    savefig(path)
    return path

def plot_risk_vs_mp(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not cfg.risk_score or not cfg.mp_count:
        return None
    if cfg.risk_score not in df.columns or cfg.mp_count not in df.columns:
        return None
    x = df[cfg.mp_count]
    y = df[cfg.risk_score]
    m = (~x.isna()) & (~y.isna())
    plt.figure()
    plt.scatter(x[m], y[m], alpha=0.6)
    plt.title("Risk Score vs mp_count_per_l")
    plt.xlabel(cfg.mp_count)
    plt.ylabel(cfg.risk_score)
    path = os.path.join(outdir, "risk_vs_mpcount.png")
    savefig(path)
    return path

def plot_polymer_distribution(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not cfg.polymer or cfg.polymer not in df.columns:
        return None
    counts = df[cfg.polymer].value_counts(dropna=False)
    plt.figure()
    counts.plot(kind="bar")  # matplotlib backend handles pd Series
    plt.title("Polymer Type Distribution")
    plt.xlabel("Polymer Type")
    plt.ylabel("Count")
    path = os.path.join(outdir, "polymer_type_distribution.png")
    savefig(path)
    return path

# -------------------------- Preprocessing --------------------------

def cap_outliers_iqr(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Winsorize numeric columns using IQR. Why: stabilize extreme tails before scaling."""
    X = X.copy()
    for col in numeric_cols:
        s = X[col]
        s = s[~s.isna()]
        if s.empty:
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower, upper)
    return X

def build_preprocessor(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create ColumnTransformer with impute->winsorize(power)->scale for numeric, one-hot for categoricals."""
    drop_cols = set((cfg.id_cols or []) + (cfg.date_cols or []))
    candidates = [c for c in df.columns if c not in drop_cols]
    numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in candidates if (not pd.api.types.is_numeric_dtype(df[c])) and c != (cfg.target or "")]
    # Pre-impute to allow transformer fit
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),  # Why: fix skew w/o scaling yet
        ("scale", RobustScaler(with_centering=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        n_jobs=None
    )
    return pre, numeric_cols, categorical_cols

# -------------------------- Feature Selection --------------------------

def run_feature_selection(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer, max_feats: int = 50) -> Dict[str, any]:
    """Compute variance threshold and mutual information ranks."""
    # Fit preprocessor to get transformed features
    Xt = pre.fit_transform(X, y)
    # Build feature names post-transform
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feature_names = list(num_cols) + cat_out

    # Variance filter (very low variance)
    vt = VarianceThreshold(threshold=1e-4)
    vt.fit(Xt)
    low_var_mask = ~vt.get_support()
    low_var_features = [feature_names[i] for i, m in enumerate(low_var_mask) if m]

    # Mutual information ranking on a sample for speed
    rs = np.random.RandomState(42)
    sample_idx = rs.choice(np.arange(Xt.shape[0]), size=min(5000, Xt.shape[0]), replace=False)
    try:
        y_s = y.iloc[sample_idx]
    except Exception:
        y_s = y.sample(n=min(5000, Xt.shape[0]), random_state=42)
        sample_idx = y_s.index
    Xt_s = Xt[sample_idx] if isinstance(sample_idx, np.ndarray) else Xt

    # If y is not encoded numeric categories yet, factorize
    if not pd.api.types.is_integer_dtype(y_s) and not pd.api.types.is_categorical_dtype(y_s):
        y_enc, _ = pd.factorize(y_s)
    else:
        y_enc = y_s.values
    mi = mutual_info_classif(Xt_s, y_enc, random_state=42, discrete_features=[False] * Xt_s.shape[1])
    mi_rank = sorted(zip(feature_names, mi), key=lambda t: t[1], reverse=True)
    top_mi = mi_rank[:max_feats]

    return {
        "feature_names": feature_names,
        "low_variance": low_var_features,
        "mi_ranking": mi_rank,
        "top_mi": top_mi,
        "pre_fitted": pre,
    }

# -------------------------- Modeling --------------------------

def build_model_pipelines(pre: ColumnTransformer, random_state: int = 42):
    """Return dict of model name -> pipeline (with SMOTE if available)."""
    models: Dict[str, any] = {}

    base_estimators = {
        "logreg": LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
    }
    if HAS_XGB:
        base_estimators["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="multi:softprob", random_state=random_state, n_jobs=-1
        )

    for name, est in base_estimators.items():
        if HAS_IMB:
            pipe = ImbPipeline(steps=[
                ("pre", pre),
                ("smote", SMOTE(random_state=random_state)),
                ("clf", est),
            ])
        else:
            pipe = Pipeline(steps=[
                ("pre", pre),
                ("clf", est),
            ])
        models[name] = pipe
    return models

def tune_logreg(pipe, X, y) -> GridSearchCV:
    """Simple grid for LR C and penalty."""
    param_grid = {
        "clf__C": [0.1, 1.0, 3.0, 10.0],
        "clf__penalty": ["l2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs

def evaluate_models(models: Dict[str, any], X_train, y_train, X_test, y_test, outdir: str) -> Tuple[pd.DataFrame, str, any]:
    """Train/evaluate, save plots, return leaderboard and best."""
    rows = []
    best_name, best_score, best_pipe = None, -np.inf, None
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            proba = pipe.predict_proba(X_test)
        except Exception:
            proba = None
        m = summarize_metrics(y_test, y_pred, proba[:, 1] if (proba is not None and proba.ndim == 2 and proba.shape[1] == 2) else proba)
        m["model"] = name
        rows.append(m)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(ticks=np.arange(len(np.unique(y_test))), labels=np.unique(y_test), rotation=45)
        plt.yticks(ticks=np.arange(len(np.unique(y_test))), labels=np.unique(y_test))
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        savefig(os.path.join(outdir, f"cm_{name}.png"))

        # ROC curves (binary only)
        if proba is not None and is_binary(y_test):
            try:
                RocCurveDisplay.from_predictions(y_test, proba[:, 1] if proba.ndim == 2 else proba)
                plt.title(f"ROC - {name}")
                savefig(os.path.join(outdir, f"roc_{name}.png"))
            except Exception:
                pass

        # Track best
        score = m.get("f1_w", -np.inf)
        if score > best_score:
            best_name, best_score, best_pipe = name, score, pipe

        # classification report text dump
        report = classification_report(y_test, y_pred)
        with open(os.path.join(outdir, f"classification_report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    leaderboard = pd.DataFrame(rows).sort_values(by="f1_w", ascending=False)
    # Bar chart comparison
    plt.figure()
    plt.bar(leaderboard["model"], leaderboard["f1_w"])
    plt.title("Model Comparison (F1-weighted)")
    plt.xlabel("Model")
    plt.ylabel("F1_weighted")
    savefig(os.path.join(outdir, "model_comparison.png"))

    return leaderboard, best_name or "", best_pipe

# -------------------------- Feature Relevance --------------------------

def compute_feature_relevance(best_pipe, X_train, y_train, outdir: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Extract importance via model internals + permutation."""
    try:
        pre: ColumnTransformer = best_pipe.named_steps["pre"]
    except Exception:
        return None, None

    # Build feature names
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feature_names = list(num_cols) + cat_out

    imp_df = None
    try:
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
        elif hasattr(clf, "coef_"):
            coefs = np.mean(np.abs(clf.coef_), axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
            imp_df = pd.DataFrame({"feature": feature_names, "importance": coefs}).sort_values("importance", ascending=False)
    except Exception:
        pass

    # Permutation importance on a small subset (why: model-agnostic)
    try:
        idx = np.random.RandomState(42).choice(np.arange(X_train.shape[0]), size=min(2000, X_train.shape[0]), replace=False)
        Xs, ys = X_train.iloc[idx], y_train.iloc[idx]
        result = permutation_importance(best_pipe, Xs, ys, n_repeats=10, random_state=42, n_jobs=-1)
        perm = pd.DataFrame({"feature": feature_names, "perm_importance": result.importances_mean}).sort_values("perm_importance", ascending=False)
        if imp_df is None:
            imp_df = perm.rename(columns={"perm_importance": "importance"})
        else:
            imp_df = imp_df.merge(perm, on="feature", how="left")
    except Exception:
        pass

    if imp_df is not None:
        plt.figure()
        top = imp_df.head(20)
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title("Top Feature Importance (model-derived)")
        plt.xlabel("Importance")
        savefig(os.path.join(outdir, "feature_importance.png"))
        return imp_df, os.path.join(outdir, "feature_importance.png")

    return None, None

# -------------------------- Summary --------------------------

def write_summary(
    outpaths: Dict[str, Optional[str]],
    leaderboard: Optional[pd.DataFrame],
    best_name: str,
    cfg: ColumnConfig,
    outdir: str
) -> str:
    """Markdown summary with links to generated artifacts."""
    md = ["# Risk Analytics & Modeling Summary", ""]
    def add_fig(title, key):
        p = outpaths.get(key)
        if p:
            md.append(f"## {title}\n![]({os.path.relpath(p, outdir)})\n")

    md.append("## Columns Used")
    md.append(f"- Target: `{cfg.target}`")
    md.append(f"- Risk Score: `{cfg.risk_score}`")
    md.append(f"- Risk Level: `{cfg.risk_level}`")
    md.append(f"- mp_count_per_l: `{cfg.mp_count}`")
    md.append(f"- Polymer Type: `{cfg.polymer}`")
    md.append("")

    add_fig("Risk Score Distribution", "risk_dist")
    add_fig("Risk Score by Risk Level", "risk_by_level")
    add_fig("Risk Score vs mp_count_per_l", "risk_vs_mp")
    add_fig("Polymer Type Distribution", "polymer_dist")
    add_fig("Model Comparison", "model_cmp")
    add_fig("Top Feature Importance", "feat_imp")

    if leaderboard is not None and not leaderboard.empty:
        md.append("## Leaderboard (F1-weighted)")
        md.append(leaderboard.to_markdown(index=False))
        md.append("")
        md.append(f"**Best model:** `{best_name}`")
    else:
        md.append("No models evaluated.")

    path = os.path.join(outdir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return path

# -------------------------- Main Flow --------------------------

def main():
    parser = argparse.ArgumentParser(description="Risk analytics system")
    parser.add_argument("--input", required=True, help="Path to CSV or Parquet table")
    parser.add_argument("--target", default=None, help="Target column for risk type classification")
    parser.add_argument("--risk-score", dest="risk_score", default=None, help="Risk score numeric column")
    parser.add_argument("--risk-level", dest="risk_level", default=None, help="Risk level categorical column")
    parser.add_argument("--mp-count", dest="mp_count", default=None, help="MP count per L numeric column")
    parser.add_argument("--polymer", default=None, help="Polymer type categorical column (optional)")
    parser.add_argument("--id-cols", default=None, help="Comma-separated ID columns to exclude from features")
    parser.add_argument("--date-cols", default=None, help="Comma-separated date columns to exclude from features")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    outdirs = ensure_dirs()
    plots_dir = outdirs["plots"]
    base_dir = outdirs["base"]

    df = load_table(args.input)
    df = sanitize_columns(df)
    cfg = detect_columns(df, args)

    # Basic checks
    if cfg.target is None or cfg.target not in df.columns:
        raise ValueError("Target column not found. Provide --target or include Risk_Type in the dataset.")

    # Feature/target split
    drop_cols = list(set((cfg.id_cols or []) + (cfg.date_cols or []) + [cfg.target]))
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[cfg.target].astype("category")

    # EDA plots
    outpaths: Dict[str, Optional[str]] = {}
    outpaths["risk_dist"] = plot_risk_score_distribution(df, cfg, plots_dir)
    outpaths["risk_by_level"] = plot_risk_score_by_level(df, cfg, plots_dir)
    outpaths["risk_vs_mp"] = plot_risk_vs_mp(df, cfg, plots_dir)
    outpaths["polymer_dist"] = plot_polymer_distribution(df, cfg, plots_dir)

    # Outlier capping before transformers (why: reduce influence on PT/scale)
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = cap_outliers_iqr(X, numeric_cols)

    # Preprocessor
    pre, num_cols, cat_cols = build_preprocessor(df.drop(columns=[cfg.target]), cfg)

    # Feature selection diagnostics (does not alter X yet)
    fs = run_feature_selection(X, y, pre, max_feats=50)
    # (Optional) Could filter low variance; keep diagnostics only to avoid feature leakage across CV.

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    # Models
    models = build_model_pipelines(pre, random_state=args.random_state)

    # Tune logistic regression quickly
    tuned_logreg = tune_logreg(models["logreg"], X_train, y_train)
    models["logreg_tuned"] = tuned_logreg.best_estimator_

    # Evaluate
    leaderboard, best_name, best_pipe = evaluate_models(models, X_train, y_train, X_test, y_test, plots_dir)
    outpaths["model_cmp"] = os.path.join(plots_dir, "model_comparison.png")

    # Feature relevance for best model
    imp_df, imp_fig = compute_feature_relevance(best_pipe, X_train, y_train, plots_dir)
    outpaths["feat_imp"] = imp_fig

    # Persist best model
    try:
        import joblib  # lazy
        joblib.dump(best_pipe, os.path.join(base_dir, "best_model.joblib"))
    except Exception:
        pass  # Environment without joblib

    # Write summary
    summary_path = write_summary(outpaths, leaderboard, best_name, cfg, base_dir)
    print(f"Done. Summary: {summary_path}")

if __name__ == "__main__":
    main()
