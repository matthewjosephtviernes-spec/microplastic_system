# file: app.py
"""
Risk analytics & modeling CLI.

Example:
  python app.py --input data.csv --target Risk_Type \
    --risk-score Risk_Score --risk-level Risk_Level --mp-count mp_count_per_l --polymer Polymer_Type
Outputs:
  outputs/plots/*.png
  outputs/summary.md
  outputs/best_model.joblib
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Optional deps
try:
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
    HAS_IMB = True
except Exception:
    HAS_IMB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")  # headless

# ---------- small utils ----------
def ensure_dirs() -> Dict[str, str]:
    base = "outputs"; plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    return {"base": base, "plots": plots}

def savefig(path: str) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def is_binary(y: pd.Series) -> bool:
    return y.nunique(dropna=True) == 2

# ---------- IO ----------
def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [c.strip().replace(" ", "_") for c in df.columns]; return df

# ---------- column config ----------
@dataclass
class ColumnConfig:
    id_cols: List[str]; date_cols: List[str]
    target: Optional[str]; risk_score: Optional[str]
    risk_level: Optional[str]; mp_count: Optional[str]; polymer: Optional[str]

def detect_columns(df: pd.DataFrame, args: argparse.Namespace) -> ColumnConfig:
    cols = set(df.columns)
    def pick(name: Optional[str], candidates: List[str]) -> Optional[str]:
        if name and name in cols: return name
        for c in candidates:
            if c in cols: return c
        return None
    target = pick(args.target, ["Risk_Type","risk_type","RISK_TYPE"])
    risk_score = pick(args.risk_score, ["Risk_Score","risk_score","score"])
    risk_level = pick(args.risk_level, ["Risk_Level","risk_level","level"])
    mp_count = pick(args.mp_count, ["mp_count_per_l","mp_count","MP_Count","mp"])
    polymer = pick(args.polymer, ["Polymer_Type","polymer_type","polymer"])
    id_cols = [c for c in (args.id_cols or "").split(",") if c in cols] if args.id_cols else []
    date_cols = [c for c in (args.date_cols or "").split(",") if c in cols] if args.date_cols else []
    return ColumnConfig(id_cols, date_cols, target, risk_score, risk_level, mp_count, polymer)

# ---------- EDA ----------
def plot_risk_score_distribution(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not cfg.risk_score or cfg.risk_score not in df: return None
    s = df[cfg.risk_score].dropna()
    plt.figure(); plt.hist(s, bins=30)
    plt.title("Risk Score Distribution"); plt.xlabel(cfg.risk_score); plt.ylabel("Count")
    p = os.path.join(outdir, "risk_score_distribution.png"); savefig(p); return p

def plot_risk_score_by_level(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not (cfg.risk_score and cfg.risk_level): return None
    if cfg.risk_score not in df or cfg.risk_level not in df: return None
    levels = df[cfg.risk_level].dropna().unique()
    data = [df.loc[df[cfg.risk_level]==lvl, cfg.risk_score].dropna().values for lvl in levels]
    if not len(data): return None
    plt.figure(); plt.boxplot(data, labels=[str(x) for x in levels], showfliers=True)
    plt.title("Risk Score by Risk Level"); plt.xlabel("Risk Level"); plt.ylabel(cfg.risk_score)
    p = os.path.join(outdir, "risk_score_by_level.png"); savefig(p); return p

def plot_risk_vs_mp(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not (cfg.risk_score and cfg.mp_count): return None
    if cfg.risk_score not in df or cfg.mp_count not in df: return None
    x, y = df[cfg.mp_count], df[cfg.risk_score]; m = (~x.isna()) & (~y.isna())
    plt.figure(); plt.scatter(x[m], y[m], alpha=0.6)
    plt.title("Risk Score vs mp_count_per_l"); plt.xlabel(cfg.mp_count); plt.ylabel(cfg.risk_score)
    p = os.path.join(outdir, "risk_vs_mpcount.png"); savefig(p); return p

def plot_polymer_distribution(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Optional[str]:
    if not (cfg.polymer and cfg.polymer in df): return None
    counts = df[cfg.polymer].value_counts(dropna=False)
    plt.figure(); counts.plot(kind="bar")
    plt.title("Polymer Type Distribution"); plt.xlabel("Polymer Type"); plt.ylabel("Count")
    p = os.path.join(outdir, "polymer_type_distribution.png"); savefig(p); return p

# ---------- preprocessing ----------
def cap_outliers_iqr(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in numeric_cols:
        s = X[c].dropna()
        if s.empty: continue
        q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
        if iqr <= 0: continue
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        X[c] = X[c].clip(lower, upper)
    return X

def build_preprocessor(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[ColumnTransformer, List[str], List[str]]:
    drop_cols = set((cfg.id_cols or []) + (cfg.date_cols or []))
    kept = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in kept if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in kept if c not in num_cols and c != (cfg.target or "")]
    num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")),
                               ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                               ("scale", RobustScaler(with_centering=True))])
    cat_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return pre, num_cols, cat_cols

# ---------- feature selection diag ----------
def feature_selection_diag(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer, max_feats: int = 50) -> Dict[str, object]:
    Xt = pre.fit_transform(X, y)
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feat_names = list(num_cols) + cat_out

    vt = VarianceThreshold(threshold=1e-4); vt.fit(Xt)
    low_var = [feat_names[i] for i, keep in enumerate(vt.get_support()) if not keep]

    y_enc, _ = pd.factorize(y)
    idx = np.random.RandomState(42).choice(np.arange(Xt.shape[0]), size=min(5000, Xt.shape[0]), replace=False)
    mi = mutual_info_classif(Xt[idx], y_enc[idx], random_state=42, discrete_features=[False]*Xt.shape[1])
    mi_rank = sorted(zip(feat_names, mi), key=lambda t: t[1], reverse=True)
    return {"feature_names": feat_names, "low_variance": low_var, "mi_ranking": mi_rank, "top_mi": mi_rank[:max_feats]}

# ---------- models ----------
def build_models(pre: ColumnTransformer, random_state: int = 42) -> Dict[str, object]:
    base = {
        "logreg": LogisticRegression(max_iter=200, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
    }
    if HAS_XGB:
        base["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="multi:softprob", random_state=random_state, n_jobs=-1
        )
    models: Dict[str, object] = {}
    for name, clf in base.items():
        if HAS_IMB:
            models[name] = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=random_state)), ("clf", clf)])
        else:
            models[name] = Pipeline([("pre", pre), ("clf", clf)])
    return models

def tune_logreg(pipe, X, y):
    grid = {"clf__C": [0.1, 1.0, 3.0, 10.0], "clf__penalty": ["l2"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y); return gs

def summarize_metrics(y_true, y_pred, proba=None) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "precision_w": p, "recall_w": r, "f1_w": f1}
    if proba is not None:
        try:
            if proba.ndim == 1: out["roc_auc"] = roc_auc_score(y_true, proba)
            else: out["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr")
        except Exception:
            pass
    return out

def evaluate_models(models: Dict[str, object], Xtr, ytr, Xte, yte, outdir: str):
    rows = []; best = ("", -np.inf, None)
    labels = np.unique(yte)
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        proba = None
        try: proba = pipe.predict_proba(Xte)
        except Exception: pass
        m = summarize_metrics(yte, yhat, proba[:,1] if (proba is not None and proba.ndim==2 and proba.shape[1]==2) else proba)
        m["model"] = name; rows.append(m)

        cm = confusion_matrix(yte, yhat, labels=labels)
        plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        for (i, j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
        savefig(os.path.join(outdir, f"cm_{name}.png"))

        if proba is not None and is_binary(yte):
            try:
                RocCurveDisplay.from_predictions(yte, proba[:,1] if proba.ndim==2 else proba)
                plt.title(f"ROC - {name}"); savefig(os.path.join(outdir, f"roc_{name}.png"))
            except Exception: pass

        with open(os.path.join(outdir, f"classification_report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(yte, yhat))

        if m.get("f1_w", -np.inf) > best[1]: best = (name, m["f1_w"], pipe)

    leaderboard = pd.DataFrame(rows).sort_values("f1_w", ascending=False)
    plt.figure(); plt.bar(leaderboard["model"], leaderboard["f1_w"])
    plt.title("Model Comparison (F1-weighted)"); plt.xlabel("Model"); plt.ylabel("F1_weighted")
    savefig(os.path.join(outdir, "model_comparison.png"))
    return leaderboard, best[0], best[2]

# ---------- feature relevance ----------
def feature_relevance(best_pipe, Xtr, ytr, outdir: str):
    try:
        pre: ColumnTransformer = best_pipe.named_steps["pre"]
    except Exception:
        return None, None
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feat_names = list(num_cols) + cat_out

    imp_df = None
    try:
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
        elif hasattr(clf, "coef_"):
            coefs = np.mean(np.abs(clf.coef_), axis=0) if getattr(clf.coef_, "ndim", 1) > 1 else np.abs(clf.coef_)
            imp_df = pd.DataFrame({"feature": feat_names, "importance": coefs}).sort_values("importance", ascending=False)
    except Exception:
        pass

    try:
        idx = np.random.RandomState(42).choice(np.arange(Xtr.shape[0]), size=min(2000, Xtr.shape[0]), replace=False)
        res = permutation_importance(best_pipe, Xtr.iloc[idx], ytr.iloc[idx], n_repeats=10, random_state=42, n_jobs=-1)
        perm = pd.DataFrame({"feature": feat_names, "perm_importance": res.importances_mean}).sort_values("perm_importance", ascending=False)
        if imp_df is None: imp_df = perm.rename(columns={"perm_importance":"importance"})
        else: imp_df = imp_df.merge(perm, on="feature", how="left")
    except Exception:
        pass

    if imp_df is not None:
        plt.figure(); top = imp_df.head(20); plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title("Top Feature Importance"); plt.xlabel("Importance")
        path = os.path.join(outdir, "feature_importance.png"); savefig(path)
        return imp_df, path
    return None, None

# ---------- summary ----------
def write_summary(figs: Dict[str, Optional[str]], leaderboard: Optional[pd.DataFrame], best_name: str, cfg: ColumnConfig, outdir: str) -> str:
    md = ["# Risk Analytics & Modeling Summary","",
          "## Columns",
          f"- Target: `{cfg.target}`",
          f"- Risk Score: `{cfg.risk_score}`",
          f"- Risk Level: `{cfg.risk_level}`",
          f"- mp_count_per_l: `{cfg.mp_count}`",
          f"- Polymer Type: `{cfg.polymer}`",""]
    def add(title, key):
        p = figs.get(key); 
        if p: md.append(f"## {title}\n![]({os.path.relpath(p, outdir)})\n")
    add("Risk Score Distribution","risk_dist")
    add("Risk Score by Level","risk_by_level")
    add("Risk Score vs mp_count_per_l","risk_vs_mp")
    add("Polymer Type Distribution","polymer_dist")
    add("Model Comparison","model_cmp")
    add("Top Feature Importance","feat_imp")

    if leaderboard is not None and not leaderboard.empty:
        md.append("## Leaderboard\n" + leaderboard.to_markdown(index=False))
        md.append(f"\n**Best model:** `{best_name}`")
    else:
        md.append("No models evaluated.")
    path = os.path.join(outdir, "summary.md")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(md))
    return path

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Risk analytics pipeline")
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--risk-score", dest="risk_score", default=None)
    ap.add_argument("--risk-level", dest="risk_level", default=None)
    ap.add_argument("--mp-count", dest="mp_count", default=None)
    ap.add_argument("--polymer", default=None)
    ap.add_argument("--id-cols", default=None)
    ap.add_argument("--date-cols", default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    out = ensure_dirs(); plots = out["plots"]; base = out["base"]

    df = load_table(args.input); df = sanitize_columns(df)
    cfg = detect_columns(df, args)
    if cfg.target is None or cfg.target not in df: raise ValueError("Target column not found. Provide --target.")
    drop_cols = list(set((cfg.id_cols or []) + (cfg.date_cols or []) + [cfg.target]))
    X = df.drop(columns=[c for c in drop_cols if c in df], errors="ignore")
    y = df[cfg.target].astype("category")

    figs: Dict[str, Optional[str]] = {}
    figs["risk_dist"] = plot_risk_score_distribution(df, cfg, plots)
    figs["risk_by_level"] = plot_risk_score_by_level(df, cfg, plots)
    figs["risk_vs_mp"] = plot_risk_vs_mp(df, cfg, plots)
    figs["polymer_dist"] = plot_polymer_distribution(df, cfg, plots)

    num_cols_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = cap_outliers_iqr(X, num_cols_all)

    pre, _, _ = build_preprocessor(df.drop(columns=[cfg.target]), cfg)
    _ = feature_selection_diag(X, y, pre, max_feats=50)  # diagnostics only

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    models = build_models(pre, random_state=args.random_state)
    tuned = tune_logreg(models["logreg"], Xtr, ytr); models["logreg_tuned"] = tuned.best_estimator_
    leaderboard, best_name, best_pipe = evaluate_models(models, Xtr, ytr, Xte, yte, plots)
    figs["model_cmp"] = os.path.join(plots, "model_comparison.png")

    _, feat_fig = feature_relevance(best_pipe, Xtr, ytr, plots)
    figs["feat_imp"] = feat_fig

    try:
        import joblib; joblib.dump(best_pipe, os.path.join(base, "best_model.joblib"))
    except Exception:
        pass

    summary = write_summary(figs, leaderboard, best_name, cfg, base)
    print(f"Done. Summary: {summary}")

if __name__ == "__main__":
    main()
