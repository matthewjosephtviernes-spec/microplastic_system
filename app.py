# file: pipeline_tasks.py
"""
All-in-one Risk Analytics Pipeline matching the requested task list.

Quick demo:
    python pipeline_tasks.py --demo --verbose

Real data:
    python pipeline_tasks.py --input data.csv --target Risk_Type \
      --risk-score Risk_Score --risk-level Risk_Level --mp-count mp_count_per_l \
      --polymer Polymer_Type --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
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

# Optional: class imbalance
try:
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
    HAS_IMB = True
except Exception:
    HAS_IMB = False

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")  # headless saves

# ---------------- Utilities ----------------
def log(msg: str, on: bool) -> None:
    if on:
        print(msg, flush=True)

def ensure_dirs() -> Dict[str, str]:
    base = "outputs"
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    return {"base": base, "plots": plots}

def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def is_binary(y: pd.Series) -> bool:
    return y.nunique(dropna=True) == 2

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

# ---------------- Demo data ----------------
def make_demo_csv(path: str, n: int = 1500, seed: int = 42) -> str:
    rng = np.random.RandomState(seed)
    risk_level = rng.choice(["Low","Medium","High"], size=n, p=[0.5,0.35,0.15])
    polymer = rng.choice(["PE","PP","PS","PET","PVC"], size=n)
    mp = np.clip(rng.normal(100, 40, size=n) + (risk_level == "High")*50 + (risk_level == "Medium")*20, 5, None)
    score = np.clip(0.02*mp + rng.normal(0, 1.5, size=n) + (risk_level == "High")*3 + (risk_level == "Medium")*1.2, 0, None)
    logits = -1.0 + 0.15*score + (polymer == "PS")*0.5 + (risk_level == "High")*0.7
    prob = 1/(1+np.exp(-logits))
    risk_type = np.where(rng.uniform(size=n) < prob, "At_Risk", "Safe")
    df = pd.DataFrame({
        "Risk_Score": score,
        "Risk_Level": risk_level,
        "mp_count_per_l": mp,
        "Polymer_Type": polymer,
        "Risk_Type": pd.Series(risk_type).astype("category"),
    })
    df.to_csv(path, index=False)
    return path

# ---------------- Column config ----------------
@dataclass
class ColumnConfig:
    target: Optional[str]
    risk_score: Optional[str]
    risk_level: Optional[str]
    mp_count: Optional[str]
    polymer: Optional[str]
    id_cols: List[str]
    date_cols: List[str]

def detect_columns(df: pd.DataFrame, args: argparse.Namespace) -> ColumnConfig:
    cols = set(df.columns)

    def pick(name: Optional[str], candidates: List[str]) -> Optional[str]:
        if name and name in cols:
            return name
        for c in candidates:
            if c in cols:
                return c
        return None

    target = pick(args.target, ["Risk_Type","risk_type","RISK_TYPE"])
    risk_score = pick(args.risk_score, ["Risk_Score","risk_score","score"])
    risk_level = pick(args.risk_level, ["Risk_Level","risk_level","level"])
    mp_count = pick(args.mp_count, ["mp_count_per_l","mp_count","MP_Count","mp"])
    polymer = pick(args.polymer, ["Polymer_Type","polymer_type","polymer"])
    id_cols = [c for c in (args.id_cols or "").split(",") if c in cols] if args.id_cols else []
    date_cols = [c for c in (args.date_cols or "").split(",") if c in cols] if args.date_cols else []
    return ColumnConfig(target, risk_score, risk_level, mp_count, polymer, id_cols, date_cols)

# ---------------- EDA ----------------
def eda_plots(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Dict[str, Optional[str]]:
    paths: Dict[str, Optional[str]] = {"risk_dist": None, "risk_by_level": None, "risk_vs_mp": None, "polymer_dist": None}
    if cfg.risk_score and cfg.risk_score in df:
        plt.figure(); plt.hist(df[cfg.risk_score].dropna(), bins=30)
        plt.title("Risk Score Distribution"); plt.xlabel(cfg.risk_score); plt.ylabel("Count")
        p = os.path.join(outdir, "risk_score_distribution.png"); savefig(p); paths["risk_dist"] = p
    if cfg.risk_score and cfg.risk_level and cfg.risk_score in df and cfg.risk_level in df:
        levels = df[cfg.risk_level].dropna().unique()
        data = [df.loc[df[cfg.risk_level]==lvl, cfg.risk_score].dropna().values for lvl in levels]
        if len(data):
            plt.figure(); plt.boxplot(data, labels=[str(x) for x in levels], showfliers=True)
            plt.title("Risk Score by Risk Level"); plt.xlabel("Risk Level"); plt.ylabel(cfg.risk_score)
            p = os.path.join(outdir, "risk_score_by_level.png"); savefig(p); paths["risk_by_level"] = p
    if cfg.risk_score and cfg.mp_count and cfg.risk_score in df and cfg.mp_count in df:
        m = df[[cfg.mp_count, cfg.risk_score]].dropna()
        plt.figure(); plt.scatter(m[cfg.mp_count], m[cfg.risk_score], alpha=0.6)
        plt.title("Risk Score vs mp_count_per_l"); plt.xlabel(cfg.mp_count); plt.ylabel(cfg.risk_score)
        p = os.path.join(outdir, "risk_vs_mpcount.png"); savefig(p); paths["risk_vs_mp"] = p
    if cfg.polymer and cfg.polymer in df:
        counts = df[cfg.polymer].value_counts(dropna=False)
        plt.figure(); counts.plot(kind="bar")
        plt.title("Polymer Type Distribution"); plt.xlabel("Polymer Type"); plt.ylabel("Count")
        p = os.path.join(outdir, "polymer_type_distribution.png"); savefig(p); paths["polymer_dist"] = p
    return paths

# ---------------- Preprocessing ----------------
def cap_outliers_iqr(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in numeric_cols:
        s = X[c].dropna()
        if s.empty:
            continue
        q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        X[c] = X[c].clip(lower, upper)
    return X

def build_preprocessor(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[ColumnTransformer, List[str], List[str]]:
    drop_cols = set((cfg.id_cols or []) + (cfg.date_cols or []))
    kept = [c for c in df.columns if c not in drop_cols and c != (cfg.target or "")]
    num_cols = [c for c in kept if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in kept if c not in num_cols]
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                    ("scale", RobustScaler(with_centering=True))])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")
    return pre, num_cols, cat_cols

# ---------------- Feature selection (diag) ----------------
def feature_selection_diag(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer, outdir: str) -> pd.DataFrame:
    Xt = pre.fit_transform(X, y)
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feat_names = list(num_cols) + cat_out

    vt = VarianceThreshold(1e-4).fit(Xt)
    low_var = [feat_names[i] for i, keep in enumerate(vt.get_support()) if not keep]

    y_enc, _ = pd.factorize(y)
    idx = np.random.RandomState(42).choice(np.arange(Xt.shape[0]), size=min(5000, Xt.shape[0]), replace=False)
    mi = mutual_info_classif(Xt[idx], y_enc[idx], random_state=42, discrete_features=[False]*Xt.shape[1])
    mi_df = pd.DataFrame({"feature": feat_names, "mi": mi}).sort_values("mi", ascending=False)

    # Plot top-20 MI
    top = mi_df.head(20)
    plt.figure(); plt.barh(top["feature"][::-1], top["mi"][::-1]); plt.title("Top Mutual Information Features"); plt.xlabel("MI")
    savefig(os.path.join(outdir, "feature_mi_top20.png"))

    return pd.DataFrame({"feature": low_var, "reason": "low_variance"}), mi_df

# ---------------- Modeling ----------------
def summarize_metrics(y_true, y_pred, proba=None) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "precision_w": p, "recall_w": r, "f1_w": f1}
    try:
        if proba is not None:
            if proba.ndim == 1: out["roc_auc"] = roc_auc_score(y_true, proba)
            else: out["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr")
    except Exception:
        pass
    return out

def build_models(pre: ColumnTransformer, use_smote: bool, seed: int) -> Dict[str, object]:
    # Why: SMOTE inside pipeline prevents leakage
    def wrap(est):
        if use_smote and HAS_IMB:
            return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=seed)), ("clf", est)])
        return Pipeline([("pre", pre), ("clf", est)])

    models = {
        "logreg": wrap(LogisticRegression(max_iter=200, solver="lbfgs")),
        "rf": wrap(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
    }
    return models

def tune_logreg(pipe, X, y) -> GridSearchCV:
    grid = {"clf__C": [0.1, 1.0, 3.0, 10.0]}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs

def evaluate(models: Dict[str, object], Xtr, ytr, Xte, yte, outdir: str) -> Tuple[pd.DataFrame, str, object]:
    rows = []
    labels = np.unique(yte)
    best_name, best_score, best_pipe = "", -1.0, None
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        proba = None
        try:
            proba = pipe.predict_proba(Xte)
        except Exception:
            pass
        met = summarize_metrics(yte, yhat, proba[:,1] if (proba is not None and proba.ndim==2 and proba.shape[1]==2) else proba)
        met["model"] = name
        rows.append(met)

        cm = confusion_matrix(yte, yhat, labels=labels)
        plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        for (i, j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
        savefig(os.path.join(outdir, f"cm_{name}.png"))

        if proba is not None and len(labels) == 2:
            try:
                RocCurveDisplay.from_predictions(yte, proba[:,1] if proba.ndim==2 else proba)
                plt.title(f"ROC - {name}")
                savefig(os.path.join(outdir, f"roc_{name}.png"))
            except Exception:
                pass

        with open(os.path.join(outdir, f"classification_report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(yte, yhat))

        if met["f1_w"] > best_score:
            best_name, best_score, best_pipe = name, met["f1_w"], pipe

    leaderboard = pd.DataFrame(rows).sort_values("f1_w", ascending=False)
    plt.figure(); plt.bar(leaderboard["model"], leaderboard["f1_w"])
    plt.title("Model Comparison (F1-weighted)"); plt.xlabel("Model"); plt.ylabel("F1_weighted")
    savefig(os.path.join(outdir, "model_comparison.png"))
    return leaderboard, best_name, best_pipe

# ---------------- Feature relevance ----------------
def feature_relevance(best_pipe, Xtr, ytr, outdir: str) -> Optional[str]:
    try:
        pre: ColumnTransformer = best_pipe.named_steps["pre"]
    except Exception:
        return None

    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feat_names = list(num_cols) + cat_out

    # Try model-derived
    imp_df = None
    try:
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
        elif hasattr(clf, "coef_"):
            coefs = np.mean(np.abs(clf.coef_), axis=0) if getattr(clf.coef_, "ndim", 1) > 1 else np.abs(clf.coef_)
            imp_df = pd.DataFrame({"feature": feat_names, "importance": coefs})
    except Exception:
        pass

    # Permutation (small subset)
    try:
        idx = np.random.RandomState(42).choice(np.arange(Xtr.shape[0]), size=min(2000, Xtr.shape[0]), replace=False)
        res = permutation_importance(best_pipe, Xtr.iloc[idx], ytr.iloc[idx], n_repeats=8, random_state=42, n_jobs=-1)
        perm = pd.DataFrame({"feature": feat_names, "perm_importance": res.importances_mean})
        if imp_df is None:
            imp_df = perm.rename(columns={"perm_importance": "importance"})
        else:
            imp_df = imp_df.merge(perm, on="feature", how="left")
    except Exception:
        pass

    if imp_df is not None:
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        plt.figure(); plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        plt.title("Top Feature Importance"); plt.xlabel("Importance")
        p = os.path.join(outdir, "feature_importance.png"); savefig(p); return p
    return None

# ---------------- Summary ----------------
def write_summary(figs: Dict[str, Optional[str]], leaderboard: pd.DataFrame, best_name: str, cfg: ColumnConfig, outdir: str, class_dist: str) -> str:
    md = [
        "# Summary",
        "",
        "## Tasks Completed",
        "- Encode categorical variables",
        "- Perform feature scaling",
        "- Address outliers",
        "- Analyze distribution of risk score",
        "- Investigate difference in risk score by risk level",
        "- Explore relationship between risk score and mp_count_per_l",
        "- Transform skewed numerical columns",
        "- Explore feature selection (variance + mutual information)",
        "- Prepare/train/evaluate/tune models for Risk_Type",
        "- Compare model performance",
        "- Extract/analyze/visualize feature relevance",
        "",
        "## Columns",
        f"- Target: `{cfg.target}`",
        f"- Risk Score: `{cfg.risk_score}`",
        f"- Risk Level: `{cfg.risk_level}`",
        f"- mp_count_per_l: `{cfg.mp_count}`",
        f"- Polymer Type: `{cfg.polymer}`",
        "",
        "## Class Distribution",
        class_dist,
        "",
    ]

    def add(title, key):
        p = figs.get(key)
        if p:
            md.append(f"## {title}\n![]({os.path.relpath(p, outdir)})\n")

    add("Risk Score Distribution", "risk_dist")
    add("Risk Score by Risk Level", "risk_by_level")
    add("Risk Score vs mp_count_per_l", "risk_vs_mp")
    add("Polymer Type Distribution", "polymer_dist")
    add("Top Mutual Information Features", "mi_top")
    add("Model Comparison", "model_cmp")
    add("Top Feature Importance", "feat_imp")

    if leaderboard is not None and not leaderboard.empty:
        md.append("## Leaderboard\n" + leaderboard.to_markdown(index=False))
        md.append(f"\n**Best model:** `{best_name}`")

    path = os.path.join(outdir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return path

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="End-to-end risk pipeline (matches task list)")
    ap.add_argument("--input", help="CSV or Parquet path")
    ap.add_argument("--target", default=None)
    ap.add_argument("--risk-score", dest="risk_score", default=None)
    ap.add_argument("--risk-level", dest="risk_level", default=None)
    ap.add_argument("--mp-count", dest="mp_count", default=None)
    ap.add_argument("--polymer", default=None)
    ap.add_argument("--id-cols", default=None)
    ap.add_argument("--date-cols", default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-smote", action="store_true", help="Disable SMOTE even if available")
    ap.add_argument("--demo", action="store_true", help="Run on synthetic data")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo = "demo.csv"
        make_demo_csv(demo)
        args.input = demo
        args.target = args.target or "Risk_Type"
        args.risk_score = args.risk_score or "Risk_Score"
        args.risk_level = args.risk_level or "Risk_Level"
        args.mp_count = args.mp_count or "mp_count_per_l"
        args.polymer = args.polymer or "Polymer_Type"

    if not args.input:
        print("ERROR: provide --input <file> or use --demo", file=sys.stderr)
        sys.exit(2)

    # Load
    log(f"Loading {args.input}", args.verbose)
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    df = sanitize_columns(df)

    cfg = detect_columns(df, args)
    log(f"Detected -> target:{cfg.target}, risk_score:{cfg.risk_score}, level:{cfg.risk_level}, mp:{cfg.mp_count}, polymer:{cfg.polymer}", args.verbose)
    if not cfg.target or cfg.target not in df:
        print("ERROR: target column missing; use --target", file=sys.stderr)
        sys.exit(2)

    outdirs = ensure_dirs()
    plots = outdirs["plots"]

    # EDA
    log("EDA: generating plots", args.verbose)
    figs = eda_plots(df, cfg, plots)

    # Prepare features
    drop_cols = list(set(cfg.id_cols + cfg.date_cols + [cfg.target]))
    X = df.drop(columns=[c for c in drop_cols if c in df], errors="ignore")
    y = df[cfg.target].astype("category")
    # Outliers (numeric only)
    num_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = cap_outliers_iqr(X, num_all)

    # Preprocess & feature selection diag
    pre, _, _ = build_preprocessor(df.drop(columns=[cfg.target]), cfg)
    log("Feature selection diagnostics", args.verbose)
    low_var_df, mi_df = feature_selection_diag(X, y, pre, plots)
    figs["mi_top"] = os.path.join(plots, "feature_mi_top20.png")

    # Split & class dist
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    class_dist = ytr.value_counts(normalize=True).rename("share").to_frame()
    class_dist["count"] = ytr.value_counts()
    class_dist_md = class_dist.to_markdown()

    # Build/train/tune/eval
    use_smote = (not args.no_smote) and HAS_IMB
    log(f"Models: SMOTE={'on' if use_smote else 'off'}", args.verbose)
    models = build_models(pre, use_smote, args.random_state)
    tuned = tune_logreg(models["logreg"], Xtr, ytr)
    models["logreg_tuned"] = tuned.best_estimator_

    leaderboard, best_name, best_pipe = evaluate(models, Xtr, ytr, Xte, yte, plots)
    figs["model_cmp"] = os.path.join(plots, "model_comparison.png")

    # Feature relevance
    figs["feat_imp"] = feature_relevance(best_pipe, Xtr, ytr, plots)

    # Persist best model
    try:
        import joblib
        joblib.dump(best_pipe, os.path.join(outdirs["base"], "best_model.joblib"))
    except Exception:
        pass  # non-fatal

    # Summary
    summary = write_summary(figs, leaderboard, best_name, cfg, outdirs["base"], class_dist_md)
    print(f"Done. Summary: {summary}")
    print("Open the images in outputs/plots/ and the markdown report at outputs/summary.md")

if __name__ == "__main__":
    main()
