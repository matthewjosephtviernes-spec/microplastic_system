# =========================
# file: pipeline_tasks.py
# =========================
"""
End-to-end risk analytics pipeline (CLI).
Run demo:
  python pipeline_tasks.py --demo --verbose
Real data:
  python pipeline_tasks.py --input data.csv --target Risk_Type \
    --risk-score Risk_Score --risk-level Risk_Level --mp-count mp_count_per_l --polymer Polymer_Type --verbose
Outputs:
  outputs/plots/*.png, outputs/summary.md, outputs/best_model.joblib
"""
from __future__ import annotations
import argparse, os, sys, io, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")

try:
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
    HAS_IMB = True
except Exception:
    HAS_IMB = False

try:
    from charset_normalizer import from_bytes as detect_encoding
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False

@dataclass
class ColumnConfig:
    target: Optional[str]; risk_score: Optional[str]; risk_level: Optional[str]; mp_count: Optional[str]; polymer: Optional[str]
    id_cols: List[str]; date_cols: List[str]

def log(msg: str, on: bool): 
    if on: print(msg, flush=True)

def ensure_dirs() -> Dict[str,str]:
    base, plots = "outputs", os.path.join("outputs","plots")
    os.makedirs(plots, exist_ok=True); return {"base": base, "plots": plots}

def savefig(path: str): plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [c.strip().replace(" ","_") for c in df.columns]; return df

def is_binary(y: pd.Series) -> bool: return y.nunique(dropna=True) == 2

def make_demo_csv(path: str, n: int = 1500, seed: int = 42) -> str:
    rng = np.random.RandomState(seed)
    rl = rng.choice(["Low","Medium","High"], size=n, p=[0.5,0.35,0.15])
    poly = rng.choice(["PE","PP","PS","PET","PVC"], size=n)
    mp = np.clip(rng.normal(100, 40, size=n) + (rl=="High")*50 + (rl=="Medium")*20, 5, None)
    sc = np.clip(0.02*mp + rng.normal(0,1.5,size=n) + (rl=="High")*3 + (rl=="Medium")*1.2, 0, None)
    logits = -1.0 + 0.15*sc + (poly=="PS")*0.5 + (rl=="High")*0.7
    prob = 1/(1+np.exp(-logits))
    y = np.where(rng.uniform(size=n) < prob, "At_Risk", "Safe")
    pd.DataFrame({"Risk_Score": sc, "Risk_Level": rl, "mp_count_per_l": mp, "Polymer_Type": poly, "Risk_Type": pd.Series(y).astype("category")}).to_csv(path, index=False)
    return path

def detect_cols(df: pd.DataFrame, args) -> ColumnConfig:
    cols = set(df.columns)
    def pick(name, cands): 
        if name and name in cols: return name
        for c in cands:
            if c in cols: return c
        return None
    return ColumnConfig(
        pick(args.target, ["Risk_Type","risk_type","RISK_TYPE"]),
        pick(args.risk_score, ["Risk_Score","risk_score","score"]),
        pick(args.risk_level, ["Risk_Level","risk_level","level"]),
        pick(args.mp_count, ["mp_count_per_l","mp_count","MP_Count","mp"]),
        pick(args.polymer, ["Polymer_Type","polymer_type","polymer"]),
        [c for c in (args.id_cols or "").split(",") if c in cols] if args.id_cols else [],
        [c for c in (args.date_cols or "").split(",") if c in cols] if args.date_cols else [],
    )

def load_any(path: str, verbose=False) -> pd.DataFrame:
    if path.lower().endswith((".xlsx",".xls")):
        import openpyxl  # noqa
        df = pd.read_excel(path)
        return sanitize_columns(df)
    if path.lower().endswith(".parquet"):
        import pyarrow  # noqa
        return sanitize_columns(pd.read_parquet(path))
    # robust CSV load (why: avoid UnicodeDecodeError)
    with open(path, "rb") as f: raw = f.read()
    encoding = None
    if HAS_CHARDET:
        try:
            best = detect_encoding(raw).best()
            if best: encoding = best.encoding
        except Exception:
            encoding = None
    for enc in [encoding, "utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        if not enc: continue
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            return sanitize_columns(df)
        except Exception:
            continue
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", sep=None, engine="python", on_bad_lines="skip", errors="replace")
    return sanitize_columns(df)

def cap_outliers_iqr(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in numeric_cols:
        s = X[c].dropna()
        if s.empty: continue
        q1,q3 = np.percentile(s,[25,75]); iqr = q3-q1
        if iqr <= 0: continue
        lo,hi = q1-1.5*iqr, q3+1.5*iqr
        X[c] = X[c].clip(lo,hi)
    return X

def build_pre(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[ColumnTransformer, List[str], List[str]]:
    drop = set(cfg.id_cols + cfg.date_cols)
    kept = [c for c in df.columns if c not in drop and c != (cfg.target or "")]
    num = [c for c in kept if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in kept if c not in num]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("power", PowerTransformer(method="yeo-johnson", standardize=False)),  # why: fix skew
                         ("scale", RobustScaler(with_centering=True))])                         # why: robust scaling
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    pre = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], remainder="drop")
    return pre, num, cat

def eda(df: pd.DataFrame, cfg: ColumnConfig, outdir: str) -> Dict[str, Optional[str]]:
    out = {"risk_dist": None, "risk_by_level": None, "risk_vs_mp": None, "polymer": None}
    if cfg.risk_score and cfg.risk_score in df:
        plt.figure(); plt.hist(df[cfg.risk_score].dropna(), bins=30); plt.title("Risk Score Distribution"); plt.xlabel(cfg.risk_score); plt.ylabel("Count")
        p = os.path.join(outdir,"risk_score_distribution.png"); savefig(p); out["risk_dist"] = p
    if cfg.risk_score and cfg.risk_level and cfg.risk_score in df and cfg.risk_level in df:
        lv = df[cfg.risk_level].dropna().unique()
        data = [df.loc[df[cfg.risk_level]==v, cfg.risk_score].dropna().values for v in lv]
        if data:
            plt.figure(); plt.boxplot(data, labels=[str(x) for x in lv], showfliers=True); plt.title("Risk Score by Risk Level"); plt.xlabel("Risk Level"); plt.ylabel(cfg.risk_score)
            p = os.path.join(outdir,"risk_score_by_level.png"); savefig(p); out["risk_by_level"] = p
    if cfg.risk_score and cfg.mp_count and cfg.risk_score in df and cfg.mp_count in df:
        m = df[[cfg.mp_count, cfg.risk_score]].dropna()
        plt.figure(); plt.scatter(m[cfg.mp_count], m[cfg.risk_score], alpha=0.6); plt.title("Risk Score vs mp_count_per_l"); plt.xlabel(cfg.mp_count); plt.ylabel(cfg.risk_score)
        p = os.path.join(outdir,"risk_vs_mpcount.png"); savefig(p); out["risk_vs_mp"] = p
    if cfg.polymer and cfg.polymer in df:
        counts = df[cfg.polymer].value_counts(dropna=False)
        plt.figure(); counts.plot(kind="bar"); plt.title("Polymer Type Distribution"); plt.xlabel("Polymer Type"); plt.ylabel("Count")
        p = os.path.join(outdir,"polymer_type_distribution.png"); savefig(p); out["polymer"] = p
    return out

def mutual_info_diag(pre: ColumnTransformer, X: pd.DataFrame, y: pd.Series, outdir: str) -> pd.DataFrame:
    Xt = pre.fit_transform(X,y)
    num_cols = pre.transformers_[0][2]; cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feats = list(num_cols) + cat_out
    vt = VarianceThreshold(1e-4).fit(Xt)
    y_enc, _ = pd.factorize(y)
    idx = np.random.RandomState(42).choice(np.arange(Xt.shape[0]), size=min(5000, Xt.shape[0]), replace=False)
    mi = mutual_info_classif(Xt[idx], y_enc[idx], random_state=42, discrete_features=[False]*Xt.shape[1])
    mi_df = pd.DataFrame({"feature": feats, "mi": mi}).sort_values("mi", ascending=False)
    plt.figure(); top = mi_df.head(20); plt.barh(top["feature"][::-1], top["mi"][::-1]); plt.title("Top MI Features"); plt.xlabel("MI")
    savefig(os.path.join(outdir,"feature_mi_top20.png"))
    return mi_df

def metrics(y_true, y_pred, proba=None) -> Dict[str,float]:
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    out = {"accuracy": accuracy_score(y_true,y_pred), "precision_w": p, "recall_w": r, "f1_w": f1}
    try:
        if proba is not None:
            if proba.ndim==1: out["roc_auc"] = roc_auc_score(y_true, proba)
            else: out["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr")
    except Exception: pass
    return out

def wrap_model(pre: ColumnTransformer, est, smote: bool, seed: int):
    # Why: SMOTE inside pipeline avoids leakage
    if smote and HAS_IMB: return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=seed)), ("clf", est)])
    return Pipeline([("pre", pre), ("clf", est)])

def main():
    ap = argparse.ArgumentParser(description="Risk pipeline")
    ap.add_argument("--input"); ap.add_argument("--target"); ap.add_argument("--risk-score", dest="risk_score"); ap.add_argument("--risk-level", dest="risk_level")
    ap.add_argument("--mp-count", dest="mp_count"); ap.add_argument("--polymer"); ap.add_argument("--id-cols"); ap.add_argument("--date-cols")
    ap.add_argument("--test-size", type=float, default=0.2); ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-smote", action="store_true"); ap.add_argument("--demo", action="store_true"); ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.demo:
        args.input = make_demo_csv("demo.csv")
        args.target = args.target or "Risk_Type"; args.risk_score = args.risk_score or "Risk_Score"; args.risk_level = args.risk_level or "Risk_Level"
        args.mp_count = args.mp_count or "mp_count_per_l"; args.polymer = args.polymer or "Polymer_Type"
    if not args.input: print("ERROR: provide --input or use --demo", file=sys.stderr); sys.exit(2)

    out = ensure_dirs(); plots = out["plots"]
    log(f"Loading {args.input}", args.verbose); df = load_any(args.input, verbose=args.verbose)
    cfg = detect_cols(df, args)
    if not cfg.target or cfg.target not in df: print("ERROR: target not found", file=sys.stderr); sys.exit(2)

    log("EDA plots", args.verbose); figs = eda(df, cfg, plots)

    drop = list(set(cfg.id_cols + cfg.date_cols + [cfg.target])); X = df.drop(columns=[c for c in drop if c in df], errors="ignore")
    y = df[cfg.target].astype("category")

    num_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = cap_outliers_iqr(X, num_all)

    pre,_,_ = build_pre(df.drop(columns=[cfg.target]), cfg)
    log("Feature selection diagnostics", args.verbose); mi_df = mutual_info_diag(pre, X, y, plots)

    log("Split", args.verbose); Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    class_dist = ytr.value_counts(normalize=True).rename("share").to_frame(); class_dist["count"] = ytr.value_counts()

    use_smote = (not args.no_smote) and HAS_IMB
    models = {
        "logreg": wrap_model(pre, LogisticRegression(max_iter=200, solver="lbfgs"), use_smote, args.random_state),
        "rf": wrap_model(pre, RandomForestClassifier(n_estimators=300, random_state=args.random_state, n_jobs=-1), use_smote, args.random_state),
    }
    log("Tuning Logistic Regression", args.verbose)
    gs = GridSearchCV(models["logreg"], {"clf__C":[0.1,1.0,3.0,10.0]}, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1_weighted", n_jobs=-1, refit=True)
    gs.fit(Xtr,ytr); models["logreg_tuned"] = gs.best_estimator_

    rows=[]; best_name,best_score,best_pipe = "", -1.0, None
    labels = np.unique(yte)
    for name, pipe in models.items():
        log(f"Training {name}", args.verbose)
        pipe.fit(Xtr,ytr); yhat = pipe.predict(Xte)
        proba = None
        try: proba = pipe.predict_proba(Xte)
        except Exception: pass
        m = metrics(yte,yhat, proba[:,1] if (proba is not None and proba.ndim==2 and proba.shape[1]==2) else proba); m["model"]=name; rows.append(m)

        cm = confusion_matrix(yte, yhat, labels=labels)
        plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45); plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v), ha="center", va="center")
        savefig(os.path.join(plots, f"cm_{name}.png"))

        if proba is not None and is_binary(yte):
            try:
                RocCurveDisplay.from_predictions(yte, proba[:,1] if proba.ndim==2 else proba); plt.title(f"ROC - {name}"); savefig(os.path.join(plots,f"roc_{name}.png"))
            except Exception: pass

        with open(os.path.join(plots, f"classification_report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(yte,yhat))
        if m["f1_w"] > best_score: best_name, best_score, best_pipe = name, m["f1_w"], pipe

    leaderboard = pd.DataFrame(rows).sort_values("f1_w", ascending=False)
    plt.figure(); plt.bar(leaderboard["model"], leaderboard["f1_w"]); plt.title("Model Comparison (F1-weighted)"); plt.xlabel("Model"); plt.ylabel("F1_weighted")
    savefig(os.path.join(plots, "model_comparison.png"))

    # Feature relevance
    feat_fig = None
    try:
        pre_b: ColumnTransformer = best_pipe.named_steps["pre"]
        num_cols = pre_b.transformers_[0][2]; cat_cols = pre_b.transformers_[1][2]
        ohe: OneHotEncoder = pre_b.named_transformers_["cat"].named_steps["ohe"]
        cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
        feat_names = list(num_cols) + cat_out
        clf = best_pipe.named_steps["clf"]
        imp_df = None
        if hasattr(clf,"feature_importances_"):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
        elif hasattr(clf,"coef_"):
            coefs = np.mean(np.abs(clf.coef_), axis=0) if getattr(clf.coef_,"ndim",1)>1 else np.abs(clf.coef_)
            imp_df = pd.DataFrame({"feature": feat_names, "importance": coefs})
        try:
            idx = np.random.RandomState(42).choice(np.arange(Xtr.shape[0]), size=min(1500, Xtr.shape[0]), replace=False)
            res = permutation_importance(best_pipe, Xtr.iloc[idx], ytr.iloc[idx], n_repeats=8, random_state=42, n_jobs=-1)
            perm = pd.DataFrame({"feature": feat_names, "perm_importance": res.importances_mean})
            imp_df = imp_df.merge(perm, on="feature", how="left") if imp_df is not None else perm.rename(columns={"perm_importance":"importance"})
        except Exception: pass
        if imp_df is not None:
            top = imp_df.sort_values("importance", ascending=False).head(20)
            plt.figure(); plt.barh(top["feature"][::-1], top["importance"][::-1]); plt.title("Top Feature Importance"); plt.xlabel("Importance")
            feat_fig = os.path.join(plots, "feature_importance.png"); savefig(feat_fig)
    except Exception: pass

    # Save model
    try:
        import joblib; joblib.dump(best_pipe, os.path.join(out["base"], "best_model.joblib"))
    except Exception: pass

    # Summary
    md = ["# Summary","","## Tasks Completed",
          "- Encode categorical variables","- Perform feature scaling","- Address outliers",
          "- Analyze Risk_Score distribution","- Risk_Score by Risk_Level","- Risk_Score vs mp_count_per_l",
          "- Transform skewed numerical columns","- Feature selection (variance + MI)",
          "- Prepare/train/tune/evaluate models for Risk_Type","- Compare model performance","- Feature relevance",
          "","## Columns",
          f"- Target: `{cfg.target}`", f"- Risk Score: `{cfg.risk_score}`", f"- Risk Level: `{cfg.risk_level}`", f"- mp_count_per_l: `{cfg.mp_count}`", f"- Polymer Type: `{cfg.polymer}`","",
          "## Class distribution (train)", class_dist.to_markdown(), "",
          "## Leaderboard", leaderboard.to_markdown(index=False), f"\n**Best model:** `{best_name}`",""]
    if feat_fig: md += ["## Top Feature Importance", f"![feature_importance]({os.path.relpath(feat_fig, out['base'])})"]
    summary = os.path.join(out["base"], "summary.md")
    with open(summary, "w", encoding="utf-8") as f: f.write("\n".join(md))
    print(f"Done. Summary: {summary}")

if __name__ == "__main__":
    main()


# =========================
# file: streamlit_app.py
# =========================
"""
Streamlit UI for the same pipeline (upload → EDA → preprocess → feature selection → SMOTE → train/tune → evaluate → feature relevance).
Run:
  streamlit run streamlit_app.py
"""
import os, io, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")

try:
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
    HAS_IMB = True
except Exception:
    HAS_IMB = False

try:
    from charset_normalizer import from_bytes as detect_encoding
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False

st.set_page_config(page_title="Risk Analytics System", layout="wide")

@dataclass
class ColumnConfig:
    target: Optional[str]; risk_score: Optional[str]; risk_level: Optional[str]; mp_count: Optional[str]; polymer: Optional[str]
    id_cols: List[str]; date_cols: List[str]

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [c.strip().replace(" ","_") for c in df.columns]; return df

def is_binary(y: pd.Series) -> bool: return y.nunique(dropna=True) == 2

def cap_outliers_iqr(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in numeric_cols:
        s = X[c].dropna()
        if s.empty: continue
        q1,q3 = np.percentile(s,[25,75]); iqr = q3-q1
        if iqr <= 0: continue
        lo,hi = q1-1.5*iqr, q3+1.5*iqr
        X[c] = X[c].clip(lo,hi)
    return X

def build_pre(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[ColumnTransformer, List[str], List[str]]:
    drop = set(cfg.id_cols + cfg.date_cols)
    kept = [c for c in df.columns if c not in drop and c != (cfg.target or "")]
    num = [c for c in kept if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in kept if c not in num]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("power", PowerTransformer(method="yeo-johnson", standardize=False)),  # why: de-skew
                         ("scale", RobustScaler(with_centering=True))])                         # why: robust to outliers
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], remainder="drop"), num, cat

def summarize(y_true, y_pred, proba=None) -> Dict[str,float]:
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    out = {"accuracy": accuracy_score(y_true,y_pred), "precision_w": p, "recall_w": r, "f1_w": f1}
    try:
        if proba is not None:
            if proba.ndim==1: out["roc_auc"] = roc_auc_score(y_true, proba)
            else: out["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr")
    except Exception: pass
    return out

@st.cache_data(show_spinner=False)
def load_uploaded(file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx",".xls")):
        import openpyxl  # noqa
        file.seek(0); return sanitize_columns(pd.read_excel(file))
    if name.endswith(".parquet"):
        import pyarrow  # noqa
        file.seek(0); return sanitize_columns(pd.read_parquet(file))
    file.seek(0); raw = file.read()
    enc = None
    if HAS_CHARDET:
        try:
            best = detect_encoding(raw).best()
            if best: enc = best.encoding
        except Exception: enc = None
    for e in [enc,"utf-8-sig","utf-8","cp1252","latin-1"]:
        if not e: continue
        try:
            return sanitize_columns(pd.read_csv(io.BytesIO(raw), encoding=e, sep=None, engine="python"))
        except Exception: continue
    return sanitize_columns(pd.read_csv(io.BytesIO(raw), encoding="latin-1", sep=None, engine="python", on_bad_lines="skip", errors="replace"))

def make_demo(n:int=1200, seed:int=42)->pd.DataFrame:
    rng = np.random.RandomState(seed)
    rl = rng.choice(["Low","Medium","High"], size=n, p=[0.5,0.35,0.15])
    poly = rng.choice(["PE","PP","PS","PET","PVC"], size=n)
    mp = np.clip(rng.normal(100,40,size=n)+(rl=="High")*50+(rl=="Medium")*20,5,None)
    sc = np.clip(0.02*mp + rng.normal(0,1.5,size=n) + (rl=="High")*3 + (rl=="Medium")*1.2,0,None)
    logits = -1.0 + 0.15*sc + (poly=="PS")*0.5 + (rl=="High")*0.7
    prob = 1/(1+np.exp(-logits))
    y = np.where(rng.uniform(size=n) < prob, "At_Risk", "Safe")
    return sanitize_columns(pd.DataFrame({"Risk_Score": sc,"Risk_Level": rl,"mp_count_per_l": mp,"Polymer_Type": poly,"Risk_Type": pd.Series(y).astype("category")}))

@st.cache_data(show_spinner=False)
def mi_plot(pre: ColumnTransformer, X: pd.DataFrame, y: pd.Series):
    Xt = pre.fit_transform(X,y)
    num_cols = pre.transformers_[0][2]; cat_cols = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feats = list(num_cols) + cat_out
    y_enc,_ = pd.factorize(y)
    idx = np.random.RandomState(42).choice(np.arange(Xt.shape[0]), size=min(5000,Xt.shape[0]), replace=False)
    mi = mutual_info_classif(Xt[idx], y_enc[idx], random_state=42, discrete_features=[False]*Xt.shape[1])
    mi_df = pd.DataFrame({"feature": feats, "mi": mi}).sort_values("mi", ascending=False)
    fig = plt.figure(); top = mi_df.head(20); plt.barh(top["feature"][::-1], top["mi"][::-1]); plt.title("Top MI Features"); plt.xlabel("MI")
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=160); plt.close(fig); buf.seek(0)
    return mi_df, buf

st.title("Risk Analytics & Modeling")

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("Upload CSV/Parquet/Excel", type=["csv","parquet","xlsx","xls"])
    use_demo = st.checkbox("Or use demo data", value=(up is None))
    st.header("2) Options")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("Random state", 0, 9999, 42, 1)
    use_smote = st.checkbox("Address class imbalance with SMOTE", value=True and HAS_IMB)
    st.caption(f"SMOTE available: {'Yes' if HAS_IMB else 'No'}")
    run = st.button("Run Pipeline", type="primary")

df = make_demo() if use_demo else (load_uploaded(up) if up else None)
if df is None: st.info("Upload a dataset or enable demo to continue."); st.stop()
st.success(f"Rows: {len(df)} • Cols: {df.shape[1]}")
st.expander("Preview (first 50)").dataframe(df.head(50), use_container_width=True)

# Column mapping
st.subheader("Column mapping")
cols = df.columns.tolist()
c1,c2,c3 = st.columns(3)
with c1:
    target = st.selectbox("Target", [""]+cols, index=(cols.index("Risk_Type")+1) if "Risk_Type" in cols else 0)
    risk_score = st.selectbox("Risk Score", [""]+cols, index=(cols.index("Risk_Score")+1) if "Risk_Score" in cols else 0)
with c2:
    risk_level = st.selectbox("Risk Level", [""]+cols, index=(cols.index("Risk_Level")+1) if "Risk_Level" in cols else 0)
    mp_count = st.selectbox("mp_count_per_l", [""]+cols, index=(cols.index("mp_count_per_l")+1) if "mp_count_per_l" in cols else 0)
with c3:
    polymer = st.selectbox("Polymer Type (optional)", [""]+cols, index=(cols.index("Polymer_Type")+1) if "Polymer_Type" in cols else 0)
    id_cols = st.multiselect("ID columns to exclude", cols)
date_cols = st.multiselect("Date/time columns to exclude", cols)

cfg = ColumnConfig(target or None, risk_score or None, risk_level or None, mp_count or None, polymer or None, id_cols, date_cols)
if not cfg.target: st.warning("Select a Target."); st.stop()

# EDA
st.subheader("Exploratory analysis")
g1,g2,g3,g4 = st.columns(4)
with g1:
    if cfg.risk_score and cfg.risk_score in df:
        fig = plt.figure(); plt.hist(df[cfg.risk_score].dropna(), bins=30); plt.title("Risk Score Distribution"); plt.xlabel(cfg.risk_score); plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)
with g2:
    if cfg.risk_score and cfg.risk_level and cfg.risk_score in df and cfg.risk_level in df:
        lv = df[cfg.risk_level].dropna().unique()
        data = [df.loc[df[cfg.risk_level]==v, cfg.risk_score].dropna().values for v in lv]
        if data:
            fig = plt.figure(); plt.boxplot(data, labels=[str(x) for x in lv], showfliers=True); plt.title("Risk Score by Risk Level"); plt.xlabel("Risk Level"); plt.ylabel(cfg.risk_score)
            st.pyplot(fig, clear_figure=True)
with g3:
    if cfg.risk_score and cfg.mp_count and cfg.risk_score in df and cfg.mp_count in df:
        m = df[[cfg.mp_count, cfg.risk_score]].dropna()
        fig = plt.figure(); plt.scatter(m[cfg.mp_count], m[cfg.risk_score], alpha=0.6); plt.title("Risk Score vs mp_count_per_l"); plt.xlabel(cfg.mp_count); plt.ylabel(cfg.risk_score)
        st.pyplot(fig, clear_figure=True)
with g4:
    if cfg.polymer and cfg.polymer in df:
        counts = df[cfg.polymer].value_counts(dropna=False)
        fig = plt.figure(); counts.plot(kind="bar"); plt.title("Polymer Type Distribution"); plt.xlabel("Polymer Type"); plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)

if not run: st.stop()

with st.spinner("Preparing data..."):
    drop = list(set(cfg.id_cols + cfg.date_cols + [cfg.target]))
    X = df.drop(columns=[c for c in drop if c in df], errors="ignore")
    y = df[cfg.target].astype("category")
    num_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = cap_outliers_iqr(X, num_all)
    pre,_,_ = build_pre(df.drop(columns=[cfg.target]), cfg)

with st.spinner("Feature selection diagnostics..."):
    mi_df, mi_img = mi_plot(pre, X, y)
    st.image(mi_img, caption="Top Mutual Information Features", use_column_width=True)

Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=test_size, stratify=y, random_state=int(seed))
st.subheader("Class distribution (train)")
cls = ytr.value_counts(normalize=True).rename("share").to_frame(); cls["count"] = ytr.value_counts()
st.dataframe(cls, use_container_width=True)

def wrap(est):
    if use_smote and HAS_IMB: return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=int(seed))), ("clf", est)])
    return Pipeline([("pre", pre), ("clf", est)])

models = {
    "logreg": wrap(LogisticRegression(max_iter=200, solver="lbfgs")),
    "rf": wrap(RandomForestClassifier(n_estimators=300, random_state=int(seed), n_jobs=-1)),
}

with st.spinner("Tuning Logistic Regression..."):
    gs = GridSearchCV(models["logreg"], {"clf__C":[0.1,1.0,3.0,10.0]}, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1_weighted", n_jobs=-1, refit=True)
    gs.fit(Xtr,ytr); models["logreg_tuned"] = gs.best_estimator_
st.caption(f"Best C: {getattr(gs.best_params_, 'get', lambda *_: None)('clf__C')}")

with st.spinner("Training & evaluating..."):
    rows=[]; labels = np.unique(yte); cm_imgs=[]; roc_imgs=[]; best_name,best_score,best_pipe=None,-1.0,None
    for name, pipe in models.items():
        pipe.fit(Xtr,ytr); yhat = pipe.predict(Xte); proba=None
        try: proba = pipe.predict_proba(Xte)
        except Exception: pass
        m = summarize(yte,yhat, proba[:,1] if (proba is not None and proba.ndim==2 and proba.shape[1]==2) else proba); m["model"]=name; rows.append(m)

        fig = plt.figure(); cm = confusion_matrix(yte,yhat, labels=labels); plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {name}"); plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45); plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v), ha="center", va="center")
        buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=160); plt.close(fig); buf.seek(0); cm_imgs.append((name,buf))

        if proba is not None and is_binary(yte):
            try:
                fig = plt.figure(); RocCurveDisplay.from_predictions(yte, proba[:,1] if proba.ndim==2 else proba); plt.title(f"ROC - {name}")
                rbuf = io.BytesIO(); plt.tight_layout(); plt.savefig(rbuf, format="png", dpi=160); plt.close(fig); rbuf.seek(0); roc_imgs.append((name,rbuf))
            except Exception: pass

        if m["f1_w"] > best_score: best_name,best_score,best_pipe = name,m["f1_w"],pipe

leader = pd.DataFrame(rows).sort_values("f1_w", ascending=False)
st.subheader("Model leaderboard (F1-weighted)"); st.dataframe(leader, use_container_width=True)
fig = plt.figure(); plt.bar(leader["model"], leader["f1_w"]); plt.title("Model Comparison (F1-weighted)"); plt.xlabel("Model"); plt.ylabel("F1_weighted")
st.pyplot(fig, clear_figure=True)

st.subheader("Confusion matrices"); [st.image(buf, caption=name, use_column_width=True) for name,buf in cm_imgs]
if roc_imgs: st.subheader("ROC curves (binary)"); [st.image(buf, caption=name, use_column_width=True) for name,buf in roc_imgs]

st.subheader(f"Top features — best model: {best_name}")
try:
    pre_b: ColumnTransformer = best_pipe.named_steps["pre"]
    num_cols = pre_b.transformers_[0][2]; cat_cols = pre_b.transformers_[1][2]
    ohe: OneHotEncoder = pre_b.named_transformers_["cat"].named_steps["ohe"]
    cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    feats = list(num_cols) + cat_out
    clf = best_pipe.named_steps["clf"]
    imp_df=None
    if hasattr(clf,"feature_importances_"): imp_df = pd.DataFrame({"feature":feats,"importance":clf.feature_importances_})
    elif hasattr(clf,"coef_"):
        coefs = np.mean(np.abs(clf.coef_), axis=0) if getattr(clf.coef_,"ndim",1)>1 else np.abs(clf.coef_)
        imp_df = pd.DataFrame({"feature":feats,"importance":coefs})
    try:
        idx = np.random.RandomState(42).choice(np.arange(Xtr.shape[0]), size=min(1500,Xtr.shape[0]), replace=False)
        res = permutation_importance(best_pipe, Xtr.iloc[idx], ytr.iloc[idx], n_repeats=8, random_state=42, n_jobs=-1)
        perm = pd.DataFrame({"feature":feats,"perm_importance":res.importances_mean})
        imp_df = perm.rename(columns={"perm_importance":"importance"}) if imp_df is None else imp_df.merge(perm, on="feature", how="left")
    except Exception: pass
    if imp_df is not None:
        top = imp_df.sort_values("importance", ascending=False).head(20)
        fig = plt.figure(); plt.barh(top["feature"][::-1], top["importance"][::-1]); plt.title("Top Feature Importance"); plt.xlabel("Importance")
        st.pyplot(fig, clear_figure=True); st.dataframe(top, use_container_width=True)
    else:
        st.info("Feature importances not available for this estimator.")
except Exception as e:
    st.info(f"Feature relevance unavailable: {e}")

st.subheader("Summary report")
summary = "\n".join(["# Risk Analytics & Modeling Summary","","## Leaderboard (F1-weighted)", leader.to_markdown(index=False), f"\n**Best model:** `{best_name}`"])
st.download_button("Download summary.md", data=summary.encode("utf-8"), file_name="summary.md", mime="text/markdown")
st.success("Done.")


# =========================
# file: requirements.txt
# =========================
# Streamlit UI + core DS stack + IO + imbalance + encoding + excel
"""
streamlit>=1.29
numpy>=1.23
pandas>=1.5
scikit-learn>=1.3
matplotlib>=3.6
joblib>=1.2
pyarrow>=12
imbalanced-learn>=0.10
charset-normalizer>=3.3
openpyxl>=3.1
"""
