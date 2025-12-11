# file: streamlit_app.py
import os
import io
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
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, RocCurveDisplay
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
    from charset_normalizer import from_bytes as detect_encoding
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")

# ---------- always render something first ----------
st.set_page_config(page_title="Risk Analytics System", layout="wide")
st.title("Risk Analytics & Modeling")

try:
    # ---------- dataclasses & small utils ----------
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

    def is_binary(y: pd.Series) -> bool:
        return y.nunique(dropna=True) == 2

    def cap_outliers_iqr(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        X = X.copy()
        for c in numeric_cols:
            s = X[c].dropna()
            if s.empty:
                continue
            q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
            if iqr <= 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            X[c] = X[c].clip(lo, hi)
        return X

    def build_preprocessor(df: pd.DataFrame, cfg: ColumnConfig) -> Tuple[ColumnTransformer, List[str], List[str]]:
        drop_cols = set(cfg.id_cols + cfg.date_cols)
        kept = [c for c in df.columns if c not in drop_cols and c != (cfg.target or "")]
        num_cols = [c for c in kept if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in kept if c not in num_cols]
        num = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson", standardize=False)),  # why: de-skew
            ("scale", RobustScaler(with_centering=True)),                         # why: robust to outliers
        ])
        cat = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
        pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")
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

    # ---------- robust file loader ----------
    @st.cache_data(show_spinner=False)
    def load_uploaded(file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
        name = file.name.lower()

        if name.endswith((".xlsx", ".xls")):
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
            except Exception:
                enc = None
        for e in [enc, "utf-8-sig", "utf-8", "cp1252", "latin-1"]:
            if not e: continue
            try:
                return sanitize_columns(pd.read_csv(io.BytesIO(raw), encoding=e, sep=None, engine="python"))
            except Exception:
                continue
        return sanitize_columns(pd.read_csv(io.BytesIO(raw), encoding="latin-1", sep=None, engine="python", on_bad_lines="skip", errors="replace"))

    # ---------- demo data ----------
    def make_demo(n: int = 1200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        level = rng.choice(["Low","Medium","High"], size=n, p=[0.5,0.35,0.15])
        polymer = rng.choice(["PE","PP","PS","PET","PVC"], size=n)
        mp = np.clip(rng.normal(100, 40, size=n) + (level=="High")*50 + (level=="Medium")*20, 5, None)
        score = np.clip(0.02*mp + rng.normal(0, 1.5, size=n) + (level=="High")*3 + (level=="Medium")*1.2, 0, None)
        logits = -1.0 + 0.15*score + (polymer=="PS")*0.5 + (level=="High")*0.7
        prob = 1/(1+np.exp(-logits))
        risk_type = np.where(rng.uniform(size=n) < prob, "At_Risk", "Safe")
        return sanitize_columns(pd.DataFrame({
            "Risk_Score": score,
            "Risk_Level": level,
            "mp_count_per_l": mp,
            "Polymer_Type": polymer,
            "Risk_Type": pd.Series(risk_type).astype("category"),
        }))

    @st.cache_data(show_spinner=False)
    def mi_plot(pre: ColumnTransformer, X: pd.DataFrame, y: pd.Series):
        Xt = pre.fit_transform(X, y)
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]
        ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
        feat_names = list(num_cols) + cat_out

        y_enc, _ = pd.factorize(y)
        idx = np.random.RandomState(42).choice(np.arange(Xt.shape[0]), size=min(5000, Xt.shape[0]), replace=False)
        mi = mutual_info_classif(Xt[idx], y_enc[idx], random_state=42, discrete_features=[False]*Xt.shape[1])
        mi_df = pd.DataFrame({"feature": feat_names, "mi": mi}).sort_values("mi", ascending=False)

        fig = plt.figure()
        top = mi_df.head(20)
        plt.barh(top["feature"][::-1], top["mi"][::-1])
        plt.title("Top Mutual Information Features")
        plt.xlabel("MI")
        buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=160); plt.close(fig); buf.seek(0)
        return mi_df, buf

    # ---------- sidebar ----------
    with st.sidebar:
        st.header("1) Data")
        up = st.file_uploader("Upload CSV/Parquet/Excel", type=["csv","parquet","xlsx","xls"])
        use_demo = st.checkbox("Or use demo data", value=(up is None))
        st.header("2) Options")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        seed = st.number_input("Random state", 0, 9999, 42, 1)
        use_smote = st.checkbox("Address class imbalance with SMOTE", value=True and HAS_IMB)
        st.caption(f"SMOTE available: {'Yes' if HAS_IMB else 'No'}")
        st.header("3) Run")
        run = st.button("Run Pipeline", type="primary")

    # ---------- load data ----------
    if use_demo:
        df = make_demo()
        st.success(f"Loaded demo • rows: {len(df)} • cols: {df.shape[1]}")
    else:
        if not up:
            st.info("Upload a dataset or enable demo to continue.")
            st.stop()
        df = load_uploaded(up)
        st.success(f"Loaded: {up.name} • rows: {len(df)} • cols: {df.shape[1]}")

    st.expander("Preview (first 50 rows)").dataframe(df.head(50), use_container_width=True)

    # ---------- column mapping ----------
    st.subheader("Column mapping")
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.selectbox("Target (classification)", [""] + cols,
                              index=(cols.index("Risk_Type")+1) if "Risk_Type" in cols else 0)
        risk_score = st.selectbox("Risk Score (numeric)", [""] + cols,
                                  index=(cols.index("Risk_Score")+1) if "Risk_Score" in cols else 0)
    with c2:
        risk_level = st.selectbox("Risk Level (categorical)", [""] + cols,
                                  index=(cols.index("Risk_Level")+1) if "Risk_Level" in cols else 0)
        mp_count = st.selectbox("mp_count_per_l (numeric)", [""] + cols,
                                index=(cols.index("mp_count_per_l")+1) if "mp_count_per_l" in cols else 0)
    with c3:
        polymer = st.selectbox("Polymer Type (optional)", [""] + cols,
                               index=(cols.index("Polymer_Type")+1) if "Polymer_Type" in cols else 0)
        id_cols = st.multiselect("ID columns to exclude", cols)
    date_cols = st.multiselect("Date/time columns to exclude", cols)

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

    # ---------- EDA ----------
    st.subheader("Exploratory analysis")
    g1, g2, g3, g4 = st.columns(4)
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
    with g4:
        if cfg.polymer and cfg.polymer in df:
            counts = df[cfg.polymer].value_counts(dropna=False)
            fig = plt.figure()
            counts.plot(kind="bar")
            plt.title("Polymer Type Distribution"); plt.xlabel("Polymer Type"); plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)

    # ---------- run pipeline ----------
    if not run:
        st.stop()

    with st.spinner("Preparing data..."):
        drop_cols = list(set(cfg.id_cols + cfg.date_cols + [cfg.target]))
        X = df.drop(columns=[c for c in drop_cols if c in df], errors="ignore")
        y = df[cfg.target].astype("category")
        num_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
        X = cap_outliers_iqr(X, num_all)
        pre, _, _ = build_preprocessor(df.drop(columns=[cfg.target]), cfg)

    with st.spinner("Feature selection diagnostics..."):
        # Variance + MI; show MI bar
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

        fig = plt.figure()
        top = mi_df.head(20)
        plt.barh(top["feature"][::-1], top["mi"][::-1])
        plt.title("Top Mutual Information Features"); plt.xlabel("MI")
        st.pyplot(fig, clear_figure=True)
        with st.expander("Low-variance features (diagnostic)"):
            st.write(low_var if low_var else "None")

    # Split + class balance
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=int(seed))
    st.subheader("Class distribution (train)")
    class_dist = ytr.value_counts(normalize=True).rename("share").to_frame()
    class_dist["count"] = ytr.value_counts()
    st.dataframe(class_dist, use_container_width=True)

    # Models
    def wrap(est):
        if use_smote and HAS_IMB:
            return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=int(seed))), ("clf", est)])
        return Pipeline([("pre", pre), ("clf", est)])

    models = {
        "logreg": wrap(LogisticRegression(max_iter=200, solver="lbfgs")),
        "rf": wrap(RandomForestClassifier(n_estimators=300, random_state=int(seed), n_jobs=-1)),
    }

    with st.spinner("Tuning Logistic Regression (C grid)..."):
        gs = GridSearchCV(models["logreg"], {"clf__C": [0.1, 1.0, 3.0, 10.0]},
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="f1_weighted", n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        models["logreg_tuned"] = gs.best_estimator_
    st.caption(f"Best C: {getattr(gs.best_params_, 'get', lambda *_: None)('clf__C')}")

    with st.spinner("Training models and evaluating..."):
        rows = []
        labels = np.unique(yte)
        cm_imgs, roc_imgs = [], []
        best_name, best_score, best_pipe = None, -1.0, None

        for name, pipe in models.items():
            pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xte)
            proba = None
            try:
                proba = pipe.predict_proba(Xte)
            except Exception:
                pass
            met = summarize_metrics(
                yte, yhat,
                proba[:, 1] if (proba is not None and proba.ndim == 2 and proba.shape[1] == 2) else proba
            )
            met["model"] = name
            rows.append(met)

            # CM
            fig = plt.figure()
            cm = confusion_matrix(yte, yhat, labels=labels)
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
            plt.yticks(ticks=np.arange(len(labels)), labels=labels)
            for (i, j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
            buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png", dpi=160); plt.close(fig); buf.seek(0)
            cm_imgs.append((name, buf))

            # ROC
            if proba is not None and is_binary(yte):
                try:
                    fig = plt.figure()
                    RocCurveDisplay.from_predictions(yte, proba[:, 1] if proba.ndim == 2 else proba)
                    plt.title(f"ROC - {name}")
                    rbuf = io.BytesIO(); plt.tight_layout(); plt.savefig(rbuf, format="png", dpi=160); plt.close(fig); rbuf.seek(0)
                    roc_imgs.append((name, rbuf))
                except Exception:
                    pass

            if met["f1_w"] > best_score:
                best_name, best_score, best_pipe = name, met["f1_w"], pipe

    leader = pd.DataFrame(rows).sort_values("f1_w", ascending=False)
    st.subheader("Model leaderboard (F1-weighted)")
    st.dataframe(leader, use_container_width=True)

    fig = plt.figure()
    plt.bar(leader["model"], leader["f1_w"])
    plt.title("Model Comparison (F1-weighted)"); plt.xlabel("Model"); plt.ylabel("F1_weighted")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Confusion matrices")
    for name, buf in cm_imgs:
        st.image(buf, caption=name, use_column_width=True)

    if roc_imgs:
        st.subheader("ROC curves (binary target)")
        for name, buf in roc_imgs:
            st.image(buf, caption=name, use_column_width=True)

    # Feature relevance
    st.subheader(f"Top features — best model: {best_name}")
    try:
        pre_b: ColumnTransformer = best_pipe.named_steps["pre"]
        num_cols = pre_b.transformers_[0][2]
        cat_cols = pre_b.transformers_[1][2]
        ohe: OneHotEncoder = pre_b.named_transformers_["cat"].named_steps["ohe"]
        cat_out = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
        feature_names = list(num_cols) + cat_out

        clf = best_pipe.named_steps["clf"]
        imp_df = None
        if hasattr(clf, "feature_importances_"):
            imp_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        elif hasattr(clf, "coef_"):
            coefs = np.mean(np.abs(clf.coef_), axis=0) if getattr(clf.coef_, "ndim", 1) > 1 else np.abs(clf.coef_)
            imp_df = pd.DataFrame({"feature": feature_names, "importance": coefs})

        try:
            rng = np.random.RandomState(42)
            idx = rng.choice(np.arange(Xtr.shape[0]), size=min(1500, Xtr.shape[0]), replace=False)
            res = permutation_importance(best_pipe, Xtr.iloc[idx], ytr.iloc[idx], n_repeats=8, random_state=42, n_jobs=-1)
            perm_df = pd.DataFrame({"feature": feature_names, "perm_importance": res.importances_mean})
            imp_df = perm_df.rename(columns={"perm_importance": "importance"}) if imp_df is None else imp_df.merge(perm_df, on="feature", how="left")
        except Exception:
            pass

        if imp_df is not None:
            imp_top = imp_df.sort_values("importance", ascending=False).head(20)
            fig = plt.figure()
            plt.barh(imp_top["feature"][::-1], imp_top["importance"][::-1])
            plt.title("Top Feature Importance"); plt.xlabel("Importance")
            st.pyplot(fig, clear_figure=True)
            st.dataframe(imp_top, use_container_width=True)
        else:
            st.info("Feature importances not available for this estimator.")
    except Exception as e:
        st.info(f"Could not compute feature relevance: {e}")

    # Summary download
    st.subheader("Summary report")
    summary_md = "\n".join([
        "# Risk Analytics & Modeling Summary",
        "",
        "## Leaderboard (F1-weighted)",
        leader.to_markdown(index=False),
        f"\n**Best model:** `{best_name}`",
    ])
    st.download_button("Download summary.md", data=summary_md.encode("utf-8"),
                       file_name="summary.md", mime="text/markdown")
    st.success("Done.")

except Exception as e:
    st.error("The app hit an error. Details below:")
    st.exception(e)
    st.stop()
