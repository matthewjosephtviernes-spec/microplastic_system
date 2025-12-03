import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SMOTE = None
    IMBLEARN_AVAILABLE = False
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
except ModuleNotFoundError:
    ImbPipeline = None

# ------------------------------------------------------------
# Page config + styling (keep a clean aesthetic similar to your original)
# ------------------------------------------------------------
st.set_page_config(page_title="Microplastic Risk Dashboard", page_icon="🧪", layout="wide")

CSS = """
<style>
:root{
  --bg: #0f172a;              /* slate-900 */
  --panel: rgba(255,255,255,0.06);
  --panel2: rgba(255,255,255,0.09);
  --border: rgba(255,255,255,0.14);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --accent: #38bdf8;          /* sky-400 */
  --accent2: #a78bfa;         /* violet-400 */
  --good: #4ade80;            /* green-400 */
  --warn: #fbbf24;            /* amber-400 */
  --bad: #fb7185;             /* rose-400 */
  --shadow: 0 10px 30px rgba(0,0,0,0.35);
}

html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 20% 0%, rgba(56,189,248,0.12), transparent 60%),
              radial-gradient(1000px 600px at 80% 10%, rgba(167,139,250,0.14), transparent 55%),
              var(--bg);
  color: var(--text);
}

[data-testid="stSidebar"]{
  background: rgba(2,6,23,0.65);
  border-right: 1px solid var(--border);
}

a { color: var(--accent); }
h1,h2,h3,h4 { color: var(--text); letter-spacing: .2px; }
p,li,span,label,div { color: var(--text); }

.small-muted{ color: var(--muted) !important; }

.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 18px 18px 8px 18px;
  margin: 10px 0 18px 0;
}
.card h3 { margin-top: 0; }

.kpi{
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 14px;
}

hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 18px 0;
}

/* Inputs */
[data-testid="stTextInput"], [data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stNumberInput"]{
  color: var(--text);
}

.stButton > button{
  background: linear-gradient(135deg, rgba(56,189,248,0.9), rgba(167,139,250,0.9));
  color: #0b1220;
  border: 0;
  border-radius: 14px;
  padding: 0.65rem 1rem;
  font-weight: 700;
  box-shadow: 0 10px 25px rgba(56,189,248,0.18);
}
.stButton > button:hover{
  filter: brightness(1.05);
  transform: translateY(-1px);
}

[data-testid="stDataFrame"]{
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}

[data-testid="stMetric"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 12px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def card_open(title=None, subtitle=None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if title:
        st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="smallmuted">{subtitle}</div>', unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        with col:
            st.metric(label, value, help=help_)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "feature_df" not in st.session_state:
    st.session_state.feature_df = None  # X after basic cleaning (no scaling leakage)
if "targets" not in st.session_state:
    st.session_state.targets = {}  # y columns stored here
if "chosen_features" not in st.session_state:
    st.session_state.chosen_features = None  # list[str]
if "models_cache" not in st.session_state:
    st.session_state.models_cache = {}  # store fitted models/pipelines + metrics

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # try case-insensitive match
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: strip column names, normalize whitespace."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def get_feature_target_split(df: pd.DataFrame):
    """Infer targets and features based on common column names. You can adjust if needed."""
    risk_type_col = detect_column(df, ["Risk_Type", "RiskType", "risk_type", "risktype"])
    risk_level_col = detect_column(df, ["Risk_Level", "RiskLevel", "risk_level", "risklevel"])
    risk_score_col = detect_column(df, ["Risk_Score", "RiskScore", "risk_score", "riskscore"])
    polymer_col = detect_column(df, ["Polymer_Type", "PolymerType", "polymer_type", "polymertype"])

    # Default: treat non-target categorical as features too
    target_cols = [c for c in [risk_type_col, risk_level_col] if c is not None]
    feature_df = df.drop(columns=target_cols, errors="ignore").copy()

    targets = {}
    if risk_type_col:
        targets["Risk_Type"] = df[risk_type_col].astype(str)
    if risk_level_col:
        targets["Risk_Level"] = df[risk_level_col].astype(str)

    aux = {
        "risk_type_col": risk_type_col,
        "risk_level_col": risk_level_col,
        "risk_score_col": risk_score_col,
        "polymer_col": polymer_col,
    }
    return feature_df, targets, aux

def build_preprocessor(X: pd.DataFrame):
    """Construct a leakage-safe preprocessing pipeline: impute + encode + scale numeric."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Numeric pipeline: impute -> scale
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: impute -> one-hot
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols

def clip_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str], k: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        out[col] = out[col].clip(lo, hi)
    return out

def transform_skew_log1p(df: pd.DataFrame, numeric_cols: list[str], skew_threshold: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        s = out[col].dropna()
        if s.empty:
            continue
        skew = s.skew()
        if np.isfinite(skew) and abs(skew) >= skew_threshold:
            min_val = out[col].min()
            if min_val <= -1:
                # shift to be >= -0.999 then log1p
                shift = (-min_val) + 1.0
                out[col] = np.log1p(out[col] + shift)
            else:
                out[col] = np.log1p(out[col])
    return out

def classification_report_dict(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"Accuracy": acc, "Precision (weighted)": p, "Recall (weighted)": r, "F1 (weighted)": f1}

def plot_hist(series: pd.Series, title: str, bins: int = 30):
    fig, ax = plt.subplots()
    ax.hist(series.dropna().astype(float), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name if series.name else "")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

def plot_bar_counts(series: pd.Series, title: str, top_n: int = 30):
    vc = series.astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots()
    ax.bar(vc.index, vc.values)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=45)
    st.pyplot(fig)
    plt.close(fig)

def plot_box_by_group(df: pd.DataFrame, score_col: str, group_col: str, title: str):
    # simple matplotlib box plot grouped
    groups = [g for g in df[group_col].dropna().unique()]
    data = [df.loc[df[group_col] == g, score_col].dropna().astype(float).values for g in groups]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=[str(g) for g in groups], showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(score_col)
    ax.tick_params(axis="x", labelrotation=45)
    st.pyplot(fig)
    plt.close(fig)

def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots()
    ax.scatter(df[x].astype(float), df[y].astype(float), alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)
    plt.close(fig)

def get_feature_names(preprocessor: ColumnTransformer, X_cols: list[str]):
    # After fitting, we can retrieve feature names for permutation importance labeling
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array(X_cols, dtype=object)

# ------------------------------------------------------------
# Sidebar navigation (correct sequence)
# ------------------------------------------------------------
st.sidebar.markdown("## 🧪 Microplastic Risk Dashboard")
tabs = [
    "Overview / About",
    "1) Data Upload & Description",
    "2) Exploratory Data Analysis (EDA)",
    "3) Data Preprocessing",
    "4) Feature Selection",
    "5) Modeling (Objective #1)",
    "6) Risk_Type Modeling (Objective #2)",
    "7) Hyperparameter Tuning & Best Model",
    "8) Feature Relevance & Summary",
]
selected_tab = st.sidebar.radio("Navigate", tabs)

st.sidebar.markdown("---")
st.sidebar.markdown('<span class="pill">Methodology order applied</span>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 0) Overview
# ------------------------------------------------------------
if selected_tab == tabs[0]:
    card_open("Overview / About the Study")
    st.markdown(
        """
This dashboard follows a **properly sequenced methodology** for your thesis:

1. **EDA** (understand distributions + relationships)  
2. **Preprocessing** (outliers → skew transforms → encoding → scaling)  
3. **Feature Selection** (identify useful predictors)  
4. **Modeling** (train/evaluate/compare models)  
5. **Objective #2** Risk_Type modeling (class imbalance → SMOTE → tuning)  
6. **Interpretability** (feature relevance)  

Use the sidebar to proceed step-by-step.
        """
    )
    card_close()

# ------------------------------------------------------------
# 1) Data Upload & Description
# ------------------------------------------------------------
elif selected_tab == tabs[1]:
    card_open("1) Data Upload & Description", "Upload your dataset and preview columns & basic info.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        df = basic_clean(df)

        # basic numeric coercion for common numeric-ish columns if present
        for col in df.columns:
            if any(k in col.lower() for k in ["lat", "lon", "longitude", "latitude", "density", "size", "count", "score", "mp_"]):
                df[col] = safe_numeric_series(df[col]) if df[col].dtype == object else df[col]

        st.session_state.raw_df = df
        X, targets, aux = get_feature_target_split(df)
        st.session_state.feature_df = X
        st.session_state.targets = targets
        st.session_state.models_cache = {}
        st.session_state.chosen_features = None

        st.success("Dataset loaded.")
        metric_row([
            ("Rows", f"{df.shape[0]:,}", "Number of records"),
            ("Columns", f"{df.shape[1]:,}", "Number of fields"),
            ("Targets detected", ", ".join(list(targets.keys())) or "None", "Detected target columns"),
        ])

        st.markdown("#### Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("#### Columns")
        st.write(list(df.columns))

        st.markdown("#### Missing values (top 15)")
        na = df.isna().sum().sort_values(ascending=False).head(15)
        st.dataframe(na.rename("Missing"), use_container_width=True)
    else:
        st.info("Upload a CSV to begin.")
    card_close()

# ------------------------------------------------------------
# 2) EDA
# ------------------------------------------------------------
elif selected_tab == tabs[2]:
    card_open("2) Exploratory Data Analysis (EDA)", "Distribution + relationships before preprocessing/modeling.")
    df = st.session_state.raw_df
    if df is None:
        st.warning("Please upload a dataset in Step 1.")
        card_close()
        st.stop()

    _, _, aux = get_feature_target_split(df)
    risk_score_col = aux["risk_score_col"]
    risk_level_col = aux["risk_level_col"]
    polymer_col = aux["polymer_col"]

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("#### General EDA")
        if polymer_col:
            st.markdown("**Polymer Type distribution**")
            plot_bar_counts(df[polymer_col], f"Distribution of {polymer_col}")
        else:
            st.info("Polymer type column not found (expected something like 'Polymer_Type').")

    with right:
        st.markdown("#### Risk score EDA")
        if risk_score_col:
            rs = safe_numeric_series(df[risk_score_col])
            plot_hist(rs.rename(risk_score_col), f"Distribution of {risk_score_col}", bins=30)

            if risk_level_col:
                st.markdown("**Risk score by Risk level**")
                tmp = df.copy()
                tmp[risk_score_col] = safe_numeric_series(tmp[risk_score_col])
                plot_box_by_group(tmp, risk_score_col, risk_level_col, f"{risk_score_col} by {risk_level_col}")
            else:
                st.info("Risk level column not found (expected something like 'Risk_Level').")
        else:
            st.warning("Risk score column not found (expected something like 'Risk_Score').")

    st.divider()

    # Relationship: risk score vs mp count per L (try best-effort detect)
    mp_col = detect_column(df, ["MP_Count_per_L", "mp_count_per_l", "MP_Count", "mp_count", "Microplastic_Count"])
    if risk_score_col and mp_col:
        tmp = df.copy()
        tmp[risk_score_col] = safe_numeric_series(tmp[risk_score_col])
        tmp[mp_col] = safe_numeric_series(tmp[mp_col])
        st.markdown(f"#### Relationship: {risk_score_col} vs {mp_col}")
        plot_scatter(tmp.dropna(subset=[risk_score_col, mp_col]), mp_col, risk_score_col, f"{risk_score_col} vs {mp_col}")
    else:
        st.info("MP count column not found for risk score relationship (expected something like 'MP_Count_per_L').")

    card_close()

# ------------------------------------------------------------
# 3) Data Preprocessing
# ------------------------------------------------------------
elif selected_tab == tabs[3]:
    card_open("3) Data Preprocessing", "Outliers → skew transform → encode categoricals → feature scaling (leakage-safe).")
    df = st.session_state.raw_df
    X = st.session_state.feature_df
    if df is None or X is None:
        st.warning("Please upload a dataset in Step 1.")
        card_close()
        st.stop()

    # Choose numeric cols to transform/clip at the raw feature stage
    X_work = X.copy()
    numeric_cols = X_work.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # attempt numeric coercion for object columns that are numeric-ish
        for col in X_work.columns:
            if X_work[col].dtype == object:
                coerced = safe_numeric_series(X_work[col])
                if coerced.notna().mean() > 0.8:  # mostly numeric
                    X_work[col] = coerced
        numeric_cols = X_work.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("#### Configure preprocessing")
    c1, c2, c3 = st.columns(3)
    with c1:
        do_outliers = st.checkbox("Address outliers (IQR clipping)", value=True)
        iqr_k = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
    with c2:
        do_skew = st.checkbox("Transform skewed numerical columns (log1p)", value=True)
        skew_thr = st.slider("Skew threshold", 0.5, 3.0, 1.0, 0.1)
    with c3:
        st.info("Encoding + scaling are applied **inside model pipelines** to prevent data leakage.")

    if st.button("Run preprocessing preview"):
        X_proc = X_work.copy()
        if do_outliers and numeric_cols:
            X_proc = clip_outliers_iqr(X_proc, numeric_cols, k=iqr_k)
        if do_skew and numeric_cols:
            X_proc = transform_skew_log1p(X_proc, numeric_cols, skew_threshold=skew_thr)

        st.session_state.feature_df = X_proc  # store cleaned features (still not encoded/scaled globally)
        st.success("Preprocessing preview stored (outliers/skew applied). Encoding/scaling will happen in modeling pipelines.")

        st.markdown("#### Preview (features only)")
        st.dataframe(X_proc.head(20), use_container_width=True)

        if numeric_cols:
            st.markdown("#### Numeric summary")
            st.dataframe(X_proc[numeric_cols].describe().T, use_container_width=True)

    card_close()

# ------------------------------------------------------------
# 4) Feature Selection
# ------------------------------------------------------------
elif selected_tab == tabs[4]:
    card_open("4) Feature Selection", "Explore methods and apply a practical selection strategy.")
    df = st.session_state.raw_df
    X = st.session_state.feature_df
    targets = st.session_state.targets
    if df is None or X is None or not targets:
        st.warning("Upload data first (Step 1) and ensure at least one target exists (Risk_Type or Risk_Level).")
        card_close()
        st.stop()

    st.markdown("#### Understand the goal")
    target_choice = st.selectbox("Choose target for feature selection", list(targets.keys()))
    y = targets[target_choice]

    st.markdown("#### Explore feature selection methods")
    method = st.radio(
        "Method",
        ["Filter (correlation with numeric risk score)", "Embedded (RandomForest importance)", "No selection (use all)"],
        horizontal=False,
    )

    # Work with a train split to avoid peeking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    if method == "No selection (use all)":
        chosen = X.columns.tolist()
        st.session_state.chosen_features = chosen
        st.success(f"Selected {len(chosen)} features (all).")
        st.dataframe(pd.DataFrame({"Selected Features": chosen}), use_container_width=True)

    elif method == "Filter (correlation with numeric risk score)":
        # Only possible if risk score exists and is numeric-ish
        _, _, aux = get_feature_target_split(df)
        score_col = aux["risk_score_col"]
        if not score_col:
            st.error("Cannot run correlation filter: Risk_Score not found.")
        else:
            score = safe_numeric_series(df[score_col])
            # numeric feature correlations:
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.error("No numeric features available for correlation filter.")
            else:
                corr = {}
                for c in num_cols:
                    corr[c] = pd.concat([X[c], score], axis=1).corr(numeric_only=True).iloc[0, 1]
                corr_s = pd.Series(corr).dropna().abs().sort_values(ascending=False)
                top_k = st.slider("Select top-K correlated numeric features", 5, min(50, len(corr_s)), min(15, len(corr_s)))
                chosen = corr_s.head(top_k).index.tolist()
                st.session_state.chosen_features = chosen
                st.success(f"Selected {len(chosen)} numeric features using correlation(|r|) with {score_col}.")
                st.dataframe(corr_s.head(top_k).rename("abs(corr)").to_frame(), use_container_width=True)

    else:
        st.markdown("Embedded selection trains a light RandomForest model and uses feature importance.")
        top_k = st.slider("Select top-K important features (after encoding)", 10, 80, 25)

        # Build a pipeline for RF importance with proper preprocessing
        preprocessor, _, _ = build_preprocessor(X_train)
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        pipe = Pipeline(steps=[("prep", preprocessor), ("rf", rf)])
        pipe.fit(X_train, y_train)

        # Extract feature names + importance
        feat_names = get_feature_names(pipe.named_steps["prep"], X_train.columns.tolist())
        importances = pipe.named_steps["rf"].feature_importances_
        imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(top_k)
        # Convert back to original columns notion is hard because of one-hot. We'll select by one-hot names.
        # For modeling later we can keep "all" and just report relevance, OR we can pick original columns via grouping.
        # Here: simple grouping by original feature prefix before '_' from OneHotEncoder output.
        grouped = {}
        for name, val in imp.items():
            base = name.split("_")[0]  # heuristic
            grouped[base] = grouped.get(base, 0.0) + float(val)
        grouped_s = pd.Series(grouped).sort_values(ascending=False)
        chosen = [c for c in X.columns if c in grouped_s.head(min(len(grouped_s), int(top_k/2)+1)).index]
        if not chosen:
            chosen = X.columns.tolist()

        st.session_state.chosen_features = chosen
        st.success(f"Selected {len(chosen)} original features (grouped from one-hot importances).")
        st.markdown("Top one-hot features:")
        st.dataframe(imp.rename("importance").to_frame(), use_container_width=True)
        st.markdown("Grouped importance (approx by original feature name):")
        st.dataframe(grouped_s.head(30).rename("grouped_importance").to_frame(), use_container_width=True)

    card_close()

# ------------------------------------------------------------
# 5) Modeling (Objective #1) - generic classification modeling
# ------------------------------------------------------------
elif selected_tab == tabs[5]:
    card_open("5) Modeling (Objective #1)", "Prepare → Choose models → Train → Evaluate → Compare performance.")
    df = st.session_state.raw_df
    X_all = st.session_state.feature_df
    targets = st.session_state.targets
    if df is None or X_all is None or not targets:
        st.warning("Upload data first (Step 1) and ensure targets exist.")
        card_close()
        st.stop()

    target_choice = st.selectbox("Choose target for Objective #1 modeling", list(targets.keys()))
    y_all = targets[target_choice]

    # Apply feature selection if set
    chosen = st.session_state.chosen_features
    if chosen:
        X_all = X_all[[c for c in chosen if c in X_all.columns]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )

    st.markdown("#### Choose classification models")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    picked = st.multiselect("Models to run", list(models.keys()), default=list(models.keys()))

    if st.button("Train & Evaluate (Objective #1)"):
        results = []
        fitted = {}

        preprocessor, _, _ = build_preprocessor(X_train)

        for name in picked:
            clf = models[name]
            pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            m = classification_report_dict(y_test, pred)
            m["Model"] = name
            results.append(m)
            fitted[name] = pipe

        res_df = pd.DataFrame(results).set_index("Model").sort_values("F1 (weighted)", ascending=False)
        st.session_state.models_cache["objective1"] = {"target": target_choice, "results": res_df, "models": fitted}

        st.success("Done.")
        st.dataframe(res_df, use_container_width=True)

        st.markdown("#### Best model (by weighted F1)")
        best_name = res_df.index[0]
        st.write(f"**{best_name}**")
        cm = confusion_matrix(y_test, fitted[best_name].predict(X_test), labels=np.unique(y_all))
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(cm, index=np.unique(y_all), columns=np.unique(y_all)), use_container_width=True)

    card_close()

# ------------------------------------------------------------
# 6) Risk_Type Modeling (Objective #2)
# ------------------------------------------------------------
elif selected_tab == tabs[6]:
    card_open("6) Risk_Type Modeling (Objective #2)", "Class distribution → SMOTE → Train/Evaluate/Compare + visuals.")
    df = st.session_state.raw_df
    X_all = st.session_state.feature_df
    targets = st.session_state.targets
    if df is None or X_all is None or "Risk_Type" not in targets:
        st.warning("This step requires a Risk_Type column. Upload a dataset that includes Risk_Type.")
        card_close()
        st.stop()

    y_all = targets["Risk_Type"]

    chosen = st.session_state.chosen_features
    if chosen:
        X_all = X_all[[c for c in chosen if c in X_all.columns]].copy()

    st.markdown("#### Check class distribution for Risk_Type")
    plot_bar_counts(y_all, "Risk_Type class distribution", top_n=50)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )

    st.markdown("#### Address Class Imbalance with SMOTE")
    use_smote = st.checkbox("Enable SMOTE", value=True)
    if use_smote and not IMBLEARN_AVAILABLE:
        st.warning("SMOTE requested but 'imbalanced-learn' is not installed. SMOTE will be skipped.")
        use_smote = False

    smote_k = st.slider("SMOTE k_neighbors", 2, 10, 5)

    st.markdown("#### Choose classification models")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    picked = st.multiselect("Models to run", list(models.keys()), default=list(models.keys()))

    if st.button("Train & Evaluate (Objective #2 - Risk_Type)"):
        results = []
        fitted = {}
        preprocessor, _, _ = build_preprocessor(X_train)

        for name in picked:
            clf = models[name]
            if use_smote:
                pipe = ImbPipeline(steps=[
                    ("prep", preprocessor),
                    ("smote", SMOTE(random_state=42, k_neighbors=smote_k)),
                    ("clf", clf),
                ])
            else:
                pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            m = classification_report_dict(y_test, pred)
            m["Model"] = name
            results.append(m)
            fitted[name] = pipe

        res_df = pd.DataFrame(results).set_index("Model").sort_values("F1 (weighted)", ascending=False)

        st.session_state.models_cache["objective2"] = {
            "use_smote": use_smote,
            "results": res_df,
            "models": fitted,
        }

        st.success("Done.")
        st.dataframe(res_df, use_container_width=True)

        st.markdown("#### Visualize model performance (weighted F1)")
        fig, ax = plt.subplots()
        ax.bar(res_df.index, res_df["F1 (weighted)"].values)
        ax.set_title("Model comparison (Risk_Type) - Weighted F1")
        ax.set_ylabel("F1 (weighted)")
        ax.tick_params(axis="x", labelrotation=30)
        st.pyplot(fig)
        plt.close(fig)

    card_close()

# ------------------------------------------------------------
# 7) Hyperparameter Tuning & Best Model
# ------------------------------------------------------------
elif selected_tab == tabs[7]:
    card_open("7) Hyperparameter Tuning & Best Model", "Tune Logistic Regression, then compare against other models.")
    df = st.session_state.raw_df
    X_all = st.session_state.feature_df
    targets = st.session_state.targets
    if df is None or X_all is None or "Risk_Type" not in targets:
        st.warning("This step requires Risk_Type and uploaded data.")
        card_close()
        st.stop()

    y_all = targets["Risk_Type"]
    chosen = st.session_state.chosen_features
    if chosen:
        X_all = X_all[[c for c in chosen if c in X_all.columns]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )

    st.markdown("#### Check Imports for Hyperparameter Tuning")
    st.code("from sklearn.model_selection import GridSearchCV\nfrom imblearn.over_sampling import SMOTE", language="python")

    st.markdown("#### Tuning settings")
    use_smote = st.checkbox("Use SMOTE during tuning", value=True)
    smote_k = st.slider("SMOTE k_neighbors (tuning)", 2, 10, 5)
    cv_folds = st.slider("CV folds", 3, 10, 5)

    preprocessor, _, _ = build_preprocessor(X_train)

    # Pipeline for tuned LR
    base_lr = LogisticRegression(max_iter=5000)
    if use_smote:
        pipe = ImbPipeline(steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=smote_k)),
            ("clf", base_lr),
        ])
        param_grid = {
            "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "saga"],
        }
    else:
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", base_lr)])
        param_grid = {
            "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "saga"],
        }

    if st.button("Run hyperparameter tuning (Logistic Regression)"):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        pred = best.predict(X_test)
        tuned_metrics = classification_report_dict(y_test, pred)

        st.success("Tuning complete.")
        st.markdown("#### Best params")
        st.write(grid.best_params_)
        st.markdown("#### Tuned Logistic Regression performance (test set)")
        st.dataframe(pd.DataFrame([tuned_metrics]).T.rename(columns={0: "Value"}), use_container_width=True)

        # Compare with other baseline models quickly
        others = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        comp_rows = [{"Model": "Tuned Logistic Regression", **tuned_metrics}]
        fitted = {"Tuned Logistic Regression": best}

        for name, clf in others.items():
            if use_smote:
                p2 = ImbPipeline(steps=[
                    ("prep", preprocessor),
                    ("smote", SMOTE(random_state=42, k_neighbors=smote_k)),
                    ("clf", clf),
                ])
            else:
                p2 = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
            p2.fit(X_train, y_train)
            pred2 = p2.predict(X_test)
            m2 = classification_report_dict(y_test, pred2)
            comp_rows.append({"Model": name, **m2})
            fitted[name] = p2

        comp = pd.DataFrame(comp_rows).set_index("Model").sort_values("F1 (weighted)", ascending=False)
        st.markdown("#### Compare Performance of Tuned Logistic Regression with Other Models")
        st.dataframe(comp, use_container_width=True)

        best_name = comp.index[0]
        st.markdown("#### Evaluate the best model")
        st.write(f"Best by weighted F1: **{best_name}**")
        st.session_state.models_cache["best_model"] = {"name": best_name, "model": fitted[best_name], "comparison": comp}

    card_close()

# ------------------------------------------------------------
# 8) Feature Relevance & Summary
# ------------------------------------------------------------
elif selected_tab == tabs[8]:
    card_open("8) Feature Relevance & Summary", "Extract → Analyze → Visualize feature relevance, then summarize findings.")
    cache = st.session_state.models_cache.get("best_model")
    df = st.session_state.raw_df
    X_all = st.session_state.feature_df
    targets = st.session_state.targets

    if df is None or X_all is None or "Risk_Type" not in targets:
        st.warning("Upload data with Risk_Type first.")
        card_close()
        st.stop()

    if not cache:
        st.info("Run Step 7 first to select and store a best model.")
        card_close()
        st.stop()

    model = cache["model"]
    model_name = cache["name"]
    st.markdown(f"**Best model stored:** `{model_name}`")

    # Recreate train-test consistently for permutation importance
    y_all = targets["Risk_Type"]
    chosen = st.session_state.chosen_features
    if chosen:
        X_all = X_all[[c for c in chosen if c in X_all.columns]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )
    model.fit(X_train, y_train)
    baseline_pred = model.predict(X_test)
    base = classification_report_dict(y_test, baseline_pred)

    st.markdown("#### Baseline performance (for context)")
    st.dataframe(pd.DataFrame([base]).T.rename(columns={0: "Value"}), use_container_width=True)

    st.markdown("#### Extract feature relevance (Permutation Importance)")
    if st.button("Compute permutation importance"):
        try:
            # Try to get encoded feature names from pipeline
            prep = model.named_steps.get("prep", None)
            if prep is None:
                st.error("Could not find preprocessing step in the stored model pipeline.")
                card_close()
                st.stop()

            feat_names = prep.get_feature_names_out()
            # Need transformed X_test for permutation importance if model expects transformed input.
            # But permutation_importance works on estimator with raw X if pipeline is passed (it permutes columns of raw X).
            # We pass the whole pipeline to permutation_importance, so feature names correspond to raw columns, not one-hot.
            # We'll therefore compute raw-column permutation importance:
            pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_weighted")
            imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)

            st.markdown("#### Top features (raw columns)")
            st.dataframe(imp.head(30).rename("importance").to_frame(), use_container_width=True)

            fig, ax = plt.subplots()
            top = imp.head(15)
            ax.bar(top.index.astype(str), top.values)
            ax.set_title("Top 15 Feature Relevance (Permutation Importance)")
            ax.set_ylabel("Mean importance (Δ f1_weighted)")
            ax.tick_params(axis="x", labelrotation=45)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### Summarize findings")
            st.markdown(
                f"""
- The model used is **{model_name}** and was evaluated using **weighted F1**.
- The chart above highlights which **input features most influence Risk_Type prediction**.
- Use these results in your thesis under **Model Interpretation / Feature Relevance**.

> Tip: If reviewers ask why you used permutation importance: it is model-agnostic and works even when features are encoded/scaled in pipelines.
                """
            )
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")

    card_close()

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown('<div class="smallmuted">Built for thesis workflow: EDA → Preprocess → Feature Select → Model → Tune → Interpret.</div>', unsafe_allow_html=True)
