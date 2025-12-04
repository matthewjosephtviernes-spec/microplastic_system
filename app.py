import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from pandas.errors import EmptyDataError, ParserError

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Microplastic Risk Analysis",
    layout="wide",
)

# Columns (adjust if your dataset uses slightly different names)
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
# DATA LOADING
# -------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None, path: str = "Microplastic.csv"):
    """
    Load CSV from uploaded file or local path.
    Tries multiple encodings to avoid UnicodeDecodeError.
    Raises EmptyDataError if the file has no readable content.
    """
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
# PREPROCESSING & FEATURE ENGINEERING
# -------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric columns: coerce to numeric and fill with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            median = s.median()
            df[col] = s.fillna(median)

    # Polymer_Type -> fill with mode
    if "Polymer_Type" in df.columns:
        mode_val = df["Polymer_Type"].mode(dropna=True)
        if len(mode_val) > 0:
            df["Polymer_Type"] = df["Polymer_Type"].fillna(mode_val.iloc[0])

    # Categorical columns: fill with mode
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
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        s_clipped = np.where(s < lower, lower, s)
        s_clipped = np.where(s_clipped > upper, upper, s_clipped)
        df[col] = s_clipped
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
        shift = 0
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
        df[col] = np.log1p(df[col] + shift)
    return df, skewness, skewed_cols


def scale_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    scaler = StandardScaler()
    cols_present = [c for c in cols if c in df.columns]
    if cols_present:
        df[cols_present] = scaler.fit_transform(df[cols_present])
    return df, scaler


def preprocess_for_model(df: pd.DataFrame):
    df = df.copy()
    available_targets = [c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns]
    if len(available_targets) == 2:
        df = df.dropna(subset=available_targets)

    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, NUMERIC_COLS)
    df, skewness, skewed_cols = transform_skewed(df, NUMERIC_COLS)
    df, _ = scale_numeric(df, NUMERIC_COLS)

    y_type = df[TARGET_RISK_TYPE] if TARGET_RISK_TYPE in df.columns else None
    y_level = df[TARGET_RISK_LEVEL] if TARGET_RISK_LEVEL in df.columns else None

    drop_cols = [c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    existing_cat_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns]
    X = pd.get_dummies(feature_df, columns=existing_cat_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df, X, y_type, y_level, skewness, skewed_cols


# -------------------------------------------------------
# SPLIT HELPERS (FIX STRATIFY FAILURES)
# -------------------------------------------------------
def merge_rare_classes(y: pd.Series, min_count: int = 2, other_label: str = "Other"):
    """
    Merge classes with frequency < min_count into 'Other'.
    Useful when stratified splitting fails due to tiny classes.
    """
    y = pd.Series(y).copy()
    counts = y.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    y = y.where(~y.isin(rare), other_label)
    return y


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Attempts stratified split with an adjusted test_size if needed.
    Falls back to non-stratified split only if stratification is impossible.
    Returns: (X_train, X_test, y_train, y_test), used_stratify(bool)
    """
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to train models.")

    counts = y.value_counts()
    min_class = int(counts.min())
    n = len(y)
    k = y.nunique()

    # If a class has only 1 sample total, stratify is impossible
    if min_class < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        return (X_train, X_test, y_train, y_test), False

    # Ensure test has at least 1 item per class: test_size >= k/n
    min_test_size = k / n
    # Ensure train has at least 1 per class: test_size <= 1 - k/n
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
            return (X_train, X_test, y_train, y_test), True
        except ValueError:
            continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    return (X_train, X_test, y_train, y_test), False


# -------------------------------------------------------
# MODELING
# -------------------------------------------------------
def train_models(X, y):
    """
    Train Logistic Regression, Random Forest, Gradient Boosting.
    Returns:
        models      - dict of trained models
        metrics_df  - performance metrics
        split_info  - dict with train/test shapes and class distributions
        split_note  - message about stratification usage
        merge_note  - info about rare-class merging
    """
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to train models.")

    # Merge rare classes to improve stratification feasibility
    before_counts = y.value_counts()
    y_merged = merge_rare_classes(y, min_count=2, other_label="Other")
    after_counts = y_merged.value_counts()

    merge_note = None
    if not before_counts.equals(after_counts):
        merge_note = {
            "before": before_counts,
            "after": after_counts,
        }

    (X_train, X_test, y_train, y_test), used_stratify = safe_train_test_split(
        X, y_merged, test_size=0.2, random_state=42
    )

    split_note = "✅ Stratified split used (class proportions preserved)." if used_stratify else \
        "⚠️ Non-stratified split used (some classes too small for stratification)."

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
    }

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="auto", n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    metrics_list = []
    for name, model in models.items():
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
    return models, metrics_df, split_info, split_note, merge_note


def smote_and_tune_logreg(X, y):
    """
    Risk_Type only:
      - safe split (prefer stratified)
      - SMOTE on train if possible
      - GridSearchCV tuning LogisticRegression
    Returns:
        best_lr, tuned_metrics, best_params, split_info, split_note, merge_note
    """
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to run SMOTE and tuning.")

    before_counts = y.value_counts()
    y_merged = merge_rare_classes(y, min_count=2, other_label="Other")
    after_counts = y_merged.value_counts()

    merge_note = None
    if not before_counts.equals(after_counts):
        merge_note = {"before": before_counts, "after": after_counts}

    (X_train, X_test, y_train, y_test), used_stratify = safe_train_test_split(
        X, y_merged, test_size=0.2, random_state=42
    )

    split_note = "✅ Stratified split used (class proportions preserved)." if used_stratify else \
        "⚠️ Non-stratified split used (some classes too small for stratification)."

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
    }

    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        smote_used = True
    except ValueError:
        st.warning("SMOTE failed due to very small minority classes. Proceeding without SMOTE.")
        X_res, y_res = X_train, y_train
        smote_used = False

    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, multi_class="auto", n_jobs=-1),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=5,
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

    return best_lr, tuned_metrics, grid.best_params_, split_info, split_note, merge_note


# -------------------------------------------------------
# VISUALIZATION HELPERS
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


def plot_bar(value_counts, title, xlabel, ylabel="Count"):
    fig, ax = plt.subplots(figsize=(6, 4))
    value_counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
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
        ax.text(0.5, 0.5, f"No usable data for {value_col} by {category_col}",
                ha="center", va="center")
        plt.tight_layout()
        return fig

    counts = data[category_col].value_counts()
    keep = counts.head(top_n).index
    data[category_col] = np.where(data[category_col].isin(keep), data[category_col], other_label)

    order = (
        data.groupby(category_col)[value_col]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    if horizontal:
        sns.boxplot(data=data, y=category_col, x=value_col, order=order, ax=ax)
        ax.set_xlabel(value_col)
        ax.set_ylabel(category_col)
    else:
        sns.boxplot(data=data, x=category_col, y=value_col, order=order, ax=ax)
        ax.set_xlabel(category_col)
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    ax.set_title(f"{value_col} by {category_col} (Top {top_n} + {other_label})")
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        """
        This app demonstrates the analysis and modeling steps for predicting **Risk_Type**
        and **Risk_Level** from environmental and microplastic parameters.
        """
    )

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
        st.error(
            "⚠️ Unable to decode the file as text.\n\n"
            "Please upload a valid CSV file (not Excel)."
        )
        st.stop()
    except EmptyDataError:
        st.error("⚠️ The file appears to be empty or unreadable as CSV.")
        st.stop()
    except ParserError:
        st.error("⚠️ The file format is not a proper CSV.")
        st.stop()
    except FileNotFoundError:
        df_raw = None

    if df_raw is None:
        st.error("❌ No dataset found. Upload your CSV or place 'Microplastic.csv' beside app.py.")
        st.stop()

    # ---------------------------------------------------
    # PAGE: Data Overview & Task 1
    # ---------------------------------------------------
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

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - This table shows a snapshot of the raw dataset to confirm correct loading and column availability.
                """
            )

        with tab2:
            if "Risk_Score" in df_raw.columns:
                st.subheader("Distribution of Risk_Score (Histogram & Boxplot)")
                st.pyplot(plot_hist_box(df_raw, "Risk_Score"))

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - The histogram shows how Risk_Score values are distributed.
                    - The boxplot summarizes spread and outliers (extreme risk scores).
                    """
                )
            else:
                st.info("Column 'Risk_Score' not found in the dataset.")

        with tab3:
            if "MP_Count_per_L" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Relationship between Risk_Score and MP_Count_per_L")
                st.pyplot(plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score"))

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Each point represents a sample. An upward trend suggests higher MP concentration may increase risk.
                    - If points are scattered, other factors may influence risk beyond MP counts.
                    """
                )
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

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - This compares Risk_Score distributions across Risk_Level categories.
                    - Higher categories should show higher median Risk_Score, while overlaps may indicate borderline thresholds.
                    """
                )
            else:
                st.info("Columns 'Risk_Level' and/or 'Risk_Score' not found.")

    # ---------------------------------------------------
    # PAGE: Preprocessing (Task 2)
    # ---------------------------------------------------
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Preprocessing")
        df_clean, X, y_type, y_level, skewness, skewed_cols = preprocess_for_model(df_raw)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Before Preprocessing",
            "After Preprocessing",
            "Skewness",
            "Encoded Features",
        ])

        with tab1:
            numeric_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
            if numeric_present:
                st.subheader("Descriptive Stats (Raw Numeric Data)")
                st.write(df_raw[numeric_present].describe())
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Baseline stats before cleaning. Large ranges suggest outliers and skewness.
                    """
                )
            else:
                st.info("No numeric columns found for descriptive stats.")

        with tab2:
            numeric_present_clean = [c for c in NUMERIC_COLS if c in df_clean.columns]
            if numeric_present_clean:
                st.subheader("Descriptive Stats (After Cleaning & Scaling)")
                st.write(df_clean[numeric_present_clean].describe())
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - After cleaning, features are more stable and comparable due to scaling.
                    """
                )
            else:
                st.info("No cleaned numeric columns found.")

        with tab3:
            st.subheader("Skewness of Numeric Columns (before transform)")
            st.write(skewness)
            if len(skewed_cols) > 0:
                st.write("Skewed columns transformed (log1p):")
                st.write(skewed_cols)
            else:
                st.write("No columns exceeded skewness threshold.")

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Skewness indicates asymmetry in feature distributions.
                - Transforming skewed features reduces long tails and helps many models.
                """
            )

        with tab4:
            st.subheader("Encoded Feature Matrix (X) – First 10 Rows")
            st.dataframe(X.head(10))
            st.write("Shape of X:", X.shape)
            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Categorical variables are encoded into numeric columns for machine learning.
                """
            )

    # ---------------------------------------------------
    # PAGE: Feature Selection & Relevance (Task 3 & 6)
    # ---------------------------------------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance")
        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        st.markdown(
            """
            A **Random Forest** model estimates feature importance. Higher importance means stronger contribution to prediction.
            """
        )

        tab_rt, tab_rl = st.tabs(["Risk_Type Feature Importance", "Risk_Level Feature Importance"])

        with tab_rt:
            if y_type is not None:
                rf_rt = RandomForestClassifier(n_estimators=200, random_state=42)
                rf_rt.fit(X, y_type)
                importances_rt = pd.Series(rf_rt.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.subheader("Top 10 Feature Importances (Risk_Type)")
                st.dataframe(importances_rt.head(10))
                st.pyplot(plot_bar(importances_rt.head(10), "Top 10 Feature Importances (Risk_Type)", "Features"))

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - The most important features are the strongest predictors for Risk_Type.
                    """
                )
            else:
                st.warning("Risk_Type column not found.")

        with tab_rl:
            if y_level is not None:
                rf_rl = RandomForestClassifier(n_estimators=200, random_state=42)
                rf_rl.fit(X, y_level)
                importances_rl = pd.Series(rf_rl.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.subheader("Top 10 Feature Importances (Risk_Level)")
                st.dataframe(importances_rl.head(10))
                st.pyplot(plot_bar(importances_rl.head(10), "Top 10 Feature Importances (Risk_Level)", "Features"))

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - The most important features are the strongest predictors for Risk_Level.
                    """
                )
            else:
                st.warning("Risk_Level column not found.")

    # ---------------------------------------------------
    # PAGE: Classification Modeling (Tasks 4, 5 & 7)
    # ---------------------------------------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Tasks 4, 5 & 7: Classification Modeling")
        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        tab1, tab2 = st.tabs(["Risk-Type Models", "Risk-Level Models"])

        with tab1:
            if y_type is None:
                st.warning("Risk_Type column not found.")
            else:
                _, metrics_rt, split_info_rt, split_note_rt, merge_note_rt = train_models(X, y_type)
                st.subheader("Performance Metrics – Risk-Type")
                st.dataframe(metrics_rt.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))

                st.info(split_note_rt)

                if merge_note_rt is not None:
                    with st.expander("See class merging details (rare classes → 'Other')"):
                        st.write("Before merging:")
                        st.write(merge_note_rt["before"])
                        st.write("After merging:")
                        st.write(merge_note_rt["after"])

                st.markdown("**Train–Test Split (Risk-Type):**")
                st.markdown(
                    f"""
                    - Training set shape: `{split_info_rt['X_train_shape']}`
                    - Test set shape: `{split_info_rt['X_test_shape']}`
                    """
                )
                st.write("Training class distribution:")
                st.write(split_info_rt["y_train_counts"])
                st.write("Test class distribution:")
                st.write(split_info_rt["y_test_counts"])

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Metrics compare model performance. F1-score is most reliable for imbalanced classes.
                    - If the split is stratified, class proportions are preserved; otherwise, evaluation may be less stable.
                    """
                )

        with tab2:
            if y_level is None:
                st.warning("Risk_Level column not found.")
            else:
                _, metrics_rl, split_info_rl, split_note_rl, merge_note_rl = train_models(X, y_level)
                st.subheader("Performance Metrics – Risk-Level")
                st.dataframe(metrics_rl.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))

                st.info(split_note_rl)

                if merge_note_rl is not None:
                    with st.expander("See class merging details (rare classes → 'Other')"):
                        st.write("Before merging:")
                        st.write(merge_note_rl["before"])
                        st.write("After merging:")
                        st.write(merge_note_rl["after"])

                st.markdown("**Train–Test Split (Risk-Level):**")
                st.markdown(
                    f"""
                    - Training set shape: `{split_info_rl['X_train_shape']}`
                    - Test set shape: `{split_info_rl['X_test_shape']}`
                    """
                )
                st.write("Training class distribution:")
                st.write(split_info_rl["y_train_counts"])
                st.write("Test class distribution:")
                st.write(split_info_rl["y_test_counts"])

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Metrics evaluate the ability to classify Risk_Level categories.
                    - Rare classes may be merged into 'Other' when sample counts are too small for stratification.
                    """
                )

        st.subheader("Overall Interpretation")
        st.markdown(
            """
            - Comparing models helps select the most reliable classifier per target (often highest F1-score).
            - Stratified splits are preferred; if data is too small, merging rare classes improves stability.
            """
        )

    # ---------------------------------------------------
    # PAGE: Polymer Type Distribution
    # ---------------------------------------------------
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution")
        df = handle_missing_values(df_raw)

        if "Polymer_Type" in df.columns:
            st.subheader("Value Counts of Polymer_Type")
            vc = df["Polymer_Type"].value_counts()
            st.dataframe(vc.rename("count"))

            st.subheader("Bar Plot of Polymer_Type Distribution")
            st.pyplot(plot_bar(vc, "Distribution of Polymer_Type", "Polymer_Type"))

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Dominant polymer types likely represent the most common sources of microplastics in the study area.
                """
            )
        else:
            st.warning("Polymer_Type column not found.")

    # ---------------------------------------------------
    # PAGE: SMOTE & Hyperparameter Tuning (Risk_Type)
    # ---------------------------------------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("Address Class Imbalance & Tune Logistic Regression (Risk-Type)")
        _, X, y_type, _, _, _ = preprocess_for_model(df_raw)

        if y_type is None:
            st.warning("Risk_Type column not found.")
            return

        tab1, tab2 = st.tabs(["Original Distribution & Base Models", "SMOTE + Tuning & Comparison"])

        with tab1:
            st.subheader("Class Distribution of Risk-Type (Original)")
            st.write(pd.Series(y_type).value_counts())

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Large differences in counts indicate class imbalance and can bias model performance.
                """
            )

            _, base_metrics, split_info, split_note, merge_note = train_models(X, y_type)

            st.subheader("Base Models Performance")
            st.dataframe(base_metrics.style.format("{:.3f}"))
            st.pyplot(plot_metrics_bar(base_metrics, "(Risk-Type – Base)"))
            st.info(split_note)

            if merge_note is not None:
                with st.expander("See class merging details (rare classes → 'Other')"):
                    st.write("Before merging:")
                    st.write(merge_note["before"])
                    st.write("After merging:")
                    st.write(merge_note["after"])

            st.markdown("**Train–Test Split:**")
            st.markdown(
                f"""
                - Training set shape: `{split_info['X_train_shape']}`
                - Test set shape: `{split_info['X_test_shape']}`
                """
            )

        with tab2:
            st.subheader("SMOTE + Logistic Regression Hyperparameter Tuning")
            best_lr, tuned_metrics, best_params, split_info2, split_note2, merge_note2 = smote_and_tune_logreg(X, y_type)

            st.write("Best Hyperparameters:")
            st.json(best_params)
            st.info(split_note2)

            if merge_note2 is not None:
                with st.expander("See class merging details (rare classes → 'Other')"):
                    st.write("Before merging:")
                    st.write(merge_note2["before"])
                    st.write("After merging:")
                    st.write(merge_note2["after"])

            st.subheader("Tuned Model Performance")
            st.dataframe(tuned_metrics.style.format("{:.3f}"))

            try:
                _, base_metrics_compare, _, _, _ = train_models(X, y_type)
                combined = pd.concat([base_metrics_compare, tuned_metrics])
                st.subheader("Comparison: Base Models vs Tuned Logistic Regression")
                st.dataframe(combined.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(combined, "(Risk-Type – Base vs Tuned)"))
            except ValueError:
                st.warning("Comparison could not be generated due to small class sizes.")


if __name__ == "__main__":
    main()
