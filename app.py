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
    """
    Returns:
        df_clean        - after missing, outliers, skew transform, scaling
        X               - feature matrix (encoded, numeric)
        y_type          - Risk_Type labels (or None)
        y_level         - Risk_Level labels (or None)
        skewness        - skewness of numeric columns used
        skewed_cols     - list of skewed numeric columns
    """
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
# MODELING
# -------------------------------------------------------
def train_models(X, y):
    """
    Train Logistic Regression, Random Forest, Gradient Boosting.
    Returns:
        models      - dict of trained models
        metrics_df  - performance metrics
        split_info  - dict with train/test shapes and class distributions
    """
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to train models.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        st.warning(
            "Stratified train-test split failed (likely due to very small class sizes). "
            "Using a non-stratified split instead."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=None,
        )

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
    }

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
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
    return models, metrics_df, split_info


def smote_and_tune_logreg(X, y):
    """
    For Risk_Type only:
      - apply SMOTE (if possible)
      - tune LogisticRegression with GridSearchCV
    Returns:
        best_lr        - tuned LogisticRegression model
        tuned_metrics  - performance metrics
        best_params    - best hyperparameters
        split_info     - train/test shapes and distributions (before SMOTE)
    """
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to run SMOTE and tuning.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    except ValueError:
        st.warning(
            "Stratified train-test split for SMOTE/tuning failed "
            "(likely due to very small class sizes). Using non-stratified split."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=None
        )

    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
    }

    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except ValueError:
        st.warning(
            "SMOTE failed due to very small or highly imbalanced classes. "
            "Continuing without SMOTE for logistic regression tuning."
        )
        X_res, y_res = X_train, y_train

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }

    base_lr = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_res, y_res)
    best_lr = grid.best_estimator_

    y_pred = best_lr.predict(X_test)

    tuned_metrics = pd.DataFrame(
        [{
            "Model": "LogReg (tuned + SMOTE)",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }]
    ).set_index("Model")

    return best_lr, tuned_metrics, grid.best_params_, split_info


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


def plot_box_by_category(df, value_col, category_col):
    val = pd.to_numeric(df[value_col], errors="coerce")
    cat = df[category_col]
    data = pd.DataFrame({value_col: val, category_col: cat}).dropna(subset=[value_col])

    fig, ax = plt.subplots(figsize=(6, 4))

    if data.empty:
        ax.text(0.5, 0.5, f"No numeric data for {value_col}",
                ha="center", va="center")
    else:
        sns.boxplot(data=data, x=category_col, y=value_col, ax=ax)
        ax.set_title(f"{value_col} by {category_col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


# ✅ NEW: readable category boxplot (fixes your Risk_Level label mess)
def plot_box_by_category_readable(
    df,
    value_col,
    category_col,
    top_n=8,
    other_label="Other",
    figsize=(12, 5),
    horizontal=True,  # <- best readability for many categories
):
    val = pd.to_numeric(df[value_col], errors="coerce")

    # Normalize category text: strip spaces, drop empty strings
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

    # Keep top N by frequency; others -> "Other"
    counts = data[category_col].value_counts()
    keep = counts.head(top_n).index
    data[category_col] = np.where(data[category_col].isin(keep), data[category_col], other_label)

    # Order categories by median value for clearer story
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

    # Sidebar navigation
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

    # -------------------------------
    # Load dataset (upload or local)
    # -------------------------------
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
            "Please make sure you uploaded a valid **CSV text file** (not Excel). "
            "If needed, re-export it from Excel/Sheets as CSV."
        )
        st.stop()
    except EmptyDataError:
        st.error(
            "⚠️ The file appears to be **empty or has no readable CSV data**.\n\n"
            "Please ensure it has a header row and data rows, then re-export as CSV."
        )
        st.stop()
    except ParserError:
        st.error(
            "⚠️ The file is not a proper CSV. "
            "Please re-export it from Excel/Sheets using 'Save As → CSV (Comma delimited)'."
        )
        st.stop()
    except FileNotFoundError:
        df_raw = None

    if df_raw is None:
        st.error(
            "❌ No dataset found.\n\n"
            "Either:\n"
            "- Upload your `Microplastic.csv` file using the sidebar, **or**\n"
            "- Place a file named `Microplastic.csv` beside `app.py`."
        )
        st.stop()

    # ---------------------------------------------------
    # PAGE: Data Overview & Task 1 (TABS)
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
                - This table shows a sample of the raw data to confirm the dataset was loaded correctly.
                - It helps verify that key variables (e.g., Risk_Score, MP_Count_per_L) are present before analysis.
                """
            )

        with tab2:
            if "Risk_Score" in df_raw.columns:
                st.subheader("Distribution of Risk_Score (Histogram & Boxplot)")
                st.pyplot(plot_hist_box(df_raw, "Risk_Score"))
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - The histogram shows how frequently different Risk_Score values occur.
                    - The boxplot summarizes the spread and highlights outliers (unusually high/low risk scores).
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
                    - Each point represents one sampling location.
                    - If a clear upward trend exists, higher microplastic concentration tends to correspond to higher risk.
                    - If the pattern is scattered, other factors likely influence risk beyond microplastic counts alone.
                    """
                )
            else:
                st.info("Columns 'MP_Count_per_L' and/or 'Risk_Score' not found.")

        with tab4:
            if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Difference in Risk_Score by Risk_Level (Boxplot)")

                # ✅ Fixed: readable version
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
                    - This plot compares Risk_Score across Risk_Level categories.
                    - Categories with higher median Risk_Score indicate generally higher risk.
                    - Overlap between categories suggests borderline cases or thresholds that may need refinement.
                    """
                )
            else:
                st.info("Columns 'Risk_Level' and/or 'Risk_Score' not found.")

    # ---------------------------------------------------
    # PAGE: Preprocessing (Task 2) – TABS
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
                st.subheader("Before Preprocessing – Descriptive Stats (Numeric Columns)")
                st.write(df_raw[numeric_present].describe())
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - These are baseline statistics for numeric variables before cleaning.
                    - Large ranges and extreme max values usually indicate outliers and skewness.
                    """
                )
            else:
                st.info("No numeric columns from NUMERIC_COLS were found in the dataset.")

        with tab2:
            numeric_present_clean = [c for c in NUMERIC_COLS if c in df_clean.columns]
            if numeric_present_clean:
                st.subheader("After Outlier Handling, Skew Transform, and Scaling – Descriptive Stats")
                st.write(df_clean[numeric_present_clean].describe())
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - After preprocessing, the numeric variables have more stable ranges and reduced influence from extreme values.
                    - Scaling makes features comparable, which helps many models learn more effectively.
                    """
                )
            else:
                st.info("No cleaned numeric columns found for display.")

        with tab3:
            st.subheader("Skewness of Numeric Columns (before transform)")
            st.write(skewness)

            if len(skewed_cols) > 0:
                st.write("Columns treated as skewed and transformed (log1p):")
                st.write(skewed_cols)
            else:
                st.write("No numeric columns exceeded the skewness threshold; no log transform applied.")

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Skewness indicates asymmetry in the distribution. High skew can influence model training.
                - Skewed features were log-transformed to reduce long tails and improve stability.
                """
            )

        with tab4:
            st.subheader("Encoded Feature Matrix (X) – First 10 Rows")
            st.dataframe(X.head(10))
            st.write("Shape of X:", X.shape)

            if y_type is not None:
                st.write(f"Samples (y_type): {len(y_type)}, classes: {list(pd.Series(y_type).unique())}")
            if y_level is not None:
                st.write(f"Samples (y_level): {len(y_level)}, classes: {list(pd.Series(y_level).unique())}")

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Categorical variables are converted into numeric features using one-hot encoding.
                - The resulting X matrix is ready for machine learning algorithms.
                """
            )

    # ---------------------------------------------------
    # PAGE: Feature Selection & Relevance (Task 3 & 6) – TABS
    # ---------------------------------------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance")

        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        st.markdown(
            """
            Here, a **Random Forest** model estimates feature importance.
            Higher importance suggests a stronger contribution to prediction.
            """
        )

        tab_rt, tab_rl = st.tabs(["Risk_Type Feature Importance", "Risk_Level Feature Importance"])

        with tab_rt:
            if y_type is not None:
                st.subheader("Random Forest Feature Importance – Risk_Type")
                rf_rt = RandomForestClassifier(n_estimators=200, random_state=42)
                rf_rt.fit(X, y_type)
                importances_rt = pd.Series(rf_rt.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.write("Top 10 features (Risk-Type):")
                st.dataframe(importances_rt.head(10))
                st.pyplot(plot_bar(importances_rt.head(10), "Top 10 Feature Importances (Risk_Type)", "Features"))

                st.markdown("**Interpretation (Risk_Type):**")
                st.markdown(
                    """
                    - The chart shows which predictors most influence Risk_Type classification.
                    - These can be discussed as key drivers of risk in your findings/discussion.
                    """
                )
            else:
                st.warning(f"Target column '{TARGET_RISK_TYPE}' not found.")

        with tab_rl:
            if y_level is not None:
                st.subheader("Random Forest Feature Importance – Risk_Level")
                rf_rl = RandomForestClassifier(n_estimators=200, random_state=42)
                rf_rl.fit(X, y_level)
                importances_rl = pd.Series(rf_rl.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.write("Top 10 features (Risk-Level):")
                st.dataframe(importances_rl.head(10))
                st.pyplot(plot_bar(importances_rl.head(10), "Top 10 Feature Importances (Risk_Level)", "Features"))

                st.markdown("**Interpretation (Risk-Level):**")
                st.markdown(
                    """
                    - These features are most influential for predicting Risk_Level.
                    - Overlapping important features across both targets indicate robust risk indicators.
                    """
                )
            else:
                st.warning(f"Target column '{TARGET_RISK_LEVEL}' not found.")

    # ---------------------------------------------------
    # PAGE: Classification Modeling (Tasks 4, 5 & 7)
    # ---------------------------------------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Tasks 4, 5 & 7: Classification Modeling")

        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        tab1, tab2 = st.tabs(["Risk-Type Models", "Risk-Level Models"])

        with tab1:
            if y_type is None:
                st.warning(f"Target column '{TARGET_RISK_TYPE}' not found; cannot train models for Risk-Type.")
            else:
                st.subheader("Models for Risk-Type")
                try:
                    _, metrics_rt, split_info_rt = train_models(X, y_type)
                except ValueError as e:
                    st.warning(f"Could not train Risk-Type models: {e}")
                    metrics_rt, split_info_rt = None, None

                if metrics_rt is not None:
                    st.write("Performance Metrics – Risk-Type")
                    st.dataframe(metrics_rt.style.format("{:.3f}"))
                    st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))

                    st.markdown("**Train–Test Split (Risk-Type):**")
                    st.markdown(
                        f"""
                        - Training set shape: `{split_info_rt['X_train_shape']}`  
                        - Test set shape: `{split_info_rt['X_test_shape']}`
                        """
                    )
                    st.write("Class distribution in **training set**:")
                    st.write(split_info_rt["y_train_counts"])
                    st.write("Class distribution in **test set**:")
                    st.write(split_info_rt["y_test_counts"])

                    st.markdown("**Interpretation (Risk-Type Models):**")
                    st.markdown(
                        """
                        - The table and bar chart summarize the performance of models predicting Risk_Type.
                        - Accuracy is overall correctness; F1-score is more informative under class imbalance.
                        - The split information confirms evaluation on unseen test data.
                        """
                    )

        with tab2:
            if y_level is None:
                st.warning(f"Target column '{TARGET_RISK_LEVEL}' not found; cannot train models for Risk-Level.")
            else:
                st.subheader("Models for Risk-Level")
                try:
                    _, metrics_rl, split_info_rl = train_models(X, y_level)
                except ValueError as e:
                    st.warning(f"Could not train Risk-Level models: {e}")
                    metrics_rl, split_info_rl = None, None

                if metrics_rl is not None:
                    st.write("Performance Metrics – Risk-Level")
                    st.dataframe(metrics_rl.style.format("{:.3f}"))
                    st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))

                    st.markdown("**Train–Test Split (Risk-Level):**")
                    st.markdown(
                        f"""
                        - Training set shape: `{split_info_rl['X_train_shape']}`  
                        - Test set shape: `{split_info_rl['X_test_shape']}`
                        """
                    )
                    st.write("Class distribution in **training set**:")
                    st.write(split_info_rl["y_train_counts"])
                    st.write("Class distribution in **test set**:")
                    st.write(split_info_rl["y_test_counts"])

                    st.markdown("**Interpretation (Risk-Level Models):**")
                    st.markdown(
                        """
                        - The metrics evaluate how well models predict Risk_Level categories.
                        - F1-score is useful for imbalanced classes; train/test distributions show representativeness.
                        """
                    )

        st.subheader("Overall Interpretation")
        st.markdown(
            """
            - Comparing models helps identify the best-performing approach for each target.
            - The explicit train–test split information improves transparency and reproducibility.
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

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - This table shows how frequently each polymer type appears.
                - Highly frequent polymers likely reflect dominant sources in the sampling environment.
                """
            )

            st.subheader("Bar Plot of Polymer_Type Distribution")
            st.pyplot(plot_bar(vc, "Distribution of Polymer_Type", "Polymer_Type"))

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - The bar chart highlights dominant polymer types.
                - This supports recommendations on which pollution sources to prioritize for mitigation.
                """
            )
        else:
            st.warning("Column 'Polymer_Type' not found in the dataset.")

    # ---------------------------------------------------
    # PAGE: SMOTE & Hyperparameter Tuning (Risk_Type) – TABS
    # ---------------------------------------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("Address Class Imbalance & Tune Logistic Regression (Risk-Type)")

        _, X, y_type, _, _, _ = preprocess_for_model(df_raw)

        if y_type is None:
            st.warning(f"Target column '{TARGET_RISK_TYPE}' not found; cannot run SMOTE or tuning.")
            return

        tab1, tab2 = st.tabs([
            "Original Distribution & Base Models",
            "SMOTE + Tuning & Comparison",
        ])

        with tab1:
            st.subheader("Class Distribution of Risk-Type (Original)")
            st.write(pd.Series(y_type).value_counts())

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Large differences between class counts indicate class imbalance.
                - Class imbalance can cause models to favor the majority class, reducing fairness and recall for minority classes.
                """
            )

            try:
                _, base_metrics_rt, split_info_base_rt = train_models(X, y_type)
            except ValueError as e:
                st.warning(f"Could not train base Risk-Type models: {e}")
                base_metrics_rt, split_info_base_rt = None, None

            if base_metrics_rt is not None:
                st.subheader("Base Models Performance (Risk-Type)")
                st.dataframe(base_metrics_rt.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(base_metrics_rt, "(Risk-Type – Base)"))

                st.markdown("**Train–Test Split (Base Risk-Type Models):**")
                st.markdown(
                    f"""
                    - Training set shape: `{split_info_base_rt['X_train_shape']}`  
                    - Test set shape: `{split_info_base_rt['X_test_shape']}`
                    """
                )
                st.write("Class distribution in **training set**:")
                st.write(split_info_base_rt["y_train_counts"])
                st.write("Class distribution in **test set**:")
                st.write(split_info_base_rt["y_test_counts"])

                st.markdown("**Interpretation (Base Models):**")
                st.markdown(
                    """
                    - These metrics represent baseline performance before class balancing or tuning.
                    - Lower recall/F1 often indicates difficulty detecting minority classes.
                    """
                )

        with tab2:
            st.subheader("SMOTE + Hyperparameter Tuning for Logistic Regression (Risk-Type)")
            try:
                with st.spinner("Running SMOTE and GridSearchCV..."):
                    best_lr, tuned_metrics, best_params, split_info_smote = smote_and_tune_logreg(X, y_type)

                st.write("Best Hyperparameters (Logistic Regression):")
                st.json(best_params)

                st.markdown("**Train–Test Split (Tuned Risk-Type Model):**")
                st.markdown(
                    f"""
                    - Training set shape (before SMOTE): `{split_info_smote['X_train_shape']}`  
                    - Test set shape: `{split_info_smote['X_test_shape']}`
                    """
                )
                st.write("Class distribution in **training set (before SMOTE)**:")
                st.write(split_info_smote["y_train_counts"])
                st.write("Class distribution in **test set**:")
                st.write(split_info_smote["y_test_counts"])

                st.markdown("**Interpretation (Tuned Logistic Regression):**")
                st.markdown(
                    """
                    - SMOTE generates synthetic training samples for minority classes.
                    - Tuning selects hyperparameters that improve generalization under the new balanced training data.
                    """
                )

                st.subheader("Performance of Tuned Logistic Regression (with SMOTE)")
                st.dataframe(tuned_metrics.style.format("{:.3f}"))

                # Comparison chart
                try:
                    _, base_metrics_rt_for_compare, _ = train_models(X, y_type)
                    combined = pd.concat([base_metrics_rt_for_compare, tuned_metrics])
                    st.subheader("Comparison: Tuned Logistic Regression vs Original Models")
                    st.dataframe(combined.style.format("{:.3f}"))
                    st.pyplot(plot_metrics_bar(combined, "(Risk-Type – Base vs Tuned + SMOTE)"))

                    st.markdown("**Interpretation (Comparison):**")
                    st.markdown(
                        """
                        - If the tuned model improves F1-score and recall, it indicates better detection across all classes.
                        - This typically means the model is less biased toward the majority class after balancing.
                        """
                    )
                except ValueError:
                    st.warning("Could not recompute base models for comparison.")
            except ValueError as e:
                st.warning(f"Could not run SMOTE/tuning: {e}")


if __name__ == "__main__":
    main()
