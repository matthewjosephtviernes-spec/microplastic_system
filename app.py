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
            # Only relevant for local path
            if uploaded_file is None:
                raise

    # If all encodings fail, re-raise the last error
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

    # Example: Polymer_Type -> fill with mode
    if "Polymer_Type" in df.columns:
        mode_val = df["Polymer_Type"].mode(dropna=True)
        if len(mode_val) > 0:
            df["Polymer_Type"] = df["Polymer_Type"].fillna(mode_val.iloc[0])

    # Simple categorical imputation: fill remaining NaNs with mode
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

    # Ensure numeric for skewness calculation
    for col in cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    skewness = df[cols_present].skew(numeric_only=True)
    # consider columns with |skew| > 1 as skewed
    skewed_cols = skewness[skewness.abs() > 1].index.tolist()

    for col in skewed_cols:
        # shift to be positive before log1p if necessary
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

    # Keep only rows where both targets are present (if they exist)
    available_targets = [c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns]
    if len(available_targets) == 2:
        df = df.dropna(subset=available_targets)

    # Handle missing (and coerce numeric)
    df = handle_missing_values(df)

    # Outlier handling (on numeric)
    df = cap_outliers_iqr(df, NUMERIC_COLS)

    # Skew transform (before scaling)
    df, skewness, skewed_cols = transform_skewed(df, NUMERIC_COLS)

    # Scale numeric
    df, scaler = scale_numeric(df, NUMERIC_COLS)

    # Separate targets
    y_type = df[TARGET_RISK_TYPE] if TARGET_RISK_TYPE in df.columns else None
    y_level = df[TARGET_RISK_LEVEL] if TARGET_RISK_LEVEL in df.columns else None

    # Drop targets from features
    drop_cols = [c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    # One-hot encode categoricals
    existing_cat_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns]
    X = pd.get_dummies(feature_df, columns=existing_cat_cols, drop_first=True)

    # Ensure X is fully numeric (sklearn requirement)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df, X, y_type, y_level, skewness, skewed_cols


# -------------------------------------------------------
# MODELING
# -------------------------------------------------------
def train_models(X, y):
    """
    Train Logistic Regression, Random Forest, Gradient Boosting.
    Returns dict of trained models and performance metrics.
    """
    # Drop any rows where y is NaN
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to train models.")

    # Try stratified split first, then fallback if it fails
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

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (weighted)": prec,
            "Recall (weighted)": rec,
            "F1-score (weighted)": f1,
        })

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    return models, metrics_df


def smote_and_tune_logreg(X, y):
    """
    For Risk_Type only:
      - apply SMOTE (if possible)
      - tune LogisticRegression with GridSearchCV
    Returns tuned model and metrics df.
    """
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to run SMOTE and tuning.")

    # Try stratified split first
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

    # Apply SMOTE on train only (if possible)
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except ValueError:
        st.warning(
            "SMOTE failed due to very small or highly imbalanced classes. "
            "Continuing without SMOTE for logistic regression tuning."
        )
        X_res, y_res = X_train, y_train

    # Hyperparameter grid
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

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    tuned_metrics = pd.DataFrame(
        [{
            "Model": "LogReg (tuned + SMOTE)",
            "Accuracy": acc,
            "Precision (weighted)": prec,
            "Recall (weighted)": rec,
            "F1-score (weighted)": f1,
        }]
    ).set_index("Model")

    return best_lr, tuned_metrics, grid.best_params_


# -------------------------------------------------------
# VISUALIZATION HELPERS
# -------------------------------------------------------
def plot_hist_box(df, col):
    # Coerce to numeric (invalid values -> NaN)
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
    # Coerce both to numeric
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]

    fig, ax = plt.subplots(figsize=(6, 4))

    if len(x_clean) == 0:
        ax.text(0.5, 0.5, f"No numeric data for {x_col} and {y_col}",
                ha="center", va="center")
    else:
        ax.scatter(x_clean, y_clean, alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")

    plt.tight_layout()
    return fig


def plot_box_by_category(df, value_col, category_col):
    # Coerce value column to numeric
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
    # Load dataset (from upload or local file)
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
            "Please open it locally and check that:\n"
            "- It has a header row with column names, and\n"
            "- There are data rows under the header.\n\n"
            "Then save/export it again as a proper CSV and re-upload."
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
            "- Place a file named `Microplastic.csv` in the same folder as `app.py`."
        )
        st.stop()

    # ---------------------------------------------------
    # PAGE: Data Overview & Task 1
    # ---------------------------------------------------
    if page == "Data Overview & Task 1":
        st.header("Data Overview & Task 1: Risk_Score Analysis")

        # ---------- Raw Dataset ----------
        st.subheader("Raw Dataset (first 10 rows)")
        st.dataframe(df_raw.head(10))

        st.markdown(
            f"**Shape of the dataset:** `{df_raw.shape[0]}` rows × `{df_raw.shape[1]}` columns"
        )

        st.markdown("**Interpretation:**")
        st.markdown(
            """
            - This table shows a sample of the raw data, including key variables such as **MP_Count_per_L**, 
              **Risk_Score**, and the risk labels.
            - It is used to verify that the dataset has been loaded correctly and that all required columns are present.
            """
        )

        # ---------- Risk_Score Distribution ----------
        if "Risk_Score" in df_raw.columns:
            st.subheader("Distribution of Risk_Score (Histogram & Boxplot)")
            fig = plot_hist_box(df_raw, "Risk_Score")
            st.pyplot(fig)

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - The **histogram** shows how often different Risk_Score values occur in the dataset, 
                  indicating whether most sites have low, moderate, or high risk.
                - The **boxplot** highlights the overall spread of Risk_Score and any **outliers**, 
                  which may correspond to locations with unusually high or low risk.
                """
            )

        # ---------- Relationship: MP_Count_per_L vs Risk_Score ----------
        if "MP_Count_per_L" in df_raw.columns and "Risk_Score" in df_raw.columns:
            st.subheader("Relationship between Risk_Score and MP_Count_per_L")
            fig = plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score")
            st.pyplot(fig)

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Each point represents one sampling site, with **microplastic concentration (MP_Count_per_L)** on the x-axis 
                  and **Risk_Score** on the y-axis.
                - A visible upward pattern would suggest that higher microplastic concentrations tend to be associated with 
                  higher risk scores.
                - If the points are widely scattered with no clear trend, it indicates that other factors (e.g., polymer type, 
                  size, or environmental conditions) also play a strong role in determining risk.
                """
            )

        # ---------- Risk_Score by Risk_Level ----------
        if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
            st.subheader("Difference in Risk_Score by Risk_Level (Boxplot)")
            fig = plot_box_by_category(df_raw, "Risk_Score", "Risk_Level")
            st.pyplot(fig)

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - This boxplot compares the distribution of **Risk_Score** across different **Risk_Level** categories 
                  (e.g., Low, Moderate, High).
                - We expect higher Risk_Level groups to show higher Risk_Score values on average.
                - If there is strong separation between the boxes, it suggests that the numerical Risk_Score and the 
                  categorical Risk_Level are consistent.
                - Large overlaps between categories may indicate borderline cases or the need to refine the risk thresholds.
                """
            )

    # ---------------------------------------------------
    # PAGE: Preprocessing (Task 2)
    # ---------------------------------------------------
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Preprocessing")

        # ---------- Before Preprocessing ----------
        numeric_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
        if numeric_present:
            st.subheader("Before Preprocessing – Descriptive Stats (Numeric Columns)")
            st.write(df_raw[numeric_present].describe())

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - These statistics summarize the original numeric variables, including their minimum, maximum, mean, 
                  and quartiles.
                - Large ranges or extreme maximum values can indicate the presence of **outliers** or highly skewed data.
                - This provides a baseline to compare with the cleaned dataset after preprocessing.
                """
            )
        else:
            st.info("No numeric columns from NUMERIC_COLS were found in the dataset.")

        # Run full preprocessing
        df_clean, X, y_type, y_level, skewness, skewed_cols = preprocess_for_model(df_raw)

        # ---------- After Preprocessing ----------
        numeric_present_clean = [c for c in NUMERIC_COLS if c in df_clean.columns]
        if numeric_present_clean:
            st.subheader("After Outlier Handling, Skew Transform, and Scaling – Descriptive Stats")
            st.write(df_clean[numeric_present_clean].describe())

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - After preprocessing, the numeric variables have more stable ranges and reduced influence from extreme values.
                - Outlier capping and transformations smooth the distributions, which helps machine learning models perform better.
                - The means and standard deviations are now more comparable across different variables due to scaling.
                """
            )

        # ---------- Skewness ----------
        st.subheader("Skewness of Numeric Columns (before transform)")
        st.write(skewness)

        st.markdown("**Interpretation:**")
        st.markdown(
            """
            - Skewness values indicate how symmetric or asymmetric each numeric variable is.
            - High positive or negative skewness suggests that values are concentrated on one side with a long tail on the other.
            - Columns flagged as skewed are transformed (e.g., log-transformed) to reduce skewness and improve model stability.
            """
        )

        if len(skewed_cols) > 0:
            st.write("Columns treated as skewed and transformed (log1p):")
            st.write(skewed_cols)
        else:
            st.write("No numeric columns exceeded the skewness threshold; no log transform applied.")

        # ---------- Encoded Feature Matrix ----------
        st.subheader("Encoded Feature Matrix (X) – First 10 Rows")
        st.dataframe(X.head(10))

        st.write("Shape of X:", X.shape)
        if y_type is not None:
            st.write(f"Number of samples (y_type): {len(y_type)}, classes: {y_type.unique()}")
        if y_level is not None:
            st.write(f"Number of samples (y_level): {len(y_level)}, classes: {y_level.unique()}")

        st.markdown("**Interpretation:**")
        st.markdown(
            """
            - The feature matrix **X** contains all predictors in fully numeric form after encoding categorical variables.
            - Each column now represents either a scaled numeric feature or a one-hot encoded category.
            - This confirms that the dataset is ready for use in machine learning algorithms that require numeric inputs.
            """
        )

    # ---------------------------------------------------
    # PAGE: Feature Selection & Relevance (Task 3 & 6)
    # ---------------------------------------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance")

        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        st.markdown("### Method")
        st.markdown(
            """
            Here, **RandomForestClassifier** is used to estimate feature importance for both **Risk_Type** and **Risk_Level**.  
            Higher importance scores indicate a stronger contribution of that feature to the prediction.
            """
        )

        # RandomForest for Risk_Type
        if y_type is not None:
            st.subheader("Random Forest Feature Importance – Risk_Type")

            rf_rt = RandomForestClassifier(n_estimators=200, random_state=42)
            rf_rt.fit(X, y_type)
            importances_rt = pd.Series(rf_rt.feature_importances_, index=X.columns)
            importances_rt = importances_rt.sort_values(ascending=False)

            st.write("Top 10 features (Risk_Type):")
            st.dataframe(importances_rt.head(10))

            fig_rt = plot_bar(importances_rt.head(10), "Top 10 Feature Importances (Risk_Type)", "Features")
            st.pyplot(fig_rt)

            st.markdown("**Interpretation (Risk_Type):**")
            st.markdown(
                """
                - The bar chart shows which features are most influential in classifying **Risk_Type**.
                - Features with higher importance scores contribute more to distinguishing between different risk types.
                - These variables can be highlighted in the discussion as key environmental drivers affecting risk classification.
                """
            )
        else:
            st.warning(f"Target column '{TARGET_RISK_TYPE}' not found; cannot compute feature importance for Risk_Type.")

        # RandomForest for Risk_Level
        if y_level is not None:
            st.subheader("Random Forest Feature Importance – Risk_Level")

            rf_rl = RandomForestClassifier(n_estimators=200, random_state=42)
            rf_rl.fit(X, y_level)
            importances_rl = pd.Series(rf_rl.feature_importances_, index=X.columns)
            importances_rl = importances_rl.sort_values(ascending=False)

            st.write("Top 10 features (Risk-Level):")
            st.dataframe(importances_rl.head(10))

            fig_rl = plot_bar(importances_rl.head(10), "Top 10 Feature Importances (Risk_Level)", "Features")
            st.pyplot(fig_rl)

            st.markdown("**Interpretation (Risk_Level):**")
            st.markdown(
                """
                - These feature importance scores indicate which variables most influence the classification of **Risk_Level**.
                - If similar features are important for both **Risk_Type** and **Risk_Level**, they are likely to be critical indicators of microplastic-related risk.
                - This information can be used to prioritize which parameters to monitor in future sampling campaigns.
                """
            )
        else:
            st.warning(f"Target column '{TARGET_RISK_LEVEL}' not found; cannot compute feature importance for Risk-Level.")

    # ---------------------------------------------------
    # PAGE: Classification Modeling (Tasks 4, 5 & 7)
    # ---------------------------------------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Tasks 4, 5 & 7: Classification Modeling")

        df_clean, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        tab1, tab2 = st.tabs(["Risk-Type Models", "Risk-Level Models"])

        # ---------- Risk-Type Models ----------
        with tab1:
            if y_type is None:
                st.warning(f"Target column '{TARGET_RISK_TYPE}' not found; cannot train models for Risk-Type.")
            else:
                st.subheader("Models for Risk-Type")
                try:
                    models_rt, metrics_rt = train_models(X, y_type)
                except ValueError as e:
                    st.warning(f"Could not train Risk-Type models: {e}")
                    models_rt, metrics_rt = None, None

                if metrics_rt is not None:
                    st.write("Performance Metrics – Risk-Type")
                    st.dataframe(metrics_rt.style.format("{:.3f}"))

                    fig_rt = plot_metrics_bar(metrics_rt, "(Risk-Type)")
                    st.pyplot(fig_rt)

                    st.markdown("**Interpretation (Risk-Type Models):**")
                    st.markdown(
                        """
                        - The table and bar chart summarize the performance of different models in predicting **Risk_Type**.
                        - **Accuracy** measures overall correctness, while **precision, recall, and F1-score** capture how well each model handles different classes.
                        - The model with the highest F1-score is usually the most balanced and is a strong candidate for deployment for Risk_Type classification.
                        """
                    )

        # ---------- Risk-Level Models ----------
        with tab2:
            if y_level is None:
                st.warning(f"Target column '{TARGET_RISK_LEVEL}' not found; cannot train models for Risk-Level.")
            else:
                st.subheader("Models for Risk-Level")
                try:
                    models_rl, metrics_rl = train_models(X, y_level)
                except ValueError as e:
                    st.warning(f"Could not train Risk-Level models: {e}")
                    models_rl, metrics_rl = None, None

                if metrics_rl is not None:
                    st.write("Performance Metrics – Risk-Level")
                    st.dataframe(metrics_rl.style.format("{:.3f}"))

                    fig_rl = plot_metrics_bar(metrics_rl, "(Risk-Level)")
                    st.pyplot(fig_rl)

                    st.markdown("**Interpretation (Risk-Level Models):**")
                    st.markdown(
                        """
                        - These results show how accurately each model predicts the categorical **Risk_Level** (e.g., Low, Moderate, High).
                        - Comparing the metrics across models helps identify which algorithm is best suited for capturing the structure of Risk_Level in the data.
                        - Again, F1-score is a useful summary for models dealing with potentially imbalanced risk categories.
                        """
                    )

        st.subheader("Overall Interpretation")
        st.markdown(
            """
            - By comparing models for both **Risk_Type** and **Risk_Level**, we can identify the most reliable algorithms for risk prediction.
            - These results can be referenced in the thesis to justify the final model choice for the decision-support system.
            """
        )

    # ---------------------------------------------------
    # PAGE: Polymer Type Distribution
    # ---------------------------------------------------
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution (Task: Load & Visualize Polymer Type)")

        df = handle_missing_values(df_raw)

        if "Polymer_Type" in df.columns:
            # ---------- Value Counts ----------
            st.subheader("Value Counts of Polymer_Type")
            vc = df["Polymer_Type"].value_counts()
            st.dataframe(vc.rename("count"))

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - This table lists how many times each **polymer type** appears in the dataset.
                - The most frequent polymers likely correspond to dominant sources of microplastics in the study area.
                """
            )

            # ---------- Bar Plot ----------
            st.subheader("Bar Plot of Polymer_Type Distribution")
            fig = plot_bar(vc, "Distribution of Polymer_Type", "Polymer_Type")
            st.pyplot(fig)

            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - The bar chart visually highlights which polymer types are most common in the samples.
                - Dominant bars suggest key contributors to microplastic pollution, such as packaging materials, textiles, or fishing-related plastics.
                - This information can support recommendations on which sectors or activities to target for pollution reduction.
                """
            )
        else:
            st.warning("Column 'Polymer_Type' not found in the dataset.")

    # ---------------------------------------------------
    # PAGE: SMOTE & Hyperparameter Tuning (Risk_Type)
    # ---------------------------------------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("Address Class Imbalance & Tune Logistic Regression (Risk-Type)")

        df_clean, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        if y_type is None:
            st.warning(f"Target column '{TARGET_RISK_TYPE}' not found; cannot run SMOTE or tuning.")
            return

        # ---------- Class Distribution ----------
        st.subheader("Class Distribution of Risk-Type (Original)")
        st.write(y_type.value_counts())

        st.markdown("**Interpretation:**")
        st.markdown(
            """
            - This shows how many samples belong to each **Risk_Type** category.
            - Large differences between classes indicate **class imbalance**, where some risk types are underrepresented.
            - Imbalanced data can bias models toward the majority class, motivating the use of techniques such as SMOTE.
            """
        )

        # ---------- Base Models ----------
        try:
            _, base_metrics_rt = train_models(X, y_type)
        except ValueError as e:
            st.warning(f"Could not train base Risk-Type models: {e}")
            base_metrics_rt = None

        if base_metrics_rt is not None:
            st.subheader("Base Models Performance (Risk-Type)")
            st.dataframe(base_metrics_rt.style.format("{:.3f}"))
            fig_base = plot_metrics_bar(base_metrics_rt, "(Risk-Type – Base)")
            st.pyplot(fig_base)

            st.markdown("**Interpretation (Base Models):**")
            st.markdown(
                """
                - These metrics show how well the initial models perform **before** any class balancing or tuning.
                - Lower performance, especially on minority classes, often indicates the need for improved handling of class imbalance.
                """
            )

        # ---------- SMOTE + Tuning ----------
        st.subheader("Applying SMOTE + Hyperparameter Tuning for Logistic Regression (Risk-Type)")

        try:
            with st.spinner("Running SMOTE and GridSearchCV (this may take a bit)..."):
                best_lr, tuned_metrics, best_params = smote_and_tune_logreg(X, y_type)

            st.write("Best Hyperparameters (Logistic Regression):")
            st.json(best_params)

            st.markdown("**Interpretation (Tuned Logistic Regression):**")
            st.markdown(
                """
                - SMOTE generates synthetic examples for minority Risk_Type classes, leading to a more balanced training set.
                - Hyperparameter tuning adjusts the logistic regression model to better fit the balanced data.
                - The resulting performance metrics reflect how effective the tuned model is under improved class balance.
                """
            )

            st.subheader("Performance of Tuned Logistic Regression (with SMOTE)")
            st.dataframe(tuned_metrics.style.format("{:.3f}"))

            # ---------- Comparison ----------
            if base_metrics_rt is not None:
                combined = pd.concat([base_metrics_rt, tuned_metrics])
                st.subheader("Comparison: Tuned Logistic Regression vs Original Models")
                st.dataframe(combined.style.format("{:.3f}"))

                fig_combined = plot_metrics_bar(combined, "(Risk-Type – Base vs TunED + SMOTE)")
                st.pyplot(fig_combined)

                st.markdown("**Interpretation (Comparison):**")
                st.markdown(
                    """
                    - This comparison shows whether the SMOTE-balanced and tuned logistic regression model 
                      improves over the original models.
                    - Increases in F1-score and recall for minority classes indicate that class balancing has made 
                      the model more **fair** and effective at detecting all risk types.
                    """
                )

        except ValueError as e:
            st.warning(f"Could not run SMOTE/tuning: {e}")


if __name__ == "__main__":
    main()
