import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone

sns.set_style("whitegrid")

st.set_page_config(page_title="Microplastic Risk Dashboard", page_icon="🧪", layout="wide")
st.title("🧪 Microplastic Risk Data Mining & Forecasting")

# =========================
# GLOBAL STYLING – GREEN THEME + BOXED SIDEBAR NAV
# =========================
st.markdown(
    """
    <style>
    :root {
        --primary: #0da95c;
        --primary-dark: #0b7d44;
        --bg-main-top: #e2ffe9;
        --bg-main-bottom: #ffffff;
        --bg-sidebar-top: #06331f;
        --bg-sidebar-bottom: #0f7b45;
        --text-dark: #042513;
        --text-muted: #476f5a;
    }

    /* MAIN BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, var(--bg-main-top) 0%, #f7fff9 45%, var(--bg-main-bottom) 100%);
    }

    /* SIDEBAR BASE */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-sidebar-top) 0%, var(--bg-sidebar-bottom) 50%, var(--bg-sidebar-top) 100%);
        color: #f0fff6;
        border-right: 1px solid rgba(255,255,255,0.15);
    }
    [data-testid="stSidebar"] * {
        color: #f0fff6 !important;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* BOXED SIDEBAR RADIO NAVIGATION */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        border-radius: 0.7rem;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(0,0,0,0.18);
        padding: 0.55rem 0.7rem;
        display: flex !important;
        align-items: center;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
        font-size: 0.86rem;
        font-weight: 500;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(0,0,0,0.25);
        border-color: rgba(255,255,255,0.45);
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        border-color: rgba(255,255,255,0.7);
        box-shadow: 0 6px 14px rgba(0,0,0,0.35);
        transform: translateX(2px);
    }

    /* Hide the default radio circle; keep text only inside box */
    [data-testid="stSidebar"] div[role="radiogroup"] > label span:first-child {
        display: none;
    }

    /* MAIN TEXT */
    h1, h2, h3, h4 {
        color: var(--text-dark);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body, p, li, span, div {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* SECTION CARDS */
    .section-card {
        background: #ffffff;
        padding: 1.6rem 1.8rem;
        border-radius: 1.2rem;
        box-shadow: 0 10px 24px rgba(0, 80, 40, 0.08);
        border: 1px solid rgba(7, 102, 61, 0.12);
        margin-bottom: 1.5rem;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.4rem 1.2rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        box-shadow: 0 6px 16px rgba(0, 80, 40, 0.35);
    }
    .stButton > button:hover {
        filter: brightness(1.03);
        transform: translateY(-1px);
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] > div {
        border-radius: 1rem;
        border: 1.8px dashed rgba(0,128,0,0.35);
        background: rgba(255,255,255,0.1);
    }

    /* METRICS */
    .stMetric {
        background: linear-gradient(135deg, #e3ffe9, #f8fff9);
        border-radius: 0.9rem;
        padding: 0.25rem 0.65rem;
        box-shadow: 0 4px 10px rgba(0, 80, 40, 0.09);
    }

    /* TABLES */
    .stDataFrame, .stTable {
        background: #ffffff !important;
        border-radius: 0.9rem;
        overflow: hidden;
        box-shadow: 0 10px 20px rgba(0, 80, 40, 0.04);
    }
    .stDataFrame * {
        color: #0f172a !important;
    }

    /* EXPANDERS */
    details {
        border-radius: 0.9rem;
        border: 1px solid rgba(7, 102, 61, 0.18);
        background: #f3fff7;
        padding: 0.5rem 0.75rem;
    }

    /* ALERTS */
    .stAlert {
        border-radius: 0.9rem;
    }

    /* FOOTER BADGE */
    .footer {
        text-align: center;
        color: #476f5a;
        font-size: 0.78rem;
        margin-top: 1.0rem;
        padding-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TABS / STEPS
# =========================
tabs = [
    "Overview / About the Study",
    "1. Data Upload & Description",
    "2. Data Preprocessing",
    "3. Preprocessed Data Results",
    "4. Predictive Modeling & Validation",
    "5. Risk Visualizations & Interpretation",
]

# =========================
# SIDEBAR NAVIGATION (minimal, boxed)
# =========================
st.sidebar.markdown("## 📌 Workflow")
selected_tab = st.sidebar.radio(
    "Navigate:",
    tabs,
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Theme:** Green Gradient Microplastic Risk Dashboard")

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False
if "df_with_preds" not in st.session_state:
    st.session_state.df_with_preds = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None

# Expected numeric columns
num_cols = ["MP_Count_per_L", "Risk_Score", "Microplastic_Size_mm_midpoint", "Density_midpoint"]

# IMPORTANT: do NOT include Risk_Type and Risk_Level here, so their names stay as labels
cat_cols = [
    "Location",
    "Shape",
    "Polymer_Type",
    "pH",
    "Salinity",
    "Industrial_Activity",
    "Population_Density",
    "Author",
]

# =========================
# HELPERS
# =========================
def get_value_counts_for_column(df, column):
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    vc = df[column].value_counts(dropna=False)
    return pd.DataFrame({column: vc.index, "count": vc.values})


def plot_value_counts_bar(df_counts, x_col=None, y_col="count", title="Value Counts"):
    if df_counts.empty:
        st.write("No data to plot.")
        return
    if x_col is None:
        x_col = df_counts.columns[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_counts[x_col].astype(str), df_counts[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)


def card_open():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 0. Overview / About
# =========================
if selected_tab == tabs[0]:
    card_open()
    st.header("Overview / About the Study")

    st.markdown(
        """
        This interactive dashboard implements a **predictive risk modeling framework**
        for **microplastic pollution**.

        ### 🎯 General Objective
        > To develop a predictive risk modeling framework for microplastic pollution using data mining techniques.

        ### 🧪 App Structure
        1. **Data Upload & Description** – Load your microplastic risk dataset.  
        2. **Data Preprocessing** – Clean, transform, and encode the data.  
        3. **Preprocessed Data Results** – Inspect the model-ready dataset.  
        4. **Predictive Modeling & Validation** – Train and compare models.  
        5. **Risk Visualizations & Interpretation** – Explore and interpret **actual and predicted risk patterns**.
        """
    )

    st.info(
        "Begin with **'1. Data Upload & Description'** using the sidebar navigation. "
        "Each step corresponds to a part of your methodology chapter."
    )
    card_close()

# =========================
# 1. Data Upload & Description
# =========================
elif selected_tab == tabs[1]:
    card_open()
    st.header("Step 1 – Data Upload & Description")

    st.markdown(
        """
        **Purpose of this step**

        - Import the **final structured dataset** on microplastic pollution.  
        - Make sure all variables to be used in preprocessing and modeling are present.

        Accepted formats: **CSV** or **Excel (.xlsx)**.
        """
    )

    uploaded_file = st.file_uploader("Upload your microplastic risk dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, encoding="latin1")
            else:
                raw_df = pd.read_excel(uploaded_file)

            st.session_state.raw_df = raw_df.copy()
            st.session_state.df = raw_df.copy()
            st.session_state.preprocessed = False
            # New dataset → reset models & predictions
            st.session_state.df_with_preds = None
            st.session_state.best_model_name = None

            st.success("✅ Dataset uploaded successfully!")

            st.subheader("Dataset Description")
            rows, cols = raw_df.shape
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", rows)
                st.metric("Columns", cols)
            with col2:
                st.write("**Column names:**")
                st.write(list(raw_df.columns))

            st.markdown(
                """
                **How to interpret**

                - *Rows* = number of sampling records or literature entries.  
                - *Columns* = number of variables (risk drivers, location, scores, etc.).  
                - Check if the key variables (e.g., **Risk_Type**, **Risk_Level**, counts, location) are present.
                """
            )

            st.subheader("Preview (First 10 Rows)")
            st.dataframe(raw_df.head(10), use_container_width=True)

            st.markdown(
                "<details><summary style='font-weight:bold'>Show full uploaded dataset</summary>",
                unsafe_allow_html=True,
            )
            st.dataframe(raw_df, use_container_width=True)
            st.markdown("</details>", unsafe_allow_html=True)

            st.info(
                "Next, open **'2. Data Preprocessing'** from the sidebar to clean "
                "and transform the dataset for modeling."
            )
        except Exception as e:
            st.error(f"Failed to read the uploaded file: {e}")
    else:
        st.warning("Please upload a CSV or Excel file to continue.")
    card_close()

# =========================
# 2. Data Preprocessing
# =========================
elif selected_tab == tabs[2]:
    card_open()
    st.header("Step 2 – Data Preprocessing")

    st.markdown(
        """
        **Purpose of this step**

        - Prepare the raw data so that it is **clean, consistent, and suitable** for machine learning.  
        - This corresponds to the **data preparation / KDD preprocessing** phase in your methodology.

        Operations applied:

        - Convert numeric columns to proper numeric types  
        - Handle missing values and outliers (IQR-based clipping)  
        - Apply log transforms for skewed numeric features  
        - Encode categorical variables (predictor variables only)  
        - Standardize numerical features  
        """
    )

    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a dataset first in **Step 1 – Data Upload & Description**.")
        card_close()
        st.stop()

    df_prep = df.copy()
    outlier_report = []

    # Numeric preprocessing
    for col in num_cols:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors="coerce")
            nan_count = df_prep[col].isna().sum()
            if df_prep[col].notna().sum() > 0:
                q1 = df_prep[col].quantile(0.25)
                q3 = df_prep[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                clipped_before = ((df_prep[col] < lower) | (df_prep[col] > upper)).sum()
                df_prep[col] = df_prep[col].clip(lower=lower, upper=upper)
                skew_before = df_prep[col].skew()
                transform_applied = False
                if skew_before > 1:
                    df_prep[col] = np.where(
                        df_prep[col] > -1,
                        np.log1p(df_prep[col] - df_prep[col].min() + 1),
                        df_prep[col],
                    )
                    transform_applied = True
                outlier_report.append(
                    f"Column '{col}': NaNs={nan_count}, outliers clipped={clipped_before}, "
                    f"skew_before={skew_before:.2f}, log_transform_applied={transform_applied}"
                )

    # Categorical preprocessing (predictors only; Risk_Type / Risk_Level are kept as names)
    for col in cat_cols:
        if col in df_prep.columns:
            try:
                df_prep[col] = LabelEncoder().fit_transform(df_prep[col].astype(str))
            except Exception:
                pass

    # Standardize numeric features
    scaler = StandardScaler()
    for col in num_cols:
        if col in df_prep.columns:
            try:
                df_prep[col] = scaler.fit_transform(df_prep[[col]])
            except Exception:
                pass

    st.session_state.df = df_prep
    st.session_state.preprocessed = True

    st.success("✅ Data preprocessing complete!")

    st.subheader("Preprocessed Dataset (First 10 Rows)")
    st.dataframe(df_prep.head(10), use_container_width=True)

    st.markdown(
        """
        **How to interpret**

        - Values may look different from the raw data because of **standardization** and **encoding**.  
        - Predictor categorical variables (e.g., *Location, Shape*) are now encoded as integers.  
        - Target variables **Risk_Type** and **Risk_Level** remain as labels so they can be interpreted easily in the results.
        """
    )

    with st.expander("Preprocessing Log (Outliers & Transforms)"):
        if outlier_report:
            for line in outlier_report:
                st.markdown(f"- {line}")
        else:
            st.write("No numeric columns from the expected list were found or processed.")

    st.info(
        "Open **'3. Preprocessed Data Results'** in the sidebar to explore the cleaned, model-ready dataset."
    )
    card_close()

# =========================
# 3. Preprocessed Data Results
# =========================
elif selected_tab == tabs[3]:
    card_open()
    st.header("Step 3 – Preprocessed Data Results")

    if st.session_state.df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please run preprocessing first in **Step 2 – Data Preprocessing**.")
        card_close()
        st.stop()

    df_prep = st.session_state.df

    # Try to treat these as numeric if present
    special_numeric = ["Latitude", "Longitude", "Microplastic_Size_mm", "Density"]
    for col in special_numeric:
        if col in df_prep.columns:
            try:
                df_prep[col] = pd.to_numeric(df_prep[col], errors="coerce")
            except Exception:
                pass

    st.markdown(
        """
        This step **summarizes** the preprocessed dataset so that you can describe it in the
        **Results – Data Description** section of your thesis.
        """
    )

    # 1. Overview
    st.subheader("1. Dataset Overview After Preprocessing")
    n_rows, n_cols = df_prep.shape
    numeric_cols_present = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols_present = [c for c in df_prep.columns if c not in numeric_cols_present]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", n_rows)
    col2.metric("Columns", n_cols)
    col3.metric("Numeric Features", len(numeric_cols_present))

    st.markdown("**Sample of final preprocessed data (first 20 rows):**")
    st.dataframe(df_prep.head(20), use_container_width=True)

    st.caption(
        "Use this table to show an example of how the cleaned, encoded dataset looks like "
        "before feeding it into the predictive models."
    )

    # 2. Numeric summary
    st.subheader("2. Numeric Feature Summary")
    if numeric_cols_present:
        st.dataframe(df_prep[numeric_cols_present].describe(), use_container_width=True)

        st.markdown(
            """
            **How to interpret**

            - `mean` and `std` describe the **central tendency and variability** of each numeric feature.  
            - `min`, `25%`, `50%`, `75%`, `max` show the **spread** and potential outliers.  
            - You can cite these numbers when describing ranges of microplastic counts, sizes, risk scores, etc.
            """
        )

        st.markdown("**Inspect distribution of a selected numeric feature:**")
        selected_num = st.selectbox("Choose a numeric column:", numeric_cols_present)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df_prep[selected_num].dropna(), kde=True, ax=axes[0], color="seagreen")
        axes[0].set_title(f"{selected_num} – Histogram")
        sns.boxplot(x=df_prep[selected_num], ax=axes[1], color="lightgreen")
        axes[1].set_title(f"{selected_num} – Boxplot")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            f"""
            **Interpretation of {selected_num} plots**

            - The **histogram** shows how frequent each range of values appears.  
            - The **boxplot** shows the median (line inside the box), the spread (box width), and potential outliers (points).  
            - Skewed histograms or many outliers may indicate that the variable needs transformation or careful discussion.
            """
        )
    else:
        st.info("No numeric columns found in the preprocessed dataset.")

    # 3. Encoded categorical
    st.subheader("3. Encoded Categorical Feature Summary")
    if categorical_cols_present:
        st.markdown(
            """
            These plots describe the **frequency** of categories after encoding.
            Use them to report how many samples belong to each location, polymer type, risk level, etc.
            """
        )
        for col in categorical_cols_present:
            vc = get_value_counts_for_column(df_prep, col)
            with st.expander(f"Distribution for {col}"):
                st.dataframe(vc, use_container_width=True)

                if len(vc) > 20:
                    vc_plot = vc.head(20)
                    st.caption("Showing top 20 categories by frequency.")
                else:
                    vc_plot = vc

                plot_value_counts_bar(vc_plot, x_col=col, title=f"{col} Encoded Distribution")

                st.markdown(
                    f"""
                    **How to interpret {col} distribution**

                    - Each bar corresponds to an encoded category.  
                    - Taller bars mean that category appears more frequently in the dataset.  
                    - You can describe which categories dominate (for example, most samples from a certain location or polymer type).
                    """
                )
    else:
        st.info("No categorical/encoded columns found.")

    # 4. Missing values
    st.subheader("4. Missing Value Assessment")
    total_missing = int(df_prep.isna().sum().sum())
    if total_missing == 0:
        st.success("No missing values detected. Data is fully ready for modeling.")
    else:
        st.warning(f"There are {total_missing} missing values left.")
        st.dataframe(df_prep.isna().sum().to_frame("missing_count"))
        st.caption(
            "Columns with remaining missing values may require additional imputation or careful interpretation."
        )

    st.info(
        "Proceed to **'4. Predictive Modeling & Validation'** using the sidebar navigation "
        "to train and compare classification models."
    )
    card_close()

# =========================
# 4. Predictive Modeling & Validation
# =========================
elif selected_tab == tabs[4]:
    card_open()
    st.header("Step 4 – Predictive Modeling & Validation")

    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please complete preprocessing in **Step 2** first.")
        card_close()
        st.stop()

    if "Risk_Type" not in df.columns or "Risk_Level" not in df.columns:
        st.warning("Required target columns 'Risk_Type' and 'Risk_Level' not found in the dataset.")
        card_close()
        st.stop()

    st.markdown(
        """
        **Purpose of this step**

        - Build **classification models** to predict:  
          - `Risk_Type` (e.g., ecological, human health, etc.)  
          - `Risk_Level` (e.g., low, medium, high)  
        - Evaluate their performance using **accuracy**, **precision**, **recall**, **F1-score**, and **cross-validation**.  
        - Select the **best model** and generate predictions for visualization in Step 5.
        """
    )

    # ---- Handle missing targets: keep only rows with valid Risk_Type and Risk_Level for training ----
    df_model = df.dropna(subset=["Risk_Type", "Risk_Level"]).copy()
    if df_model.empty:
        st.error(
            "All rows have missing values in Risk_Type and/or Risk_Level. "
            "The model cannot be trained. Please check your dataset."
        )
        card_close()
        st.stop()

    # Features and targets for training / validation
    X_model = df_model.drop(columns=["Risk_Type", "Risk_Level"], errors="ignore")
    y_type = df_model["Risk_Type"]
    y_level = df_model["Risk_Level"]

    X_model = X_model.select_dtypes(include=[np.number]).fillna(0)

    X_train, X_test, y_train_type, y_test_type = train_test_split(
        X_model, y_type, test_size=0.2, random_state=42
    )
    _, _, y_train_level, y_test_level = train_test_split(
        X_model, y_level, test_size=0.2, random_state=42
    )

    st.subheader("Train–Test Split (rows with non-missing targets)")
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"X_test shape: {X_test.shape}")
    st.caption(
        "Around 80% of the rows with valid Risk_Type and Risk_Level are used for training and 20% for testing, "
        "which allows us to evaluate how well the model generalizes."
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    model_tabs = st.tabs(models.keys())
    cv_mean_scores = {}

    for (model_name, model), tab_model in zip(models.items(), model_tabs):
        with tab_model:
            st.subheader(f"Model: {model_name}")

            # --- Risk_Type ---
            model_t = clone(model)
            model_t.fit(X_train, y_train_type)
            pred_type = model_t.predict(X_test)

            acc_t = accuracy_score(y_test_type, pred_type)
            prec_t = precision_score(y_test_type, pred_type, average="weighted", zero_division=0)
            rec_t = recall_score(y_test_type, pred_type, average="weighted", zero_division=0)
            f1_t = f1_score(y_test_type, pred_type, average="weighted", zero_division=0)

            st.markdown("### Performance on Risk_Type (Test Set)")
            df_metrics_type = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision (weighted)",
                        "Recall (weighted)",
                        "F1-score (weighted)",
                    ],
                    "Score": [acc_t, prec_t, rec_t, f1_t],
                }
            )
            st.table(df_metrics_type.set_index("Metric"))

            st.caption(
                """
                - **Accuracy**: overall percentage of correct predictions.  
                - **Precision**: when the model predicts a given risk type, how often is it correct?  
                - **Recall**: how many of the true risk types are captured by the model?  
                - **F1-score**: balance between precision and recall (useful when classes are imbalanced).
                """
            )

            # --- Risk_Level ---
            model_l = clone(model)
            model_l.fit(X_train, y_train_level)
            pred_level = model_l.predict(X_test)

            acc_l = accuracy_score(y_test_level, pred_level)
            prec_l = precision_score(y_test_level, pred_level, average="weighted", zero_division=0)
            rec_l = recall_score(y_test_level, pred_level, average="weighted", zero_division=0)
            f1_l = f1_score(y_test_level, pred_level, average="weighted", zero_division=0)

            st.markdown("### Performance on Risk_Level (Test Set)")
            df_metrics_level = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision (weighted)",
                        "Recall (weighted)",
                        "F1-score (weighted)",
                    ],
                    "Score": [acc_l, prec_l, rec_l, f1_l],
                }
            )
            st.table(df_metrics_level.set_index("Metric"))

            st.caption(
                """
                Ideally, you want **high and balanced values** across accuracy, precision, recall,
                and F1-score. Very high accuracy but low recall might indicate that the model
                is ignoring minority risk classes.
                """
            )

            # --- Cross Validation on Risk_Type ---
            st.markdown("### K-Fold Cross Validation (Risk_Type)")
            try:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clone(model), X_model, y_type, cv=kf, scoring="accuracy")
                cv_mean = cv_scores.mean()
                cv_mean_scores[model_name] = cv_mean

                st.write(f"CV Scores: {cv_scores}")
                st.write(f"Mean CV accuracy: {cv_mean:.3f} ± {cv_scores.std():.3f}")
                st.bar_chart(cv_scores)

                st.caption(
                    """
                    Cross-validation repeats training and testing on multiple folds of the data.
                    A **high mean CV accuracy with small variation** indicates a **stable and robust** model.
                    """
                )
            except Exception as e:
                st.error(f"Cross-validation failed: {e}")

    # Choose best model by mean CV accuracy and generate predictions for full dataset
    if cv_mean_scores:
        best_model_name = max(cv_mean_scores, key=cv_mean_scores.get)
        best_model = models[best_model_name]

        st.session_state.best_model_name = best_model_name

        st.markdown("---")
        st.success(
            f"📌 Best model based on CV accuracy: **{best_model_name}**. "
            "This model will be used to generate predicted Risk_Type and Risk_Level."
        )

        st.markdown(
            """
            The selected model has the **highest average cross-validated accuracy** and is therefore
            considered the most reliable for predicting microplastic risk categories in this dataset.
            """
        )

        # ---- Train on all rows with valid targets ----
        best_model_type = clone(best_model).fit(X_model, y_type)
        best_model_level = clone(best_model).fit(X_model, y_level)

        # ---- Predict for ALL rows (including those with missing targets) ----
        X_all = df.drop(columns=["Risk_Type", "Risk_Level"], errors="ignore")
        X_all = X_all.select_dtypes(include=[np.number]).fillna(0)

        df_with_preds = df.copy()
        df_with_preds["Pred_Risk_Type"] = best_model_type.predict(X_all)
        df_with_preds["Pred_Risk_Level"] = best_model_level.predict(X_all)

        st.session_state.df_with_preds = df_with_preds

        st.info(
            "Predicted **Risk_Type** and **Risk_Level** are now stored and can be visualized "
            "in **Step 5 – Risk Visualizations & Interpretation** to show classification performance."
        )
    else:
        st.warning("Could not compute cross-validation scores; predictions for Step 5 were not generated.")

    card_close()

# =========================
# 5. Risk Visualizations & Interpretation
# =========================
elif selected_tab == tabs[5]:
    card_open()
    st.header("Step 5 – Risk Visualizations & Interpretation")

    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first in **Step 2**.")
        card_close()
        st.stop()

    # If predictions exist, use df_with_preds; otherwise, just use df
    df_vis = st.session_state.df_with_preds if st.session_state.df_with_preds is not None else df
    best_model_name = st.session_state.best_model_name

    if st.session_state.df_with_preds is None:
        st.warning(
            "There are no stored predictions yet. Run **Step 4 – Predictive Modeling & Validation** "
            "to generate predicted Risk_Type and Risk_Level."
        )
    else:
        st.success(f"Using predictions from best model: **{best_model_name}**.")

    st.markdown(
        """
        **Purpose of this step**

        - Provide **visual evidence** of microplastic risk patterns.  
        - Show how the **predicted risk types and levels** compare to the **actual labels**.  
        - These plots are ideal for the **Results and Discussion** chapter (figures plus narrative).
        """
    )

    # Always show all options
    vis_options = [
        "Risk Score Distribution",
        "Risk Score vs MP_Count_per_L",
        "Risk Score by Risk Level (Actual)",
        "Class Distribution (Risk_Type & Risk_Level – Actual)",
        "Predicted vs Actual Risk_Type",
        "Predicted vs Actual Risk_Level",
    ]

    vis_choice = st.sidebar.selectbox("Visualization type:", vis_options)

    # ---------- Risk Score Distribution ----------
    if vis_choice == "Risk Score Distribution":
        st.subheader("Risk Score Distribution")
        if "Risk_Score" in df_vis.columns:
            rs = df_vis["Risk_Score"].dropna()
            fig, ax = plt.subplots()
            sns.histplot(rs, kde=True, ax=ax, color="seagreen")
            ax.set_xlabel("Risk_Score")
            ax.set_title("Distribution of Risk_Score")
            st.pyplot(fig)
            plt.close(fig)

            mean_rs = rs.mean()
            med_rs = rs.median()
            min_rs = rs.min()
            max_rs = rs.max()

            st.markdown(
                f"""
                **Interpretation**

                - Risk scores range from **{min_rs:.2f}** to **{max_rs:.2f}**, with an average value of about **{mean_rs:.2f}**  
                  and a median of **{med_rs:.2f}**.  
                - The overall shape of the distribution indicates whether most sampling sites are concentrated at low,
                  moderate, or high risk levels.  
                - A long right tail (many values on the far right) suggests a few **very high–risk** locations compared to the rest.
                """
            )
        else:
            st.warning("Risk_Score column not found.")

    # ---------- Risk Score vs MP_Count_per_L ----------
    elif vis_choice == "Risk Score vs MP_Count_per_L":
        st.subheader("Risk Score vs MP_Count_per_L")
        if "Risk_Score" in df_vis.columns and "MP_Count_per_L" in df_vis.columns:
            sub = df_vis[["Risk_Score", "MP_Count_per_L"]].dropna()
            fig, ax = plt.subplots()
            ax.scatter(sub["Risk_Score"], sub["MP_Count_per_L"], alpha=0.7, c="seagreen")
            ax.set_xlabel("Risk_Score")
            ax.set_ylabel("MP_Count_per_L")
            ax.set_title("Risk Score vs Microplastic Count per Liter")
            st.pyplot(fig)
            plt.close(fig)

            corr = sub["Risk_Score"].corr(sub["MP_Count_per_L"])
            st.markdown(
                f"""
                **Interpretation**

                - Each point represents one sampling record (risk score vs. microplastic count per liter).  
                - The Pearson correlation between risk score and MP count is approximately **{corr:.2f}**.  
                  Values close to +1 indicate a strong positive relationship, values near 0 indicate little or no linear relationship.  
                - A positive correlation suggests that sites with higher microplastic counts tend to receive higher risk scores,
                  which supports the validity of the scoring system.
                """
            )
        else:
            st.warning("Required columns (Risk_Score, MP_Count_per_L) not found.")

    # ---------- Risk Score by Risk Level (Actual) ----------
    elif vis_choice == "Risk Score by Risk Level (Actual)":
        st.subheader("Risk Score by Risk Level (Actual)")
        if "Risk_Score" in df_vis.columns and "Risk_Level" in df_vis.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x="Risk_Level", y="Risk_Score", data=df_vis, ax=ax, palette="Greens")
            ax.set_title("Risk Score Distribution by Risk Level (Actual)")
            st.pyplot(fig)
            plt.close(fig)

            # Dynamic interpretation using actual level names
            try:
                medians = df_vis.groupby("Risk_Level")["Risk_Score"].median().sort_values()
                if not medians.empty:
                    lowest_level = medians.index[0]
                    highest_level = medians.index[-1]
                    lowest_med = medians.iloc[0]
                    highest_med = medians.iloc[-1]
                    st.markdown(
                        f"""
                        **Interpretation**

                        - The median risk score is lowest for **{lowest_level}** (≈ {lowest_med:.2f})  
                          and highest for **{highest_level}** (≈ {highest_med:.2f}).  
                        - This pattern indicates that the assigned risk levels are consistent with the numerical risk scores.  
                        - If the boxes for two levels (for example, *Medium* and *High*) overlap strongly, it suggests that the
                          boundary between those levels may not be very sharp in the data.
                        """
                    )
                else:
                    st.markdown(
                        """
                        **Interpretation**

                        - The risk score distribution by risk level cannot be summarized because no valid values were found.
                        """
                    )
            except Exception:
                st.markdown(
                    """
                    **Interpretation**

                    - The distribution of risk scores by risk level shows whether higher levels
                      generally correspond to higher scores.
                    """
                )
        else:
            st.warning("Required columns (Risk_Score, Risk_Level) not found.")

    # ---------- Class Distribution (Risk_Type & Risk_Level – Actual) ----------
    elif vis_choice == "Class Distribution (Risk_Type & Risk_Level – Actual)":
        st.subheader("Class Distributions – Actual")
        for target in ["Risk_Type", "Risk_Level"]:
            if target in df_vis.columns:
                vc = df_vis[target].value_counts()
                st.write(f"### {target} (Actual)")
                st.bar_chart(vc)

                total = vc.sum()
                if total > 0:
                    main_class = vc.idxmax()
                    main_count = vc.max()
                    main_pct = (main_count / total) * 100
                    st.markdown(
                        f"""
                        **Interpretation of {target}**

                        - The most frequent {target.lower()} in the dataset is **{main_class}**,  
                          with **{main_count}** records (≈ {main_pct:.1f}% of all cases).  
                        - Classes with very few observations may be more difficult for the model to learn,
                          which can lead to lower accuracy for those specific risk types or levels.  
                        - In your discussion, you can highlight whether the dataset is dominated by
                          certain risk categories (for example, mostly *medium* or *high* risk).
                        """
                    )
                else:
                    st.markdown(
                        f"""
                        **Interpretation of {target}**

                        - No valid {target.lower()} values were found in the dataset.
                        """
                    )
            else:
                st.warning(f"{target} not found in dataset.")

    # ---------- Predicted vs Actual Risk_Type ----------
    elif vis_choice == "Predicted vs Actual Risk_Type":
        st.subheader("Predicted vs Actual Risk_Type")

        if "Pred_Risk_Type" in df_vis.columns and "Risk_Type" in df_vis.columns:
            # Only use rows where both actual and predicted are not missing
            mask_valid = df_vis[["Risk_Type", "Pred_Risk_Type"]].notna().all(axis=1)
            df_cm = df_vis.loc[mask_valid].copy()

            if df_cm.empty:
                st.warning("No rows with both actual and predicted Risk_Type. Cannot build confusion matrix.")
            else:
                cm = pd.crosstab(
                    df_cm["Risk_Type"],
                    df_cm["Pred_Risk_Type"],
                    rownames=["Actual"],
                    colnames=["Predicted"],
                )

                st.markdown("**Confusion Matrix (Counts)**")
                st.dataframe(cm)

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
                ax.set_title("Confusion Matrix – Risk_Type")
                st.pyplot(fig)
                plt.close(fig)

                # Dynamic interpretation
                try:
                    overall_acc = accuracy_score(df_cm["Risk_Type"], df_cm["Pred_Risk_Type"])
                    vc_actual = df_cm["Risk_Type"].value_counts()
                    vc_pred = df_cm["Pred_Risk_Type"].value_counts()
                    top_actual = vc_actual.idxmax() if not vc_actual.empty else None
                    top_pred = vc_pred.idxmax() if not vc_pred.empty else None

                    st.markdown(
                        f"""
                        **Interpretation**

                        - Considering only rows with valid labels, the overall accuracy for predicting **Risk_Type**
                          is approximately **{overall_acc:.2%}**.  
                        - The most common actual risk type is **{top_actual}**, while the most frequently predicted type is **{top_pred}**.  
                        - Cells along the diagonal of the confusion matrix (where *Actual = Predicted*) represent **correct classifications**.  
                        - Large values outside the diagonal indicate **systematic confusion** between specific risk types
                          (for example, ecological risk misclassified as human health risk), which you can discuss as model limitations.
                        """
                    )
                except Exception:
                    st.markdown(
                        """
                        **Interpretation**

                        - High values along the diagonal indicate that the model correctly identifies most risk types.  
                        - Off-diagonal cells represent misclassifications, which are important to discuss when assessing model reliability.
                        """
                    )

                # Distribution comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Actual Risk_Type Distribution (valid rows)**")
                    st.bar_chart(df_cm["Risk_Type"].value_counts())
                with col2:
                    st.markdown("**Predicted Risk_Type Distribution (valid rows)**")
                    st.bar_chart(df_cm["Pred_Risk_Type"].value_counts())

                st.markdown(
                    """
                    **Additional interpretation**

                    - If the predicted distribution is more concentrated in a single risk type than the actual distribution,
                      the model may be **over-predicting** that category.  
                    - This helps explain whether the classifier is conservative or biased toward particular risk types.
                    """
                )
        else:
            st.warning(
                "Columns `Risk_Type` and/or `Pred_Risk_Type` are missing. "
                "Make sure you have run **Step 4 – Predictive Modeling & Validation** successfully."
            )

    # ---------- Predicted vs Actual Risk_Level ----------
    elif vis_choice == "Predicted vs Actual Risk_Level":
        st.subheader("Predicted vs Actual Risk_Level")

        if "Pred_Risk_Level" in df_vis.columns and "Risk_Level" in df_vis.columns:
            # Only use rows where both actual and predicted are not missing
            mask_valid = df_vis[["Risk_Level", "Pred_Risk_Level"]].notna().all(axis=1)
            df_cm = df_vis.loc[mask_valid].copy()

            if df_cm.empty:
                st.warning("No rows with both actual and predicted Risk_Level. Cannot build confusion matrix.")
            else:
                cm = pd.crosstab(
                    df_cm["Risk_Level"],
                    df_cm["Pred_Risk_Level"],
                    rownames=["Actual"],
                    colnames=["Predicted"],
                )

                st.markdown("**Confusion Matrix (Counts)**")
                st.dataframe(cm)

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
                ax.set_title("Confusion Matrix – Risk_Level")
                st.pyplot(fig)
                plt.close(fig)

                # Dynamic interpretation
                try:
                    overall_acc = accuracy_score(df_cm["Risk_Level"], df_cm["Pred_Risk_Level"])
                    vc_actual = df_cm["Risk_Level"].value_counts()
                    vc_pred = df_cm["Pred_Risk_Level"].value_counts()
                    top_actual = vc_actual.idxmax() if not vc_actual.empty else None
                    top_pred = vc_pred.idxmax() if not vc_pred.empty else None

                    st.markdown(
                        f"""
                        **Interpretation**

                        - Considering only rows with valid labels, the overall accuracy for predicting **Risk_Level**
                          is approximately **{overall_acc:.2%}**.  
                        - The most common actual level is **{top_actual}**, while the most frequently predicted level is **{top_pred}**.  
                        - A strong diagonal pattern (most counts on the diagonal) indicates that the model can distinguish
                          between **Low**, **Medium**, and **High** risk levels reasonably well.  
                        - If many *High* risk cases are predicted as *Medium*, the model tends to **underestimate high risk**,
                          which you should mention when discussing potential management implications.
                        """
                    )
                except Exception:
                    st.markdown(
                        """
                        **Interpretation**

                        - Correct predictions appear on the diagonal of the matrix, while off-diagonal cells represent misclassified levels.  
                        - Concentrated errors between two levels (for example, many *Medium* cases predicted as *Low*)
                          suggest that the boundary between these levels is not very clear for the model.
                        """
                    )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Actual Risk_Level Distribution (valid rows)**")
                    st.bar_chart(df_cm["Risk_Level"].value_counts())
                with col2:
                    st.markdown("**Predicted Risk_Level Distribution (valid rows)**")
                    st.bar_chart(df_cm["Pred_Risk_Level"].value_counts())

                st.markdown(
                    """
                    **Additional interpretation**

                    - Comparing the actual and predicted distributions shows whether the model is
                      **over-assigning** or **under-assigning** certain levels (for example, predicting too many *Medium* cases).  
                    - This helps you argue whether the classifier is conservative or aggressive in labeling high-risk situations.
                    """
                )
        else:
            st.warning(
                "Columns `Risk_Level` and/or `Pred_Risk_Level` are missing. "
                "Make sure you have run **Step 4 – Predictive Modeling & Validation** successfully."
            )

    card_close()

# Footer
st.markdown(
    '<div class="footer">Microplastic Risk Dashboard · Green Gradient Theme · Built with Streamlit</div>',
    unsafe_allow_html=True,
)
