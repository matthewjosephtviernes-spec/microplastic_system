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
# GLOBAL STYLING (GREEN THEME)
# =========================
st.markdown(
    """
    <style>
    /* Main background: soft green gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #e5ffe8 0, #f7fff9 40%, #ffffff 100%);
    }

    /* Sidebar background: darker green gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b5330 0%, #0f7b45 50%, #0b5330 100%);
        color: #f0fff6;
    }
    [data-testid="stSidebar"] * {
        color: #f0fff6 !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #06331c;
    }

    /* Section cards */
    .section-card {
        background: linear-gradient(135deg, #e7ffe9 0%, #f5fff7 50%, #ffffff 100%);
        padding: 1.5rem 1.8rem;
        border-radius: 1.2rem;
        box-shadow: 0 8px 20px rgba(0, 80, 40, 0.08);
        border: 1px solid rgba(10, 100, 60, 0.12);
        margin-bottom: 1.5rem;
    }

    /* Top horizontal navigation styled as green pill buttons
       Only target radios in the main content, not in the sidebar. */
    section.main div[role="radiogroup"] > label {
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        padding: 0.45rem 1.3rem;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #d6f7dc, #c2f1cf);
        border: 1px solid #7edb93;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.9rem;
        color: #064422 !important;
    }

    section.main div[role="radiogroup"] > label:hover {
        background: linear-gradient(135deg, #c1f2cd, #a9ebba);
    }

    section.main div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #0da95c, #0b7d44);
        color: #ffffff !important;
        border-color: #0b7d44;
        box-shadow: 0 4px 10px rgba(0, 80, 40, 0.35);
    }

    /* Metrics and small chips */
    .stMetric {
        background: linear-gradient(135deg, #e3ffe9, #f8fff9);
        border-radius: 0.9rem;
        padding: 0.3rem 0.7rem;
        box-shadow: 0 4px 10px rgba(0, 80, 40, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR: STEP PROGRESS
# =========================
def show_step_indicator(current_step_index: int, tabs_list):
    """Render a vertical step-by-step progress indicator in the sidebar."""
    st.sidebar.markdown("### Workflow Progress")
    for i, label in enumerate(tabs_list):
        if i < current_step_index:
            icon = "✅"
        elif i == current_step_index:
            icon = "🟢"
        else:
            icon = "⚪"
        st.sidebar.markdown(f"{icon} {label}")
    st.sidebar.markdown("---")


# =========================
# WORKFLOW NAVIGATION
# =========================
tabs = [
    "Overview / About the Study",
    "1. Data Upload & Description",
    "2. Data Preprocessing",
    "3. Preprocessed Data Results",
    "4. Predictive Modeling & Validation",
    "5. Risk Visualizations & Interpretation",
]

st.markdown("### Workflow Navigation")
selected_tab = st.radio(
    "Go to step:",
    tabs,
    horizontal=True,
    label_visibility="collapsed",
)

current_step_index = tabs.index(selected_tab)
show_step_indicator(current_step_index, tabs)

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False

# Expected columns
num_cols = ["MP_Count_per_L", "Risk_Score", "Microplastic_Size_mm_midpoint", "Density_midpoint"]
cat_cols = [
    "Location",
    "Shape",
    "Polymer_Type",
    "pH",
    "Salinity",
    "Industrial_Activity",
    "Population_Density",
    "Risk_Type",
    "Risk_Level",
    "Author",
]


# =========================
# HELPER FUNCTIONS
# =========================
def get_value_counts_for_column(df, column):
    """Return value counts as a clean DataFrame with unique column names."""
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    vc = df[column].value_counts(dropna=False)
    return pd.DataFrame({column: vc.index, "count": vc.values})


def plot_value_counts_bar(df_counts, x_col=None, y_col="count", title="Value Counts"):
    """Plot bar chart of value counts dataframe."""
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


# Helper to open and close a green gradient card
def card_open():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 0. Overview / About the Study
# =========================
if selected_tab == tabs[0]:
    card_open()
    st.header("Overview / About the Study")

    st.markdown(
        """
        This interactive dashboard implements the proposed **predictive risk modeling framework**
        for **microplastic pollution**.

        ### General Objective
        > To develop a predictive risk modeling framework for microplastic pollution using data mining techniques.

        ### How this app is structured:
        1. **Data Upload & Description** – Load the structured microplastic risk dataset derived from literature.  
        2. **Data Preprocessing** – Clean, transform, and encode the data (KDD preprocessing stage).  
        3. **Preprocessed Data Results** – Show what a *model-ready* dataset looks like.  
        4. **Predictive Modeling & Validation** – Train classification models and validate them with cross-validation.  
        5. **Risk Visualizations & Interpretation** – Visualize risk scores, categories, and distributions.
        """
    )

    st.info(
        "Start the workflow by going to **'1. Data Upload & Description'** in the navigation above. "
        "Each subsequent step depends on the previous one."
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
        In this step, you upload the **structured dataset** of microplastic pollution risk.
        This dataset is assumed to be the result of your **text mining / literature review** phase
        (extraction of risk information from journal articles and reports).

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

            # Keep raw copy in session and initialize working df
            st.session_state.raw_df = raw_df.copy()
            st.session_state.df = raw_df.copy()
            st.session_state.preprocessed = False

            st.success("✅ Dataset uploaded successfully!")

            # Basic description
            st.subheader("Dataset Description")
            rows, cols = raw_df.shape
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {rows}")
                st.write(f"**Columns:** {cols}")
            with col2:
                st.write("**Column names:**")
                st.write(list(raw_df.columns))

            st.subheader("Preview (First 10 Rows)")
            st.dataframe(raw_df.head(10), use_container_width=True)

            st.markdown(
                "<details><summary style='font-weight:bold'>Show full uploaded dataset</summary>",
                unsafe_allow_html=True,
            )
            st.dataframe(raw_df, use_container_width=True)
            st.markdown("</details>", unsafe_allow_html=True)

            st.info(
                "Next, go to **'2. Data Preprocessing'** using the navigation above "
                "to clean and transform the dataset for predictive modeling."
            )
        except Exception as e:
            st.error(f"Failed to read the uploaded file: {e}")
    card_close()

# =========================
# 2. Data Preprocessing
# =========================
elif selected_tab == tabs[2]:
    card_open()
    st.header("Step 2 – Data Preprocessing")

    st.markdown(
        """
        This step prepares your dataset for machine learning by:

        - Converting numeric columns to proper numeric types  
        - Handling missing values and outliers (IQR-based clipping)  
        - Applying log transforms for strongly skewed numeric features  
        - Encoding categorical variables into integer labels  
        - Standardizing numerical features (mean ≈ 0, std ≈ 1)  

        After this, the dataset becomes **model-ready**.
        """
    )

    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a dataset first in **Step 1 – Data Upload & Description**.")
        card_close()
        st.stop()

    df_prep = df.copy()
    outlier_report = []

    # Numeric conversions, outlier clipping, optional log transform
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
                    # Apply log1p transform if positive skew
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

    # Categorical encodings using LabelEncoder
    for col in cat_cols:
        if col in df_prep.columns:
            try:
                df_prep[col] = LabelEncoder().fit_transform(df_prep[col].astype(str))
            except Exception:
                pass

    # Scaling numeric columns
    scaler = StandardScaler()
    for col in num_cols:
        if col in df_prep.columns:
            try:
                df_prep[col] = scaler.fit_transform(df_prep[[col]])
            except Exception:
                pass

    # Save preprocessed into session
    st.session_state.df = df_prep
    st.session_state.preprocessed = True

    st.success("✅ Data preprocessing complete!")

    st.subheader("Preprocessed Dataset (First 10 Rows)")
    st.dataframe(df_prep.head(10), use_container_width=True)

    with st.expander("Preprocessing Log (Outliers & Transforms)"):
        if outlier_report:
            for line in outlier_report:
                st.markdown(f"- {line}")
        else:
            st.write("No numeric columns from the expected list were found or processed.")

    st.info(
        "Proceed to **'3. Preprocessed Data Results'** using the navigation above "
        "to inspect the final cleaned and model-ready dataset in more detail."
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
    raw = st.session_state.raw_df

    st.markdown(
        """
        This step summarizes the **final state of your preprocessed dataset**.  
        It shows that the data is now:

        - ✅ Numerically cleaned and standardized  
        - ✅ Categorical variables encoded as integers  
        - ✅ Free from invalid values and ready for modeling  
        """
    )

    # 1. Dataset Overview
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

    # 2. Numeric Summary
    st.subheader("2. Numeric Feature Summary")
    if numeric_cols_present:
        st.markdown("Descriptive statistics for all numeric features:")
        st.dataframe(df_prep[numeric_cols_present].describe(), use_container_width=True)

        # Single feature distribution
        st.markdown("**Inspect distribution of a selected numeric feature:**")
        selected_num = st.selectbox("Choose a numeric column:", numeric_cols_present)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df_prep[selected_num].dropna(), kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title(f"{selected_num} – Histogram")
        sns.boxplot(x=df_prep[selected_num], ax=axes[1], color="orange")
        axes[1].set_title(f"{selected_num} – Boxplot")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns found in the preprocessed dataset.")

    # 3. Encoded Categorical Summary
    st.subheader("3. Encoded Categorical Feature Summary")
    if categorical_cols_present:
        for col in categorical_cols_present:
            vc = get_value_counts_for_column(df_prep, col)
            with st.expander(f"Distribution for {col}"):
                st.dataframe(vc, use_container_width=True)
                plot_value_counts_bar(vc, x_col=col, title=f"{col} Encoded Distribution")
    else:
        st.info("No categorical/encoded columns found.")

    # 4. Missing Value Check
    st.subheader("4. Missing Value Assessment")
    total_missing = int(df_prep.isna().sum().sum())
    if total_missing == 0:
        st.success("No missing values detected. Data is fully ready for modeling.")
    else:
        st.warning(f"There are {total_missing} missing values left.")
        st.dataframe(df_prep.isna().sum().to_frame("missing_count"))

    st.markdown("---")
    st.info(
        "Next, go to **'4. Predictive Modeling & Validation'** using the navigation above "
        "to build and validate classification models on this dataset."
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
        In this step, we train **classification models** to predict:

        - **Risk_Type** (e.g., Ecological, Human health, etc.)  
        - **Risk_Level** (e.g., Low, Medium, High)  

        and we evaluate them using:

        - Accuracy  
        - Precision, Recall, F1-score (weighted)  
        - K-Fold Cross Validation (for robustness and generalizability)  
        """
    )

    # Prepare data
    X = df.drop(columns=["Risk_Type", "Risk_Level"], errors="ignore")
    y_type = df["Risk_Type"]
    y_level = df["Risk_Level"]

    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Train/test split
    X_train, X_test, y_train_type, y_test_type = train_test_split(
        X, y_type, test_size=0.2, random_state=42
    )
    _, _, y_train_level, y_test_level = train_test_split(
        X, y_level, test_size=0.2, random_state=42
    )

    st.subheader("Train–Test Split")
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"X_test shape: {X_test.shape}")
    st.write(f"y_train_type length: {len(y_train_type)}")
    st.write(f"y_test_type length: {len(y_test_type)}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    model_tabs = st.tabs(models.keys())

    for (model_name, model), tab_model in zip(models.items(), model_tabs):
        with tab_model:
            st.subheader(f"Model: {model_name}")

            # --- Risk_Type ---
            model_t = clone(model)
            model_t.fit(X_train, y_train_type)
            pred_type = model_t.predict(X_test)

            st.markdown("### Performance on Risk_Type")
            st.write("Accuracy:", accuracy_score(y_test_type, pred_type))
            st.write(
                "Precision:",
                precision_score(y_test_type, pred_type, average="weighted", zero_division=0),
            )
            st.write(
                "Recall:",
                recall_score(y_test_type, pred_type, average="weighted", zero_division=0),
            )
            st.write(
                "F1 Score:",
                f1_score(y_test_type, pred_type, average="weighted", zero_division=0),
            )

            # --- Risk_Level ---
            model_l = clone(model)
            model_l.fit(X_train, y_train_level)
            pred_level = model_l.predict(X_test)

            st.markdown("### Performance on Risk_Level")
            st.write("Accuracy:", accuracy_score(y_test_level, pred_level))
            st.write(
                "Precision:",
                precision_score(y_test_level, pred_level, average="weighted", zero_division=0),
            )
            st.write(
                "Recall:",
                recall_score(y_test_level, pred_level, average="weighted", zero_division=0),
            )
            st.write(
                "F1 Score:",
                f1_score(y_test_level, pred_level, average="weighted", zero_division=0),
            )

            # --- Cross Validation on Risk_Type ---
            st.markdown("### K-Fold Cross Validation (Risk_Type)")
            try:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clone(model), X, y_type, cv=kf, scoring="accuracy")
                st.write(f"CV Scores: {cv_scores}")
                st.write(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                st.bar_chart(cv_scores)
            except Exception as e:
                st.error(f"Cross-validation failed: {e}")

    st.info(
        "Use these metrics and model comparisons in the **Results & Discussion** section "
        "to justify which algorithm is most suitable for microplastic risk prediction."
    )
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

    st.markdown(
        """
        This final step focuses on **visualizing risk patterns** that can be used in your
        thesis **Results and Discussion** chapter.

        Choose a visualization type from the sidebar.
        """
    )

    vis_options = [
        "Risk Score Distribution",
        "Risk Score vs MP_Count_per_L",
        "Risk Score by Risk Level",
        "Class Distribution (Risk_Type & Risk_Level)",
    ]
    vis_choice = st.sidebar.selectbox("Choose visualization:", vis_options)

    if vis_choice == "Risk Score Distribution":
        st.subheader("Risk Score Distribution")
        if "Risk_Score" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df["Risk_Score"], kde=True, ax=ax)
            ax.set_xlabel("Risk_Score")
            ax.set_title("Distribution of Risk_Score")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Risk_Score column not found.")

    elif vis_choice == "Risk Score vs MP_Count_per_L":
        st.subheader("Risk Score vs MP_Count_per_L")
        if "Risk_Score" in df.columns and "MP_Count_per_L" in df.columns:
            fig, ax = plt.subplots()
            ax.scatter(df["Risk_Score"], df["MP_Count_per_L"], alpha=0.7)
            ax.set_xlabel("Risk_Score")
            ax.set_ylabel("MP_Count_per_L")
            ax.set_title("Risk Score vs Microplastic Count per Liter")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Required columns (Risk_Score, MP_Count_per_L) not found.")

    elif vis_choice == "Risk Score by Risk Level":
        st.subheader("Risk Score by Risk Level")
        if "Risk_Score" in df.columns and "Risk_Level" in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x="Risk_Level", y="Risk_Score", data=df, ax=ax)
            ax.set_title("Risk Score Distribution by Risk Level")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Required columns (Risk_Score, Risk_Level) not found.")

    elif vis_choice == "Class Distribution (Risk_Type & Risk_Level)":
        st.subheader("Class Distributions")
        for target in ["Risk_Type", "Risk_Level"]:
            if target in df.columns:
                vc = df[target].value_counts()
                st.write(f"### {target}")
                st.bar_chart(vc)
            else:
                st.warning(f"{target} not found in dataset.")
    card_close()
