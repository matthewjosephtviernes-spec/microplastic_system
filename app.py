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
st.title("🧪 Microplastic Risk Analysis — Enhanced Interactive Dashboard")

st.sidebar.title("Navigation")
tabs = [
    "1. Upload & Preview",
    "2. Data Preprocessing",
    "3. Preprocessed Results",
    "4. Modeling & Performance",
    "5. Visualizations"
]
selected_tab = st.sidebar.radio("Go to step:", tabs)

# Step progress indicator for clarity
def show_step_indicator(current_step_index: int, tabs_list):
    """Render a simple step-by-step progress indicator at the top of each page."""
    st.markdown("### Workflow Progress")
    cols = st.columns(len(tabs_list))
    for i, label in enumerate(tabs_list):
        with cols[i]:
            if i < current_step_index:
                icon = "✅"
            elif i == current_step_index:
                icon = "🟢"
            else:
                icon = "⚪"
            st.markdown(
                f"<div style='text-align:center'>{icon}<br/><span style='font-size:0.8rem'>{label}</span></div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")  # visual separator before main content

# Session state for data
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False

# Columns expected (used throughout)
num_cols = ["MP_Count_per_L", "Risk_Score", "Microplastic_Size_mm_midpoint", "Density_midpoint"]
cat_cols = ["Location", "Shape", "Polymer_Type", "pH", "Salinity", "Industrial_Activity",
            "Population_Density", "Risk_Type", "Risk_Level", "Author"]

# Helper functions
def get_value_counts_for_column(df, column):
    """Utility to get value counts as a DataFrame."""
    if column in df.columns:
        vc = df[column].value_counts(dropna=False)
        return vc.reset_index().rename(columns={"index": column, column: "count"})
    else:
        return pd.DataFrame(columns=[column, "count"])

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

# -----------------------------
# 1. Upload & Preview
# -----------------------------
if selected_tab == tabs[0]:
    show_step_indicator(0, tabs)
    st.header("Step 1: Upload Your Dataset")
    st.markdown(
        """
        Upload your microplastic dataset in CSV or Excel format.
        After uploading, you’ll see a preview and can proceed to **Step 2: Data Preprocessing**.
        """
    )
    uploaded_file = st.file_uploader("Upload CSV or Excel Dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                raw_df = pd.read_excel(uploaded_file)
            # keep raw copy in session and initialize df
            st.session_state.raw_df = raw_df.copy()
            st.session_state.df = raw_df.copy()
            st.session_state.preprocessed = False
            st.success("✅ Dataset uploaded successfully! Preview below:")
            st.subheader("Dataset Preview (First 10 Rows)")
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.markdown(
                "<details><summary style='font-weight:bold'>Show full uploaded dataset</summary>",
                unsafe_allow_html=True,
            )
            st.dataframe(raw_df, use_container_width=True)
            st.markdown("</details>", unsafe_allow_html=True)
            st.info("Next: go to **2. Data Preprocessing** in the sidebar to clean and transform the data.")
        except Exception as e:
            st.error(f"Failed to read the uploaded file: {e}")

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
elif selected_tab == tabs[1]:
    show_step_indicator(1, tabs)
    st.header("Step 2: Data Preprocessing")
    st.markdown(
        """
        In this step, the app will:
        - Convert numeric columns to numeric types  
        - Handle missing values and outliers  
        - Optionally apply log transforms for strongly skewed distributions  
        - Encode categorical variables  
        - Scale numerical features  

        After this, your dataset becomes *machine-learning ready*.
        """
    )

    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a dataset in Step 1 first.")
        st.stop()

    df_prep = df.copy()
    outlier_report = []

    # Numeric conversions, outlier clipping and optional log transform for skew
    for col in num_cols:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
            nan_count = df_prep[col].isna().sum()
            if df_prep[col].notna().sum() > 0:
                q1 = df_prep[col].quantile(0.25)
                q3 = df_prep[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                clipped_before = ((df_prep[col] < lower) | (df_prep[col] > upper)).sum()
                df_prep[col] = df_prep[col].clip(lower=lower, upper=upper)
                clipped_after = ((df_prep[col] < lower) | (df_prep[col] > upper)).sum()
                skew_before = df_prep[col].skew()
                transform_applied = False
                if skew_before > 1:
                    # apply log1p transform if positive skew
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

    st.markdown("---")
    st.info("Proceed to Step 3 to view the full results of preprocessing.")

# -----------------------------
# 3. Preprocessed Results – final cleaned & modeling-ready dataset
# -----------------------------
elif selected_tab == tabs[2]:
    show_step_indicator(2, tabs)
    st.header("Step 3: Preprocessed Data Results")

    if st.session_state.df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first.")
        st.stop()

    df_prep = st.session_state.df
    raw = st.session_state.raw_df

    st.markdown(
        """
        The dataset is now **fully preprocessed** and ready for machine learning.
        
        ### ✔ What This Means:
        - All numeric columns have been cleaned, clipped for outliers, and standardized  
        - Strongly skewed numeric columns were log-transformed  
        - Categorical variables are now encoded as integers  
        - No invalid values remain  
        """
    )

    # ---------------------------
    # 1. Dataset Overview
    # ---------------------------
    st.subheader("1. Dataset Overview After Preprocessing")

    n_rows, n_cols = df_prep.shape
    numeric_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_prep.columns if c not in numeric_cols]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", n_rows)
    col2.metric("Columns", n_cols)
    col3.metric("Numeric Features", len(numeric_cols))

    st.dataframe(df_prep.head(20), use_container_width=True)

    # ---------------------------
    # 2. Numeric Summary
    # ---------------------------
    st.subheader("2. Numeric Feature Summary")
    if numeric_cols:
        st.dataframe(df_prep[numeric_cols].describe(), use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # ---------------------------
    # 3. Encoded Categorical Summary
    # ---------------------------
    st.subheader("3. Encoded Categorical Feature Summary")
    for col in categorical_cols:
        vc = get_value_counts_for_column(df_prep, col)
        with st.expander(f"Distribution for {col}"):
            st.dataframe(vc)
            plot_value_counts_bar(vc, x_col=col, title=f"{col} Encoded Distribution")

    # ---------------------------
    # 4. Missing Value Check
    # ---------------------------
    st.subheader("4. Missing Value Assessment")
    total_missing = df_prep.isna().sum().sum()
    if total_missing == 0:
        st.success("No missing values detected. Data is fully ready for modeling.")
    else:
        st.warning(f"There are {total_missing} missing values left.")
        st.dataframe(df_prep.isna().sum().to_frame("missing_count"))

    st.markdown("---")
    st.info("Proceed to **Step 4: Modeling & Performance** to train classification models.")

# -----------------------------
# 4. Modeling & Performance
# -----------------------------
elif selected_tab == tabs[3]:
    show_step_indicator(3, tabs)
    st.header("Step 4: Modeling & Performance")

    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first.")
        st.stop()

    if "Risk_Type" not in df.columns or "Risk_Level" not in df.columns:
        st.warning("Required target columns not found.")
        st.stop()

    # Prepare data
    X = df.drop(columns=["Risk_Type", "Risk_Level"], errors="ignore")
    y_type = df["Risk_Type"]
    y_level = df["Risk_Level"]

    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Split
    X_train, X_test, y_train_type, y_test_type = train_test_split(
        X, y_type, test_size=0.2, random_state=42
    )
    _, _, y_train_level, y_test_level = train_test_split(
        X, y_level, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    tabs_models = st.tabs(models.keys())

    for (model_name, model), tab_model in zip(models.items(), tabs_models):
        with tab_model:

            st.subheader(f"{model_name}")

            # Risk_Type
            model_t = clone(model)
            model_t.fit(X_train, y_train_type)
            pred_type = model_t.predict(X_test)

            st.markdown("### Performance on Risk_Type")
            st.write("Accuracy:", accuracy_score(y_test_type, pred_type))
            st.write("Precision:", precision_score(y_test_type, pred_type, average="weighted", zero_division=0))
            st.write("Recall:", recall_score(y_test_type, pred_type, average="weighted", zero_division=0))
            st.write("F1 Score:", f1_score(y_test_type, pred_type, average="weighted", zero_division=0))

            # Risk_Level
            model_l = clone(model)
            model_l.fit(X_train, y_train_level)
            pred_level = model_l.predict(X_test)

            st.markdown("### Performance on Risk_Level")
            st.write("Accuracy:", accuracy_score(y_test_level, pred_level))
            st.write("Precision:", precision_score(y_test_level, pred_level, average="weighted", zero_division=0))
            st.write("Recall:", recall_score(y_test_level, pred_level, average="weighted", zero_division=0))
            st.write("F1 Score:", f1_score(y_test_level, pred_level, average="weighted", zero_division=0))

            # Cross Validation
            st.markdown("### 5-Fold Cross Validation (Risk_Type)")
            try:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clone(model), X, y_type, cv=kf, scoring="accuracy")

                st.write(f"CV Scores: {cv_scores}")
                st.write(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

                st.bar_chart(cv_scores)
            except Exception as e:
                st.error(f"Cross-validation failed: {e}")

# -----------------------------
# 5. Visualizations
# -----------------------------
elif selected_tab == tabs[4]:
    show_step_indicator(4, tabs)
    st.header("Step 5: Visualizations & Data Interpretations")

    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first.")
        st.stop()

    st.markdown("Select a visualization from the sidebar.")

    vis_options = [
        "Risk Score Distribution",
        "Risk Score vs MP_Count_per_L",
        "Risk Score by Risk Level",
        "Class Distribution",
    ]
    vis_choice = st.sidebar.selectbox("Choose visualization:", vis_options)

    if vis_choice == "Risk Score Distribution":
        st.subheader("Risk Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Risk_Score"], kde=True, ax=ax)
        st.pyplot(fig)

    elif vis_choice == "Risk Score vs MP_Count_per_L":
        st.subheader("Risk Score vs MP_Count_per_L")
        fig, ax = plt.subplots()
        ax.scatter(df["Risk_Score"], df["MP_Count_per_L"])
        st.pyplot(fig)

    elif vis_choice == "Risk Score by Risk Level":
        st.subheader("Risk Score by Risk Level")
        fig, ax = plt.subplots()
        sns.boxplot(x="Risk_Level", y="Risk_Score", data=df, ax=ax)
        st.pyplot(fig)

    elif vis_choice == "Class Distribution":
        st.subheader("Class Distributions")
        for target in ["Risk_Type", "Risk_Level"]:
            vc = df[target].value_counts()
            st.write(f"### {target}")
            st.bar_chart(vc)
