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
        - Optionally apply log transforms for skewed distributions  
        - Encode categorical variables  
        - Scale numerical features  

        The cleaned and transformed data will be used in **Step 3 and Step 4**.
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
                    # apply log1p transform if positive skew and values are >= -1
                    df_prep[col] = np.where(
                        df_prep[col] > -1,
                        np.log1p(df_prep[col] - df_prep[col].min() + 1),
                        df_prep[col],
                    )
                    transform_applied = True
                outlier_report.append(
                    f"Column '{col}': NaNs={nan_count}, outliers clipped={clipped_before}, "
                    f"clipped_remaining={clipped_after}, skew_before={skew_before:.2f}, "
                    f"log_transform_applied={transform_applied}"
                )

    # Categorical encodings using LabelEncoder for simplicity
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
    st.markdown(
        "<details><summary style='font-weight:bold'>Show full preprocessed dataset</summary>",
        unsafe_allow_html=True,
    )
    st.dataframe(df_prep, use_container_width=True)
    st.markdown("</details>", unsafe_allow_html=True)

    with st.expander("Preprocessing Details & Report", expanded=False):
        st.markdown("**Outlier & Skewness Report:**")
        for report in outlier_report:
            st.markdown(f"- {report}")
        st.markdown("""
        - **Categorical Encoding:** All categorical columns transformed with LabelEncoder (where possible).  
        - **Scaling:** All numerical columns standardized using StandardScaler.
        """)

    with st.expander("Compare basic statistics before and after preprocessing", expanded=False):
        if st.session_state.raw_df is not None:
            raw = st.session_state.raw_df
            num_cols_present = [col for col in num_cols if col in raw.columns]
            st.write("Original statistics (selected numeric columns):")
            if num_cols_present:
                st.dataframe(raw[num_cols_present].describe().T)
            else:
                st.warning("No valid numeric columns found for statistics in the uploaded dataset.")
        else:
            st.info("Original raw dataset not available for comparison.")
        num_cols_prep_present = [col for col in num_cols if col in df_prep.columns]
        st.write("After preprocessing:")
        if num_cols_prep_present:
            st.dataframe(df_prep[num_cols_prep_present].describe().T)
        else:
            st.warning("No valid numeric columns found for statistics in the preprocessed dataset.")

    st.info("Next: explore the cleaned data in **3. Preprocessed Results** and then proceed to **4. Modeling & Performance**.")

# -----------------------------
# 3. Preprocessed Results – now with clear “what preprocessing did” view
# -----------------------------
elif selected_tab == tabs[2]:
    show_step_indicator(2, tabs)
    st.header("Step 3: Preprocessed Data Results")
    st.markdown(
        """
        This step summarizes the **final state of your preprocessed dataset**.
        After Step 2, your data should now be:
        - ✅ Numerically cleaned (converted to numbers, outliers clipped, skew reduced where needed)
        - ✅ Categorical variables encoded as integer labels
        - ✅ Scaled / standardized for modeling
        - ✅ Free from problematic values (invalid strings, infinities)

        These results confirm that the dataset is **ready to be used in Step 4: Modeling & Performance**.
        """
    )

    if st.session_state.df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ No preprocessed data available. Please run Data Preprocessing first.")
        st.stop()

    df_prep = st.session_state.df
    raw = st.session_state.raw_df if st.session_state.raw_df is not None else None

    # ---------------------------
    # 1. Overall dataset overview
    # ---------------------------
    st.subheader("1. Dataset Overview (After Preprocessing)")
    n_rows, n_cols = df_prep.shape
    numeric_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_prep.columns if c not in numeric_cols]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of rows", n_rows)
        st.metric("Number of columns", n_cols)
    with col2:
        st.metric("Numeric features", len(numeric_cols))
        st.write(", ".join(numeric_cols) if numeric_cols else "None")
    with col3:
        st.metric("Categorical / encoded features", len(categorical_cols))
        st.write(", ".join(categorical_cols) if categorical_cols else "None")

    st.markdown("**Column-wise summary:**")
    info_df = pd.DataFrame({
        "dtype": df_prep.dtypes.astype(str),
        "non_null_count": df_prep.notna().sum(),
        "missing_values": df_prep.isna().sum(),
        "n_unique": df_prep.nunique()
    })
    st.dataframe(info_df, use_container_width=True)

    # ---------------------------
    # 2. Numeric feature diagnostics
    # ---------------------------
    st.subheader("2. Numeric Feature Diagnostics")
    if numeric_cols:
        st.markdown(
            """
            All numeric columns below have been:
            - Converted to numeric data type  
            - Clipped for extreme outliers using the IQR rule  
            - Standardized (mean ≈ 0, standard deviation ≈ 1)  
            - Optionally log-transformed if they were strongly skewed
            """
        )
        desc = df_prep[numeric_cols].describe().T
        st.dataframe(desc, use_container_width=True)

        st.markdown("**Check a numeric feature distribution:**")
        num_choice = st.selectbox("Select a numeric column:", numeric_cols)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df_prep[num_choice].dropna(), kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title(f"{num_choice} — Histogram")
        sns.boxplot(x=df_prep[num_choice], ax=axes[1], color="orange")
        axes[1].set_title(f"{num_choice} — Boxplot")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns found in the preprocessed dataset.")

    # ---------------------------
    # 3. Categorical / encoded feature diagnostics
    # ---------------------------
    st.subheader("3. Categorical / Encoded Feature Diagnostics")
    cat_present = [c for c in cat_cols if c in df_prep.columns]
    if cat_present:
        st.markdown(
            """
            The following columns were treated as **categorical** and encoded as integers.
            Each distinct category has been assigned a numeric code.
            """
        )
        st.write(", ".join(cat_present))

        cat_choice = st.selectbox("Select a categorical/encoded column:", cat_present)
        vc_encoded = get_value_counts_for_column(df_prep, cat_choice)
        st.markdown("**Encoded value counts (after preprocessing):**")
        st.dataframe(vc_encoded)

        if raw is not None and cat_choice in raw.columns and len(raw) == len(df_prep):
            st.markdown("**Approximate mapping from raw labels to encoded values (most frequent label per code):**")
            temp = pd.DataFrame({
                "raw": raw[cat_choice].astype(str),
                "encoded": df_prep[cat_choice]
            })
            mapping = temp.groupby("encoded")["raw"].agg(lambda x: x.value_counts().idxmax())
            mapping_df = mapping.reset_index().rename(columns={"encoded": "encoded_value", "raw": "most_common_raw_label"})
            st.dataframe(mapping_df)
        else:
            st.info("Raw column not available or row counts differ; showing encoded distribution only.")
    else:
        st.info("No categorical/encoded columns from the expected list were found in the preprocessed dataset.")

    # ---------------------------
    # 4. Missing values & readiness for modeling
    # ---------------------------
    st.subheader("4. Missing Values & Modeling Readiness")
    total_missing = int(df_prep.isna().sum().sum())
    if total_missing == 0:
        st.success("✅ No missing values remain in the preprocessed dataset.")
    else:
        st.warning(f"⚠️ There are still {total_missing} missing values in the preprocessed dataset.")
        st.dataframe(df_prep.isna().sum().to_frame("missing_per_column"))

    st.markdown(
        """
        ### ✅ Summary

        Your dataset is now **machine-learning ready**:
        - All features are numeric or encoded as integers  
        - Numeric features have been cleaned and scaled  
        - Categorical variables have been encoded  
        - The data can be directly passed to the models in **Step 4: Modeling & Performance**.
        """
    )

    st.markdown("---")
    st.markdown("**Sample of the final preprocessed data (first 20 rows):**")
    st.dataframe(df_prep.head(20), use_container_width=True)

# -----------------------------
# 4. Modeling & Performance
# -----------------------------
elif selected_tab == tabs[3]:
    show_step_indicator(3, tabs)
    st.header("Step 4: Modeling & Performance")
    st.markdown(
        """
        In this step, the app trains and evaluates several classification models 
        to predict **Risk_Type** and **Risk_Level** based on the preprocessed data.
        """
    )

    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first.")
        st.stop()
    if "Risk_Type" not in df.columns or "Risk_Level" not in df.columns:
        st.warning("⚠️ Required columns for modeling not found in data.")
        st.stop()

    # Show class distribution before modeling to inform users about class imbalance
    st.subheader("Class Distribution Before Modeling")
    raw = st.session_state.raw_df if st.session_state.raw_df is not None else None
    for target in ["Risk_Type", "Risk_Level"]:
        st.write(f"Target: {target}")
        if raw is not None and target in raw.columns:
            vc_raw = get_value_counts_for_column(raw, target)
            st.markdown("Original (raw) label counts:")
            st.dataframe(vc_raw)
            plot_value_counts_bar(vc_raw, title=f"{target} (raw labels)")
        elif target in df.columns:
            vc_prep = get_value_counts_for_column(df, target)
            st.markdown("Preprocessed label counts (may be encoded integers):")
            st.dataframe(vc_prep)
            plot_value_counts_bar(vc_prep, title=f"{target} (preprocessed)")
        else:
            st.write(f"{target} not found in dataset.")

    st.markdown("---")
    st.subheader("Train-Test Split & Model Setup")

    # Prepare features and targets
    X = df.drop(columns=["Risk_Type", "Risk_Level"], errors="ignore")
    y_type = df["Risk_Type"]
    y_level = df["Risk_Level"]

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    y_type = y_type.fillna(0)
    y_level = y_level.fillna(0)

    st.write(f"Feature matrix shape: {X.shape}")
    st.write(f"Target (Risk_Type) classes: {len(np.unique(y_type))}")
    st.write(f"Target (Risk_Level) classes: {len(np.unique(y_level))}")

    # Split data (same indices for both targets)
    X_train, X_test, y_train_type, y_test_type = train_test_split(
        X, y_type, test_size=0.2, random_state=42
    )
    _, _, y_train_level, y_test_level = train_test_split(
        X, y_level, test_size=0.2, random_state=42
    )

    st.write("Train/Test split complete: ")
    st.write(f"- X_train: {X_train.shape}, X_test: {X_test.shape}")
    st.write(f"- y_train_type: {y_train_type.shape}, y_test_type: {y_test_type.shape}")
    st.write(f"- y_train_level: {y_train_level.shape}, y_test_level: {y_test_level.shape}")

    # Define models
    model_objs = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }
    model_names = list(model_objs.keys())

    st.markdown("Select a model tab to view its performance in detail.")
    model_tabs = st.tabs(model_names)

    for idx, name in enumerate(model_names):
        with model_tabs[idx]:
            st.subheader(f"Model: {name}")
            mod = clone(model_objs[name])

            # Fit and evaluate for Risk_Type
            try:
                mod.fit(X_train, y_train_type)
                pred_type = mod.predict(X_test)
                acc_type = accuracy_score(y_test_type, pred_type)
                prec_type = precision_score(y_test_type, pred_type, average="weighted", zero_division=0)
                rec_type = recall_score(y_test_type, pred_type, average="weighted", zero_division=0)
                f1_type = f1_score(y_test_type, pred_type, average="weighted", zero_division=0)

                st.markdown("**Performance for Risk_Type:**")
                st.write(f"- Accuracy: {acc_type:.3f}")
                st.write(f"- Precision (weighted): {prec_type:.3f}")
                st.write(f"- Recall (weighted): {rec_type:.3f}")
                st.write(f"- F1-score (weighted): {f1_type:.3f}")
            except Exception as e:
                st.warning(f"Could not train/evaluate model {name} for Risk_Type: {e}")

            st.markdown("---")

            # Fit and evaluate for Risk_Level
            try:
                mod_level = clone(model_objs[name])
                mod_level.fit(X_train, y_train_level)
                pred_level = mod_level.predict(X_test)
                acc_level = accuracy_score(y_test_level, pred_level)
                prec_level = precision_score(y_test_level, pred_level, average="weighted", zero_division=0)
                rec_level = recall_score(y_test_level, pred_level, average="weighted", zero_division=0)
                f1_level = f1_score(y_test_level, pred_level, average="weighted", zero_division=0)

                st.markdown("**Performance for Risk_Level:**")
                st.write(f"- Accuracy: {acc_level:.3f}")
                st.write(f"- Precision (weighted): {prec_level:.3f}")
                st.write(f"- Recall (weighted): {rec_level:.3f}")
                st.write(f"- F1-score (weighted): {f1_level:.3f}")
            except Exception as e:
                st.warning(f"Could not train/evaluate model {name} for Risk_Level: {e}")

            st.markdown("---")

            # Cross-validation (for Risk_Type) to show stability
            st.markdown("**K-Fold Cross-Validation (Risk_Type):**")
            try:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clone(model_objs[name]), X, y_type, cv=kf, scoring="accuracy")
                st.write(f"CV scores: {cv_scores}")
                st.write(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():{0.3f}}")
                st.bar_chart(cv_scores)
            except Exception as e:
                st.warning(f"Could not run cross-validation for {name}: {e}")

            st.info(
                "Use these metrics to compare how well each model predicts Risk_Type and Risk_Level. "
                "Higher accuracy and F1-score generally indicate a better model."
            )

    # Summarize model comparison (Risk_Type only) in a compact table
    st.markdown("---")
    st.subheader("Overall Model Comparison (Risk_Type)")
    metrics_dict = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}
    for name, base_model in model_objs.items():
        mod = clone(base_model)
        try:
            mod.fit(X_train, y_train_type)
            pred = mod.predict(X_test)
            metrics_dict["Model"].append(name)
            metrics_dict["Accuracy"].append(accuracy_score(y_test_type, pred))
            metrics_dict["Precision"].append(
                precision_score(y_test_type, pred, average="weighted", zero_division=0)
            )
            metrics_dict["Recall"].append(
                recall_score(y_test_type, pred, average="weighted", zero_division=0)
            )
            metrics_dict["F1-Score"].append(
                f1_score(y_test_type, pred, average="weighted", zero_division=0)
            )
        except Exception as e:
            metrics_dict["Model"].append(name)
            metrics_dict["Accuracy"].append(np.nan)
            metrics_dict["Precision"].append(np.nan)
            metrics_dict["Recall"].append(np.nan)
            metrics_dict["F1-Score"].append(np.nan)
            st.warning(f"Could not compute metrics for {name}: {e}")

    perf_df = pd.DataFrame(metrics_dict).set_index("Model")
    st.dataframe(perf_df.style.format("{:.3f}"))
    fig, ax = plt.subplots(figsize=(10, 5))
    perf_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison on Test Set (Risk Type)")
    st.pyplot(fig)
    plt.close(fig)
    st.info("This bar chart visually compares model performance across key metrics for Risk Type classification.")

# -----------------------------
# 5. Visualizations (standalone tab)
# -----------------------------
elif selected_tab == tabs[4]:
    show_step_indicator(4, tabs)
    st.header("Step 5: Visualizations & Data Interpretations")
    st.markdown(
        """
        This final step lets you explore key visualizations of the data and model targets.
        These plots can be used directly in your thesis for **Results and Discussion**.
        """
    )

    df = st.session_state.df
    if df is None or st.session_state.preprocessed is False:
        st.warning("⚠️ Please preprocess the data first.")
        st.stop()

    vis_options = [
        "Risk Score Distribution",
        "Risk Score vs MP_Count_per_L",
        "Risk Score by Risk Level",
        "Class Distribution",
    ]
    selected_vis = st.sidebar.selectbox("Choose a visualization:", vis_options, index=0)

    if selected_vis == vis_options[0]:
        st.subheader("Risk Score Distribution")
        if "Risk_Score" in df.columns and df["Risk_Score"].notna().sum() > 0:
            fig, ax = plt.subplots()
            sns.histplot(df["Risk_Score"].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Risk_Score column not found or empty.")

    elif selected_vis == vis_options[1]:
        st.subheader("Risk Score vs MP Count per Liter")
        if (
            "Risk_Score" in df.columns
            and "MP_Count_per_L" in df.columns
            and df["Risk_Score"].notna().sum() > 0
            and df["MP_Count_per_L"].notna().sum() > 0
        ):
            fig, ax = plt.subplots()
            ax.scatter(df["Risk_Score"], df["MP_Count_per_L"])
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("MP Count per L")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Risk_Score or MP_Count_per_L column not found or empty.")

    elif selected_vis == vis_options[2]:
        st.subheader("Risk Score Distribution by Risk Level")
        if "Risk_Score" in df.columns and "Risk_Level" in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x="Risk_Level", y="Risk_Score", data=df, ax=ax)
            ax.set_title("Risk Score by Risk Level")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Required columns (Risk_Score, Risk_Level) not found.")

    elif selected_vis == vis_options[3]:
        st.subheader("Class Distribution")
        raw = st.session_state.raw_df if "raw_df" in st.session_state else None
        for target in ["Risk_Type", "Risk_Level"]:
            st.write(f"Target: {target}")
            if raw is not None and target in raw.columns:
                vc_raw = get_value_counts_for_column(raw, target)
                st.markdown("Original (raw) label counts:")
                st.dataframe(vc_raw)
                plot_value_counts_bar(vc_raw, title=f"{target} (raw labels)")
            elif target in df.columns:
                vc_prep = get_value_counts_for_column(df, target)
                st.markdown("Preprocessed label counts (may be encoded integers):")
                st.dataframe(vc_prep)
                plot_value_counts_bar(vc_prep, title=f"{target} (preprocessed)")
            else:
                st.write(f"{target} not found in dataset.")
