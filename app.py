import os
import io
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif


# ---------------------------
# Utility functions
# ---------------------------

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop rows with missing targets, fill other missing values."""
    df = df.copy()

    # Drop rows where targets are missing
    target_cols = ["Risk_Level", "Risk_Type"]
    existing_targets = [c for c in target_cols if c in df.columns]
    if existing_targets:
        df = df.dropna(subset=existing_targets)

    # Simple fill for remaining missing values
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


@st.cache_data
def preprocess_and_select_features(
    df: pd.DataFrame,
    top_n_features: int = 20,
    fs_target_col: str = "Risk_Level",
):
    """
    Full preprocessing:
    - basic cleaning
    - one-hot encode categoricals
    - scale numeric columns
    - mutual information feature selection

    Returns:
      X (reduced feature matrix),
      y_level, y_type,
      selected_features, scaler, mi_scores_df
    """
    df = basic_cleaning(df)

    # Targets
    if "Risk_Level" not in df.columns or "Risk_Type" not in df.columns:
        raise ValueError("Dataset must contain 'Risk_Level' and 'Risk_Type' columns.")

    y_level = df["Risk_Level"]
    y_type = df["Risk_Type"]

    # Feature matrix (drop targets)
    X = df.drop(columns=["Risk_Level", "Risk_Type"])

    # Identify categorical & numeric features
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale numeric columns
    scaler = StandardScaler()
    num_cols_encoded = [c for c in X_encoded.columns if c in num_cols]
    if num_cols_encoded:
        X_encoded[num_cols_encoded] = scaler.fit_transform(X_encoded[num_cols_encoded])

    # Mutual information for feature selection (use Risk_Level by default)
    if fs_target_col == "Risk_Level":
        fs_target = y_level
    else:
        fs_target = y_type

    mi = mutual_info_classif(X_encoded, fs_target, random_state=42)
    mi_scores_df = (
        pd.DataFrame({"feature": X_encoded.columns, "mi_score": mi})
        .sort_values(by="mi_score", ascending=False)
        .reset_index(drop=True)
    )

    # Select top N features
    top_n_features = min(top_n_features, len(mi_scores_df))
    selected_features = mi_scores_df["feature"].head(top_n_features).tolist()
    X_reduced = X_encoded[selected_features].copy()

    return X_reduced, y_level, y_type, selected_features, scaler, mi_scores_df


def split_data(
    X: pd.DataFrame,
    y_level: pd.Series,
    y_type: pd.Series,
    test_size: float = 0.3,
    val_size: float = 0.5,
):
    """
    Create train/val/test splits for both targets in a consistent way.

    If stratified split by y_level fails (e.g. very small classes),
    it falls back to a non-stratified split.
    """
    # ---------- 1st split: train vs temp (val+test) ----------
    try:
        X_train, X_temp, y_level_train, y_level_temp, y_type_train, y_type_temp = train_test_split(
            X,
            y_level,
            y_type,
            test_size=test_size,
            random_state=42,
            stratify=y_level,   # try stratified
        )
    except ValueError:
        # Fall back: no stratification
        X_train, X_temp, y_level_train, y_level_temp, y_type_train, y_type_temp = train_test_split(
            X,
            y_level,
            y_type,
            test_size=test_size,
            random_state=42,
            stratify=None,
        )

    # ---------- 2nd split: temp into val vs test ----------
    try:
        X_val, X_test, y_level_val, y_level_test, y_type_val, y_type_test = train_test_split(
            X_temp,
            y_level_temp,
            y_type_temp,
            test_size=val_size,
            random_state=42,
            stratify=y_level_temp,  # try stratified again
        )
    except ValueError:
        # Fall back: no stratification
        X_val, X_test, y_level_val, y_level_test, y_type_val, y_type_test = train_test_split(
            X_temp,
            y_level_temp,
            y_type_temp,
            test_size=val_size,
            random_state=42,
            stratify=None,
        )

    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_level_train": y_level_train,
        "y_level_val": y_level_val,
        "y_level_test": y_level_test,
        "y_type_train": y_type_train,
        "y_type_val": y_type_val,
        "y_type_test": y_type_test,
    }

    return splits


def train_candidate_models(X_train, y_train, X_val, y_val):
    """
    Train several candidate models and automatically pick the best by F1-macro.
    Returns:
      best_model, best_name, results_df
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
        ),
    }

    rows = []
    best_model = None
    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        rows.append(
            {
                "model": name,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    results_df = pd.DataFrame(rows).sort_values(by="f1_macro", ascending=False)

    return best_model, best_name, results_df


def get_feature_importance(model, feature_names):
    """Return feature importance DataFrame if model supports it."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        df = (
            pd.DataFrame({"feature": feature_names, "importance": fi})
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )
        return df
    return None


def preprocess_single_input(sample_dict, df_template, selected_features, scaler):
    """
    Given a dict of user inputs, align with training preprocessing:
    - Create single-row DataFrame
    - One-hot encode using same columns
    - Scale numeric columns using fitted scaler
    - Select and order columns to match selected_features
    """
    df = df_template.copy().iloc[:0]  # empty frame with same columns
    df = pd.concat([df, pd.DataFrame([sample_dict])], ignore_index=True)

    # Basic cleaning like we did before
    df = basic_cleaning(df)

    # Separate targets if they exist in template
    drop_cols = [c for c in ["Risk_Level", "Risk_Type"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # One-hot encode using new data
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Re-align columns to training columns (selected_features)
    for col in selected_features:
        if col not in X_encoded.columns:
            X_encoded[col] = 0.0

    X_encoded = X_encoded[selected_features]

    # Scale numeric columns (intersection of numeric + selected_features)
    num_cols_sel = [c for c in selected_features if c in num_cols]
    if num_cols_sel:
        X_encoded[num_cols_sel] = scaler.transform(X_encoded[num_cols_sel])

    return X_encoded


# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.set_page_config(
        page_title="Microplastic Risk Modeling",
        page_icon="🧪",
        layout="wide",
    )

    # Simple CSS tweak
    st.markdown(
        """
        <style>
        .big-title { font-size: 2rem; font-weight: 700; }
        .sub-title { font-size: 1.2rem; opacity: 0.8; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Navigation")

    # CSV UPLOAD
    uploaded_file = st.sidebar.file_uploader(
        "Upload MicroPlastic CSV", type=["csv"]
    )

    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Data & EDA",
            "Preprocess & Feature Selection",
            "Model Training",
            "Model Validation",
            "Feature Importance",
            "Inference",
        ],
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file in the sidebar to begin.")
        return

    # Read the uploaded CSV
    try:
        df_raw = pd.read_csv(uploaded_file, encoding="latin1")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return

    # Preprocess and select features once
    try:
        X, y_level, y_type, selected_features, scaler, mi_scores_df = (
            preprocess_and_select_features(df_raw)
        )
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return

    # Split data
    splits = split_data(X, y_level, y_type)
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    y_level_train = splits["y_level_train"]
    y_level_val = splits["y_level_val"]
    y_level_test = splits["y_level_test"]
    y_type_train = splits["y_type_train"]
    y_type_val = splits["y_type_val"]
    y_type_test = splits["y_type_test"]

    # Train models for Risk_Level
    best_model_level, best_name_level, results_level_df = train_candidate_models(
        X_train, y_level_train, X_val, y_level_val
    )

    # Train models for Risk_Type
    best_model_type, best_name_type, results_type_df = train_candidate_models(
        X_train, y_type_train, X_val, y_type_val
    )

    # --------------------
    # Pages
    # --------------------

    if page == "Home":
        st.markdown(
            '<div class="big-title">🏠 Microplastic Risk Modeling</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sub-title">
            End-to-end pipeline: from uploaded microplastic data to Risk Level & Risk Type prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Pipeline Overview")
        st.markdown(
            """
            1. **Data Loading & Basic Cleaning** – upload CSV, handle missing values  
            2. **EDA & Risk Exploration** – visualize distributions & class balance  
            3. **Preprocess & Feature Engineering** – encode categoricals, scale numerics  
            4. **Feature Selection** – select top features using mutual information  
            5. **Model Training** – train multiple classifiers for `Risk_Level` & `Risk_Type`  
            6. **Model Validation** – compare models and evaluate on test set  
            7. **Feature Importance** – inspect which variables matter most  
            8. **Inference** – predict risk for new samples  
            """
        )

        st.markdown("### Dataset Snapshot")
        st.write(df_raw.head())

        st.markdown("### Target Variables")
        if "Risk_Level" in df_raw.columns and "Risk_Type" in df_raw.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.write("`Risk_Level` counts")
                st.write(df_raw["Risk_Level"].value_counts())
            with col2:
                st.write("`Risk_Type` counts")
                st.write(df_raw["Risk_Type"].value_counts())
        else:
            st.warning("Dataset is missing 'Risk_Level' and/or 'Risk_Type' columns.")

    elif page == "Data & EDA":
        st.header("📂 Data & EDA")

        st.subheader("Raw Data")
        st.write(df_raw.head())

        st.subheader("Basic Info")
        st.write(df_raw.describe(include="all"))

        st.subheader("Missing Values")
        st.write(df_raw.isna().sum())

        col1, col2 = st.columns(2)
        with col1:
            if "Risk_Score" in df_raw.columns:
                st.subheader("Risk_Score Distribution")
                st.bar_chart(df_raw["Risk_Score"].value_counts().sort_index())
        with col2:
            if "MP_Count_per_L" in df_raw.columns:
                st.subheader("MP_Count_per_L Distribution")
                st.bar_chart(df_raw["MP_Count_per_L"])

        if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
            st.subheader("Risk_Score by Risk_Level (summary)")
            st.write(
                df_raw.groupby("Risk_Level")["Risk_Score"]
                .describe()[["mean", "std", "min", "max"]]
            )

    elif page == "Preprocess & Feature Selection":
        st.header("🧹 Preprocess & Feature Selection")

        st.subheader("Cleaned Data Sample")
        st.write(basic_cleaning(df_raw).head())

        st.subheader("Mutual Information Scores (Top 20)")
        st.dataframe(mi_scores_df.head(20))

        st.subheader("Selected Features")
        st.write(selected_features)

        st.subheader("Shape")
        st.write(f"X shape after selection: {X.shape}")
        st.write(f"Number of samples: {X.shape[0]}")

    elif page == "Model Training":
        st.header("🤖 Model Training")

        st.subheader("Train / Validation / Test Sizes")
        st.write(f"Train: {X_train.shape[0]} rows")
        st.write(f"Validation: {X_val.shape[0]} rows")
        st.write(f"Test: {X_test.shape[0]} rows")

        st.subheader("Candidate Models for Risk_Level")
        st.dataframe(results_level_df)

        st.subheader("Candidate Models for Risk_Type")
        st.dataframe(results_type_df)

        st.success(
            f"Best for Risk_Level: **{best_name_level}** | "
            f"Best for Risk_Type: **{best_name_type}**"
        )

    elif page == "Model Validation":
        st.header("✅ Model Validation")

        st.subheader("Validation Performance – Risk_Level")
        st.dataframe(results_level_df)

        st.subheader("Validation Performance – Risk_Type")
        st.dataframe(results_type_df)

        st.markdown("---")

        st.subheader("Test Set Evaluation – Risk_Level")
        y_level_pred_test = best_model_level.predict(X_test)
        report_level = classification_report(
            y_level_test,
            y_level_pred_test,
            zero_division=0,
            output_dict=False,
        )
        st.text(report_level)

        st.subheader("Test Set Evaluation – Risk_Type")
        y_type_pred_test = best_model_type.predict(X_test)
        report_type = classification_report(
            y_type_test,
            y_type_pred_test,
            zero_division=0,
            output_dict=False,
        )
        st.text(report_type)

    elif page == "Feature Importance":
        st.header("🔬 Feature Importance")

        st.subheader(f"Best Model for Risk_Level: {best_name_level}")
        fi_level = get_feature_importance(best_model_level, selected_features)
        if fi_level is not None:
            st.write(fi_level.head(20))
            st.bar_chart(fi_level.set_index("feature")["importance"].head(20))
        else:
            st.info(
                "Selected Risk_Level model does not expose feature_importances_. "
                "Try a tree-based model like RandomForest or GradientBoosting."
            )

        st.subheader(f"Best Model for Risk_Type: {best_name_type}")
        fi_type = get_feature_importance(best_model_type, selected_features)
        if fi_type is not None:
            st.write(fi_type.head(20))
            st.bar_chart(fi_type.set_index("feature")["importance"].head(20))
        else:
            st.info(
                "Selected Risk_Type model does not expose feature_importances_. "
                "Try a tree-based model like RandomForest or GradientBoosting."
            )

    elif page == "Inference":
        st.header("🚀 Inference – Predict for New Sample")

        st.markdown(
            "Provide input values below (simplified set). "
            "You can extend this form to all relevant features."
        )

        df_clean = basic_cleaning(df_raw)
        sample = {}

        # Numeric fields
        if "MP_Count_per_L" in df_clean.columns:
            sample["MP_Count_per_L"] = st.number_input(
                "MP_Count_per_L",
                float(df_clean["MP_Count_per_L"].min()),
                float(df_clean["MP_Count_per_L"].max()),
                float(df_clean["MP_Count_per_L"].median()),
            )

        if "Risk_Score" in df_clean.columns:
            sample["Risk_Score"] = st.number_input(
                "Risk_Score",
                float(df_clean["Risk_Score"].min()),
                float(df_clean["Risk_Score"].max()),
                float(df_clean["Risk_Score"].median()),
            )

        # Categorical options
        if "Location" in df_clean.columns:
            sample["Location"] = st.selectbox(
                "Location",
                sorted(df_clean["Location"].dropna().unique()),
            )
        if "Shape" in df_clean.columns:
            sample["Shape"] = st.selectbox(
                "Shape",
                sorted(df_clean["Shape"].dropna().unique()),
            )
        if "Polymer_Type" in df_clean.columns:
            sample["Polymer_Type"] = st.selectbox(
                "Polymer_Type",
                sorted(df_clean["Polymer_Type"].dropna().unique()),
            )

        # Fill any other columns with median/mode defaults
        for col in df_clean.columns:
            if col in ["Risk_Level", "Risk_Type"]:
                continue
            if col not in sample:
                if df_clean[col].dtype == "object":
                    sample[col] = df_clean[col].mode().iloc[0]
                else:
                    sample[col] = float(df_clean[col].median())

        if st.button("Predict"):
            X_new = preprocess_single_input(
                sample,
                df_template=df_clean,
                selected_features=selected_features,
                scaler=scaler,
            )

            pred_level = best_model_level.predict(X_new)[0]
            pred_type = best_model_type.predict(X_new)[0]

            st.success(f"Predicted Risk Level: **{pred_level}**")
            st.success(f"Predicted Risk Type: **{pred_type}**")

            # Optional: export models & preprocessing as pickle
            with io.BytesIO() as buf:
                pickle.dump(
                    {
                        "best_model_level": best_model_level,
                        "best_model_type": best_model_type,
                        "selected_features": selected_features,
                        "scaler": scaler,
                    },
                    buf,
                )
                buf.seek(0)
                st.download_button(
                    "Download models & preprocessing (pickle)",
                    data=buf,
                    file_name="microplastic_models.pkl",
                )


if __name__ == "__main__":
    main()
