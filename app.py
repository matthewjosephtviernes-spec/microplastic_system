import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# APP TITLE AND DESCRIPTION
# -------------------------------
st.set_page_config(page_title="Microplastic Pollution System", layout="wide")
st.title("🌊 Predictive Risk Modeling for Microplastic Pollution")
st.caption("Developed by Matthew Joseph Viernes | Agusan del Sur State College of Agriculture and Technology")

# -------------------------------
# LOAD DATA (from GitHub or local)
# -------------------------------
url = "https://raw.githubusercontent.com/<username>/<repo>/main/data/microplastic_data.csv"

data = None  # placeholder

# Try loading from GitHub
try:
    data = pd.read_csv(url)
    st.success("✅ Successfully loaded dataset from GitHub.")
except Exception as e:
    st.warning("⚠️ Could not load data from GitHub. Attempting to load local file...")
    if os.path.exists("data/microplastic_data.csv"):
        data = pd.read_csv("data/microplastic_data.csv")
        st.info("📂 Loaded local dataset successfully.")
    else:
        st.error("❌ No dataset found! Please upload your CSV file in the 'data' folder or update the GitHub link.")
        st.stop()

# -------------------------------
# DISPLAY DATA
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# SUMMARY STATISTICS
# -------------------------------
if not data.empty:
    st.subheader("📈 Summary Statistics")
    st.write(data.describe())
else:
    st.warning("Dataset is empty. Please check your CSV content.")

# -------------------------------
# DATA VISUALIZATION
# -------------------------------
if "MP_Count" in data.columns:
    st.subheader("🧠 Microplastic Count Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["MP_Count"], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.error("⚠️ Column 'MP_Count' not found in dataset. Check your CSV headers.")

# -------------------------------
# GROUPED CHART
# -------------------------------
if all(col in data.columns for col in ["Study_Location", "MP_Count"]):
    st.subheader("🌍 Microplastic Count by Study Location")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Study_Location", y="MP_Count", data=data, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
else:
    st.warning("Missing 'Study_Location' or 'MP_Count' columns for grouped chart.")
