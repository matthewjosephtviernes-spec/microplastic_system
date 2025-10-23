import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# APP TITLE AND DESCRIPTION
# -------------------------------
st.set_page_config(page_title="Microplastic Pollution Dashboard", layout="wide")

st.title("🌊 Predictive Risk Modeling for Microplastic Pollution")
st.caption("Developed by Matthew Joseph Viernes | Agusan del Sur State College of Agriculture and Technology")

# -------------------------------
# LOAD DATA FROM GITHUB
# -------------------------------
# 🔽 Replace the link below with your RAW GitHub CSV link
url = "https://raw.githubusercontent.com/matthewjosephtviernes-spec/microplastic_system/refs/heads/main/data/Data1_Microplastic.csv"

try:
    data = pd.read_csv(url)
    st.success("✅ Successfully loaded dataset from GitHub!")
except Exception as e:
    st.warning("⚠️ Could not load data from GitHub. Attempting to load local file...")
    try:
        data = pd.read_csv("Data1_Microplastic.csv")
        st.info("📁 Loaded local dataset successfully.")
    except Exception as e2:
        st.error("❌ Failed to load any dataset. Please check your file path or GitHub link.")
        st.stop()

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# SUMMARY STATISTICS
# -------------------------------
st.subheader("📈 Summary Statistics")
st.write(data.describe(include='all'))

# -------------------------------
# DATA VISUALIZATION
# -------------------------------
st.subheader("🧠 Microplastic Count Distribution")

if "MP_Count" in data.columns:
    fig, ax = plt.subplots()
    sns.histplot(data["MP_Count"], kde=True, ax=ax)
    ax.set_title("Distribution of Microplastic Count")
    st.pyplot(fig)
else:
    st.error("Column 'MP_Count' not found in dataset.")

# -------------------------------
# GROUPED CHART EXAMPLE
# -------------------------------
if "Study_Location" in data.columns and "MP_Count" in data.columns:
    st.subheader("🌍 Microplastic Count by Study Location")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Study_Location", y="MP_Count", data=data, ax=ax2)
    plt.xticks(rotation=45)
    ax2.set_title("Average Microplastic Count per Location")
    st.pyplot(fig2)
else:
    st.warning("Columns 'Study_Location' or 'MP_Count' missing — cannot generate grouped chart.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("© 2025 | Predictive Risk Modeling for Microplastic Pollution | Streamlit App")






