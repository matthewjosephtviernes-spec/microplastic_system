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
# LOAD DATA FROM GITHUB OR LOCAL
# -------------------------------
# Replace this URL with your actual GitHub Raw CSV link
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
# COLUMN NAME FIXING
# -------------------------------
# Handle variations in column names
data.columns = data.columns.str.strip()  # remove spaces

# Find the count column (whatever its exact name)
mp_count_col = None
for col in data.columns:
    if "MP_Count" in col or "items/individual" in col:
        mp_count_col = col
        break

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
# MICROPLASTIC COUNT DISTRIBUTION
# -------------------------------
st.subheader("🧠 Microplastic Count Distribution")

if mp_count_col:
    fig, ax = plt.subplots()
    sns.histplot(data[mp_count_col], kde=True, ax=ax)
    ax.set_title("Distribution of Microplastic Count (items/individual)")
    st.pyplot(fig)
else:
    st.warning("⚠️ No 'MP_Count' column detected for visualization.")

# -------------------------------
# MICROPLASTIC COUNT BY LOCATION
# -------------------------------
if "Study_Location" in data.columns and mp_count_col:
    st.subheader("🌍 Microplastic Count by Study Location")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Study_Location", y=mp_count_col, data=data, ax=ax2)
    ax2.set_title("Average Microplastic Count per Study Location")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
else:
    st.warning("⚠️ Missing columns needed for grouped chart visualization.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("© 2025 | Predictive Risk Modeling for Microplastic Pollution | Streamlit App")









