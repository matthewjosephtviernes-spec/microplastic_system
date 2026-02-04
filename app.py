import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data
def load_data():
    # Replace this with your data loading logic, e.g., loading a CSV file
    # Here we're creating a sample dataframe for demonstration
    data = {
        'Risk_Score': [50, 60, 55, 70, 80, 65, 85, 90, 50, 45, 40, 75, 60, 100, 95]
    }
    df = pd.DataFrame(data)
    return df

# Load the data
df = load_data()

# Set the title of the app
st.title('Risk Score Distribution Analysis')

# Create tabs
tabs = ['Home', 'Tab 2', 'Tab 3']  # Add more tabs as needed
selected_tab = st.selectbox('Choose a tab', tabs)

if selected_tab == 'Home':
    st.header('Data Loading and Visualization')

    # Data Loading Section
    st.subheader('Loaded Data')
    st.write(df.head())  # Show the first few rows of the dataset

    # Plotting the distribution of the Risk Score

    # Histogram
    st.subheader('Histogram of Risk Score')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Risk_Score', kde=True, bins=30, ax=ax)
    ax.set_title('Distribution of Risk Score')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Box Plot
    st.subheader('Box Plot of Risk Score')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, y='Risk_Score', ax=ax)
    ax.set_title('Box Plot of Risk Score')
    ax.set_ylabel('Risk Score')
    st.pyplot(fig)

elif selected_tab == 'Tab 2':
    st.header('Tab 2 Content')
    # Add content for Tab 2

elif selected_tab == 'Tab 3':
    st.header('Tab 3 Content')
    # Add content for Tab 3
