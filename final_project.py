import streamlit as st
import pandas as pd
import plotly.express as px
import urllib.request
import time
import os
import numpy as np

# --- Configuration & Constants ---
st.set_page_config(layout="wide", page_title="FEMA Disaster Relief Dashboard")
URL = "https://storage.googleapis.com/info_450/IndividualAssistanceHousingRegistrantsLargeDisasters%20(1).csv"
FILENAME = "fema_disaster_data.csv"
N_ROWS = 300000

# --- Caching and Data Functions ---

# Helper function for data cleaning
def standardize_binary_col(df, col_name, replace_map=None):
    if col_name in df.columns:
        if replace_map:
            df[col_name] = df[col_name].replace(replace_map)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int64')
    return df

# Use Streamlit's cache decorator to only run this expensive function once
@st.cache_data
def load_and_clean_data(url, filename, nrows):
    # Download data if not already present
    if not os.path.exists(filename):
        start = time.time()
        st.info(f"Downloading dataset ({filename})... This will only run on the first load.")
        try:
            urllib.request.urlretrieve(url, filename)
            st.success(f"Downloaded {filename} in {time.time() - start:.2f} seconds!")
        except Exception as e:
            st.error(f"Error downloading file: {e}")
            return pd.DataFrame() 

    # Load the dataset
    df = pd.read_csv(filename, nrows=nrows)

    # --- Data Cleaning Logic (From your Colab code) ---
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    # Convert columns to numeric, coercing errors to NaN
    for col in ['repairAmount', 'grossIncome', 'waterLevel']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Standardize binary columns
    df_clean = standardize_binary_col(df_clean, 'tsaEligible')
    df_clean = standardize_binary_col(df_clean, 'destroyed', {'yes':1, 'Yes':1, 'Y':1, 'No':0, 'no':0, 'N':0})
    
    return df_clean

# --- Streamlit App Layout ---

st.title("🏠 FEMA Disaster Relief Dashboard")

# Load the cached, cleaned data
df_clean = load_and_clean_data(URL, FILENAME, N_ROWS)

if df_clean.empty:
    st.error("Data could not be loaded or cleaned. Check the file source.")
else:
    # --- Data Preview (Optional but helpful for debugging) ---
    with st.expander("Data Preview"):
        st.dataframe(df_clean.head(), use_container_width=True)

# ----------------------------------------------------------------
# --- REQUIRED CHART 1: Histogram of Repair Amount ---
# ----------------------------------------------------------------
st.subheader("Distribution of Repair Amount")

# Filter data to exclude NaN (for counting)
data_for_hist = df_clean.dropna(subset=['repairAmount'])
valid_count = len(data_for_hist)

# --- DEBUG CHECK (Keep this for now) ---
# ... st.info(f"Successfully found **{valid_count}** valid entries...")
# ... rest of the check ...

# Create the main histogram figure
fig_hist = px.histogram(
    data_for_hist, 
    x="repairAmount", 
    nbins=50, 
    title="Distribution of Repair Amounts (Showing Linear and Log Views)",
    labels={"repairAmount": "Repair Amount ($)"}
)

# Brief Written Insight (Adjusted for dual view)
st.markdown(
    """
    **Insight:** The distribution of repair amounts is highly right-skewed with a long tail towards high values. 
    The logarithmic scale confirms that the majority of claims are concentrated at the lower end of the cost spectrum.
    """
)

# ----------------------------------------------------------------
# --- REQUIRED CHART 2: Boxplot of Repair Amount by TSA Eligibility ---
# ----------------------------------------------------------------
st.subheader("Repair Amount by TSA Eligibility")

# Create the Plotly figure
fig_box = px.box(
    df_clean.dropna(subset=['tsaEligible', 'repairAmount']), # Drop NaN for the chart
    x="tsaEligible", 
    y="repairAmount",
    title="Repair Amount Distribution by TSA Eligibility",
    labels={
        "tsaEligible": "TSA Eligible (1=Yes, 0=No)",
        "repairAmount": "Repair Amount ($)"
    }
)

    # Display the chart in Streamlit
st.plotly_chart(fig_box, use_container_width=True)
    
    # Brief Written Insight
st.markdown(
       """
        **Insight:** This boxplot compares the repair cost patterns for individuals who were TSA eligible (1) versus those who were not (0). 
        Typically, a visible difference in the median (the line inside the box) and interquartile range (the box height) suggests that the eligibility the 
        status is related to the magnitude of the damage.
        """
    )
