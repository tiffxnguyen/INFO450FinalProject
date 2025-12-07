import streamlit as st
import pandas as pd
import plotly.express as px
import urllib.request
import time
import os
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FEMA Disaster Relief Dashboard")

# Constants
URL = "https://storage.googleapis.com/info_450/IndividualAssistanceHousingRegistrantsLargeDisasters%20(1).csv"
FILENAME = "fema_disaster_data.csv"
N_ROWS = 300000

# --- Helper Functions (From your Colab notebook) ---

def standardize_binary_col(df, col_name, replace_map=None):
    if col_name in df.columns:
        if replace_map:
            df[col_name] = df[col_name].replace(replace_map)
        # Using 'Int64' (Pandas' nullable integer) for binary data
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int64')
    return df

@st.cache_data
def load_and_clean_data(url, filename, nrows):
    # 1. Download and load data
    if not os.path.exists(filename):
        start = time.time()
        st.info(f"Downloading dataset ({filename})... This may take a moment.")
        try:
            # Use urlretrieve to download to a local file
            urllib.request.urlretrieve(url, filename)
            st.success(f"Downloaded {filename} in {time.time() - start:.2f} seconds!")
        except Exception as e:
            st.error(f"Error downloading file: {e}")
            return pd.DataFrame() # Return empty DataFrame on failure

    # Load the dataset (using the local file path)
    df = pd.read_csv(filename, nrows=nrows)

    # 2. Handle missing values & cleaning
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    for col in ['repairAmount', 'grossIncome', 'waterLevel']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean = standardize_binary_col(df_clean, 'tsaEligible')
    df_clean = standardize_binary_col(df_clean, 'destroyed', {'yes':1, 'Yes':1, 'Y':1, 'No':0, 'no':0, 'N':0})

    return df_clean

# --- Streamlit App Layout ---

st.title("🏠 FEMA Disaster Relief Dashboard")
st.markdown("Analyzing FEMA Individual Assistance Housing Registrants data.")

# Load the cached, cleaned data
df_clean = load_and_clean_data(URL, FILENAME, N_ROWS)

if df_clean.empty:
    st.error("Could not load data. Please check the source URL and internet connection.")
else:
    # --- Data Preview ---
    st.subheader("📊 Data Preview and Summary")
    st.write(f"Showing **{len(df_clean)}** rows of data.")
    st.dataframe(df_clean.head(), use_container_width=True)

    ---

    # --- Visualizations ---

    st.subheader("📈 Key Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart: TSA eligibility rate by state (from your Colab code)
        state_rates = pd.crosstab(df_clean['damagedStateAbbreviation'], df_clean['tsaEligible'], normalize='index').fillna(0)
        state_rates = state_rates.reset_index().rename(columns={0:'not_eligible', 1:'eligible'})
        fig_bar = px.bar(
            state_rates.sort_values('eligible', ascending=False).head(30),
            x='damagedStateAbbreviation', y='eligible',
            title='Top 30 States by TSA Eligibility Rate',
            labels={'eligible':'TSA Eligible Rate'},
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Histogram: distribution of repairAmount (from your Colab code)
        fig_hist = px.histogram(
            df_clean.dropna(subset=['repairAmount']),
            x='repairAmount', nbins=50,
            title='Distribution of Repair Amount (Log Scale)',
            height=500
        )
        fig_hist.update_xaxes(type='log')
        st.plotly_chart(fig_hist, use_container_width=True)

    ---

    st.subheader("Comparisons")

    col3, col4 = st.columns(2)

    with col3:
        # Boxplot: repairAmount across residence types (from your Colab code)
        fig_box_res = px.box(
            df_clean.dropna(subset=['repairAmount','residenceType']),
            x='residenceType', y='repairAmount',
            title='Repair Amount by Residence Type',
            height=500
        )
        st.plotly_chart(fig_box_res, use_container_width=True)

    with col4:
        # Bar chart: TSA Eligibility Rate by Income Quintile (from your Colab code)
        df_bin = df_clean.dropna(subset=['grossIncome', 'tsaEligible'])
        # Only proceed if there is enough data after dropping NaNs
        if not df_bin.empty:
            df_bin['income_bin'] = pd.qcut(df_bin['grossIncome'].rank(method='first'), q=5, labels=['Q1','Q2','Q3','Q4','Q5'])
            income_rates = pd.crosstab(df_bin['income_bin'], df_bin['tsaEligible'], normalize='index').reset_index().rename(columns={1:'eligible'})
            fig_income = px.bar(
                income_rates, x='income_bin', y='eligible',
                title='TSA Eligibility Rate by Income Quintile',
                height=500
            )
            st.plotly_chart(fig_income, use_container_width=True)
        else:
            st.warning("Not enough data to generate Income Quintile chart.")

    # Optional: Display the cleaned data summary
    with st.expander("Show Data Cleaning Details"):
        relevant = ['tsaEligible', 'repairAmount', 'grossIncome', 'residenceType', 'damagedStateAbbreviation']
        missing_summary = df_clean[relevant].isna().sum()
        st.write("Missing values summary after cleaning:")
        st.dataframe(missing_summary)
