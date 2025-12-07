import streamlit as st
import pandas as pd
import plotly.express as px

st.title("FEMA Disaster Relief Dashboard")

# --- Load FEMA dataset ---
# df = pd.read_csv("https://raw.githubusercontent.com/username/repo/main/IndividualAssistanceHousingRegistrantsLargeDisasters.csv", nrows=300000)

st.subheader("Data Preview")
st.write(df.head())

# --- Initial cleaning ---
# Make a copy for cleaning
df_clean = df.copy()

# Example cleaning function for binary column
def standardize_binary_col(df, col):
    df[col] = df[col].apply(lambda x: 1 if x == True else 0)
    return df

# Apply cleaning
df_clean = standardize_binary_col(df_clean, 'tsaEligible')

# Show missing values after cleaning
st.write("Missing values summary after initial cleaning:")
st.write(df_clean.isnull().sum())

# --- Histogram of Repair Amount ---
st.subheader("Histogram of Repair Amount")
fig_hist = px.histogram(
    df_clean,
    x="repairAmount",
    nbins=30,
    title="Distribution of Repair Amounts"
)
st.plotly_chart(fig_hist)

# --- Boxplot of Repair Amount by TSA Eligibility ---
st.subheader("Boxplot: Repair Amount by TSA Eligibility")
fig_box = px.box(
    df_clean,
    x="tsaEligible",
    y="repairAmount",
    title="Repair Amount by TSA Eligibility",
    labels={
        "tsaEligible": "TSA Eligible (1=Yes, 0=No)",
        "repairAmount": "Repair Amount"
    }
)
st.plotly_chart(fig_box)

# --- Optional text summary ---
st.markdown(
    "*Insight:* Compare the central tendency and spread of repair amounts "
    "for TSA-eligible vs. non-eligible households."
)
