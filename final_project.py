# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.request
import os
import time
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.title("FEMA Disaster Relief Dashboard")

# --- Download dataset ---
url = "https://storage.googleapis.com/info_450/IndividualAssistanceHousingRegistrantsLargeDisasters%20(1).csv"
filename = "fema_disaster_data.csv"

if not os.path.exists(filename):
    start = time.time()
    st.info(f"Downloading dataset ({filename})...")
    urllib.request.urlretrieve(url, filename)
    st.success(f"Downloaded {filename} in {time.time() - start:.2f} seconds!")

# --- Load dataset ---
df = pd.read_csv(filename, nrows=300000)

# --- Clean & preprocess ---
df_clean = df.copy()
df_clean.columns = df_clean.columns.str.strip()

def standardize_binary_col(df, col_name, replace_map=None):
    if col_name in df.columns:
        if replace_map:
            df[col_name] = df[col_name].replace(replace_map)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int64')
    return df

for col in ['repairAmount', 'grossIncome', 'waterLevel']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = standardize_binary_col(df_clean, 'tsaEligible')
df_clean = standardize_binary_col(df_clean, 'destroyed', {'yes':1, 'Yes':1, 'Y':1, 'No':0, 'no':0, 'N':0})

# --- Missing values summary ---
relevant = ['tsaEligible', 'repairAmount', 'grossIncome', 'residenceType', 'damagedStateAbbreviation']
missing_summary = df_clean[relevant].isna().sum()
st.subheader("Missing values summary after initial cleaning")
st.dataframe(missing_summary)

# --- Crosstab ---
ct_residence = pd.crosstab(df_clean['residenceType'], df_clean['tsaEligible'], normalize='index').fillna(0)
ct_state = pd.crosstab(df_clean['damagedStateAbbreviation'], df_clean['tsaEligible'], normalize='index').fillna(0)

st.subheader("Crosstab: TSA Eligibility by Residence Type")
st.dataframe(ct_residence.head())
st.subheader("Crosstab: TSA Eligibility by State")
st.dataframe(ct_state.head())

# --- Groupby ---
avg_repair_by_state = df_clean.dropna(subset=['repairAmount','damagedStateAbbreviation']).groupby('damagedStateAbbreviation')['repairAmount'].mean().sort_values(ascending=False)
st.subheader("Top 20 States by Average Repair Amount")
st.dataframe(avg_repair_by_state.head(20))

# --- Charts ---
# Bar chart: TSA eligibility rate by state
state_rates = pd.crosstab(df_clean['damagedStateAbbreviation'], df_clean['tsaEligible'], normalize='index').fillna(0)
state_rates = state_rates.reset_index().rename(columns={0:'not_eligible', 1:'eligible'})
fig_bar = px.bar(
    state_rates.sort_values('eligible', ascending=False).head(30),
    x='damagedStateAbbreviation', y='eligible',
    title='Top 30 States by TSA Eligibility Rate',
    labels={'eligible':'TSA eligible rate'}
)
st.plotly_chart(fig_bar)

# Histogram: repairAmount
fig_hist = px.histogram(
    df_clean.dropna(subset=['repairAmount']),
    x='repairAmount',
    nbins=50,
    title='Distribution of Repair Amount'
)
fig_hist.update_xaxes(type='log')
st.plotly_chart(fig_hist)

# Boxplot: repairAmount across residence types
fig_box_res = px.box(
    df_clean.dropna(subset=['repairAmount','residenceType']),
    x='residenceType',
    y='repairAmount',
    title='Repair Amount by Residence Type'
)
st.plotly_chart(fig_box_res)

# TSA Eligibility by Income Quintile
df_bin = df_clean.dropna(subset=['grossIncome']).copy()
df_bin['income_bin'] = pd.qcut(df_bin['grossIncome'].rank(method='first'), q=5, labels=['Q1','Q2','Q3','Q4','Q5'])
income_rates = pd.crosstab(df_bin['income_bin'], df_bin['tsaEligible'], normalize='index').reset_index().rename(columns={1:'eligible'})
fig_income = px.bar(income_rates, x='income_bin', y='eligible', title='TSA Eligibility Rate by Income Quintile')
st.plotly_chart(fig_income)

# --- Inferential stats ---
tsa_yes = df_clean[df_clean['tsaEligible'] == 1]['repairAmount']
tsa_no = df_clean[df_clean['tsaEligible'] == 0]['repairAmount']

def get_ci(series):
    mean = series.mean()
    sd = series.std()
    n = len(series)
    sem = sd / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
    return mean, ci

mean_yes, ci_yes = get_ci(tsa_yes.dropna())
mean_no, ci_no = get_ci(tsa_no.dropna())

st.subheader("95% Confidence Intervals")
st.write(f"TSA Eligible Mean Repair Amount: ${mean_yes:.2f}")
st.write(f"95% CI: {ci_yes}")
st.write(f"Not TSA Eligible Mean Repair Amount: ${mean_no:.2f}")
st.write(f"95% CI: {ci_no}")

state1, state2 = "FL", "TX"
s1 = df_clean[df_clean['damagedStateAbbreviation'] == state1]['repairAmount']
s2 = df_clean[df_clean['damagedStateAbbreviation'] == state2]['repairAmount']

mean_s1, ci_s1 = get_ci(s1.dropna())
mean_s2, ci_s2 = get_ci(s2.dropna())

st.subheader("State Comparison")
st.write(f"{state1} Mean Repair Amount = ${mean_s1:.2f}, 95% CI = {ci_s1}")
st.write(f"{state2} Mean Repair Amount = ${mean_s2:.2f}, 95% CI = {ci_s2}")

# TSA t-test
t_tsa, p_tsa = stats.ttest_ind(tsa_yes.dropna(), tsa_no.dropna(), equal_var=False)
# State t-test
t_state, p_state = stats.ttest_ind(s1.dropna(), s2.dropna(), equal_var=False)

st.subheader("Hypothesis Tests")
st.write("TSA Eligible vs Non-Eligible (Repair Amount)")
st.write(f"T-statistic = {t_tsa:.3f}, p-value = {p_tsa:.4f}")
st.write(f"{state1} vs {state2} (Repair Amount)")
st.write(f"T-statistic = {t_state:.3f}, p-value = {p_state:.4f}")

# --- Predictive Modeling ---
df_m = df_clean.copy()
cat_cols = ['residenceType', 'damagedStateAbbreviation']
enc = OrdinalEncoder()
df_m[cat_cols] = enc.fit_transform(df_m[cat_cols].astype(str))

X = df_m[['grossIncome','repairAmount','destroyed','waterLevel'] + cat_cols]
y = df_m['tsaEligible'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

st.subheader("Decision Tree Metrics")
st.dataframe({
    'accuracy': [accuracy_score(y_test, y_pred_dt)],
    'precision': [precision_score(y_test, y_pred_dt, zero_division=0)],
    'recall': [recall_score(y_test, y_pred_dt, zero_division=0)],
    'confusion_matrix': [confusion_matrix(y_test, y_pred_dt).tolist()]
})

st.subheader("Random Forest Metrics")
st.dataframe({
    'accuracy': [accuracy_score(y_test, y_pred_rf)],
    'precision': [precision_score(y_test, y_pred_rf, zero_division=0)],
    'recall': [recall_score(y_test, y_pred_rf, zero_division=0)],
    'confusion_matrix': [confusion_matrix(y_test, y_pred_rf).tolist()]
})

st.subheader("Random Forest Feature Importances")
feat_importances = dict(zip(X.columns, rf.feature_importances_))
st.dataframe(sorted(feat_importances.items(), key=lambda x: x[1], reverse=True))
