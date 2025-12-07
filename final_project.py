# app.py
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

# --- Download data if not exists ---
url = "https://storage.googleapis.com/info_450/IndividualAssistanceHousingRegistrantsLargeDisasters%20(1).csv"
filename = "fema_disaster_data.csv"

if not os.path.exists(filename):
    start = time.time()
    st.info(f"Downloading dataset ({filename})...")
    urllib.request.urlretrieve(url, filename)
    st.success(f"Downloaded {filename} in {time.time() - start:.2f} seconds!")

# --- Load dataset ---
df = pd.read_csv(filename, nrows=300000)

# --- Clean data ---
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

# --- Data preview ---
st.subheader("Data Preview")
st.write(df_clean.head())

# --- Charts ---
st.subheader("Top 30 States by TSA Eligibility Rate")
state_rates = pd.crosstab(df_clean['damagedStateAbbreviation'], df_clean['tsaEligible'], normalize='index').fillna(0)
state_rates = state_rates.reset_index().rename(columns={0:'not_eligible', 1:'eligible'})
fig_bar = px.bar(state_rates.sort_values('eligible', ascending=False).head(30),
                 x='damagedStateAbbreviation', y='eligible',
                 title='Top 30 States by TSA Eligibility Rate', labels={'eligible':'TSA eligible rate'})
st.plotly_chart(fig_bar)

st.subheader("Distribution of Repair Amount")
fig_hist = px.histogram(df_clean.dropna(subset=['repairAmount']), x='repairAmount', nbins=50, title='Distribution of Repair Amount')
fig_hist.update_xaxes(type='log')
st.plotly_chart(fig_hist)

st.subheader("Repair Amount by Residence Type")
fig_box_res = px.box(df_clean.dropna(subset=['repairAmount','residenceType']),
                     x='residenceType', y='repairAmount', title='Repair Amount by Residence Type')
st.plotly_chart(fig_box_res)

# --- TSA Eligibility by Income Quintile ---
df_bin = df_clean.dropna(subset=['grossIncome'])
df_bin['income_bin'] = pd.qcut(df_bin['grossIncome'].rank(method='first'), q=5, labels=['Q1','Q2','Q3','Q4','Q5'])
income_rates = pd.crosstab(df_bin['income_bin'], df_bin['tsaEligible'], normalize='index').reset_index().rename(columns={1:'eligible'})
fig_income = px.bar(income_rates, x='income_bin', y='eligible', title='TSA Eligibility Rate by Income Quintile')
st.plotly_chart(fig_income)

# --- Inferential Stats ---
st.subheader("TSA Eligible vs Non-Eligible Repair Amount")
tsa_yes = df_clean[df_clean['tsaEligible'] == 1]['repairAmount'].dropna()
tsa_no = df_clean[df_clean['tsaEligible'] == 0]['repairAmount'].dropna()

def get_ci(series):
    mean = series.mean()
    sd = series.std()
    n = len(series)
    sem = sd / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
    return mean, ci

mean_yes, ci_yes = get_ci(tsa_yes)
mean_no, ci_no = get_ci(tsa_no)

st.write(f"TSA Eligible Mean Repair Amount: ${mean_yes:.2f}, 95% CI: {ci_yes}")
st.write(f"Not TSA Eligible Mean Repair Amount: ${mean_no:.2f}, 95% CI: {ci_no}")

# --- Predictive Modeling ---
st.subheader("Predictive Modeling: TSA Eligibility")

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

st.write("**Decision Tree Metrics**", {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt, zero_division=0),
    'recall': recall_score(y_test, y_pred_dt, zero_division=0),
    'confusion_matrix': confusion_matrix(y_test, y_pred_dt).tolist()
})

st.write("**Random Forest Metrics**", {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf, zero_division=0),
    'recall': recall_score(y_test, y_pred_rf, zero_division=0),
    'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist()
})

st.write("**Random Forest Feature Importances**")
feat_importances = dict(zip(X.columns, rf.feature_importances_))
st.write(sorted(feat_importances.items(), key=lambda x: x[1], reverse=True))
