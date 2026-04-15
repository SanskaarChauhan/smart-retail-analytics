# =============================
# IMPORTS (FIXED)
# =============================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess
from src.clustering import run_clustering
from src.utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products, get_top_countries,
    check_required_files
)

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import joblib

# =============================
# AUTO PIPELINE (FIXED)
# =============================

try:
    if not check_required_files():
        preprocess()
        run_clustering()
except Exception as e:
    st.error(f"Pipeline failed: {e}")

# =============================
# SESSION AUTH
# =============================

if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

def login():
    st.title("Smart Retail Analytics - Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.user = username
            st.session_state.role = "admin"
            st.rerun()
        elif username == "user" and password == "user":
            st.session_state.user = username
            st.session_state.role = "user"
            st.rerun()
        else:
            st.error("Invalid credentials")

# =============================
# LOGIN GATE
# =============================

if st.session_state.user is None:
    login()
    st.stop()

# =============================
# LOAD DATA
# =============================

@st.cache_data
def load_all():
    try:
        df = load_clean_data()
        rfm = load_rfm()
        return df, rfm, True
    except Exception as e:
        st.warning(f"Data not loaded: {e}")
        return None, None, False

df, rfm, data_loaded = load_all()

if data_loaded:
    metrics = summary_metrics(df)
    monthly = get_monthly_revenue(df)
else:
    metrics = {}
    monthly = None

# =============================
# LOAD MODELS (SAFE)
# =============================

@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists("models/random_forest.pkl"):
            models["rf"] = joblib.load("models/random_forest.pkl")
    except Exception as e:
        st.warning(f"Models not loaded: {e}")
    return models

models = load_models()

# =============================
# SIDEBAR
# =============================

st.sidebar.title("Smart Retail Analytics")
st.sidebar.write(f"Welcome, **{st.session_state.user}** ({st.session_state.role})")

pages = ["Overview", "Sales Trends", "Product Analysis", "Demand Forecast", "Customer Segments", "Upload Data"]

if st.session_state.role == "admin":
    pages.append("Admin Panel")

page = st.sidebar.radio("Navigation", pages)

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# =============================
# OVERVIEW
# =============================

if page == "Overview":
    st.title("Dashboard Overview")
    
    if not data_loaded:
        st.info("No data available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"£{metrics['Total Revenue']:,.2f}")
        c2.metric("Orders", f"{metrics['Total Orders']:,}")
        c3.metric("Customers", f"{metrics['Unique Customers']:,}")
        c4.metric("Products", f"{metrics['Unique Products']:,}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(monthly, x="MonthYear", y="Revenue", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top = get_top_products(df, 10)
            fig = px.bar(top, x="Revenue", y="Product", orientation='h')
            st.plotly_chart(fig, use_container_width=True)

# =============================
# SALES
# =============================

elif page == "Sales Trends":
    st.title("Sales Trends")
    
    if data_loaded:
        fig = px.line(monthly, x="MonthYear", y="Revenue", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# =============================
# PRODUCT
# =============================

elif page == "Product Analysis":
    st.title("Product Analysis")
    
    if data_loaded:
        top = get_top_products(df, 10)
        fig = px.bar(top, x="Revenue", y="Product", orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# =============================
# FORECAST (LIGHTWEIGHT)
# =============================

elif page == "Demand Forecast":
    st.title("Demand Forecast")
    
    if data_loaded:
        X = monthly[["MonthIndex"]]
        y = monthly["Revenue"]
        
        model = LinearRegression().fit(X, y)
        
        last_idx = monthly["MonthIndex"].max()
        
        future = pd.DataFrame({
            "MonthIndex": list(range(last_idx + 1, last_idx + 7))
        })
        
        pred = model.predict(future)
        
        future_dates = pd.date_range(
            start=monthly["MonthYear"].max(),
            periods=6, freq="MS"
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["MonthYear"], y=y, name="Actual"))
        fig.add_trace(go.Scatter(x=future_dates, y=pred, name="Forecast"))
        
        st.plotly_chart(fig, use_container_width=True)

# =============================
# CUSTOMER SEGMENTS
# =============================

elif page == "Customer Segments":
    st.title("Customer Segments")
    
    if data_loaded:
        fig = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(rfm.head(100))

# =============================
# UPLOAD
# =============================

elif page == "Upload Data":
    st.title("Upload Data")
    
    file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    
    if file:
        if file.name.endswith(".csv"):
            df_new = pd.read_csv(file)
        else:
            df_new = pd.read_excel(file)
        
        os.makedirs("data/raw", exist_ok=True)
        df_new.to_excel("data/raw/retail_sales.xlsx", index=False)
        
        st.success("Uploaded!")

# =============================
# ADMIN
# =============================

elif page == "Admin Panel":
    st.title("Admin Panel")
    
    check_required_files()
