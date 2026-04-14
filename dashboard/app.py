# dashboard/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess
from src.utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products
)
from src.auth import (
    init_files, login, signup_request,
    get_pending_requests, approve_user, reject_user
)
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# INIT AUTH FILES
init_files()

# PAGE CONFIG
st.set_page_config(page_title="Smart Retail Analytics", layout="wide")

# SESSION INIT
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user = None

# ---------------------------
# LOGIN / SIGNUP UI
# ---------------------------
if not st.session_state.logged_in:
    st.title("Login to Smart Retail Analytics")
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    # LOGIN
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, role = login(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.role = role
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    # SIGNUP
    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Request Access"):
            msg = signup_request(new_user, new_pass)
            st.info(msg)
    
    st.stop()

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_all_data():
    df = load_clean_data()
    try:
        rfm = load_rfm()
    except:
        rfm = None
    return df, rfm

@st.cache_resource
def load_models():
    models = {}
    paths = {
        "lr": "models/linear_regression.pkl",
        "rf": "models/random_forest.pkl",
        "km": "models/kmeans.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

df, rfm = load_all_data()
models = load_models()
metrics = summary_metrics(df)
monthly = get_monthly_revenue(df)

# ============================================================
# 🔥 SIDEBAR - MUST COME BEFORE PAGE LOGIC
# ============================================================
st.sidebar.title(f"Welcome {st.session_state.user}")

# ROLE BASED NAVIGATION
pages = ["Overview", "Sales Trends", "Product Analysis"]

if st.session_state.role == "admin":
    pages += [
        "Demand Forecast",
        "Customer Segments",
        "Admin Panel",
        "Data Management"
    ]

# 👉 CRITICAL: Page selector MUST be here
page = st.sidebar.radio("Navigation", pages)

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ============================================================
# PAGE CONTENT - ALL CONDITIONALS MUST USE 'page' VARIABLE
# ============================================================

# ---------------------------
# OVERVIEW
# ---------------------------
if page == "Overview":
    st.title("Smart Retail Analytics Dashboard")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"£{metrics['Total Revenue']:,.2f}")
    c2.metric("Orders", f"{metrics['Total Orders']:,}")
    c3.metric("Customers", f"{metrics['Unique Customers']:,}")
    c4.metric("Products", f"{metrics['Unique Products']:,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        ax.plot(monthly["MonthYear"], monthly["Revenue"], marker="o")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        top = get_top_products(df, 10)
        fig, ax = plt.subplots()
        ax.barh(top["Product"][::-1], top["Revenue"][::-1])
        st.pyplot(fig)

# ---------------------------
# SALES TRENDS
# ---------------------------
elif page == "Sales Trends":
    st.title("Sales Trends")
    
    years = sorted(df["Year"].unique())
    selected_year = st.selectbox("Year", ["All"] + [str(y) for y in years])
    
    filtered = df if selected_year == "All" else df[df["Year"] == int(selected_year)]
    monthly_f = get_monthly_revenue(filtered)
    
    fig, ax = plt.subplots()
    ax.plot(monthly_f["MonthYear"], monthly_f["Revenue"], marker="o")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------------------
# PRODUCT ANALYSIS
# ---------------------------
elif page == "Product Analysis":
    st.title("Product Analysis")
    
    n = st.slider("Top products", 5, 20, 10)
    top = get_top_products(df, n)
    
    fig, ax = plt.subplots()
    ax.barh(top["Product"][::-1], top["Revenue"][::-1])
    st.pyplot(fig)

# ---------------------------
# DEMAND FORECAST (ADMIN ONLY)
# ---------------------------
elif page == "Demand Forecast" and st.session_state.role == "admin":
    st.title("Demand Forecast")
    
    model_choice = st.selectbox("Model", [
        "Linear Regression", "Random Forest", "ARIMA"
    ])
    
    months_ahead = st.slider("Months", 3, 12, 6)
    
    if model_choice == "Linear Regression":
        if "lr" in models:
            model = models["lr"]
        else:
            model = LinearRegression()
            X = monthly[["MonthIndex"]]
            y = monthly["Revenue"]
            model.fit(X, y)
        
        last_idx = monthly["MonthIndex"].max()
        future_idx = pd.DataFrame({
            "MonthIndex": range(last_idx + 1, last_idx + months_ahead + 1)
        })
        future_pred = model.predict(future_idx)
        
        last_date = monthly["MonthYear"].max()
        future_dates = [
            last_date + pd.DateOffset(months=i)
            for i in range(1, months_ahead + 1)
        ]
        
        fig, ax = plt.subplots()
        ax.plot(monthly["MonthYear"], monthly["Revenue"], marker="o", label="Historical")
        ax.plot(future_dates, future_pred, marker="o", linestyle="--", label="Forecast")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif model_choice == "Random Forest":
        if "rf" in models:
            model = models["rf"]
            monthly_rf = monthly.copy()
            monthly_rf["Month"] = monthly_rf["MonthYear"].dt.month
            monthly_rf["Quarter"] = monthly_rf["MonthYear"].dt.quarter
            monthly_rf["RevLag1"] = monthly_rf["Revenue"].shift(1)
            monthly_rf["RevLag2"] = monthly_rf["Revenue"].shift(2)
            monthly_rf["RollingMean"] = monthly_rf["Revenue"].rolling(3).mean()
            monthly_rf = monthly_rf.dropna().reset_index(drop=True)
            
            features = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", "RollingMean"]
            X = monthly_rf[features]
            y = monthly_rf["Revenue"]
            preds = model.predict(X)
            
            fig, ax = plt.subplots()
            ax.plot(monthly_rf["MonthYear"], y, marker="o", label="Actual")
            ax.plot(monthly_rf["MonthYear"], preds, marker="o", linestyle="--", label="Predicted")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Random Forest model not found. Please train the model first.")
    
    elif model_choice == "ARIMA":
        revenue_series = monthly.set_index("MonthYear")["Revenue"]
        model = ARIMA(revenue_series, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=months_ahead)
        forecast_index = pd.date_range(
            start=revenue_series.index[-1] + pd.DateOffset(months=1),
            periods=months_ahead,
            freq="MS"
        )
        
        fig, ax = plt.subplots()
        ax.plot(revenue_series.index, revenue_series.values, marker="o", label="Historical")
        ax.plot(forecast_index, forecast.values, marker="o", linestyle="--", label="Forecast")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ---------------------------
# CUSTOMER SEGMENTS (ADMIN ONLY)
# ---------------------------
elif page == "Customer Segments" and st.session_state.role == "admin":
    st.title("Customer Segments")
    
    if rfm is None:
        st.warning("Run clustering first")
    else:
        fig, ax = plt.subplots()
        for cluster in rfm["Cluster"].unique():
            subset = rfm[rfm["Cluster"] == cluster]
            ax.scatter(subset["Frequency"], subset["Monetary"], alpha=0.5, label=f"Cluster {cluster}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Monetary")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(rfm.head())

# ---------------------------
# ADMIN PANEL
# ---------------------------
elif page == "Admin Panel" and st.session_state.role == "admin":
    st.title("Admin Panel")
    
    pending = get_pending_requests()
    
    if not pending:
        st.success("No pending requests")
    else:
        for req in pending:
            st.subheader(f"User: {req['username']}")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            role = col1.selectbox(
                f"Assign role for {req['username']}",
                ["employee", "admin"],
                key=req["username"]
            )
            
            if col2.button("Approve", key=f"approve_{req['username']}"):
                approve_user(req["username"], role)
                st.success(f"{req['username']} approved as {role}")
                st.rerun()
            
            if col3.button("Reject", key=f"reject_{req['username']}"):
                reject_user(req["username"])
                st.warning(f"{req['username']} rejected")
                st.rerun()

# ---------------------------
# DATA MANAGEMENT
# ---------------------------
elif page == "Data Management" and st.session_state.role == "admin":
    st.title("Data Management")
    
    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df_new = pd.read_excel(uploaded_file)
            else:
                df_new = pd.read_csv(uploaded_file)
            
            required_cols = [
                "InvoiceNo", "StockCode", "Description",
                "Quantity", "InvoiceDate", "UnitPrice",
                "CustomerID", "Country"
            ]
            
            if not all(col in df_new.columns for col in required_cols):
                st.error("Invalid dataset: Missing required columns")
                st.stop()
            
            try:
                pd.to_datetime(df_new["InvoiceDate"])
            except:
                st.error("Invalid date format in InvoiceDate")
                st.stop()
            
            st.success("Dataset validated successfully")
            
            os.makedirs("data/raw", exist_ok=True)
            filepath = "data/raw/retail_sales.xlsx"
            df_new.to_excel(filepath, index=False)
            
            st.info("File saved. Running preprocessing...")
            preprocess()
            st.success("Data updated successfully")
            
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing file: {e}")