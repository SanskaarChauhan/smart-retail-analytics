
from src.data_preprocessing import preprocess
from src.clustering import run_clustering
from src.forecasting import run_forecasting
from src.utils import check_required_files

if not check_required_files():
    preprocess()
    run_clustering()
    run_forecasting()
import sys
import os

# Make src folder accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products, get_top_countries,
    check_required_files
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import joblib

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
# LOAD MODELS
# =============================

@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists("models/linear_regression.pkl"):
            models["lr"] = joblib.load("models/linear_regression.pkl")
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
st.sidebar.write(f"Welcome, **{st.session_state.user}** (Role: {st.session_state.role})")

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
        st.info("No data available. Please upload data in 'Upload Data' page.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"£{metrics['Total Revenue']:,.2f}")
        c2.metric("Orders", f"{metrics['Total Orders']:,}")
        c3.metric("Customers", f"{metrics['Unique Customers']:,}")
        c4.metric("Products", f"{metrics['Unique Products']:,}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Trend")
            fig = px.line(monthly, x="MonthYear", y="Revenue", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Products")
            top = get_top_products(df, 10)
            fig = px.bar(top, x="Revenue", y="Product", orientation='h')
            st.plotly_chart(fig, use_container_width=True)

# =============================
# SALES TRENDS
# =============================

elif page == "Sales Trends":
    st.title("Sales Trends Analysis")
    
    if not data_loaded:
        st.info("No data available. Please upload data in 'Upload Data' page.")
    else:
        fig = px.line(monthly, x="MonthYear", y="Revenue", markers=True,
                     title="Monthly Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add year filter
        if "Year" in df.columns:
            years = sorted(df["Year"].unique())
            selected_year = st.selectbox("Filter by Year", ["All"] + list(years))
            if selected_year != "All":
                filtered = df[df["Year"] == selected_year]
                monthly_filtered = get_monthly_revenue(filtered)
                fig2 = px.line(monthly_filtered, x="MonthYear", y="Revenue", markers=True,
                              title=f"Revenue Trend - {selected_year}")
                st.plotly_chart(fig2, use_container_width=True)

# =============================
# PRODUCT ANALYSIS
# =============================

elif page == "Product Analysis":
    st.title("Product Performance Analysis")
    
    if not data_loaded:
        st.info("No data available. Please upload data in 'Upload Data' page.")
    else:
        n = st.slider("Number of top products", 5, 30, 10)
        top = get_top_products(df, n)
        
        fig = px.bar(top, x="Revenue", y="Product", orientation='h',
                    title=f"Top {n} Products by Revenue")
        st.plotly_chart(fig, use_container_width=True)
        
        # Country analysis
        st.subheader("Top Countries")
        countries = get_top_countries(df, 10)
        fig2 = px.bar(countries, x="Country", y="Revenue", title="Revenue by Country")
        st.plotly_chart(fig2, use_container_width=True)

# =============================
# DEMAND FORECAST - COMPLETE
# =============================

elif page == "Demand Forecast":
    st.title("Demand Forecast - Past, Present & Future")
    
    if not data_loaded:
        st.info("No data available. Please upload data in 'Upload Data' page.")
        st.stop()
    
    model_choice = st.radio(
        "Select Forecasting Model",
        ["Linear Regression", "Random Forest", "ARIMA"],
        horizontal=True
    )
    
    # ============================================
    # LINEAR REGRESSION
    # ============================================
    if model_choice == "Linear Regression":
        st.subheader("📈 Linear Regression Forecast")
        
        try:
            # Prepare data
            X = monthly[["MonthIndex"]]
            y = monthly["Revenue"]
            
            # Train model
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            
            # 1. PAST PREDICTIONS
            st.write("### 📊 Past Performance")
            past_pred = lr_model.predict(X)
            
            # Metrics
            mae = mean_absolute_error(y, past_pred)
            rmse = np.sqrt(mean_squared_error(y, past_pred))
            r2 = r2_score(y, past_pred)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"£{mae:,.2f}")
            c2.metric("RMSE", f"£{rmse:,.2f}")
            c3.metric("R² Score", f"{r2:.3f}")
            
            # 2. PRESENT
            st.write("### 🎯 Present Status")
            latest_actual = monthly.iloc[-1]
            present_pred = lr_model.predict([[latest_actual["MonthIndex"]]])[0]
            
            col1, col2 = st.columns(2)
            col1.metric("Latest Actual Revenue", f"£{latest_actual['Revenue']:,.2f}")
            col2.metric("Model Prediction", f"£{present_pred:,.2f}",
                       delta=f"£{present_pred - latest_actual['Revenue']:,.2f}")
            
            # 3. FUTURE
            st.write("### 🔮 Future Forecast")
            months_ahead = st.slider("Months to forecast", 1, 24, 6)
            
            if st.button("Generate Forecast"):
                last_idx = monthly["MonthIndex"].max()
                future_idx = pd.DataFrame({
                    "MonthIndex": range(last_idx + 1, last_idx + months_ahead + 1)
                })
                future_pred = lr_model.predict(future_idx)
                
                future_dates = pd.date_range(
                    start=monthly["MonthYear"].max() + pd.DateOffset(months=1),
                    periods=months_ahead, freq="MS"
                )
                
                # Plot all
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly["MonthYear"], y=monthly["Revenue"], 
                                        name="Actual", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=monthly["MonthYear"], y=past_pred, 
                                        name="Predicted (Past)", mode="lines+markers", line=dict(dash="dot")))
                fig.add_trace(go.Scatter(x=future_dates, y=future_pred, 
                                        name="Forecast (Future)", mode="lines+markers", line=dict(dash="dash")))
                fig.add_vline(x=monthly["MonthYear"].max(), line_dash="dash", line_color="gray",
                             annotation_text="Present", annotation_position="top")
                fig.update_layout(title="Linear Regression: Complete Forecast", xaxis_title="Date", yaxis_title="Revenue (£)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                results = pd.DataFrame({
                    "Month": future_dates.strftime("%b %Y"),
                    "Forecasted Revenue": future_pred.round(2)
                })
                st.dataframe(results)
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    # ============================================
    # RANDOM FOREST
    # ============================================
    elif model_choice == "Random Forest":
        st.subheader("🌲 Random Forest Forecast")
        
        if "rf" not in models:
            st.warning("Random Forest model not found. Run forecasting.py first.")
        else:
            try:
                # Prepare features (CORRECTED NAMES)
                monthly_rf = monthly.copy()
                monthly_rf["Month"] = monthly_rf["MonthYear"].dt.month
                monthly_rf["Quarter"] = monthly_rf["MonthYear"].dt.quarter
                monthly_rf["RevLag1"] = monthly_rf["Revenue"].shift(1)  # Fixed name
                monthly_rf["RevLag2"] = monthly_rf["Revenue"].shift(2)  # Fixed name
                monthly_rf["RollingMean"] = monthly_rf["Revenue"].rolling(3).mean()
                monthly_rf = monthly_rf.dropna().reset_index(drop=True)
                
                features = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", "RollingMean"]
                X_rf = monthly_rf[features]
                
                # 1. PAST PREDICTIONS
                st.write("### 📊 Past Performance")
                past_pred = models["rf"].predict(X_rf)
                
                # Metrics
                mae = mean_absolute_error(monthly_rf["Revenue"], past_pred)
                rmse = np.sqrt(mean_squared_error(monthly_rf["Revenue"], past_pred))
                r2 = r2_score(monthly_rf["Revenue"], past_pred)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"£{mae:,.2f}")
                c2.metric("RMSE", f"£{rmse:,.2f}")
                c3.metric("R² Score", f"{r2:.3f}")
                
                # 2. PRESENT
                st.write("### 🎯 Present Status")
                latest_actual = monthly.iloc[-1]
                present_features = pd.DataFrame({
                    "MonthIndex": [latest_actual["MonthIndex"]],
                    "Month": [latest_actual["MonthYear"].month],
                    "Quarter": [latest_actual["MonthYear"].quarter],
                    "RevLag1": [monthly["Revenue"].iloc[-2] if len(monthly) > 1 else monthly["Revenue"].iloc[-1]],
                    "RevLag2": [monthly["Revenue"].iloc[-3] if len(monthly) > 2 else monthly["Revenue"].iloc[-1]],
                    "RollingMean": [monthly["Revenue"].iloc[-3:].mean() if len(monthly) >= 3 else monthly["Revenue"].mean()]
                })
                present_pred = models["rf"].predict(present_features)[0]
                
                col1, col2 = st.columns(2)
                col1.metric("Latest Actual Revenue", f"£{latest_actual['Revenue']:,.2f}")
                col2.metric("Model Prediction", f"£{present_pred:,.2f}",
                           delta=f"£{present_pred - latest_actual['Revenue']:,.2f}")
                
                # 3. FUTURE
                st.write("### 🔮 Future Forecast")
                months_ahead = st.slider("Months to forecast", 1, 24, 6)
                
                if st.button("Generate Forecast"):
                    future_dates = [monthly["MonthYear"].max() + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]
                    future_preds = []
                    last_revenues = monthly["Revenue"].tail(3).tolist()
                    
                    for i, future_date in enumerate(future_dates):
                        future_features = pd.DataFrame({
                            "MonthIndex": [monthly["MonthIndex"].max() + i + 1],
                            "Month": [future_date.month],
                            "Quarter": [(future_date.month - 1) // 3 + 1],
                            "RevLag1": [last_revenues[-1]],
                            "RevLag2": [last_revenues[-2]],
                            "RollingMean": [np.mean(last_revenues[-3:])]
                        })
                        pred = models["rf"].predict(future_features)[0]
                        future_preds.append(pred)
                        last_revenues.append(pred)
                        if len(last_revenues) > 3:
                            last_revenues.pop(0)
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=monthly_rf["MonthYear"], y=monthly_rf["Revenue"], name="Actual", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=monthly_rf["MonthYear"], y=past_pred, name="Predicted", mode="lines+markers", line=dict(dash="dot")))
                    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Forecast", mode="lines+markers", line=dict(dash="dash")))
                    fig.add_vline(x=monthly["MonthYear"].max(), line_dash="dash", line_color="gray", annotation_text="Present")
                    fig.update_layout(title="Random Forest: Complete Forecast", xaxis_title="Date", yaxis_title="Revenue (£)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    results = pd.DataFrame({"Month": [d.strftime("%b %Y") for d in future_dates], "Forecasted Revenue": [round(p, 2) for p in future_preds]})
                    st.dataframe(results)
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    # ============================================
    # ARIMA
    # ============================================
    elif model_choice == "ARIMA":
        st.subheader("📊 ARIMA Forecast")
        
        try:
            series = monthly.set_index("MonthYear")["Revenue"]
            
            # Train ARIMA model
            model = ARIMA(series, order=(1, 1, 1))
            fitted = model.fit()
            
            # 1. PAST (Fitted values)
            st.write("### 📊 Past Performance")
            past_pred = fitted.fittedvalues
            
            # Metrics (excluding initial NaN)
            valid_idx = ~np.isnan(past_pred)
            y_valid = series[valid_idx]
            pred_valid = past_pred[valid_idx]
            
            mae = mean_absolute_error(y_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
            
            c1, c2 = st.columns(2)
            c1.metric("MAE", f"£{mae:,.2f}")
            c2.metric("RMSE", f"£{rmse:,.2f}")
            
            # 2. PRESENT
            st.write("### 🎯 Present Status")
            latest_actual = series.iloc[-1]
            # Get in-sample prediction for last point
            present_pred = past_pred.iloc[-1] if not np.isnan(past_pred.iloc[-1]) else latest_actual
            
            col1, col2 = st.columns(2)
            col1.metric("Latest Actual Revenue", f"£{latest_actual:,.2f}")
            col2.metric("Model Fit Value", f"£{present_pred:,.2f}",
                       delta=f"£{present_pred - latest_actual:,.2f}")
            
            # 3. FUTURE
            st.write("### 🔮 Future Forecast")
            months_ahead = st.slider("Months to forecast", 1, 24, 6)
            
            if st.button("Generate Forecast"):
                forecast = fitted.forecast(steps=months_ahead)
                future_dates = pd.date_range(
                    start=series.index[-1] + pd.DateOffset(months=1),
                    periods=months_ahead, freq="MS"
                )
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=series.index, y=fitted.fittedvalues, name="Fitted (Past)", mode="lines+markers", line=dict(dash="dot")))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast (Future)", mode="lines+markers", line=dict(dash="dash")))
                fig.add_vline(x=series.index[-1], line_dash="dash", line_color="gray", annotation_text="Present")
                fig.update_layout(title="ARIMA: Complete Forecast", xaxis_title="Date", yaxis_title="Revenue (£)")
                st.plotly_chart(fig, use_container_width=True)
                
                results = pd.DataFrame({
                    "Month": future_dates.strftime("%b %Y"),
                    "Forecasted Revenue": forecast.values.round(2)
                })
                st.dataframe(results)
                
                # Show model summary
                with st.expander("Model Summary"):
                    st.text(fitted.summary().as_text())
                    
        except Exception as e:
            st.error(f"Error: {e}")

# =============================
# CUSTOMER SEGMENTS
# =============================

elif page == "Customer Segments":
    st.title("Customer Segmentation Analysis")
    
    if not data_loaded or rfm is None:
        st.info("No customer segments data available. Run clustering.py first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment",
                           title="Customer Segments", hover_data=["CustomerID"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            seg_counts = rfm["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Customers"]
            fig = px.pie(seg_counts, values="Customers", names="Segment", title="Segment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("RFM Segment Summary")
        summary = rfm.groupby("Segment").agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean")
        ).round(2).reset_index()
        st.dataframe(summary)
        
        st.subheader("Customer Data with Segments")
        st.dataframe(rfm.head(100))

# =============================
# UPLOAD DATA
# =============================

elif page == "Upload Data":
    st.title("Upload New Dataset")
    
    uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file)
            
            # Validate required columns
            required_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", 
                           "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
            
            missing = [col for col in required_cols if col not in df_new.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                os.makedirs("data/raw", exist_ok=True)
                df_new.to_excel("data/raw/retail_sales.xlsx", index=False)
                st.success("File uploaded!")
                
                if st.button("Run Preprocessing"):
                    with st.spinner("Processing..."):
                        from data_preprocessing import preprocess
                        preprocess()
                        st.success("Done! Refresh the page.")
                        st.cache_data.clear()
                        st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# =============================
# ADMIN PANEL
# =============================

elif page == "Admin Panel":
    st.title("Admin Panel")
    
    if st.session_state.role != "admin":
        st.warning("Access Denied. Admin privileges required.")
    else:
        st.success(f"Welcome Admin {st.session_state.user}")
        
        st.subheader("System Status")
        check_required_files()
        
        if data_loaded:
            st.subheader("Data Information")
            st.write(f"Total Records: {len(df):,}")
            st.write(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
            st.write(f"Total Customers: {df['CustomerID'].nunique():,}")
