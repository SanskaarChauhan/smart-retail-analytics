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
    st.title("💹 Demand Forecasting")
    st.markdown("---")
    
    if data_loaded and len(monthly) > 5:
        # Model selection dropdown
        model_options = ["Linear Trend", "Random Forest", "Simple Average"]
        selected_model = st.selectbox("Select Forecasting Model:", model_options)
        
        # Prediction horizon
        horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Professional chart with proper sizing
            fig = go.Figure()
            
            # Actual data
            fig.add_trace(go.Scatter(
                x=monthly["MonthYear"], 
                y=monthly["Revenue"],
                mode='lines+markers',
                name='Actual Revenue',
                line=dict(color='royalblue', width=3),
                marker=dict(size=6)
            ))
            
            # Forecast
            last_idx = monthly["MonthIndex"].max()
            future_dates = pd.date_range(
                start=monthly["MonthYear"].max() + pd.DateOffset(months=1),
                periods=horizon, freq="MS"
            )
            
            if selected_model == "Linear Trend":
                X_all = monthly[["MonthIndex"]]
                model = LinearRegression().fit(X_all, monthly["Revenue"])
                future_idx = np.array(range(last_idx + 1, last_idx + horizon + 1)).reshape(-1, 1)
                pred = model.predict(future_idx)
                
            elif selected_model == "Random Forest" and "rf" in models:
                # Simplified RF prediction
                pred = [monthly["Revenue"].tail(3).mean()] * horizon  # Fallback
                try:
                    feature_df = create_time_features(monthly)
                    # Use last few predictions
                    pred = models["rf"].predict(feature_df[feature_cols].tail(horizon))
                except:
                    pred = [monthly["Revenue"].tail(3).mean()] * horizon
            
            else:  # Simple Average
                avg = monthly["Revenue"].tail(3).mean()
                pred = [avg] * horizon
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=pred,
                mode='lines+markers',
                name=f'{selected_model} Forecast',
                line=dict(color='orange', width=3, dash='dash'),
                marker=dict(size=6),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title=f"Revenue Forecast ({horizon} months ahead)",
                xaxis_title="Date",
                yaxis_title="Revenue (£)",
                height=500,
                showlegend=True,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Last Month Revenue", f"£{monthly['Revenue'].iloc[-1]:,.0f}")
            st.metric("3-Month Average", f"£{monthly['Revenue'].tail(3).mean():,.0f}")
            st.metric("Forecasted Next Month", f"£{pred[0]:,.0f}")
            
            if st.button("Export Forecast"):
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Revenue': pred
                })
                st.download_button(
                    "Download CSV",
                    forecast_df.to_csv(index=False),
                    "revenue_forecast.csv",
                    "text/csv"
                )
    else:
        st.warning("Not enough data for forecasting.")

# =============================
# CUSTOMER SEGMENTS (FIXED TABLE)
# =============================

elif page == "Customer Segments":
    st.title("👥 Customer Segmentation (RFM Analysis)")
    st.markdown("---")
    
    if data_loaded and rfm is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Professional RFM scatter plot
            fig = px.scatter(
                rfm, 
                x="Recency", 
                y="Monetary", 
                size="Frequency",
                color="Segment",
                hover_data=["CustomerID"],
                title="RFM Customer Segments",
                height=500
            )
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fixed summary table - KEY METRICS ONLY
            st.subheader("Segment Summary")
            summary = rfm.groupby('Segment').agg({
                'CustomerID': 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).round(1)
            summary.columns = ['Customers', 'Avg Recency (days)', 'Avg Frequency', 'Avg Spend (£)']
            summary = summary.reset_index()
            
            st.dataframe(
                summary, 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Segment": st.column_config.TextColumn("Segment"),
                    "Customers": st.column_config.NumberColumn("Customers", format="%d"),
                    "Avg Spend (£)": st.column_config.NumberColumn("Avg Spend", format="£%,.0f")
                }
            )
        
        # Top customers table (simple and clean)
        st.subheader("Top 10 Valuable Customers")
        top_customers = rfm.nlargest(10, 'Monetary')[['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary']]
        top_customers['Monetary'] = top_customers['Monetary'].round(0)
        st.dataframe(
            top_customers,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CustomerID": st.column_config.TextColumn("Customer ID"),
                "Monetary": st.column_config.NumberColumn("Total Spend", format="£%,.0f")
            }
        )
    else:
        st.warning("RFM data not available. Run clustering pipeline.")

# =============================
# Fix chart sizes globally - Add this at the top after imports
# =============================

# Set page config for professional look
st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
