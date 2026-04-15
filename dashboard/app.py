# =============================
# IMPORTS 
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
from src.forecasting import create_time_features  

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pickle

# =============================
# PAGE CONFIG (PROFESSIONAL)
# =============================
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
    .stMetric > label {
        color: #1f77b4;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# AUTO PIPELINE (FIXED)
# =============================
try:
    if not check_required_files():
        print("Running preprocessing...")
        preprocess()
        print("Running clustering...")
        run_clustering()
        print("Pipeline complete!")
except Exception as e:
    st.error(f"Pipeline failed: {e}")

# =============================
# SESSION AUTH (SIMPLE)
# =============================
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

def login():
    st.title("🔐 Smart Retail Analytics - Login")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        username = st.text_input("Username", placeholder="admin / user")
    with col2:
        password = st.text_input("Password", type="password", placeholder="admin123 / user123")
    
    if st.button("Login", type="primary"):
        if username == "admin" and password == "admin123":
            st.session_state.user = username
            st.session_state.role = "admin"
            st.success("Logged in as Admin!")
            st.rerun()
        elif username == "user" and password == "user123":
            st.session_state.user = username
            st.session_state.role = "user"
            st.success("Logged in as User!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

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
    feature_cols = None
    try:
        if os.path.exists("models/random_forest.pkl"):
            models["rf"] = joblib.load("models/random_forest.pkl")
            if os.path.exists("models/rf_features.pkl"):
                with open("models/rf_features.pkl", "rb") as f:
                    feature_cols = pickle.load(f)
                models["feature_cols"] = feature_cols
    except Exception as e:
        st.warning(f"RF model not loaded: {e}")
    return models

models = load_models()

# =============================
# SIDEBAR (PROFESSIONAL)
# =============================
st.sidebar.title("📊 Smart Retail Analytics")
st.sidebar.markdown(f"👋 **Welcome, {st.session_state.user}**")
st.sidebar.markdown(f"🔑 **Role:** {st.session_state.role}")

pages = ["📈 Overview", "📉 Sales Trends", "📦 Product Analysis", 
         "🔮 Demand Forecast", "👥 Customer Segments", "📤 Upload Data"]

if st.session_state.role == "admin":
    pages.append("⚙️ Admin Panel")

page = st.sidebar.radio("Navigation", pages, index=0)

if st.sidebar.button("🚪 Logout", type="secondary"):
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

# =============================
# OVERVIEW (PROFESSIONAL)
# =============================
if page == "📈 Overview":
    st.title("📊 Dashboard Overview")
    st.markdown("---")
    
    if not data_loaded:
        st.info("⚠️ No data available. Please upload data first.")
    else:
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Total Revenue", f"£{metrics['Total Revenue']:,.0f}")
        col2.metric("📋 Total Orders", f"{metrics['Total Orders']:,}")
        col3.metric("👤 Unique Customers", f"{metrics['Unique Customers']:,}")
        col4.metric("📦 Unique Products", f"{metrics['Unique Products']:,}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(monthly, x="MonthYear", y="Revenue", 
                         markers=True, title="Monthly Revenue Trend")
            fig.update_layout(height=400, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top = get_top_products(df, 10)
            fig = px.bar(top, x="Revenue", y="Product", orientation='h',
                        title="Top 10 Products by Revenue")
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

# =============================
# SALES TRENDS
# =============================
elif page == "📉 Sales Trends":
    st.title("📉 Sales Trends Analysis")
    st.markdown("---")
    
    if data_loaded:
        fig = px.line(monthly, x="MonthYear", y="Revenue", 
                     markers=True, title="Monthly Revenue Over Time")
        fig.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# =============================
# PRODUCT ANALYSIS
# =============================
elif page == "📦 Product Analysis":
    st.title("📦 Product Performance")
    st.markdown("---")
    
    if data_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            top = get_top_products(df, 10)
            fig = px.bar(top, x="Revenue", y="Product", orientation='h',
                        title="Top 10 Products")
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_countries = get_top_countries(df, 10)
            fig = px.bar(top_countries, x="Revenue", y="Country", orientation='h',
                        title="Top Countries by Revenue")
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

# =============================
# DEMAND FORECAST (DROPDOWN FIXED)
# =============================
elif page == "🔮 Demand Forecast":
    st.title("🔮 Demand Forecasting")
    st.markdown("---")
    
    if data_loaded and len(monthly) > 5:
        col1, col2 = st.columns(2)
        
        with col1:
            model_options = ["Linear Trend", "Recent Average", "Random Forest"]
            selected_model = st.selectbox("🎯 Select Model:", model_options)
            horizon = st.slider("📅 Forecast Horizon (months)", 1, 12, 6)
        
        # Generate forecast
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
            
        elif selected_model == "Recent Average":
            avg = monthly["Revenue"].tail(3).mean()
            pred = [avg] * horizon
            
        else:  # Random Forest
            pred = [monthly["Revenue"].tail(3).mean()] * horizon
            try:
                if "rf" in models and "feature_cols" in models:
                    feature_df = create_time_features(monthly)
                    future_features = feature_df[models["feature_cols"]].tail(horizon)
                    pred = models["rf"].predict(future_features)
            except:
                pass
        
        # Professional chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly["MonthYear"], 
            y=monthly["Revenue"],
            mode='lines+markers',
            name='Actual Revenue',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred,
            mode='lines+markers',
            name=f'{selected_model} Forecast',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.2)'
        ))
        
        fig.update_layout(
            title=f"Revenue Forecast - {horizon} Months Ahead",
            xaxis_title="Date",
            yaxis_title="Revenue (£)",
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Last Month", f"£{monthly['Revenue'].iloc[-1]:,.0f}")
        col2.metric("3-Month Avg", f"£{monthly['Revenue'].tail(3).mean():,.0f}")
        col3.metric("Next Month", f"£{pred[0]:,.0f}")
        
        # Export
        if st.button("📥 Download Forecast"):
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Revenue': pred})
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV", csv, "revenue_forecast.csv", "text/csv"
            )
    else:
        st.warning("❌ Not enough data for forecasting (need 6+ months)")

# =============================
# CUSTOMER SEGMENTS (TABLE FIXED)
# =============================
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation (RFM Analysis)")
    st.markdown("---")
    
    if data_loaded and rfm is not None and not rfm.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Scatter Plot
            fig = px.scatter(
                rfm.sample(min(1000, len(rfm))),  # Sample for performance
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
            # Segment Summary Table (KEY METRICS ONLY)
            st.subheader("📋 Segment Summary")
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
        
        # Top Customers (SIMPLE TABLE)
        st.subheader("⭐ Top 10 Valuable Customers")
        top_customers = rfm.nlargest(10, 'Monetary')[['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary']]
        top_customers['Monetary'] = top_customers['Monetary'].round(0)
        top_customers.columns = ['ID', 'Segment', 'Recency (days)', 'Frequency', 'Total Spend (£)']
        
        st.dataframe(
            top_customers,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total Spend (£)": st.column_config.NumberColumn(format="£%,.0f")
            }
        )
    else:
        st.warning("❌ RFM data not available. Run: `python src/clustering.py`")

# =============================
# UPLOAD DATA
# =============================
elif page == "📤 Upload Data":
    st.title("📤 Data Upload")
    st.markdown("---")
    
    file = st.file_uploader("Choose Excel/CSV file", type=["csv", "xlsx"])
    
    if file:
        try:
            if file.name.endswith(".csv"):
                df_new = pd.read_csv(file)
            else:
                df_new = pd.read_excel(file)
            
            os.makedirs("data/raw", exist_ok=True)
            output_path = "data/raw/retail_sales.xlsx"
            df_new.to_excel(output_path, index=False)
            
            st.success(f"✅ File uploaded! Saved to {output_path}")
            st.info("🔄 Run preprocessing: `python src/data_preprocessing.py`")
            
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

# =============================
# ADMIN PANEL
# =============================
elif page == "⚙️ Admin Panel":
    st.title("⚙️ Admin Panel")
    st.markdown("---")
    
    check_required_files()
    
    st.subheader("🔄 Pipeline Status")
    if st.button("🔄 Re-run Full Pipeline", type="primary"):
        try:
            preprocess()
            run_clustering()
            st.success("✅ Pipeline re-run complete!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
