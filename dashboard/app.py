import sys
import os

# Make src folder accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products, get_top_countries
)

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# =============================
# SESSION AUTH
# =============================

if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

def login():
    st.title("Login")

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
    df = load_clean_data()
    rfm = load_rfm()
    return df, rfm

df, rfm = load_all()
metrics = summary_metrics(df)
monthly = get_monthly_revenue(df)

# =============================
# SIDEBAR
# =============================

st.sidebar.title("Smart Retail Analytics")

page = st.sidebar.radio("Navigation", [
    "Overview",
    "Sales Trends",
    "Product Analysis",
    "Demand Forecast",
    "Customer Segments",
    "Upload Data",
    "Admin Panel"
])

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# =============================
# OVERVIEW
# =============================

if page == "Overview":
    st.title("Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", f"£{metrics['Total Revenue']:,.2f}")
    c2.metric("Orders", f"{metrics['Total Orders']:,}")
    c3.metric("Customers", f"{metrics['Unique Customers']:,}")

    st.subheader("Sales Trend")

    fig = px.line(monthly, x="MonthYear", y="Revenue")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# SALES
# =============================

elif page == "Sales Trends":
    st.title("Sales Trends")

    fig = px.line(monthly, x="MonthYear", y="Revenue")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# PRODUCT ANALYSIS
# =============================

elif page == "Product Analysis":
    st.title("Top Products")

    top = get_top_products(df, 10)

    fig = px.bar(top, x="Product", y="Revenue")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# DEMAND FORECAST
# =============================

elif page == "Demand Forecast":
    st.title("Demand Forecast")

    model_choice = st.selectbox("Select Model", [
        "Linear Regression",
        "Random Forest",
        "ARIMA"
    ])

    months_ahead = st.slider("Forecast Months", 3, 12, 6)

    # ---------- Linear Regression ----------
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(monthly[["MonthIndex"]], monthly["Revenue"])

        last_idx = monthly["MonthIndex"].max()
        future_idx = pd.DataFrame({
            "MonthIndex": range(last_idx + 1, last_idx + months_ahead + 1)
        })

        preds = model.predict(future_idx)

        future_dates = pd.date_range(
            start=monthly["MonthYear"].max() + pd.DateOffset(months=1),
            periods=months_ahead,
            freq="MS"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["MonthYear"], y=monthly["Revenue"], name="Actual"))
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Month": future_dates.strftime("%b %Y"),
            "Forecast": preds
        }))

    # ---------- Random Forest ----------
    elif model_choice == "Random Forest":
        st.info("Random Forest uses trained model")

    # ---------- ARIMA ----------
    elif model_choice == "ARIMA":
        series = monthly.set_index("MonthYear")["Revenue"]

        model = ARIMA(series, order=(1,1,1))
        fitted = model.fit()

        forecast = fitted.forecast(steps=months_ahead)

        future_dates = pd.date_range(
            start=series.index[-1] + pd.DateOffset(months=1),
            periods=months_ahead,
            freq="MS"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual"))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Month": future_dates.strftime("%b %Y"),
            "Forecast": forecast
        }))

# =============================
# CUSTOMER SEGMENTS
# =============================

elif page == "Customer Segments":
    st.title("Customer Segments")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        fig = px.bar(seg_counts, x="Segment", y="Customers")
        st.plotly_chart(fig, use_container_width=True)

    # RFM TABLE
    st.subheader("RFM Summary")
    summary = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    ).round(2).reset_index()

    st.dataframe(summary)

    # TOP PRODUCT PER CUSTOMER
    st.subheader("Top Product per Customer")

    cust_prod = df.groupby(["CustomerID", "Description"])["Quantity"].sum().reset_index()
    idx = cust_prod.groupby("CustomerID")["Quantity"].idxmax()
    top_customer = cust_prod.loc[idx]

    st.dataframe(top_customer)

# =============================
# UPLOAD DATA
# =============================

elif page == "Upload Data":
    st.title("Upload Data")

    file = st.file_uploader("Upload file")

    if file:
        df_new = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

        os.makedirs("data/raw", exist_ok=True)
        df_new.to_excel("data/raw/retail_sales.xlsx", index=False)

        st.success("File uploaded successfully")

# =============================
# ADMIN PANEL
# =============================

elif page == "Admin Panel":
    st.title("Admin Panel")

    if st.session_state.role != "admin":
        st.warning("Access Denied")
    else:
        st.success("Admin Access Granted")
