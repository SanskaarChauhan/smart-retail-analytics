# =============================
# IMPORTS
# =============================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.linear_model import LinearRegression

from src.data_preprocessing import preprocess
from src.clustering import run_clustering
from src.utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products,
    check_required_files
)

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Smart Retail Analytics",
    layout="wide"
)

# =============================
# AUTO PIPELINE
# =============================
try:
    if not check_required_files():
        preprocess()
        run_clustering()
except:
    pass

# =============================
# SESSION
# =============================
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

def login():
    st.title("Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "admin":
            st.session_state.user = u
            st.session_state.role = "admin"
            st.rerun()
        elif u == "user" and p == "user":
            st.session_state.user = u
            st.session_state.role = "user"
            st.rerun()
        else:
            st.error("Invalid")

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
    except:
        return None, None, False

df, rfm, ok = load_all()

if ok:
    metrics = summary_metrics(df)
    monthly = get_monthly_revenue(df)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("Smart Retail Analytics")
st.sidebar.write(f"{st.session_state.user}")

pages = [
    "Overview",
    "Sales Trends",
    "Product Analysis",
    "Demand Forecast",
    "Customer Segments"
]

if st.session_state.role == "admin":
    pages += ["Upload Data"]

page = st.sidebar.radio("Navigation", pages)

# =============================
# OVERVIEW
# =============================
if page == "Overview":
    st.title("Overview")

    if ok:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"£{metrics['Total Revenue']:,.0f}")
        c2.metric("Orders", metrics["Total Orders"])
        c3.metric("Customers", metrics["Unique Customers"])
        c4.metric("Products", metrics["Unique Products"])

# =============================
# SALES
# =============================
elif page == "Sales Trends":
    st.title("Sales Trends")

    if ok:
        fig = px.line(monthly, x="MonthYear", y="Revenue")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# =============================
# PRODUCT
# =============================
elif page == "Product Analysis":
    st.title("Product Analysis")

    if ok:
        top = get_top_products(df)

        fig = px.bar(top, x="Revenue", y="Product", orientation="h")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# =============================
# DEMAND FORECAST (FIXED)
# =============================
elif page == "Demand Forecast":
    st.title("Demand Forecast")

    if ok:
        model_type = st.selectbox(
            "Select Model",
            ["Linear Trend", "Recent Average"]
        )

        months = st.slider("Months", 1, 12, 6)

        last_idx = monthly["MonthIndex"].max()

        future_idx = pd.DataFrame({
            "MonthIndex": range(last_idx + 1, last_idx + months + 1)
        })

        if model_type == "Linear Trend":
            model = LinearRegression()
            model.fit(monthly[["MonthIndex"]], monthly["Revenue"])
            pred = model.predict(future_idx)
        else:
            avg = monthly["Revenue"].tail(3).mean()
            pred = [avg] * months

        future_dates = pd.date_range(
            start=monthly["MonthYear"].max(),
            periods=months + 1,
            freq="MS"
        )[1:]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly["MonthYear"],
            y=monthly["Revenue"],
            name="Actual"
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred,
            name="Forecast"
        ))

        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

# =============================
# CUSTOMER SEGMENTS (FIXED)
# =============================
elif page == "Customer Segments":
    st.title("Customer Segments")

    if ok:
        fig = px.scatter(
            rfm,
            x="Frequency",
            y="Monetary",
            color="Segment"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # CLEAN TABLE (IMPORTANT)
        summary = rfm.groupby("Segment").agg({
            "CustomerID": "count",
            "Monetary": "mean",
            "Frequency": "mean"
        }).reset_index()

        summary.columns = ["Segment", "Customers", "Avg Spend", "Avg Frequency"]

        st.subheader("Segment Summary")
        st.dataframe(summary, use_container_width=True)

# =============================
# UPLOAD
# =============================
elif page == "Upload Data":
    st.title("Upload Data")

    file = st.file_uploader("Upload", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df_new = pd.read_csv(file)
        else:
            df_new = pd.read_excel(file)

        os.makedirs("data/raw", exist_ok=True)
        df_new.to_excel("data/raw/retail_sales.xlsx", index=False)

        st.success("Uploaded")
