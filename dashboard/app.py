import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# import your modules
from src.analysis import load_data
from src.forecasting import linear_regression_forecast, random_forest_forecast, arima_forecast
from src.clustering import perform_clustering

st.set_page_config(layout="wide")

# LOAD DATA
df = load_data()

st.title("Smart Retail Analytics Dashboard")

# -------------------------------
# KPI METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

revenue = df["TotalPrice"].sum()
orders = df["InvoiceNo"].nunique()
customers = df["CustomerID"].nunique()

col1.metric("Revenue", f"£{revenue:,.2f}")
col2.metric("Orders", orders)
col3.metric("Customers", customers)

# -------------------------------
# SALES TREND
# -------------------------------
st.subheader("Sales Trend")

fig = px.line(
    df,
    x="InvoiceDate",
    y="TotalPrice",
    title="Sales Trend Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# MONTHLY SALES
# -------------------------------
st.subheader("Monthly Sales")

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly = df.groupby("Month")["TotalPrice"].sum().reset_index()

fig = px.line(
    monthly,
    x="Month",
    y="TotalPrice",
    title="Monthly Sales Trend"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TOP PRODUCTS
# -------------------------------
st.subheader("Top Products")

top_products = df.groupby("Description")["Quantity"].sum().nlargest(10).reset_index()

fig = px.bar(
    top_products,
    x="Description",
    y="Quantity",
    title="Top Selling Products"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DEMAND FORECASTING
# -------------------------------
st.subheader("Demand Forecasting")

model = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "ARIMA"])

if model == "Linear Regression":
    future_dates, preds = linear_regression_forecast(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="Prediction"))

    fig.update_layout(title="Linear Regression Forecast")
    st.plotly_chart(fig, use_container_width=True)

elif model == "Random Forest":
    future_dates, preds = random_forest_forecast(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="Prediction"))

    fig.update_layout(title="Random Forest Forecast")
    st.plotly_chart(fig, use_container_width=True)

elif model == "ARIMA":
    future_dates, preds = arima_forecast(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="Prediction"))

    fig.update_layout(title="ARIMA Forecast")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------
st.subheader("Customer Segmentation")

cluster_df = perform_clustering(df)

fig = px.scatter(
    cluster_df,
    x="AnnualSpend",
    y="Frequency",
    color="Cluster",
    title="Customer Segmentation"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DISTRIBUTION
# -------------------------------
st.subheader("Revenue Distribution")

fig = px.histogram(
    df,
    x="TotalPrice",
    title="Revenue Distribution"
)

st.plotly_chart(fig, use_container_width=True)
