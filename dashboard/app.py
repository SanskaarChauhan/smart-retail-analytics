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
import joblib

# ✅ FIXED IMPORT
from utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products, get_top_countries
)

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="📊",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────

@st.cache_data
def load_all_data():
    df  = load_clean_data()
    rfm = load_rfm()
    return df, rfm

@st.cache_resource
def load_models():
    models = {}
    paths = {
        "lr" : "models/linear_regression.pkl",
        "rf" : "models/random_forest.pkl",
        "km" : "models/kmeans.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

df, rfm  = load_all_data()
models   = load_models()
metrics  = summary_metrics(df)
monthly  = get_monthly_revenue(df)

# ── Sidebar ───────────────────────────────────────────────────────

st.sidebar.title("📊 Smart Retail Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Sales Trends",
    "Product Analysis",
    "Demand Forecast",
    "Customer Segments"
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {metrics['Date From']} → {metrics['Date To']}")
st.sidebar.caption("BCA Final Year Project")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("📊 Smart Retail Analytics Dashboard")
    st.markdown("AI-Based Retail Sales Analysis and Demand Forecasting System")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue",    f"£{metrics['Total Revenue']:,.2f}")
    c2.metric("Total Orders",     f"{metrics['Total Orders']:,}")
    c3.metric("Unique Customers", f"{metrics['Unique Customers']:,}")
    c4.metric("Unique Products",  f"{metrics['Unique Products']:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Revenue Trend")
        fig = px.line(
            monthly, x="MonthYear", y="Revenue",
            markers=True, labels={"MonthYear": "Month", "Revenue": "Revenue (£)"}
        )
        fig.update_traces(hovertemplate="Month: %{x}<br>Revenue: £%{y:,.2f}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Products by Revenue")
        top = get_top_products(df, 10)
        fig = px.bar(
            top.sort_values("Revenue"), x="Revenue", y="Product",
            orientation="h", labels={"Revenue": "Revenue (£)", "Product": ""}
        )
        fig.update_traces(hovertemplate="Product: %{y}<br>Revenue: £%{x:,.2f}")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — SALES TRENDS
# ══════════════════════════════════════════════════════════════════

elif page == "Sales Trends":
    st.title("📈 Sales Trends")
    st.markdown("---")

    years = sorted(df["Year"].unique())
    selected_year = st.selectbox("Filter by Year", ["All"] + [str(y) for y in years])
    filtered = df if selected_year == "All" else df[df["Year"] == int(selected_year)]
    monthly_f = get_monthly_revenue(filtered)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Revenue")
        fig = px.line(
            monthly_f, x="MonthYear", y="Revenue",
            markers=True, labels={"MonthYear": "Month", "Revenue": "Revenue (£)"}
        )
        fig.update_traces(hovertemplate="Month: %{x}<br>Revenue: £%{y:,.2f}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Orders by Day of Week")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = filtered.groupby("DayOfWeek")["InvoiceNo"].nunique().reset_index()
        dow.columns = ["DayOfWeek", "Orders"]
        dow["DayName"] = dow["DayOfWeek"].apply(lambda x: days[x])
        fig = px.bar(
            dow, x="DayName", y="Orders",
            labels={"DayName": "Day", "Orders": "Number of Orders"}
        )
        fig.update_traces(hovertemplate="Day: %{x}<br>Orders: %{y}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue Growth (%)")
    monthly_f = monthly_f.copy()
    monthly_f["Growth"] = monthly_f["Revenue"].pct_change() * 100
    fig = px.bar(
        monthly_f.dropna(), x="MonthYear", y="Growth",
        labels={"MonthYear": "Month", "Growth": "Growth (%)"}
    )
    fig.update_traces(hovertemplate="Month: %{x}<br>Growth: %{y:.2f}%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Revenue Table")
    st.dataframe(
        monthly_f[["MonthYear", "Revenue"]].rename(
            columns={"MonthYear": "Month", "Revenue": "Revenue (£)"}
        ), use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT ANALYSIS
# ══════════════════════════════════════════════════════════════════

elif page == "Product Analysis":
    st.title("🛒 Product Analysis")
    st.markdown("---")

    n = st.slider("Number of top products to show", 5, 20, 10)
    top      = get_top_products(df, n)
    countries = get_top_countries(df, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top {n} Products by Revenue")
        fig = px.bar(
            top.sort_values("Revenue"), x="Revenue", y="Product",
            orientation="h", labels={"Revenue": "Revenue (£)", "Product": ""}
        )
        fig.update_traces(hovertemplate="Product: %{y}<br>Revenue: £%{x:,.2f}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Countries by Revenue")
        fig = px.bar(
            countries, x="Country", y="Revenue",
            labels={"Revenue": "Revenue (£)"}
        )
        fig.update_traces(hovertemplate="Country: %{x}<br>Revenue: £%{y:,.2f}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Revenue Table")
    st.dataframe(top, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════

elif page == "Demand Forecast":
    st.title("🔮 Demand Forecast")
    st.markdown("---")

    model_choice = st.selectbox("Select Forecast Model", [
        "Linear Regression", "Random Forest", "ARIMA"
    ])
    months_ahead = st.slider("Months to forecast", 3, 12, 6)
    st.markdown("---")

    if model_choice == "Linear Regression":
        st.subheader("Linear Regression Forecast")
        model = models.get("lr") or LinearRegression().fit(
            monthly[["MonthIndex"]], monthly["Revenue"]
        )
        last_idx     = monthly["MonthIndex"].max()
        future_idx   = pd.DataFrame({"MonthIndex": range(last_idx + 1, last_idx + months_ahead + 1)})
        future_pred  = model.predict(future_idx)
        last_date    = monthly["MonthYear"].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["MonthYear"], y=monthly["Revenue"],
            mode="lines+markers", name="Actual",
            hovertemplate="Month: %{x}<br>Revenue: £%{y:,.2f}"
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_pred,
            mode="lines+markers", name="Forecast",
            line=dict(dash="dash"),
            hovertemplate="Month: %{x}<br>Forecast: £%{y:,.2f}"
        ))
        fig.update_layout(
            title="Linear Regression — Revenue Forecast",
            xaxis_title="Month", yaxis_title="Revenue (£)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Table")
        st.dataframe(pd.DataFrame({
            "Month"        : [d.strftime("%b %Y") for d in future_dates],
            "Forecast (£)" : [round(v, 2) for v in future_pred]
        }), use_container_width=True)

    elif model_choice == "Random Forest":
        st.subheader("Random Forest — Actual vs Predicted")
        if "rf" in models:
            monthly_rf = monthly.copy()
            monthly_rf["Month"]       = monthly_rf["MonthYear"].dt.month
            monthly_rf["Quarter"]     = monthly_rf["MonthYear"].dt.quarter
            monthly_rf["RevLag1"]     = monthly_rf["Revenue"].shift(1)
            monthly_rf["RevLag2"]     = monthly_rf["Revenue"].shift(2)
            monthly_rf["RollingMean"] = monthly_rf["Revenue"].rolling(3).mean()
            monthly_rf = monthly_rf.dropna().reset_index(drop=True)

            features = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", "RollingMean"]
            y_pred   = models["rf"].predict(monthly_rf[features])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_rf["MonthYear"], y=monthly_rf["Revenue"],
                mode="lines+markers", name="Actual",
                hovertemplate="Month: %{x}<br>Revenue: £%{y:,.2f}"
            ))
            fig.add_trace(go.Scatter(
                x=monthly_rf["MonthYear"], y=y_pred,
                mode="lines+markers", name="Predicted",
                line=dict(dash="dash"),
                hovertemplate="Month: %{x}<br>Predicted: £%{y:,.2f}"
            ))
            fig.update_layout(
                title="Random Forest — Actual vs Predicted",
                xaxis_title="Month", yaxis_title="Revenue (£)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Random Forest model not found. Run src/forecasting.py first.")

    elif model_choice == "ARIMA":
        st.subheader("ARIMA Forecast")
        with st.spinner("Running ARIMA model..."):
            revenue_series = monthly.set_index("MonthYear")["Revenue"]
            fitted = ARIMA(revenue_series, order=(1, 1, 1)).fit()
            forecast = fitted.forecast(steps=months_ahead)
            forecast_index = pd.date_range(
                start=revenue_series.index[-1] + pd.DateOffset(months=1),
                periods=months_ahead, freq="MS"
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=revenue_series.index, y=revenue_series.values,
            mode="lines+markers", name="Actual",
            hovertemplate="Month: %{x}<br>Revenue: £%{y:,.2f}"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_index, y=forecast.values,
            mode="lines+markers", name="Forecast",
            line=dict(dash="dash"),
            hovertemplate="Month: %{x}<br>Forecast: £%{y:,.2f}"
        ))
        fig.update_layout(
            title="ARIMA — Revenue Forecast",
            xaxis_title="Month", yaxis_title="Revenue (£)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Table")
        st.dataframe(pd.DataFrame({
            "Month"        : [d.strftime("%b %Y") for d in forecast_index],
            "Forecast (£)" : [round(v, 2) for v in forecast.values]
        }), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════

elif page == "Customer Segments":
    st.title("👥 Customer Segments")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(rfm):,}")
    c2.metric("Avg Frequency",   f"{rfm['Frequency'].mean():.1f} orders")
    c3.metric("Avg Revenue",     f"£{rfm['Monetary'].mean():,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        fig = px.bar(
            seg_counts, x="Segment", y="Customers",
            labels={"Customers": "Number of Customers"}
        )
        fig.update_traces(hovertemplate="Segment: %{x}<br>Customers: %{y}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Frequency vs Monetary")
        fig = px.scatter(
            rfm, x="Frequency", y="Monetary",
            color="Segment", opacity=0.6,
            labels={"Frequency": "Order Frequency", "Monetary": "Revenue (£)"},
            hover_data=["CustomerID", "Recency"]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("RFM Segment Summary")
    summary = rfm.groupby("Segment").agg(
        Customers     = ("CustomerID", "count"),
        Avg_Recency   = ("Recency",    "mean"),
        Avg_Frequency = ("Frequency",  "mean"),
        Avg_Monetary  = ("Monetary",   "mean")
    ).round(2).reset_index()
    st.dataframe(summary, use_container_width=True)
