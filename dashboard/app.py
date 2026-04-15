# ============================================================
# dashboard/app.py
# ============================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import pickle

from src.utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products, get_top_countries,
    check_required_files
)
from src.auth import (
    init_files, load_users, load_pending,
    approve_user, reject_user, get_pending_requests, delete_user
)
from src.data_preprocessing import preprocess
from src.clustering import run_clustering

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# ── Try importing optional advanced models ────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stMetric > label { color: #1f77b4; font-size: 14px; font-weight: bold; }
    .pending-card {
        background: #fff3cd;
        padding: 1rem 1.2rem;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        margin: 0.6rem 0;
    }
    .user-card {
        background: #f8f9fa;
        padding: 0.8rem 1.2rem;
        border-left: 5px solid #0d6efd;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .admin-badge {
        background: #dc3545;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .employee-badge {
        background: #0d6efd;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# AUTH INIT
# ============================================================
init_files()

if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.title("Smart Retail Analytics")
    st.markdown("Company Internal Portal")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("Sign In")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        st.markdown("")

        if st.button("Login", type="primary", use_container_width=True):
            from src.auth import login as auth_login
            success, role = auth_login(username, password)
            if success:
                st.session_state.user = username
                st.session_state.role = role
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials. Contact your administrator.")

        st.markdown("---")
        st.caption("Don't have an account? Contact your company admin.")

        with st.expander("Request Access"):
            new_user = st.text_input("New Username", key="req_user")
            new_pass = st.text_input("New Password", type="password", key="req_pass")
            if st.button("Submit Request", key="submit_req"):
                if new_user and new_pass:
                    from src.auth import signup_request
                    result = signup_request(new_user, new_pass)
                    st.info(f"{result}")
                else:
                    st.warning("Please fill in both fields.")

if st.session_state.user is None:
    show_login()
    st.stop()

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_all_data():
    try:
        df  = load_clean_data()
        rfm = load_rfm()
        return df, rfm, True
    except Exception as e:
        return None, None, False

@st.cache_resource
def load_models():
    models = {}
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paths = {
        "lr"  : os.path.join(base, "models/linear_regression.pkl"),
        "rf"  : os.path.join(base, "models/random_forest.pkl"),
        "xgb" : os.path.join(base, "models/xgboost.pkl"),
        "prophet": os.path.join(base, "models/prophet.pkl"),
    }
    for key, path in paths.items():
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except Exception:
                pass
    feat_path = os.path.join(base, "models/rf_features.pkl")
    if os.path.exists(feat_path):
        try:
            with open(feat_path, "rb") as f:
                models["feature_cols"] = pickle.load(f)
        except Exception:
            pass
    return models

df, rfm, data_loaded = load_all_data()
models = load_models()

if data_loaded:
    metrics = summary_metrics(df)
    monthly = get_monthly_revenue(df)
else:
    metrics = {}
    monthly = None

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Smart Retail Analytics")
st.sidebar.markdown(f"Hello {st.session_state.user.title()}")

if st.session_state.role == "admin":
    st.sidebar.markdown('<span class="admin-badge">ADMIN</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="employee-badge">EMPLOYEE</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")

if st.session_state.role == "admin":
    pages = [
        "Overview",
        "Sales Trends",
        "Product Analysis",
        "Demand Forecast",
        "Customer Segments",
        "Upload Data",
        "Admin Panel"
    ]
else:
    pages = [
        "Overview",
        "Sales Trends",
        "Product Analysis",
        "Demand Forecast",
        "Customer Segments"
    ]

page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
if data_loaded:
    st.sidebar.caption(f"Data: {metrics.get('Date From','N/A')} → {metrics.get('Date To','N/A')}")
status = "Online" if data_loaded else "Offline"
st.sidebar.caption(f"Status: {status}")

if st.sidebar.button("Logout", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ============================================================
# ADMIN GUARD
# ============================================================
def admin_only():
    if st.session_state.role != "admin":
        st.error("Admin access required.")
        st.stop()

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Executive Dashboard")
    st.markdown("AI-Based Smart Retail Analytics and Demand Forecasting System")
    st.markdown("---")

    if not data_loaded:
        st.warning("No data available. Admin must upload and run the pipeline first.")
    else:
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
                markers=True,
                labels={"MonthYear": "Month", "Revenue": "Revenue (£)"}
            )
            fig.update_traces(hovertemplate="<b>Month:</b> %{x}<br><b>Revenue:</b> £%{y:,.2f}<extra></extra>")
            fig.update_layout(height=420, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 10 Products by Revenue")
            top = get_top_products(df, 10)
            fig = px.bar(
                top.sort_values("Revenue"),
                x="Revenue", y="Product", orientation="h",
                labels={"Revenue": "Revenue (£)", "Product": ""}
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.2f}<extra></extra>")
            fig.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Revenue by Country")
            countries = get_top_countries(df, 8)
            fig = px.pie(
                countries, values="Revenue", names="Country",
                hole=0.4
            )
            fig.update_traces(hovertemplate="<b>%{label}</b><br>Revenue: £%{value:,.2f}<br>%{percent}<extra></extra>")
            fig.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Orders by Day of Week")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow = df.groupby("DayOfWeek")["InvoiceNo"].nunique().reset_index()
            dow.columns = ["DayOfWeek", "Orders"]
            dow["Day"] = dow["DayOfWeek"].apply(lambda x: days[x])
            fig = px.bar(
                dow, x="Day", y="Orders",
                labels={"Day": "Day of Week", "Orders": "Number of Orders"}
            )
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Orders: %{y}<extra></extra>")
            fig.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("Hourly Sales Distribution")
            hourly = df.groupby("Hour")["TotalPrice"].sum().reset_index()
            hourly.columns = ["Hour", "Revenue"]
            fig = px.area(
                hourly, x="Hour", y="Revenue",
                labels={"Hour": "Hour of Day", "Revenue": "Revenue (£)"}
            )
            fig.update_traces(hovertemplate="<b>Hour %{x}:00</b><br>Revenue: £%{y:,.2f}<extra></extra>")
            fig.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 — SALES TRENDS
# ============================================================
elif page == "Sales Trends":
    st.title("Sales Trends Analysis")
    st.markdown("---")

    if not data_loaded:
        st.warning("No data available.")
    else:
        years = sorted(df["Year"].unique())
        selected_year = st.selectbox("Filter by Year", ["All"] + [str(y) for y in years])
        filtered = df if selected_year == "All" else df[df["Year"] == int(selected_year)]
        monthly_f = get_monthly_revenue(filtered)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Monthly Revenue")
            fig = px.line(
                monthly_f, x="MonthYear", y="Revenue",
                markers=True,
                labels={"MonthYear": "Month", "Revenue": "Revenue (£)"}
            )
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Revenue: £%{y:,.2f}<extra></extra>")
            fig.update_layout(height=400, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Orders by Day of Week")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow = filtered.groupby("DayOfWeek")["InvoiceNo"].nunique().reset_index()
            dow.columns = ["DayOfWeek", "Orders"]
            dow["Day"] = dow["DayOfWeek"].apply(lambda x: days[x])
            fig = px.bar(
                dow, x="Day", y="Orders",
                labels={"Day": "Day", "Orders": "Orders"}
            )
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Orders: %{y}<extra></extra>")
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Revenue Growth (%)")
        monthly_g = monthly_f.copy()
        monthly_g["Growth"] = monthly_g["Revenue"].pct_change() * 100
        monthly_g = monthly_g.dropna()
        fig = px.bar(
            monthly_g, x="MonthYear", y="Growth",
            labels={"MonthYear": "Month", "Growth": "Growth (%)"},
            color="Growth",
            color_continuous_scale=["red", "lightgray", "green"],
            color_continuous_midpoint=0
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>")
        fig.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Revenue Data")
        st.dataframe(
            monthly_f[["MonthYear", "Revenue"]].rename(
                columns={"MonthYear": "Month", "Revenue": "Revenue (£)"}
            ).reset_index(drop=True),
            use_container_width=True
        )

# ============================================================
# PAGE 3 — PRODUCT ANALYSIS
# ============================================================
elif page == "Product Analysis":
    st.title("Product Performance Analysis")
    st.markdown("---")

    if not data_loaded:
        st.warning("No data available.")
    else:
        n = st.slider("Number of top products", 5, 20, 10)
        top      = get_top_products(df, n)
        countries = get_top_countries(df, 10)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Top {n} Products by Revenue")
            fig = px.bar(
                top.sort_values("Revenue"),
                x="Revenue", y="Product", orientation="h",
                labels={"Revenue": "Revenue (£)", "Product": ""}
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.2f}<extra></extra>")
            fig.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 10 Countries by Revenue")
            fig = px.bar(
                countries.sort_values("Revenue"),
                x="Revenue", y="Country", orientation="h",
                labels={"Revenue": "Revenue (£)", "Country": ""}
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.2f}<extra></extra>")
            fig.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Product Revenue Table")
        st.dataframe(top.reset_index(drop=True), use_container_width=True)

# ============================================================
# PAGE 4 — DEMAND FORECAST
# ============================================================
elif page == "Demand Forecast":
    st.title("Demand Forecasting")
    st.markdown("---")

    if not data_loaded:
        st.warning("No data available.")
    elif len(monthly) < 6:
        st.warning("Not enough data for forecasting (need at least 6 months).")
    else:
        available_models = ["Linear Regression", "Random Forest", "ARIMA"]

        if XGB_AVAILABLE:
            available_models.append("XGBoost")

        if PROPHET_AVAILABLE:
            available_models.append("Prophet")

        col1, col2 = st.columns([2, 1])
        with col1:
            model_choice = st.selectbox("Select Forecast Model", available_models)
        with col2:
            months_ahead = st.slider("Months to Forecast", 3, 12, 6)

        st.markdown("---")

        if model_choice == "Linear Regression":
            st.subheader("Linear Regression Forecast")

            model = models.get("lr") or LinearRegression().fit(
                monthly[["MonthIndex"]], monthly["Revenue"]
            )

            last_idx = monthly["MonthIndex"].max()

            future_idx = pd.DataFrame({
                "MonthIndex": list(range(last_idx + 1, last_idx + months_ahead + 1))
            })

            future_pred = model.predict(future_idx)

            last_date = monthly["MonthYear"].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["MonthYear"], y=monthly["Revenue"],
                mode="lines+markers", name="Actual",
                line=dict(color="royalblue", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=future_pred,
                mode="lines+markers", name="Forecast",
                line=dict(color="orange", dash="dash")
            ))

            st.plotly_chart(fig, use_container_width=True)

        elif model_choice == "Random Forest":
            st.subheader("Random Forest — Actual vs Predicted")

            if "rf" in models:
                monthly_rf = monthly.copy()

                monthly_rf["Month"] = monthly_rf["MonthYear"].dt.month
                monthly_rf["Quarter"] = monthly_rf["MonthYear"].dt.quarter
                monthly_rf["RevLag1"] = monthly_rf["Revenue"].shift(1)
                monthly_rf["RevLag2"] = monthly_rf["Revenue"].shift(2)
                monthly_rf["RollingMean"] = monthly_rf["Revenue"].rolling(3).mean()

                monthly_rf = monthly_rf.dropna().reset_index(drop=True)

                features = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", "RollingMean"]

                y_pred = models["rf"].predict(monthly_rf[features])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_rf["MonthYear"], y=monthly_rf["Revenue"], name="Actual"
                ))
                fig.add_trace(go.Scatter(
                    x=monthly_rf["MonthYear"], y=y_pred, name="Predicted"
                ))

                st.plotly_chart(fig, use_container_width=True)

        elif model_choice == "ARIMA":
            st.subheader("ARIMA Forecast")

            revenue_series = monthly.set_index("MonthYear")["Revenue"]

            fitted = ARIMA(revenue_series, order=(1, 1, 1)).fit()

            forecast = fitted.forecast(steps=months_ahead)

            forecast_index = pd.date_range(
                start=revenue_series.index[-1] + pd.DateOffset(months=1),
                periods=months_ahead, freq="MS"
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=revenue_series.index, y=revenue_series.values, name="Actual"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_index, y=forecast.values, name="Forecast"
            ))

            st.plotly_chart(fig, use_container_width=True)

        elif model_choice == "XGBoost":
            st.subheader("XGBoost Forecast")

            if not XGB_AVAILABLE:
                st.error("XGBoost not installed.")
            else:
                monthly_xgb = monthly.copy()

                monthly_xgb["Month"] = monthly_xgb["MonthYear"].dt.month
                monthly_xgb["Quarter"] = monthly_xgb["MonthYear"].dt.quarter
                monthly_xgb["RevLag1"] = monthly_xgb["Revenue"].shift(1)
                monthly_xgb["RevLag2"] = monthly_xgb["Revenue"].shift(2)
                monthly_xgb["RollingMean"] = monthly_xgb["Revenue"].rolling(3).mean()

                monthly_xgb = monthly_xgb.dropna()

                features = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", "RollingMean"]

                model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
                model.fit(monthly_xgb[features], monthly_xgb["Revenue"])

                y_pred = model.predict(monthly_xgb[features])

                last_idx = monthly_xgb["MonthIndex"].max()

                future = pd.DataFrame({
                    "MonthIndex": range(last_idx + 1, last_idx + months_ahead + 1),
                    "Month": [(monthly_xgb["Month"].iloc[-1] + i) % 12 + 1 for i in range(months_ahead)],
                    "Quarter": [(monthly_xgb["Quarter"].iloc[-1] + i) % 4 + 1 for i in range(months_ahead)],
                    "RevLag1": [monthly_xgb["Revenue"].iloc[-1]] * months_ahead,
                    "RevLag2": [monthly_xgb["Revenue"].iloc[-2]] * months_ahead,
                    "RollingMean": [monthly_xgb["Revenue"].tail(3).mean()] * months_ahead
                })

                future_pred = model.predict(future)

                last_date = monthly_xgb["MonthYear"].iloc[-1]
                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=monthly_xgb["MonthYear"],
                    y=monthly_xgb["Revenue"],
                    name="Actual",
                    line=dict(color="blue")
                ))

                fig.add_trace(go.Scatter(
                    x=monthly_xgb["MonthYear"],
                    y=y_pred,
                    name="XGB Fit",
                    line=dict(color="orange")
                ))

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_pred,
                    name="Future Forecast",
                    line=dict(color="green", dash="dash")
                ))

                st.plotly_chart(fig, use_container_width=True)

        elif model_choice == "Prophet":
            st.subheader("Prophet Forecast")

            if not PROPHET_AVAILABLE:
                st.error("Prophet not installed.")
            else:
                prophet_df = monthly[["MonthYear", "Revenue"]].rename(
                    columns={"MonthYear": "ds", "Revenue": "y"}
                )

                model = Prophet()
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=months_ahead, freq="MS")
                forecast = model.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))

                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5 — CUSTOMER SEGMENTS
# ============================================================
elif page == "Customer Segments":
    st.title("Customer Segmentation (RFM Analysis)")
    st.markdown("---")

    if not data_loaded or rfm is None or rfm.empty:
        st.warning("RFM data not available. Run `python src/clustering.py` first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers",  f"{len(rfm):,}")
        c2.metric("Avg Order Freq",   f"{rfm['Frequency'].mean():.1f} orders")
        c3.metric("Avg Revenue",      f"£{rfm['Monetary'].mean():,.2f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Segments — RFM Scatter")
            fig = px.scatter(
                rfm.sample(min(1500, len(rfm)), random_state=42),
                x="Recency", y="Monetary",
                size="Frequency", color="Segment",
                hover_data=["CustomerID", "Frequency"],
                labels={"Recency": "Recency (days)", "Monetary": "Revenue (£)"},
                opacity=0.7
            )
            fig.update_layout(height=460, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Segment Distribution")
            seg_counts = rfm["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Customers"]
            fig = px.bar(
                seg_counts, x="Segment", y="Customers",
                labels={"Customers": "Number of Customers"},
                color="Segment"
            )
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Customers: %{y}<extra></extra>")
            fig.update_layout(height=460, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("RFM Segment Summary")
        summary = rfm.groupby("Segment").agg(
            Customers     = ("CustomerID", "count"),
            Avg_Recency   = ("Recency",    "mean"),
            Avg_Frequency = ("Frequency",  "mean"),
            Avg_Revenue   = ("Monetary",   "mean")
        ).round(2).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Top 10 Most Valuable Customers")
        top_customers = rfm.nlargest(10, "Monetary")[
            ["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]
        ].copy()
        top_customers["Monetary"] = top_customers["Monetary"].round(2)
        top_customers.columns = ["Customer ID", "Segment", "Recency (days)", "Orders", "Total Spend (£)"]
        st.dataframe(top_customers.reset_index(drop=True), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 6 — UPLOAD DATA (ADMIN ONLY)
# ============================================================
elif page == "Upload Data":
    admin_only()

    st.title("Upload New Data")
    st.markdown("---")
    st.info("Admin only. Upload a new retail dataset to update the dashboard.")

    file = st.file_uploader("Choose Excel or CSV file", type=["csv", "xlsx"])

    if file:
        try:
            if file.name.endswith(".csv"):
                df_new = pd.read_csv(file)
            else:
                df_new = pd.read_excel(file)

            st.success(f"File read successfully — {df_new.shape[0]:,} rows, {df_new.shape[1]} columns")
            st.dataframe(df_new.head(5), use_container_width=True)

            os.makedirs("data/raw", exist_ok=True)
            df_new.to_excel("data/raw/retail_sales.xlsx", index=False)
            st.success("Saved to `data/raw/retail_sales.xlsx`")

            st.markdown("---")
            st.subheader("Run Pipeline")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Preprocessing", type="primary"):
                    with st.spinner("Running preprocessing..."):
                        preprocess()
                    st.success("Preprocessing complete!")
                    st.cache_data.clear()
            with col2:
                if st.button("Run Clustering", type="secondary"):
                    with st.spinner("Running clustering..."):
                        run_clustering()
                    st.success("Clustering complete!")
                    st.cache_data.clear()

        except Exception as e:
            st.error(f"Upload failed: {e}")

# ============================================================
# PAGE 7 — ADMIN PANEL (ADMIN ONLY)
# ============================================================
elif page == "Admin Panel":
    admin_only()

    st.title("Admin Control Panel")
    st.markdown("---")
    st.success(f"Logged in as {st.session_state.user} — Full Admin Access")

    st.subheader("Pipeline Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Re-run Preprocessing", use_container_width=True):
            with st.spinner("Running..."):
                preprocess()
            st.success("Done!")
            st.cache_data.clear()
    with col2:
        if st.button("Re-run Clustering", use_container_width=True):
            with st.spinner("Running..."):
                run_clustering()
            st.success("Done!")
            st.cache_data.clear()
    with col3:
        if st.button("Check File Status", use_container_width=True):
            check_required_files()

    st.markdown("---")

    st.subheader("User Management")

    pending_requests = get_pending_requests()

    if pending_requests:
        st.markdown(f"Pending Requests ({len(pending_requests)})")
        for req in pending_requests:
            st.markdown(f"""
            <div class="pending-card">
                <strong>{req['username']}</strong> — wants to join as <em>{req.get('role','employee')}</em>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                assigned_role = st.selectbox(
                    "Assign role",
                    ["employee", "admin"],
                    key=f"role_sel_{req['username']}",
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("Approve", key=f"approve_{req['username']}", use_container_width=True):
                    approve_user(req["username"], assigned_role)
                    st.success(f"{req['username']} approved as {assigned_role}!")
                    st.rerun()
            with col3:
                if st.button("Reject", key=f"reject_{req['username']}", use_container_width=True):
                    reject_user(req["username"])
                    st.warning(f"{req['username']} rejected.")
                    st.rerun()
    else:
        st.info("No pending access requests.")

    st.markdown("---")

    st.markdown("Existing Users")
    users = load_users()

    if not users:
        st.warning("No users found.")
    else:
        for username, data in users.items():
            role_badge = "admin" if data["role"] == "admin" else "employee"
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.markdown(f"""
                <div class="user-card">
                    <strong>{username}</strong><br>
                    <small>{role_badge}</small>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("")
                st.markdown("")
                st.caption(f"Role: {data['role'].upper()}")
            with col3:
                st.markdown("")
                st.markdown("")
                if username == st.session_state.user:
                    st.caption("(You — cannot delete)")
                else:
                    if st.button(f"Remove {username}", key=f"del_{username}", use_container_width=True):
                        delete_user(username)
                        st.error(f"{username} has been removed.")
                        st.rerun()
