import sys
import os

# Make src folder accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# utils
from utils import (
    load_clean_data, load_rfm, summary_metrics,
    get_monthly_revenue, get_top_products, get_top_countries
)

# ==============================
# SIMPLE AUTH SYSTEM
# ==============================

if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

def login_page():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # simple demo login
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

# ==============================
# DATA LOADING
# ==============================

@st.cache_data
def load_all():
    df = load_clean_data()
    rfm = load_rfm()
    return df, rfm

# ==============================
# MAIN APP
# ==============================

if st.session_state.user is None:
    login_page()
    st.stop()

df, rfm = load_all()
metrics = summary_metrics(df)
monthly = get_monthly_revenue(df)

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("Smart Retail Analytics")

page = st.sidebar.radio("Navigation", [
    "Overview",
    "Sales Trends",
    "Product Analysis",
    "Customer Segments",
    "Upload Data",
    "Admin Panel"
])

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# ==============================
# OVERVIEW
# ==============================

if page == "Overview":
    st.title("Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", f"£{metrics['Total Revenue']:,.2f}")
    c2.metric("Orders", f"{metrics['Total Orders']:,}")
    c3.metric("Customers", f"{metrics['Unique Customers']:,}")

    st.subheader("Sales Trend")

    fig = px.line(
        monthly,
        x="MonthYear",
        y="Revenue",
        title="Sales Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# SALES
# ==============================

elif page == "Sales Trends":
    st.title("Sales Trends")

    fig = px.line(
        monthly,
        x="MonthYear",
        y="Revenue",
        title="Monthly Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# PRODUCTS
# ==============================

elif page == "Product Analysis":
    st.title("Top Products")

    top = get_top_products(df, 10)

    fig = px.bar(top, x="Product", y="Revenue")
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# SEGMENTS
# ==============================

elif page == "Customer Segments":
    st.title("Customer Segments")

    fig = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color="Segment"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# UPLOAD DATA
# ==============================

elif page == "Upload Data":
    st.title("Upload Data")

    uploaded_file = st.file_uploader("Upload CSV or Excel")

    if uploaded_file:
        df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        os.makedirs("data/raw", exist_ok=True)
        filepath = "data/raw/retail_sales.xlsx"
        df_new.to_excel(filepath, index=False)

        st.success("File uploaded successfully")

# ==============================
# ADMIN PANEL
# ==============================

elif page == "Admin Panel":
    st.title("Admin Panel")

    if st.session_state.role != "admin":
        st.warning("Access denied")
    else:
        st.success("Admin access granted")
        st.write("User approvals coming soon...")
