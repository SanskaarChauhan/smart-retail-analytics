# src/analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def load_clean_data():
    """Load the preprocessed dataset."""
    df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["InvoiceDate"])
    print(f"Loaded clean data: {df.shape[0]} rows")
    return df

def overall_summary(df):
    """Print key business metrics."""
    print("\n===== OVERALL SUMMARY =====")
    print(f"Total Revenue       : £{df['TotalPrice'].sum():,.2f}")
    print(f"Total Orders        : {df['InvoiceNo'].nunique():,}")
    print(f"Unique Customers    : {df['CustomerID'].nunique():,}")
    print(f"Unique Products     : {df['StockCode'].nunique():,}")
    print(f"Date Range          : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")

def monthly_sales_trend(df):
    """Plot monthly revenue trend."""
    df["MonthYear"] = pd.to_datetime(df["MonthYear"])

    monthly = df.groupby("MonthYear")["TotalPrice"].sum().reset_index()
    monthly = monthly.sort_values("MonthYear")

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["MonthYear"], monthly["TotalPrice"], marker="o")
    plt.title("Monthly Sales Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/monthly_sales_trend.png", dpi=150)
    plt.show()

    print("Chart saved: reports/figures/monthly_sales_trend.png")
    return monthly

def top_products(df, n=10):
    """Top N products by revenue."""
    top = (df.groupby("Description")["TotalPrice"]
             .sum()
             .sort_values(ascending=False)
             .head(n)
             .reset_index())

    top.columns = ["Product", "Revenue"]

    plt.figure(figsize=(10, 6))
    plt.barh(top["Product"][::-1], top["Revenue"][::-1])
    plt.title(f"Top {n} Products by Revenue")
    plt.xlabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig("reports/figures/top_products.png", dpi=150)
    plt.show()

    print("Chart saved: reports/figures/top_products.png")
    return top

def sales_by_country(df, n=10):
    """Top countries by revenue."""
    country = (df.groupby("Country")["TotalPrice"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(n)
                 .reset_index())

    country.columns = ["Country", "Revenue"]

    print("\n===== TOP COUNTRIES =====")
    print(country.to_string(index=False))
    return country

def daily_orders(df):
    """Orders per day of week."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    dow = df.groupby("DayOfWeek")["InvoiceNo"].nunique().reset_index()
    dow.columns = ["DayOfWeek", "Orders"]

    dow["DayName"] = dow["DayOfWeek"].apply(lambda x: days[x])

    plt.figure(figsize=(8, 4))
    plt.bar(dow["DayName"], dow["Orders"])
    plt.title("Orders by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.savefig("reports/figures/orders_by_day.png", dpi=150)
    plt.show()

    print("Chart saved: reports/figures/orders_by_day.png")
    return dow

def revenue_growth(df):
    """Monthly revenue growth (%)"""
    df["MonthYear"] = pd.to_datetime(df["MonthYear"])

    monthly = df.groupby("MonthYear")["TotalPrice"].sum().reset_index()
    monthly = monthly.sort_values("MonthYear")

    monthly["Growth"] = monthly["TotalPrice"].pct_change() * 100

    print("\n===== MONTHLY GROWTH (%) =====")
    print(monthly.head(10))

    # Optional graph
    plt.figure(figsize=(10,5))
    plt.plot(monthly["MonthYear"], monthly["Growth"], marker="o")
    plt.title("Monthly Revenue Growth (%)")
    plt.xlabel("Month")
    plt.ylabel("Growth %")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/revenue_growth.png", dpi=150)
    plt.show()

    return monthly

def run_analysis():
    os.makedirs("reports/figures", exist_ok=True)

    df = load_clean_data()
    overall_summary(df)

    monthly = monthly_sales_trend(df)
    top = top_products(df)
    country = sales_by_country(df)
    dow = daily_orders(df)
    growth = revenue_growth(df)

    print("\n✅ Analysis complete. Charts saved in reports/figures/")
    return df, monthly, top, country, dow, growth

if __name__ == "__main__":
    run_analysis()