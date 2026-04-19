# src/utils.py

import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# DIRECTORY MANAGEMENT

def ensure_dirs():
    """Ensure all required project directories exist."""
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "reports/figures",
        "reports/research_paper"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("All directories verified.")

# DATA LOADING

def load_clean_data():
    path = "data/processed/retail_clean.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Clean dataset not found.\n"
            "Run: python src/data_preprocessing.py"
        )

    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"Loaded clean data: {df.shape[0]} rows")
    return df

def load_rfm():
    path = "data/processed/rfm_segments.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(
            "RFM dataset not found.\n"
            "Run: python src/clustering.py"
        )

    df = pd.read_csv(path)
    print(f"Loaded RFM data: {df.shape[0]} customers")
    return df

# FILE VALIDATION

def check_required_files():
    required = {
        "Clean dataset"   : "data/processed/retail_clean.csv",
        "RFM segments"    : "data/processed/rfm_segments.csv",
        "Linear Reg model": "models/linear_regression.pkl",
        "Random Forest"   : "models/random_forest.pkl",
        "KMeans model"    : "models/kmeans.pkl",
        "RFM scaler"      : "models/rfm_scaler.pkl",
    }

    print("\n===== FILE CHECK =====")
    all_ok = True

    for name, path in required.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"{status:8s} {name:20s} -> {path}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\nAll required files found. Dashboard ready.")
    else:
        print("\nSome files are missing. Run pipeline first.")

    return all_ok

# METRICS

def summary_metrics(df):
    return {
        "Total Revenue"    : round(df["TotalPrice"].sum(), 2),
        "Total Orders"     : df["InvoiceNo"].nunique(),
        "Unique Customers" : df["CustomerID"].nunique(),
        "Unique Products"  : df["StockCode"].nunique(),
        "Avg Order Value"  : round(df.groupby("InvoiceNo")["TotalPrice"].sum().mean(), 2),
        "Date From"        : str(df["InvoiceDate"].min().date()),
        "Date To"          : str(df["InvoiceDate"].max().date()),
    }

def print_metrics(metrics):
    print("\n===== BUSINESS METRICS =====")
    for k, v in metrics.items():
        print(f"{k:22s}: {v}")

# DATA TRANSFORM HELPERS

def get_monthly_revenue(df):
    df = df.copy()

    df["MonthYear"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("M")

    monthly = (df.groupby("MonthYear")["TotalPrice"]
                 .sum()
                 .reset_index())

    monthly.columns = ["MonthYear", "Revenue"]
    monthly["MonthYear"] = monthly["MonthYear"].dt.to_timestamp()
    monthly = monthly.sort_values("MonthYear").reset_index(drop=True)
    monthly["MonthIndex"] = range(len(monthly))

    return monthly

def get_top_products(df, n=10):
    top = (df.groupby("Description")["TotalPrice"]
             .sum()
             .sort_values(ascending=False)
             .head(n)
             .reset_index())

    top.columns = ["Product", "Revenue"]
    return top

def get_top_countries(df, n=10):
    country = (df.groupby("Country")["TotalPrice"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(n)
                 .reset_index())

    country.columns = ["Country", "Revenue"]
    return country

# PLOT SAVE

def save_figure(filename):
    os.makedirs("reports/figures", exist_ok=True)
    path = f"reports/figures/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {path}")

# TEST

if __name__ == "__main__":
    ensure_dirs()
    df = load_clean_data()
    metrics = summary_metrics(df)
    print_metrics(metrics)
    check_required_files()

# src/utils.py — add this function

REQUIRED_COLUMNS = {
    "InvoiceNo"   : "object",
    "StockCode"   : "object",
    "Description" : "object",
    "Quantity"    : "int64",
    "InvoiceDate" : "datetime64",
    "UnitPrice"   : "float64",
    "CustomerID"  : "float64",
    "Country"     : "object",
}

def validate_dataset(df) -> tuple:
    """
    Validate uploaded dataset before processing.
    Returns (is_valid, list_of_errors).
    """
    errors = []

    # Check required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return False, errors

    # Check minimum rows
    if len(df) < 100:
        errors.append(f"Dataset too small: {len(df)} rows (minimum 100 required).")

    # Check InvoiceDate is parseable
    try:
        pd.to_datetime(df["InvoiceDate"])
    except Exception:
        errors.append("Column 'InvoiceDate' cannot be parsed as dates.")

    # Check numeric columns
    for col in ["Quantity", "UnitPrice"]:
        if col in df.columns:
            if not pd.to_numeric(df[col], errors="coerce").notna().any():
                errors.append(f"Column '{col}' has no valid numeric values.")

    # Check for completely empty critical columns
    for col in ["InvoiceNo", "CustomerID"]:
        if col in df.columns and df[col].isna().all():
            errors.append(f"Column '{col}' is entirely empty.")

    return len(errors) == 0, errors
