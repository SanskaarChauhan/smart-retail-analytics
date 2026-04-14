# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os

def update_dates(df):
    print("\nUpdating dates to modern years (2021–2025)...")
    np.random.seed(42)  # add this line ← makes results reproducible
    df['InvoiceDate'] = df['InvoiceDate'].apply(
        lambda x: x.replace(year=np.random.choice([2021, 2022, 2023, 2024, 2025]))
    )
    return df

def load_data(filepath):
    """Load the raw Excel dataset."""
    print("Loading dataset...")
    df = pd.read_excel(filepath, engine='openpyxl')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """Clean and preprocess the retail dataset."""
    print("\nCleaning data...")

    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    print(f"After removing missing CustomerID: {df.shape[0]} rows")

    # Remove cancelled orders (InvoiceNo starts with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"After removing cancellations: {df.shape[0]} rows")

    # Remove rows with negative or zero Quantity and UnitPrice
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    print(f"After removing invalid quantities/prices: {df.shape[0]} rows")

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Clean CustomerID
    df['CustomerID'] = df['CustomerID'].astype(int)

    return df

def feature_engineering(df):
    """Create new useful features."""
    print("\nCreating new features...")

    # Total price per transaction line
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Date features
    df['Year']    = df['InvoiceDate'].dt.year
    df['Month']   = df['InvoiceDate'].dt.month
    df['Day']     = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour']    = df['InvoiceDate'].dt.hour

    # Month-Year label for plotting
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    print("Features created: TotalPrice, Year, Month, Day, DayOfWeek, Hour, MonthYear")
    return df

def save_processed_data(df, output_path):
    """Save cleaned data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

def preprocess():
    """Main preprocessing pipeline."""
    raw_path       = "data/raw/retail_sales.xlsx"
    processed_path = "data/processed/retail_clean.csv"

    df = load_data(raw_path)
    df = clean_data(df)
    df = update_dates(df)
    df = feature_engineering(df)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(3))

    save_processed_data(df, processed_path)
    return df

if __name__ == "__main__":
    preprocess()