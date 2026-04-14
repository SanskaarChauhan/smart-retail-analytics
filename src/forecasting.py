# src/forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def load_clean_data():
    df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["InvoiceDate"])
    print(f"Loaded clean data: {df.shape[0]} rows")
    return df

def prepare_monthly_data(df):
    df["MonthYear"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("M")
    monthly = df.groupby("MonthYear")["TotalPrice"].sum().reset_index()
    monthly.columns = ["MonthYear", "Revenue"]
    monthly["MonthYear"] = monthly["MonthYear"].dt.to_timestamp()
    monthly = monthly.sort_values("MonthYear").reset_index(drop=True)
    monthly["MonthIndex"] = range(len(monthly))

    print(f"\nMonthly data prepared: {len(monthly)} months")
    return monthly

def linear_regression_forecast(monthly):
    print("\n===== LINEAR REGRESSION FORECAST =====")

    X = monthly[["MonthIndex"]]
    y = monthly["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"MAE  : £{mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE : £{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R²   : {r2_score(y_test, y_pred):.4f}")

    last_idx = monthly["MonthIndex"].max()
    future_idx = pd.DataFrame({"MonthIndex": range(last_idx + 1, last_idx + 7)})
    future_pred = model.predict(future_idx)

    last_date = monthly["MonthYear"].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["MonthYear"], monthly["Revenue"], marker="o", label="Actual")
    plt.plot(future_dates, future_pred, marker="o", linestyle="--", label="Forecast")
    plt.title("Linear Regression — Revenue Forecast")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/lr_forecast.png", dpi=150)
    plt.show()

    print("Chart saved: reports/figures/lr_forecast.png")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_regression.pkl")

    return model

def random_forest_forecast(monthly):
    print("\n===== RANDOM FOREST FORECAST =====")

    monthly["Month"] = monthly["MonthYear"].dt.month
    monthly["Quarter"] = monthly["MonthYear"].dt.quarter
    monthly["RevLag1"] = monthly["Revenue"].shift(1)
    monthly["RevLag2"] = monthly["Revenue"].shift(2)
    monthly["RollingMean"] = monthly["Revenue"].rolling(3).mean()
    monthly = monthly.dropna().reset_index(drop=True)

    features = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", "RollingMean"]
    X = monthly[features]
    y = monthly["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"MAE  : £{mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE : £{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R²   : {r2_score(y_test, y_pred):.4f}")

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, marker="o", label="Actual")
    plt.plot(y_pred, marker="o", linestyle="--", label="Predicted")
    plt.title("Random Forest — Actual vs Predicted")
    plt.xlabel("Test Months")
    plt.ylabel("Revenue (£)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/rf_forecast.png", dpi=150)
    plt.show()

    print("Chart saved: reports/figures/rf_forecast.png")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")

    return model

def arima_forecast(monthly, steps=6):
    print("\n===== ARIMA FORECAST =====")

    revenue_series = monthly.set_index("MonthYear")["Revenue"]

    model = ARIMA(revenue_series, order=(1, 1, 1))
    fitted = model.fit()

    forecast = fitted.forecast(steps=steps)

    forecast_index = pd.date_range(
        start=revenue_series.index[-1] + pd.DateOffset(months=1),
        periods=steps, freq="MS"
    )

    for date, val in zip(forecast_index, forecast):
        print(f"{date.strftime('%b %Y')} : £{val:,.2f}")

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(revenue_series.index, revenue_series.values, marker="o", label="Actual")
    plt.plot(forecast_index, forecast.values, marker="o", linestyle="--", label="Forecast")
    plt.title("ARIMA Forecast")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/arima_forecast.png", dpi=150)
    plt.show()

    print("Chart saved: reports/figures/arima_forecast.png")

    return fitted

def run_forecasting():
    df = load_clean_data()
    monthly = prepare_monthly_data(df)

    lr_model = linear_regression_forecast(monthly)
    rf_model = random_forest_forecast(monthly)
    arima_model = arima_forecast(monthly)

    print("\n✅ Forecasting complete!")
    print("Models saved in: models/")
    print("Charts saved in: reports/figures/")

if __name__ == "__main__":
    run_forecasting()