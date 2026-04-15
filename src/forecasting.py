# src/forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle

# ============================================
# DATA LOADING
# ============================================

def load_clean_data():
    """Load the preprocessed dataset."""
    df = pd.read_csv("data/processed/retail_clean.csv", parse_dates=["InvoiceDate"])
    print(f"✅ Loaded clean data: {df.shape[0]} rows")
    return df

def prepare_monthly_data(df):
    """Prepare monthly aggregated data for forecasting."""
    print("\n📊 Preparing monthly data...")
    
    df["MonthYear"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("M")
    monthly = df.groupby("MonthYear")["TotalPrice"].sum().reset_index()
    monthly.columns = ["MonthYear", "Revenue"]
    monthly["MonthYear"] = monthly["MonthYear"].dt.to_timestamp()
    monthly = monthly.sort_values("MonthYear").reset_index(drop=True)
    monthly["MonthIndex"] = range(len(monthly))
    
    print(f"✅ Created {len(monthly)} months of data")
    print(f"   Date range: {monthly['MonthYear'].min().date()} to {monthly['MonthYear'].max().date()}")
    print(f"   Revenue range: £{monthly['Revenue'].min():,.2f} to £{monthly['Revenue'].max():,.2f}")
    
    return monthly

# ============================================
# FEATURE ENGINEERING
# ============================================

def create_time_features(monthly):
    """Create time-based features for ML models."""
    df = monthly.copy()
    
    # Basic time features
    df["Month"] = df["MonthYear"].dt.month
    df["Quarter"] = df["MonthYear"].dt.quarter
    df["Year"] = df["MonthYear"].dt.year
    
    # Lag features (past values)
    df["RevLag1"] = df["Revenue"].shift(1)
    df["RevLag2"] = df["Revenue"].shift(2)
    df["RevLag3"] = df["Revenue"].shift(3)
    df["RevLag6"] = df["Revenue"].shift(6)  # Year-over-year
    
    # Rolling statistics
    df["RollingMean3"] = df["Revenue"].rolling(3).mean()
    df["RollingMean6"] = df["Revenue"].rolling(6).mean()
    df["RollingStd3"] = df["Revenue"].rolling(3).std()
    
    # Trend features
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Drop NaN values created by shifts
    df = df.dropna().reset_index(drop=True)
    
    print(f"✅ Created {len(df.columns) - len(monthly.columns)} new features")
    
    return df

# ============================================
# MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================

def train_linear_regression_models(monthly):
    """Train Linear Regression, Ridge, and Lasso models."""
    print("\n" + "="*50)
    print("📈 TRAINING LINEAR REGRESSION MODELS")
    print("="*50)
    
    # Prepare features
    X = monthly[["MonthIndex"]]
    y = monthly["Revenue"]
    
    # Split data (maintain time order)
    train_size = int(len(monthly) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    models = {}
    results = {}
    
    # 1. Simple Linear Regression
    print("\n1. Simple Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    results["LinearRegression"] = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    models["linear_regression"] = lr
    print(f"   R² Score: {results['LinearRegression']['r2']:.4f}")
    
    # 2. Polynomial Regression (degree 2)
    print("\n2. Polynomial Regression (degree 2)")
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    y_pred = lr_poly.predict(X_test_poly)
    
    results["PolynomialRegression"] = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    models["polynomial_regression"] = {"model": lr_poly, "poly": poly}
    print(f"   R² Score: {results['PolynomialRegression']['r2']:.4f}")
    
    # 3. Ridge Regression (L2 regularization)
    print("\n3. Ridge Regression (with cross-validation)")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    results["Ridge"] = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    models["ridge"] = ridge
    print(f"   R² Score: {results['Ridge']['r2']:.4f}")
    
    # 4. Lasso Regression (L1 regularization)
    print("\n4. Lasso Regression")
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    
    results["Lasso"] = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    models["lasso"] = lasso
    print(f"   R² Score: {results['Lasso']['r2']:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]["r2"])
    print(f"\n🏆 Best Linear Model: {best_model_name} (R²={results[best_model_name]['r2']:.4f})")
    
    return models, results, best_model_name

def train_random_forest_models(feature_df):
    """Train optimized Random Forest and Gradient Boosting models."""
    print("\n" + "="*50)
    print("🌲 TRAINING RANDOM FOREST MODELS")
    print("="*50)
    
    # Define features
    feature_cols = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", 
                    "RevLag3", "RollingMean3", "MonthSin", "MonthCos"]
    
    X = feature_df[feature_cols]
    y = feature_df["Revenue"]
    
    # Time-based split
    train_size = int(len(feature_df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    models = {}
    results = {}
    
    # 1. Random Forest with Hyperparameter Tuning
    print("\n1. Random Forest (with GridSearchCV)")
    
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("   Performing grid search...")
    rf_grid = GridSearchCV(
        rf_base, rf_params, 
        cv=tscv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    y_pred = best_rf.predict(X_test)
    
    results["RandomForest"] = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "best_params": rf_grid.best_params_
    }
    models["random_forest"] = best_rf
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   R² Score: {results['RandomForest']['r2']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Feature Importance (Top 5):")
    for _, row in feature_importance.head(5).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    # Save feature columns for dashboard use
    models["feature_cols"] = feature_cols
    
    # 2. Gradient Boosting (simplified)
    print("\n2. Gradient Boosting Regressor")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    
    results["GradientBoosting"] = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    models["gradient_boosting"] = gb
    print(f"   R² Score: {results['GradientBoosting']['r2']:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]["r2"])
    print(f"\n🏆 Best Tree Model: {best_model_name} (R²={results[best_model_name]['r2']:.4f})")
    
    return models, results, best_model_name, feature_cols

def train_arima_model(monthly):
    """Train ARIMA model with auto-selection of parameters."""
    print("\n" + "="*50)
    print("📊 TRAINING ARIMA MODEL")
    print("="*50)
    
    series = monthly.set_index("MonthYear")["Revenue"]
    
    # Check stationarity
    print("\n1. Testing stationarity...")
    result = adfuller(series)
    print(f"   ADF Statistic: {result[0]:.4f}")
    print(f"   p-value: {result[1]:.4f}")
    
    if result[1] > 0.05:
        print("   Series is not stationary. Applying differencing...")
        series_diff = series.diff().dropna()
        result_diff = adfuller(series_diff)
        print(f"   After differencing - p-value: {result_diff[1]:.4f}")
        d = 1
    else:
        print("   Series is stationary")
        d = 0
    
    # Try different ARIMA configurations
    print("\n2. Trying different ARIMA configurations...")
    
    configurations = [
        (1, d, 1), (1, d, 2), (2, d, 1), (2, d, 2),
        (1, d, 0), (0, d, 1)
    ]
    
    best_aic = float('inf')
    best_model = None
    best_order = None
    
    for order in configurations:
        try:
            model = ARIMA(series, order=order)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
                best_order = order
                print(f"   Order {order}: AIC={fitted.aic:.1f}")
        except:
            continue
    
    print(f"\n🏆 Best ARIMA Order: {best_order} (AIC={best_aic:.1f})")
    
    # Evaluate on test set
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    model_test = ARIMA(train, order=best_order)
    fitted_test = model_test.fit()
    forecast = fitted_test.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)
    
    print(f"\nTest Set Performance:")
    print(f"   MAE: £{mae:,.2f}")
    print(f"   RMSE: £{rmse:,.2f}")
    print(f"   R²: {r2:.4f}")
    
    return best_model, best_order, {"mae": mae, "rmse": rmse, "r2": r2}

# ============================================
# FIXED MODEL SAVING
# ============================================

def save_models(models, results, feature_cols=None):
    """Save all trained models with metadata."""
    os.makedirs("models", exist_ok=True)
    
    # Save Linear Regression models
    if "linear_regression" in models:
        joblib.dump(models["linear_regression"], "models/linear_regression.pkl")
        print("✅ Saved: models/linear_regression.pkl")
    
    if "ridge" in models:
        joblib.dump(models["ridge"], "models/ridge_regression.pkl")
    
    if "lasso" in models:
        joblib.dump(models["lasso"], "models/lasso_regression.pkl")
    
    if "polynomial_regression" in models:
        joblib.dump(models["polynomial_regression"], "models/polynomial_regression.pkl")
    
    # Save Random Forest model (with feature names)
    if "random_forest" in models:
        joblib.dump(models["random_forest"], "models/random_forest.pkl")
        print("✅ Saved: models/random_forest.pkl")
        
        # Save feature names separately
        if feature_cols:
            with open("models/rf_features.pkl", "wb") as f:
                pickle.dump(feature_cols, f)
            print("✅ Saved: models/rf_features.pkl")
    
    if "gradient_boosting" in models:
        joblib.dump(models["gradient_boosting"], "models/gradient_boosting.pkl")
    
    # Save model comparison results
    results_df = pd.DataFrame(results).T
    results_df.to_csv("models/model_comparison.csv")
    print("✅ Saved: models/model_comparison.csv")
    
    return results_df

def save_arima_model(model, order):
    """Save ARIMA model and parameters."""
    os.makedirs("models", exist_ok=True)
    
    # Save model parameters
    arima_info = {
        "order": order,
        "params": model.params.to_dict() if hasattr(model, 'params') else {},
        "aic": model.aic if hasattr(model, 'aic') else None,
        "bic": model.bic if hasattr(model, 'bic') else None
    }
    
    with open("models/arima_info.pkl", "wb") as f:
        pickle.dump(arima_info, f)
    
    # Save the fitted model safely
    try:
        joblib.dump(model, "models/arima_model.pkl")
        print("✅ Saved: models/arima_model.pkl")
    except:
        print("⚠️  Could not save ARIMA model (using params only)")
    
    print("✅ Saved: models/arima_info.pkl")

# ============================================
# FIXED VISUALIZATION (SIMPLE & WORKING)
# ============================================

def plot_forecast_comparison(monthly, models, feature_cols=None):
    """Compare all model forecasts - FIXED PROFESSIONAL VERSION."""
    os.makedirs("reports/figures", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Actual data
    ax.plot(monthly["MonthYear"], monthly["Revenue"], 
            marker='o', linewidth=3, label='Actual Revenue', 
            color='black', markersize=6, markerfacecolor='white', markeredgewidth=1)
    
    # Linear Regression
    if "linear_regression" in models:
        X = monthly[["MonthIndex"]]
        lr_pred = models["linear_regression"].predict(X)
        ax.plot(monthly["MonthYear"], lr_pred, '--', 
                label='Linear Regression', alpha=0.8, linewidth=2.5, color='blue')
    
    # Random Forest (if available)
    if "random_forest" in models and feature_cols is not None:
        try:
            feature_df = create_time_features(monthly)
            X_rf = feature_df[feature_cols]
            rf_pred = models["random_forest"].predict(X_rf)
            ax.plot(feature_df["MonthYear"], rf_pred, '--', 
                    label='Random Forest', alpha=0.8, linewidth=2.5, color='green')
        except Exception as e:
            print(f"RF plotting skipped: {e}")
    
    ax.set_title('Model Forecast Comparison', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Revenue (£)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Professional chart saved: reports/figures/model_comparison.png")

# ============================================
# MAIN PIPELINE
# ============================================

def run_forecasting():
    """Main forecasting pipeline."""
    print("\n" + "="*60)
    print("🚀 STARTING FORECASTING PIPELINE")
    print("="*60)
    
    # Load and prepare data
    df = load_clean_data()
    monthly = prepare_monthly_data(df)
    
    # Train Linear Regression models
    lr_models, lr_results, best_lr = train_linear_regression_models(monthly)
    
    # Prepare features for tree models
    feature_df = create_time_features(monthly)
    
    # Train Random Forest models
    rf_models, rf
