# src/forecasting.py - COMPLETE WITH XGBOOST + PROPHET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

# Core ML imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# NEW: XGBoost
import xgboost as xgb

# NEW: Prophet
from prophet import Prophet
from prophet.plot import plot_plotly

# Statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import pickle

print("✅ All forecasting libraries loaded: sklearn, xgboost, prophet, statsmodels")

# ============================================
# [KEEP ALL EXISTING FUNCTIONS: load_clean_data, prepare_monthly_data, 
# create_time_features, train_linear_regression_models, train_random_forest_models,
# train_arima_model, save_models, save_arima_model, plot_forecast_comparison]
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
    return monthly

def create_time_features(monthly):
    """Create time-based features for ML models."""
    df = monthly.copy()
    df["Month"] = df["MonthYear"].dt.month
    df["Quarter"] = df["MonthYear"].dt.quarter
    df["Year"] = df["MonthYear"].dt.year
    df["RevLag1"] = df["Revenue"].shift(1)
    df["RevLag2"] = df["Revenue"].shift(2)
    df["RevLag3"] = df["Revenue"].shift(3)
    df["RollingMean3"] = df["Revenue"].rolling(3).mean()
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df = df.dropna().reset_index(drop=True)
    return df

# ============================================
# NEW: XGBOOST MODEL
# ============================================
def train_xgboost_model(feature_df):
    """Train XGBoost with hyperparameter tuning."""
    print("\n" + "="*50)
    print(" TRAINING XGBOOST MODEL")
    print("="*50)
    
    feature_cols = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", 
                    "RevLag3", "RollingMean3", "MonthSin", "MonthCos"]
    
    X = feature_df[feature_cols]
    y = feature_df["Revenue"]
    
    train_size = int(len(feature_df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # XGBoost with GridSearch
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("   Tuning XGBoost...")
    xgb_grid = GridSearchCV(xgb_base, xgb_params, cv=tscv, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    
    best_xgb = xgb_grid.best_estimator_
    y_pred = best_xgb.predict(X_test)
    
    results = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "best_params": xgb_grid.best_params_
    }
    
    print(f"   Best params: {xgb_grid.best_params_}")
    print(f"   R² Score: {results['r2']:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Top Features:")
    for _, row in importance.head(5).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    return best_xgb, results, feature_cols

# ============================================
# NEW: PROPHET MODEL (Meta's Algorithm)
# ============================================
def train_prophet_model(monthly):
    """Train Facebook Prophet model."""
    print("\n" + "="*50)
    print(" TRAINING PROPHET (Meta) MODEL")
    print("="*50)
    
    # Prepare Prophet data
    prophet_df = monthly[['MonthYear', 'Revenue']].rename(columns={
        'MonthYear': 'ds',
        'Revenue': 'y'
    })
    
    # Train/test split
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:train_size]
    test_df = prophet_df[train_size:]
    
    # Train Prophet
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(train_df)
    
    # Forecast test period
    future = prophet_model.make_future_dataframe(periods=len(test_df), freq='MS')
    forecast = prophet_model.predict(future)
    y_pred = forecast['yhat'][-len(test_df):].values
    y_true = test_df['y'].values
    
    results = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    
    print(f"   MAE: £{results['mae']:,.0f}")
    print(f"   RMSE: £{results['rmse']:,.0f}")
    print(f"   R²: {results['r2']:.4f}")
    
    return prophet_model, results

# ============================================
# ENHANCED MODEL TRAINING PIPELINE
# ============================================
def train_all_models(monthly):
    """Train ALL models including new ones."""
    print("\n Training ALL forecasting models...")
    
    # 1. Linear models
    lr_models, lr_results, _ = train_linear_regression_models(monthly)
    
    # 2. Tree models + features
    feature_df = create_time_features(monthly)
    rf_models, rf_results, _, feature_cols = train_random_forest_models(feature_df)
    
    # 3. NEW: XGBoost
    xgb_model, xgb_results, _ = train_xgboost_model(feature_df)
    
    # 4. NEW: Prophet
    prophet_model, prophet_results = train_prophet_model(monthly)
    
    # 5. ARIMA
    arima_model, arima_order, arima_results = train_arima_model(monthly)
    
    # Combine ALL results
    all_results = {
        **lr_results, **rf_results, 
        "XGBoost": xgb_results,
        "Prophet": prophet_results,
        "ARIMA": arima_results
    }
    
    all_models = {**lr_models, **rf_models}
    all_models["xgboost"] = xgb_model
    all_models["prophet"] = prophet_model
    all_models["arima"] = arima_model
    all_models["feature_cols"] = feature_cols
    
    return all_models, all_results, feature_cols

# ============================================
# ENHANCED SAVING
# ============================================
def save_all_models(models, results, feature_cols):
    """Save ALL models."""
    os.makedirs("models", exist_ok=True)
    
    # Existing models
    save_models(models, results, feature_cols)
    
    # NEW: XGBoost
    if "xgboost" in models:
        joblib.dump(models["xgboost"], "models/xgboost.pkl")
        print("✅ Saved: models/xgboost.pkl")
    
    # NEW: Prophet
    if "prophet" in models:
        joblib.dump(models["prophet"], "models/prophet.pkl")
        print("✅ Saved: models/prophet.pkl")
    
    # Save comparison
    results_df = pd.DataFrame(results).T
    results_df.to_csv("models/model_comparison.csv")
    print("✅ Updated: models/model_comparison.csv")
    
    return results_df

# ============================================
# MAIN PIPELINE (UPDATED)
# ============================================
def run_forecasting():
    """Enhanced pipeline with XGBoost + Prophet."""
    print("\n" + "="*70)
    print(" ADVANCED FORECASTING PIPELINE (7 Models)")
    print("="*70)
    
    df = load_clean_data()
    monthly = prepare_monthly_data(df)
    
    # Train ALL models
    all_models, all_results, feature_cols = train_all_models(monthly)
    
    # Save everything
    print("\n💾 Saving models...")
    results_df = save_all_models(all_models, all_results, feature_cols)
    save_arima_model(all_models["arima"], (1,1,1))  # Default order
    
    # Plot comparison
    plot_forecast_comparison(monthly, all_models, feature_cols)
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print(" MODEL PERFORMANCE RANKING")
    print("="*70)
    top_models = results_df.sort_values('r2', ascending=False)[['r2', 'rmse', 'mae']]
    print(top_models.round(4))
    
    print("\n✅ Pipeline complete! 7 models trained & saved:")
    print("   LinearRegression, Polynomial, Ridge, Lasso")
    print("   RandomForest, XGBoost, Prophet, ARIMA")
    
    return all_models, all_results

# ============================================
# [KEEP OTHER FUNCTIONS: train_linear_regression_models, etc. - too long for this response]
# ============================================

# Run pipeline
if __name__ == "__main__":
    run_forecasting()
