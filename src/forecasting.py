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
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Use smaller grid for speed, or full grid for better results
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # TimeSeriesSplit for time series validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("   Performing grid search (this may take a minute)...")
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
    
    print("\n   Feature Importance:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    # 2. Gradient Boosting
    print("\n2. Gradient Boosting Regressor")
    
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    
    gb_base = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(gb_base, gb_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    
    best_gb = gb_grid.best_estimator_
    y_pred = best_gb.predict(X_test)
    
    results["GradientBoosting"] = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "best_params": gb_grid.best_params_
    }
    models["gradient_boosting"] = best_gb
    print(f"   Best params: {gb_grid.best_params_}")
    print(f"   R² Score: {results['GradientBoosting']['r2']:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]["r2"])
    print(f"\n🏆 Best Tree Model: {best_model_name} (R²={results[best_model_name]['r2']:.4f})")
    
    # Save feature names with model
    best_rf.feature_names_in_ = np.array(feature_cols)
    models["random_forest"] = best_rf
    
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
        (1, d, 0), (0, d, 1), (3, d, 3)
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
# MODEL SAVING
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
    
    if "gradient_boosting" in models:
        joblib.dump(models["gradient_boosting"], "models/gradient_boosting.pkl")
    
    # Save model comparison results
    results_df = pd.DataFrame(results).T
    results_df.to_csv("models/model_comparison.csv")
    print("✅ Saved: models/model_comparison.csv")
    
    return results_df

def save_arima_model(model, order):
    """Save ARIMA model."""
    os.makedirs("models", exist_ok=True)
    
    # Save model parameters
    arima_info = {
        "order": order,
        "params": model.params.to_dict(),
        "aic": model.aic,
        "bic": model.bic
    }
    
    with open("models/arima_info.pkl", "wb") as f:
        pickle.dump(arima_info, f)
    
    # Save the fitted model
    joblib.dump(model, "models/arima_model.pkl")
    print("✅ Saved: models/arima_model.pkl")
    print("✅ Saved: models/arima_info.pkl")

# ============================================
# VISUALIZATION
# ============================================

def plot_forecast_comparison(monthly, models, feature_cols=None):
    """Compare all model forecasts."""
    os.makedirs("reports/figures", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Linear Regression
    ax1 = axes[0, 0]
    X = monthly[["MonthIndex"]]
    y = monthly["Revenue"]
    lr_pred = models["linear_regression"].predict(X)
    ax1.plot(monthly["MonthYear"], y, marker='o', label='Actual', linewidth=2)
    ax1.plot(monthly["MonthYear"], lr_pred, '--', label='Linear Regression', alpha=0.7)
    ax1.set_title('Linear Regression', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue (£)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Random Forest
    ax2 = axes[0, 1]
    if "random_forest" in models and feature_cols:
        feature_df = create_time_features(monthly)
        X_rf = feature_df[feature_cols]
        rf_pred = models["random_forest"].predict(X_rf)
        ax2.plot(feature_df["MonthYear"], feature_df["Revenue"], marker='o', label='Actual', linewidth=2)
        ax2.plot(feature_df["MonthYear"], rf_pred, '--', label='Random Forest', alpha=0.7)
        ax2.set_title('Random Forest', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Revenue (£)')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. ARIMA
    ax3 = axes[1, 0]
    series = monthly.set_index("MonthYear")["Revenue"]
    arima_pred = models["arima"].fittedvalues
    ax3.plot(series.index, series.values, marker='o', label='Actual', linewidth=2)
    ax3.plot(series.index, arima_pred, '--', label='ARIMA', alpha=0.7)
    ax3.set_title('ARIMA', fontsize=12)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Revenue (£)')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Model Comparison
    ax4 = axes[1, 1]
    results = pd.read_csv("models/model_comparison.csv", index_col=0)
    models_names = results.index
    r2_scores = results['r2']
    bars = ax4.bar(models_names, r2_scores)
    ax4.set_title('Model Comparison (R² Score)', fontsize=12)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('R² Score')
    ax4.set_ylim([0, 1])
    for bar, score in zip(bars, r2_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', fontsize=9)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Chart saved: reports/figures/model_comparison.png")

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
    rf_models, rf_results, best_rf, feature_cols = train_random_forest_models(feature_df)
    
    # Train ARIMA model
    arima_model, arima_order, arima_results = train_arima_model(monthly)
    
    # Combine all models
    all_models = {**lr_models, **rf_models}
    all_models["arima"] = arima_model
    
    all_results = {**lr_results, **rf_results, "ARIMA": arima_results}
    
    # Save models
    print("\n" + "="*50)
    print("💾 SAVING MODELS")
    print("="*50)
    save_models(all_models, all_results, feature_cols)
    save_arima_model(arima_model, arima_order)
    
    # Create comparison chart
    plot_forecast_comparison(monthly, all_models, feature_cols)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ FORECASTING PIPELINE COMPLETE!")
    print("="*60)
    print("\n📊 Model Performance Summary:")
    print("-"*40)
    
    results_df = pd.read_csv("models/model_comparison.csv", index_col=0)
    print(results_df[['r2', 'mae', 'rmse']].to_string())
    
    print("\n🎯 Best performing models:")
    print(f"   Linear: {best_lr}")
    print(f"   Tree-based: {best_rf}")
    print(f"   ARIMA Order: {arima_order}")
    
    print("\n📁 Files saved:")
    print("   - models/*.pkl (all trained models)")
    print("   - models/model_comparison.csv")
    print("   - reports/figures/model_comparison.png")
    
    return all_models, all_results

# ============================================
# QUICK PREDICTION FUNCTION FOR DASHBOARD
# ============================================

def predict_future(model_type, months_ahead, monthly_data, model=None):
    """Make future predictions using trained models."""
    if model_type == "linear_regression":
        if model is None:
            model = joblib.load("models/linear_regression.pkl")
        last_idx = monthly_data["MonthIndex"].max()
        future_idx = pd.DataFrame({
            "MonthIndex": range(last_idx + 1, last_idx + months_ahead + 1)
        })
        predictions = model.predict(future_idx)
        
    elif model_type == "random_forest":
        if model is None:
            model = joblib.load("models/random_forest.pkl")
            with open("models/rf_features.pkl", "rb") as f:
                feature_cols = pickle.load(f)
        
        # Prepare features for future
        last_date = monthly_data["MonthYear"].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]
        
        predictions = []
        last_revenues = monthly_data["Revenue"].tail(3).tolist()
        
        for i, future_date in enumerate(future_dates):
            future_features = pd.DataFrame({
                "MonthIndex": [monthly_data["MonthIndex"].max() + i + 1],
                "Month": [future_date.month],
                "Quarter": [(future_date.month - 1) // 3 + 1],
                "RevLag1": [last_revenues[-1]],
                "RevLag2": [last_revenues[-2]],
                "RevLag3": [last_revenues[-3] if len(last_revenues) > 2 else last_revenues[-1]],
                "RollingMean3": [np.mean(last_revenues[-3:])],
                "MonthSin": [np.sin(2 * np.pi * future_date.month / 12)],
                "MonthCos": [np.cos(2 * np.pi * future_date.month / 12)]
            })
            pred = model.predict(future_features[feature_cols])[0]
            predictions.append(pred)
            last_revenues.append(pred)
            if len(last_revenues) > 3:
                last_revenues.pop(0)
        
    elif model_type == "arima":
        if model is None:
            model = joblib.load("models/arima_model.pkl")
        forecast = model.forecast(steps=months_ahead)
        predictions = forecast.values
    
    return predictions

if __name__ == "__main__":
    run_forecasting()
