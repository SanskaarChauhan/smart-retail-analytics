#train_models.py
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.utils import load_clean_data, get_monthly_revenue

def train_forecasting_models():
    """Train LR, RF, XGBoost and save models."""
    print("Training forecasting models...")
    
    # Load latest data
    df = load_clean_data()
    monthly = get_monthly_revenue(df)
    
    # Feature engineering
    monthly = monthly.copy()
    monthly["Month"] = monthly["MonthYear"].dt.month
    monthly["Quarter"] = monthly["MonthYear"].dt.quarter
    monthly["RevLag1"] = monthly["Revenue"].shift(1)
    monthly["RevLag2"] = monthly["Revenue"].shift(2)
    monthly["RollingMean"] = monthly["Revenue"].rolling(3).mean()
    monthly = monthly.dropna().reset_index(drop=True)
    
    features = ["MonthIndex", "Month", "Quarter", "RevLag1", "RevLag2", "RollingMean"]
    X = monthly[features]
    y = monthly["Revenue"]
    
    # Train models
    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X, y)
    
    # Save models + features
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr, "models/linear_regression.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")
    joblib.dump(features, "models/rf_features.pkl")
    
    print("✅ All models trained and saved")
    return True
