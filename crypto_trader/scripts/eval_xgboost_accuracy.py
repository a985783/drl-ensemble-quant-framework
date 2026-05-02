import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add crypto_trader to path
from crypto_trader.models.signal_model import SignalPredictor
from crypto_trader.config import load_config
from crypto_trader.data_loader import DataLoader
from crypto_trader.features import FeatureEngineer

def evaluate_xgboost():
    from sklearn.metrics import accuracy_score, classification_report
    print("Evaluating XGBoost strictly on Train(80) / OOS(20) split...")
    
    train_path = "crypto_trader/data_moe_20200101_20260216_train80.csv"
    oos_path = "crypto_trader/data_moe_20200101_20260216_oos20.csv"
    
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    oos_df = pd.read_csv(oos_path, index_col=0, parse_dates=True)
    
    predictor = SignalPredictor()
    print("="*60)
    print(f"TRAINING XGBOOST ON TRAIN SET ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print("="*60)
    
    # Train fits the model and scaler
    predictor.train(train_df)
    
    print("\n" + "="*60)
    print(f"EVALUATING XGBOOST DIRECTLY ON OOS SET ({oos_df.index[0].date()} to {oos_df.index[-1].date()})")
    print("="*60)
    
    # Prepare OOS data
    X_oos, y_oos = predictor._prepare_features_and_labels(oos_df, training=True)
    
    # Scale and predict
    # X_oos = X_oos.reindex(columns=predictor.scaler.feature_names_in_, fill_value=0.0)
    X_oos_scaled = predictor.scaler.transform(X_oos)
    
    y_pred = predictor.model.predict(X_oos_scaled)
    acc = accuracy_score(y_oos, y_pred)
    report = classification_report(y_oos, y_pred)
    
    print(f"Signal Model Accuracy (STRICT OOS): {acc:.4f}")
    print("Classification Report (STRICT OOS):\n", report)
    
    print("\nNote: ")
    print("- Class 0 means next day Close <= current day Close (Down/Flat)")
    print("- Class 1 means next day Close > current day Close (Up)")
    print("- If Accuracy ~ 50%, it's basically a coin flip.")
    print("- If Precision for Class 1 > 53%, it has a measurable edge in crypto.")

if __name__ == "__main__":
    evaluate_xgboost()
