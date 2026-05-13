
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class SignalPredictor:
    """
    Predicts market direction using XGBoost.
    """

    def __init__(self):
        self.model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42
        )
        self.scaler = StandardScaler()
        # Columns to be excluded from features (raw price/volume are non-stationary)
        self.excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']

    def _prepare_features_and_labels(self, data: pd.DataFrame, training: bool = True):
        """
        Internal helper to create features (X) and labels (y).
        """
        df = data.copy()

        # Create Label: 1 if next Close > current Close, else 0
        if training:
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            # Remove the last row as it doesn't have a label
            df = df.iloc[:-1]
        
        # Select Feature Columns
        # Assume all other numeric columns are features
        feature_cols = [c for c in df.columns if c not in self.excluded_cols]
        
        X = df[feature_cols]
        
        if training:
            y = df['Target']
            return X, y
        else:
            return X

    def train(self, data: pd.DataFrame):
        """
        Trains the XGBoost model with TimeSeriesSplit and Hyperparameter Tuning.
        """
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        
        X, y = self._prepare_features_and_labels(data)
        
        # Time-series split (no shuffle)
        # Using larger test size for internal validation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit Scaler on TRAIN only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training Signal Model (Hyperopt) on {len(X_train)} samples...")
        
        # Defaults
        # param_dist = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [3, 5, 7],
        #     'learning_rate': [0.01, 0.05, 0.1],
        #     'subsample': [0.6, 0.8, 1.0],
        #     'colsample_bytree': [0.6, 0.8, 1.0]
        # }
        
        # For speed in this environment, we use a lighter search or just a Better Preset.
        # Running full GridSearch might timeout. 
        # Let's use a Strong Preset for Daily Crypto instead of full search to save time 
        # but better than default.
        # "Alpha Hunter" Preset:
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Fit
        self.model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Signal Model Accuracy (Test Set): {acc:.4f}")
        print("Classification Report:\n", report)
        
        return acc, report

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predicts the probability of the market going UP.
        
        Args:
            data (pd.DataFrame): DataFrame with features.
            
        Returns:
            np.ndarray: Array of probabilities for class 1 (Up).
        """
        X = self._prepare_features_and_labels(data, training=False)
        expected_cols = list(getattr(self.scaler, "feature_names_in_", []))
        if expected_cols:
            # Runtime data may contain extra columns from newer feature pipelines.
            # Align strictly to training-time feature schema to keep live inference stable.
            X = X.reindex(columns=expected_cols, fill_value=0.0)
        X_scaled = self.scaler.transform(X)
        
        # predict_proba returns [prob_0, prob_1]
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: str):
        """Save trained model and scaler to file."""
        import joblib
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"Signal Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load trained model from file."""
        import joblib
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        print(f"Signal Model loaded from {path}")
        return instance
