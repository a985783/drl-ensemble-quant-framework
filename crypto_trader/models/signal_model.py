"""
Signal Model Module | 信号模型模块

Predicts market direction using XGBoost classification.
Outputs probability of price going UP, used as a feature for RL agents.

使用 XGBoost 分类预测市场方向。
输出价格上涨的概率，作为 RL 代理的特征输入。
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import signal_model_config


class SignalPredictor:
    """
    Market Direction Predictor using XGBoost | 使用 XGBoost 的市场方向预测器
    
    This model predicts the probability of the market going UP (Close[t+1] > Close[t]).
    The prediction is used as an additional feature for the RL ensemble.
    
    本模型预测市场上涨的概率（Close[t+1] > Close[t]）。
    预测结果作为 RL 集成的额外特征使用。
    
    Attributes:
        model: XGBoost classifier instance
        scaler: StandardScaler for feature normalization
        excluded_cols: Columns to exclude from features
        
    Configuration (from config.yaml):
        - n_estimators: Number of boosting rounds (default: 100)
        - max_depth: Maximum tree depth (default: 3)
        - learning_rate: Boosting learning rate (default: 0.1)
        - subsample: Subsample ratio (default: 0.8)
        - colsample_bytree: Column subsample ratio (default: 0.8)
    """

    def __init__(self):
        """Initialize SignalPredictor with configurable hyperparameters."""
        cfg = signal_model_config()
        
        self.model = xgb.XGBClassifier(
            n_estimators=cfg.get('n_estimators', 100),
            max_depth=cfg.get('max_depth', 3),
            learning_rate=cfg.get('learning_rate', 0.1),
            subsample=cfg.get('subsample', 0.8),
            colsample_bytree=cfg.get('colsample_bytree', 0.8),
            random_state=cfg.get('random_state', 42),
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.scaler = StandardScaler()
        # Columns excluded from features (raw price/volume are non-stationary)
        # 从特征中排除的列（原始价格/成交量是非平稳的）
        self.excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']

    def _prepare_features_and_labels(self, data: pd.DataFrame, training: bool = True):
        """
        Prepare features and labels for training/prediction.
        为训练/预测准备特征和标签。
        
        Args:
            data: DataFrame with technical indicators
            training: If True, create labels. If False, return features only.
            
        Returns:
            X: Feature DataFrame
            y: Labels (only if training=True)
        """
        df = data.copy()

        # Create Label: 1 if next Close > current Close, else 0
        # 创建标签：如果下一个收盘价 > 当前收盘价则为 1，否则为 0
        if training:
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.iloc[:-1]  # Remove last row (no label)
        
        # Select feature columns | 选择特征列
        feature_cols = [c for c in df.columns if c not in self.excluded_cols]
        
        X = df[feature_cols]
        
        if training:
            y = df['Target']
            return X, y
        else:
            return X

    def train(self, data: pd.DataFrame):
        """
        Train the XGBoost model with time-series aware split.
        使用时间序列感知的划分训练 XGBoost 模型。
        
        Args:
            data: DataFrame with features and OHLCV data
            
        Returns:
            tuple: (accuracy, classification_report)
        """
        X, y = self._prepare_features_and_labels(data)
        
        # Time-series split (no shuffle) | 时间序列划分（不打乱）
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit scaler on training data only | 仅在训练数据上拟合缩放器
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training Signal Model on {len(X_train)} samples...")
        print(f"正在使用 {len(X_train)} 个样本训练信号模型...")
        
        # Train model | 训练模型
        self.model.fit(
            X_train_scaled, y_train, 
            eval_set=[(X_test_scaled, y_test)], 
            verbose=False
        )
        
        # Evaluation | 评估
        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Signal Model Accuracy: {acc:.4f}")
        print(f"信号模型准确率: {acc:.4f}")
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

