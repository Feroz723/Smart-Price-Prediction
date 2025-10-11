import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import pickle
import os

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"Warning: LightGBM not available ({str(e)}), will skip LightGBM model")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except (ImportError, OSError) as e:
    CATBOOST_AVAILABLE = False
    print(f"Warning: CatBoost not available ({str(e)}), will skip CatBoost model")

class PricePredictor:
    """Ensemble model for price prediction combining multiple regressors."""
    
    def __init__(self, model_type='ensemble'):
        """
        Initialize price predictor.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'catboost', 'rf', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.weights = {}
        
        if model_type == 'ensemble':
            self.models['xgboost'] = XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            if LIGHTGBM_AVAILABLE:
                self.models['lightgbm'] = LGBMRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            
            if CATBOOST_AVAILABLE:
                self.models['catboost'] = CatBoostRegressor(
                    iterations=200,
                    depth=8,
                    learning_rate=0.05,
                    random_seed=42,
                    verbose=False
                )
            
            self.models['rf'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            total_weight = 1.0
            xgb_weight = 0.4
            lgb_weight = 0.3 if LIGHTGBM_AVAILABLE else 0
            cat_weight = 0.2 if CATBOOST_AVAILABLE else 0
            rf_weight = 0.1
            
            if not LIGHTGBM_AVAILABLE:
                xgb_weight += 0.15
                rf_weight += 0.15
            if not CATBOOST_AVAILABLE:
                xgb_weight += 0.1
                rf_weight += 0.1
            
            self.weights = {
                'xgboost': xgb_weight,
                'rf': rf_weight
            }
            if LIGHTGBM_AVAILABLE:
                self.weights['lightgbm'] = lgb_weight
            if CATBOOST_AVAILABLE:
                self.weights['catboost'] = cat_weight
        
        elif model_type == 'xgboost':
            self.models['xgboost'] = XGBRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            self.weights['xgboost'] = 1.0
        
        elif model_type == 'lightgbm':
            if LIGHTGBM_AVAILABLE:
                self.models['lightgbm'] = LGBMRegressor(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                self.weights['lightgbm'] = 1.0
            else:
                print("LightGBM not available, falling back to XGBoost")
                self.models['xgboost'] = XGBRegressor(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                self.weights['xgboost'] = 1.0
        
        elif model_type == 'catboost':
            if CATBOOST_AVAILABLE:
                self.models['catboost'] = CatBoostRegressor(
                    iterations=300,
                    depth=10,
                    learning_rate=0.03,
                    random_seed=42,
                    verbose=False
                )
                self.weights['catboost'] = 1.0
            else:
                print("CatBoost not available, falling back to XGBoost")
                self.models['xgboost'] = XGBRegressor(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                self.weights['xgboost'] = 1.0
        
        elif model_type == 'rf':
            self.models['rf'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.weights['rf'] = 1.0
    
    def fit(self, X_train, y_train):
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print(f"\nTraining {self.model_type} model(s)...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            print(f"  {name} training complete")
    
    def predict(self, X_test):
        """
        Predict using ensemble of models.
        
        Args:
            X_test: Test features
        
        Returns:
            np.array: Predicted prices
        """
        if self.model_type == 'ensemble':
            predictions = np.zeros(len(X_test))
            
            for name, model in self.models.items():
                pred = model.predict(X_test)
                predictions += self.weights[name] * pred
            
            predictions = np.maximum(predictions, 0)
            return predictions
        
        else:
            model_name = list(self.models.keys())[0]
            predictions = self.models[model_name].predict(X_test)
            predictions = np.maximum(predictions, 0)
            return predictions
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        
        Returns:
            PricePredictor: Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model


def smape_metric(y_true, y_pred):
    """
    Custom SMAPE metric for model evaluation.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        float: SMAPE score
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / (denominator + 1e-10)) * 100
    return smape
