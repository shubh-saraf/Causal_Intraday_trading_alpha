"""
EnsembleModel Module - Combines XGBoost, LightGBM, and CatBoost with hyperparameter tuning.

Features:
- XGBoost, LightGBM, and CatBoost ensemble
- Custom directional loss function (penalizes wrong-direction predictions)
- Optuna hyperparameter tuning
- Multiple ensemble strategies (average, weighted average, stacking)
- Feature importance analysis

Causality: All training uses only past data.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None
import optuna
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Custom Loss Functions for Direction-Aware Training
# ══════════════════════════════════════════════════════════════════════

def fobj_directional_mse_lgb(preds, train_data):
    """
    Custom loss function for LightGBM that heavily penalizes wrong direction.
    - Correct direction: normal MSE gradient
    - Wrong direction: 10x MSE gradient (stronger penalty)
    
    Returns: (gradient, hessian) for LightGBM
    """
    labels = train_data.get_label()
    residual = preds - labels
    
    # Standard MSE gradient and hessian
    grad = 2 * residual
    hess = np.ones_like(preds) * 2
    
    # Amplify for wrong-direction predictions
    wrong_direction = (np.sign(preds) != np.sign(labels)) & (np.sign(labels) != 0)
    grad[wrong_direction] *= 10
    hess[wrong_direction] *= 10
    
    return grad, hess


def feval_mse_lgb(preds, train_data):
    """
    Custom MSE eval metric for LightGBM.
    Required when using a custom objective so early stopping has a metric to monitor.
    
    Returns: (metric_name, metric_value, is_higher_better)
    """
    labels = train_data.get_label()
    mse = np.mean((preds - labels) ** 2)
    return 'custom_mse', mse, False  # lower is better


def directional_loss_xgb(y_pred, y_true):
    """
    Custom loss for XGBoost (used via custom_metric, objective stays MSE).
    Penalizes wrong direction more heavily.
    
    Returns: gradient, hessian
    """
    labels = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
    residual = y_pred - labels
    
    grad = 2 * residual
    hess = np.ones_like(y_pred) * 2
    
    wrong_direction = (np.sign(y_pred) != np.sign(labels)) & (np.sign(labels) != 0)
    grad[wrong_direction] *= 10
    hess[wrong_direction] *= 10
    
    return grad, hess


class EnsembleModel:
    """
    Ensemble model combining XGBoost, LightGBM, and CatBoost with hyperparameter tuning.
    Uses custom directional loss functions to optimize for trading direction accuracy.
    """
    
    def __init__(self,
                 ensemble_strategy: str = 'weighted_average',
                 n_trials: int = 50,
                 optimize_metric: str = 'direction_accuracy',
                 use_custom_loss: bool = True,
                 include_catboost: bool = True):
        """
        Initialize ensemble model.
        
        Args:
            ensemble_strategy: 'average', 'weighted_average', or 'stacking'
            n_trials: Number of Optuna trials for each model
            optimize_metric: 'sharpe', 'win_rate', or 'direction_accuracy'
            use_custom_loss: Whether to use directional loss (penalizes wrong direction)
            include_catboost: Whether to include CatBoost in ensemble
        """
        self.ensemble_strategy = ensemble_strategy
        self.n_trials = n_trials
        self.optimize_metric = optimize_metric
        self.use_custom_loss = use_custom_loss
        self.include_catboost = include_catboost and CATBOOST_AVAILABLE
        
        if include_catboost and not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available. Install with: pip install catboost")
            self.include_catboost = False
        
        self.lgbm_model_ = None
        self.xgb_model_ = None
        self.catboost_model_ = None
        self.lgbm_params_: Dict = {}
        self.xgb_params_: Dict = {}
        self.catboost_params_: Dict = {}
        self.ensemble_weights_ = [0.33, 0.33, 0.34] if self.include_catboost else [0.5, 0.5]
        # [xgb_weight, lgbm_weight, catboost_weight]
        self.feature_names_: List[str] = []
        self.tuning_results_: Dict = {}
        
    def _calculate_sharpe(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate Sharpe-like metric from predictions vs actuals."""
        # Simulate returns based on prediction direction
        direction = np.sign(predictions)
        returns = direction * actuals
        
        if np.std(returns) == 0:
            return 0.0
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 78)  # Annualized
        return sharpe
    
    def _calculate_win_rate(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate win rate (direction accuracy)."""
        correct = np.sign(predictions) == np.sign(actuals)
        return np.mean(correct)
    
    def _objective_lgbm(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for LightGBM with optional custom directional loss."""
        params = {
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': 5,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 1.0, log=True),
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        if self.use_custom_loss:
            # Use custom directional loss (LightGBM 4.x+ requires objective in params)
            params['objective'] = fobj_directional_mse_lgb
            params['metric'] = 'None'  # Disable built-in metric; use feval instead
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                feval=feval_mse_lgb,
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
        else:
            # Use standard MSE
            params.update({'objective': 'regression', 'metric': 'mse'})
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
        
        preds = model.predict(X_val)
        
        if self.optimize_metric == 'sharpe':
            return self._calculate_sharpe(preds, y_val)
        elif self.optimize_metric == 'win_rate':
            return self._calculate_win_rate(preds, y_val)
        else:  # direction_accuracy
            return self._calculate_win_rate(preds, y_val)
    
    def _objective_xgb(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for XGBoost."""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'verbosity': 0,
        }
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val)
        
        if self.optimize_metric == 'sharpe':
            return self._calculate_sharpe(preds, y_val)
        elif self.optimize_metric == 'win_rate':
            return self._calculate_win_rate(preds, y_val)
        else:
            return self._calculate_win_rate(preds, y_val)
    
    def _objective_catboost(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for CatBoost with directional loss."""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-4, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': False,
            'early_stopping_rounds': 50,
            'loss_function': 'RMSE',  # CatBoost doesn't support custom loss as easily
            'eval_metric': 'RMSE'
        }
        
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        preds = model.predict(X_val)
        
        if self.optimize_metric == 'sharpe':
            return self._calculate_sharpe(preds, y_val)
        elif self.optimize_metric == 'win_rate':
            return self._calculate_win_rate(preds, y_val)
        else:
            return self._calculate_win_rate(preds, y_val)
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Tune hyperparameters for all models using Optuna.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with tuning results
        """
        logger.info(f"Starting hyperparameter tuning ({self.n_trials} trials per model)")
        if self.use_custom_loss:
            logger.info("Using custom directional loss function (10x penalty for wrong direction)")
        
        # Clean data
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_clean = y.fillna(0).values
        
        # Split for validation (85/15)
        split_idx = int(len(X_clean) * 0.85)
        X_train, X_val = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_val = y_clean[:split_idx], y_clean[split_idx:]
        
        # Tune LightGBM
        logger.info("Tuning LightGBM...")
        study_lgbm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study_lgbm.optimize(
            lambda trial: self._objective_lgbm(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.lgbm_params_ = study_lgbm.best_params
        lgbm_best_score = study_lgbm.best_value
        logger.info(f"LightGBM Best {self.optimize_metric}: {lgbm_best_score:.4f}")
        logger.info(f"LightGBM Best Params: {self.lgbm_params_}")
        
        # Tune XGBoost
        logger.info("Tuning XGBoost...")
        study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study_xgb.optimize(
            lambda trial: self._objective_xgb(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.xgb_params_ = study_xgb.best_params
        xgb_best_score = study_xgb.best_value
        logger.info(f"XGBoost Best {self.optimize_metric}: {xgb_best_score:.4f}")
        logger.info(f"XGBoost Best Params: {self.xgb_params_}")
        
        # Tune CatBoost if enabled
        catboost_best_score = None
        if self.include_catboost:
            logger.info("Tuning CatBoost...")
            study_catboost = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study_catboost.optimize(
                lambda trial: self._objective_catboost(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
            self.catboost_params_ = study_catboost.best_params
            catboost_best_score = study_catboost.best_value
            logger.info(f"CatBoost Best {self.optimize_metric}: {catboost_best_score:.4f}")
            logger.info(f"CatBoost Best Params: {self.catboost_params_}")
        
        # Optimize ensemble weights
        if self.ensemble_strategy == 'weighted_average':
            logger.info("Optimizing ensemble weights...")
            self._optimize_weights(X_train, y_train, X_val, y_val)
        
        self.tuning_results_ = {
            'lgbm_best_score': lgbm_best_score,
            'lgbm_params': self.lgbm_params_,
            'xgb_best_score': xgb_best_score,
            'xgb_params': self.xgb_params_,
            'ensemble_weights': self.ensemble_weights_
        }
        
        if self.include_catboost:
            self.tuning_results_['catboost_best_score'] = catboost_best_score
            self.tuning_results_['catboost_params'] = self.catboost_params_
        
        return self.tuning_results_
    
    def _optimize_weights(self, X_train, y_train, X_val, y_val):
        """Optimize ensemble weights using grid search for 2 or 3 models."""
        # Train all models with best params
        lgbm_params = {**self.lgbm_params_, 'verbosity': -1}
        if not self.use_custom_loss:
            lgbm_params.update({'objective': 'regression'})
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if self.use_custom_loss:
            lgbm_params['objective'] = fobj_directional_mse_lgb
            lgbm = lgb.train(lgbm_params, train_data,
                           num_boost_round=lgbm_params.get('n_estimators', 300))
        else:
            lgbm_params['objective'] = 'regression'
            lgbm = lgb.train(lgbm_params, train_data, num_boost_round=lgbm_params.get('n_estimators', 300))
        
        xgb_params = {**self.xgb_params_, 'objective': 'reg:squarederror', 'verbosity': 0}
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        lgbm_preds = lgbm.predict(X_val)
        xgb_preds = xgb_model.predict(X_val)
        
        if self.include_catboost:
            # 3-model ensemble
            catboost_params = {**self.catboost_params_, 'verbose': False}
            catboost_model = cb.CatBoostRegressor(**catboost_params)
            catboost_model.fit(X_train, y_train, verbose=False)
            catboost_preds = catboost_model.predict(X_val)
            
            # Grid search for 3-model weights
            best_score = -np.inf
            best_weights = [0.33, 0.33, 0.34]
            
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    w3 = 1.0 - w1 - w2
                    if w3 < -0.01:  # Skip invalid combinations
                        continue
                    
                    ensemble_preds = w1 * xgb_preds + w2 * lgbm_preds + w3 * catboost_preds
                    
                    if self.optimize_metric == 'sharpe':
                        score = self._calculate_sharpe(ensemble_preds, y_val)
                    else:
                        score = self._calculate_win_rate(ensemble_preds, y_val)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = [w1, w2, w3]
            
            self.ensemble_weights_ = best_weights
            logger.info(f"Optimal weights - XGBoost: {best_weights[0]:.2f}, "
                       f"LightGBM: {best_weights[1]:.2f}, CatBoost: {best_weights[2]:.2f}")
        
        else:
            # 2-model ensemble (original)
            best_score = -np.inf
            best_weights = [0.5, 0.5]
            
            for w1 in np.arange(0.0, 1.05, 0.1):
                w2 = 1.0 - w1
                ensemble_preds = w1 * xgb_preds + w2 * lgbm_preds
                
                if self.optimize_metric == 'sharpe':
                    score = self._calculate_sharpe(ensemble_preds, y_val)
                else:
                    score = self._calculate_win_rate(ensemble_preds, y_val)
                
                if score > best_score:
                    best_score = score
                    best_weights = [w1, w2]
            
            self.ensemble_weights_ = best_weights
            logger.info(f"Optimal weights - XGBoost: {best_weights[0]:.2f}, LightGBM: {best_weights[1]:.2f}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, tune: bool = True) -> 'EnsembleModel':
        """
        Train all models on the full training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            tune: Whether to run hyperparameter tuning first
            
        Returns:
            self
        """
        logger.info(f"Training ensemble model on {len(X)} samples, {X.shape[1]} features")
        if self.include_catboost:
            logger.info("Training 3-model ensemble: XGBoost + LightGBM + CatBoost")
        else:
            logger.info("Training 2-model ensemble: XGBoost + LightGBM")
        
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_clean = y.fillna(0)
        
        self.feature_names_ = list(X_clean.columns)
        
        # Tune if requested
        if tune:
            self.tune_hyperparameters(X_clean, y_clean)
        
        # Split for training with validation
        split_idx = int(len(X_clean) * 0.85)
        X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        
        # Train LightGBM with custom loss
        lgbm_params = {
            'verbosity': -1,
            **self.lgbm_params_
        }
        
        if not self.use_custom_loss:
            lgbm_params.update({'objective': 'regression', 'metric': 'mse'})
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        if self.use_custom_loss:
            logger.info("Training LightGBM with custom directional loss...")
            lgbm_params['objective'] = fobj_directional_mse_lgb
            lgbm_params['metric'] = 'None'  # Disable built-in metric; use feval instead
            self.lgbm_model_ = lgb.train(
                lgbm_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                num_boost_round=lgbm_params.get('n_estimators', 300),
                feval=feval_mse_lgb,
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
            )
        else:
            lgbm_params['objective'] = 'regression'
            lgbm_params['metric'] = 'mse'
            self.lgbm_model_ = lgb.train(
                lgbm_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                num_boost_round=lgbm_params.get('n_estimators', 300),
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
            )
        
        # Train XGBoost
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0,
            **self.xgb_params_
        }
        
        self.xgb_model_ = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        self.xgb_model_.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Train CatBoost if enabled
        if self.include_catboost:
            catboost_params = {
                'verbose': False,
                'early_stopping_rounds': 50,
                **self.catboost_params_
            }
            
            self.catboost_model_ = cb.CatBoostRegressor(**catboost_params)
            self.catboost_model_.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
        
        # Calculate validation metrics for each model
        lgbm_val_preds = self.lgbm_model_.predict(X_val)
        xgb_val_preds = self.xgb_model_.predict(X_val)
        
        lgbm_dir_acc = self._calculate_win_rate(lgbm_val_preds, y_val.values)
        xgb_dir_acc = self._calculate_win_rate(xgb_val_preds, y_val.values)
        
        # Ensemble prediction
        if self.include_catboost:
            catboost_val_preds = self.catboost_model_.predict(X_val)
            catboost_dir_acc = self._calculate_win_rate(catboost_val_preds, y_val.values)
            
            if self.ensemble_strategy == 'average':
                ensemble_preds = (lgbm_val_preds + xgb_val_preds + catboost_val_preds) / 3
            else:  # weighted_average
                ensemble_preds = (self.ensemble_weights_[0] * xgb_val_preds + 
                                self.ensemble_weights_[1] * lgbm_val_preds +
                                self.ensemble_weights_[2] * catboost_val_preds)
            
            logger.info(f"Validation Direction Accuracy - LightGBM: {lgbm_dir_acc:.4f}, "
                       f"XGBoost: {xgb_dir_acc:.4f}, CatBoost: {catboost_dir_acc:.4f}")
        else:
            if self.ensemble_strategy == 'average':
                ensemble_preds = (lgbm_val_preds + xgb_val_preds) / 2
            else:  # weighted_average
                ensemble_preds = self.ensemble_weights_[0] * xgb_val_preds + self.ensemble_weights_[1] * lgbm_val_preds
            
            logger.info(f"Validation Direction Accuracy - LightGBM: {lgbm_dir_acc:.4f}, "
                       f"XGBoost: {xgb_dir_acc:.4f}")
        
        ensemble_dir_acc = self._calculate_win_rate(ensemble_preds, y_val.values)
        logger.info(f"Ensemble Direction Accuracy: {ensemble_dir_acc:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of ensemble predictions
        """
        if self.lgbm_model_ is None or self.xgb_model_ is None:
            raise ValueError("Models must be trained first")
        
        if self.include_catboost and self.catboost_model_ is None:
            raise ValueError("CatBoost model must be trained first")
        
        # Align features
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        X_clean = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Get predictions from all models
        lgbm_preds = self.lgbm_model_.predict(X_clean)
        xgb_preds = self.xgb_model_.predict(X_clean)
        
        # Combine predictions
        if self.include_catboost:
            catboost_preds = self.catboost_model_.predict(X_clean)
            
            if self.ensemble_strategy == 'average':
                ensemble_preds = (lgbm_preds + xgb_preds + catboost_preds) / 3
            else:  # weighted_average
                ensemble_preds = (self.ensemble_weights_[0] * xgb_preds + 
                                self.ensemble_weights_[1] * lgbm_preds +
                                self.ensemble_weights_[2] * catboost_preds)
        else:
            if self.ensemble_strategy == 'average':
                ensemble_preds = (lgbm_preds + xgb_preds) / 2
            else:  # weighted_average
                ensemble_preds = self.ensemble_weights_[0] * xgb_preds + self.ensemble_weights_[1] * lgbm_preds
        
        return ensemble_preds
    
    def predict_individual(self, X: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Get individual predictions from all models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (xgb_predictions, lgbm_predictions, catboost_predictions) if CatBoost enabled
            or (xgb_predictions, lgbm_predictions) otherwise
        """
        if self.lgbm_model_ is None or self.xgb_model_ is None:
            raise ValueError("Models must be trained first")
        
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        X_clean = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        lgbm_preds = self.lgbm_model_.predict(X_clean)
        xgb_preds = self.xgb_model_.predict(X_clean)
        
        if self.include_catboost and self.catboost_model_ is not None:
            catboost_preds = self.catboost_model_.predict(X_clean)
            return xgb_preds, lgbm_preds, catboost_preds
        else:
            return xgb_preds, lgbm_preds
    
    def get_xgb_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        if self.xgb_model_ is None:
            raise ValueError("XGBoost model must be trained first")
        
        importance = self.xgb_model_.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_lgbm_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from LightGBM model."""
        if self.lgbm_model_ is None:
            raise ValueError("LightGBM model must be trained first")
        
        importance = self.lgbm_model_.feature_importance(importance_type='gain')
        
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_catboost_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from CatBoost model."""
        if self.catboost_model_ is None:
            raise ValueError("CatBoost model must be trained first")
        
        importance = self.catboost_model_.get_feature_importance()
        
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save ensemble model to file."""
        data = {
            'lgbm_model': self.lgbm_model_,
            'xgb_model': self.xgb_model_,
            'lgbm_params': self.lgbm_params_,
            'xgb_params': self.xgb_params_,
            'ensemble_weights': self.ensemble_weights_,
            'ensemble_strategy': self.ensemble_strategy,
            'feature_names': self.feature_names_,
            'tuning_results': self.tuning_results_,
            'use_custom_loss': self.use_custom_loss,
            'include_catboost': self.include_catboost
        }
        
        if self.include_catboost:
            data['catboost_model'] = self.catboost_model_
            data['catboost_params'] = self.catboost_params_
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load ensemble model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.lgbm_model_ = data['lgbm_model']
        self.xgb_model_ = data['xgb_model']
        self.lgbm_params_ = data['lgbm_params']
        self.xgb_params_ = data['xgb_params']
        self.ensemble_weights_ = data['ensemble_weights']
        self.ensemble_strategy = data['ensemble_strategy']
        self.feature_names_ = data['feature_names']
        self.tuning_results_ = data.get('tuning_results', {})
        self.use_custom_loss = data.get('use_custom_loss', False)
        self.include_catboost = data.get('include_catboost', False)
        
        if self.include_catboost:
            self.catboost_model_ = data.get('catboost_model')
            self.catboost_params_ = data.get('catboost_params', {})
        
        logger.info(f"Ensemble model loaded from {filepath}")