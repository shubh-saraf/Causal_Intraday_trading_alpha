"""
FeatureSelector Module - Selects best features for the model.

Methods:
- Variance threshold
- Correlation filtering
- Feature importance from initial model
- Mutual information

Causality: Feature selection is done on training data only.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
import lightgbm as lgb

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Selects the most important features for prediction.
    """
    
    def __init__(self, 
                 n_features: int = 100,
                 variance_threshold: float = 0.001,
                 correlation_threshold: float = 0.95):
        """
        Initialize the feature selector.
        
        Args:
            n_features: Maximum number of features to select
            variance_threshold: Minimum variance for features
            correlation_threshold: Max correlation for feature pairs
        """
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        
        self.selected_features_: List[str] = []
        self.feature_importance_: pd.DataFrame = None
        self.fitted_ = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            self
        """
        logger.info(f"Fitting feature selector on {X.shape[1]} features")
        
        # Handle NaN/Inf
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_clean = y.fillna(0)
        
        # Step 1: Remove low variance features
        high_var_cols = self._filter_by_variance(X_clean)
        logger.info(f"After variance filter: {len(high_var_cols)} features")
        
        # Step 2: Remove highly correlated features
        uncorr_cols = self._filter_by_correlation(X_clean[high_var_cols])
        logger.info(f"After correlation filter: {len(uncorr_cols)} features")
        
        # Step 3: Rank by importance using LightGBM
        importance_df = self._get_feature_importance(
            X_clean[uncorr_cols], y_clean
        )
        
        # Step 4: Select top N features
        top_features = importance_df.head(self.n_features)['feature'].tolist()
        
        self.selected_features_ = top_features
        self.feature_importance_ = importance_df
        self.fitted_ = True
        
        logger.info(f"Selected {len(self.selected_features_)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform to selected features only.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        if not self.fitted_:
            raise ValueError("FeatureSelector must be fitted first")
        
        # Use only features that exist in both selected and input
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _filter_by_variance(self, X: pd.DataFrame) -> List[str]:
        """Remove features with variance below threshold."""
        variances = X.var()
        high_var = variances[variances > self.variance_threshold]
        return high_var.index.tolist()
    
    def _filter_by_correlation(self, X: pd.DataFrame) -> List[str]:
        """Remove highly correlated features, keeping the first."""
        # Sample for efficiency
        sample = X.iloc[:min(5000, len(X))]
        
        corr_matrix = sample.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find columns to drop
        to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        
        return [c for c in X.columns if c not in to_drop]
    
    def _get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get feature importance using LightGBM.
        """
        # Train a quick model
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100
        )
        
        # Get importance
        importance = model.feature_importance(importance_type='gain')
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_feature_report(self) -> pd.DataFrame:
        """Return feature importance report."""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        return self.feature_importance_.copy()