"""
FeatureSanitizer Module - Handles the 5 critical sanitation issues.

Issues addressed:
1. Constant Features (67) - REMOVE
2. Extreme Autocorrelation (147) - INVESTIGATE/REMOVE
3. Scale Differences (13 orders) - NORMALIZE
4. Duplicate Features (2+ pairs) - REMOVE DUPLICATES
5. Cumulative Features (27) - DIFFERENCE or REMOVE

Causality: All normalization uses only past data (expanding or rolling windows).
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureSanitizer:
    """
    Sanitizes features to address data quality issues while maintaining causality.
    """
    
    def __init__(self, 
                 constant_threshold: float = 1e-10,
                 autocorr_threshold: float = 0.99,
                 correlation_threshold: float = 0.999,
                 normalize_method: str = 'robust'):
        """
        Initialize the FeatureSanitizer.
        
        Args:
            constant_threshold: Variance threshold for constant detection
            autocorr_threshold: Threshold for extreme autocorrelation
            correlation_threshold: Threshold for duplicate detection
            normalize_method: 'robust', 'standard', or 'minmax'
        """
        self.constant_threshold = constant_threshold
        self.autocorr_threshold = autocorr_threshold
        self.correlation_threshold = correlation_threshold
        self.normalize_method = normalize_method
        
        # Store learned parameters for consistent application
        self.constant_features_: List[str] = []
        self.duplicate_features_: List[str] = []
        self.autocorr_features_: List[str] = []
        self.cumulative_features_: List[str] = []
        self.normalization_params_: Dict = {}
        self.fitted_ = False
        
    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> 'FeatureSanitizer':
        """
        Fit the sanitizer on training data.
        
        Args:
            df: Training DataFrame
            feature_cols: List of feature columns to sanitize
            
        Returns:
            self (for method chaining)
        """
        logger.info(f"Fitting sanitizer on {len(feature_cols)} features")
        
        X = df[feature_cols].copy()
        
        # 1. Identify constant features
        self.constant_features_ = self._find_constant_features(X)
        logger.info(f"Found {len(self.constant_features_)} constant features")
        
        # Remove constant features before further analysis
        remaining = [c for c in feature_cols if c not in self.constant_features_]
        X = X[remaining]
        
        # 2. Identify duplicate features
        self.duplicate_features_ = self._find_duplicate_features(X)
        logger.info(f"Found {len(self.duplicate_features_)} duplicate features")
        
        # Remove duplicates before further analysis
        remaining = [c for c in remaining if c not in self.duplicate_features_]
        X = X[remaining]
        
        # 3. Identify highly autocorrelated features
        self.autocorr_features_ = self._find_autocorrelated_features(X)
        logger.info(f"Found {len(self.autocorr_features_)} highly autocorrelated features")
        
        # 4. Identify cumulative features
        self.cumulative_features_ = self._find_cumulative_features(X, remaining)
        logger.info(f"Found {len(self.cumulative_features_)} cumulative features")
        
        # 5. Calculate normalization parameters
        self._calculate_normalization_params(X, remaining)
        
        self.fitted_ = True
        return self
    
    def transform(self, df: pd.DataFrame, feature_cols: List[str], 
                  causal: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform features by applying sanitation.
        
        Args:
            df: DataFrame to transform
            feature_cols: Feature columns to process
            causal: If True, use causal normalization (only past data)
            
        Returns:
            Tuple of (transformed DataFrame, list of remaining feature columns)
        """
        if not self.fitted_:
            raise ValueError("Sanitizer must be fitted before transform")
        
        result = df.copy()
        
        # Get columns to keep (remove constant and duplicate)
        features_to_remove = set(self.constant_features_) | set(self.duplicate_features_)
        remaining_cols = [c for c in feature_cols if c not in features_to_remove]
        
        # Apply differencing to cumulative features
        for col in self.cumulative_features_:
            if col in remaining_cols:
                # CAUSAL: diff only uses current and previous value
                result[f'{col}_diff'] = result[col].diff().fillna(0)
                remaining_cols.remove(col)
                remaining_cols.append(f'{col}_diff')
        
        # Apply normalization
        if causal:
            result = self._causal_normalize(result, remaining_cols)
        else:
            result = self._batch_normalize(result, remaining_cols)
        
        # Handle any remaining NaN/Inf values
        for col in remaining_cols:
            if col in result.columns:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                result[col] = result[col].fillna(0)
        
        logger.info(f"Transformed to {len(remaining_cols)} features")
        return result, remaining_cols
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str],
                      causal: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Fit and transform in one step."""
        self.fit(df, feature_cols)
        return self.transform(df, feature_cols, causal)
    
    def _find_constant_features(self, X: pd.DataFrame) -> List[str]:
        """Identify features with near-zero variance."""
        variances = X.var()
        constant = variances[variances < self.constant_threshold].index.tolist()
        return constant
    
    def _find_duplicate_features(self, X: pd.DataFrame) -> List[str]:
        """Identify duplicate features via correlation."""
        duplicates = []
        cols = list(X.columns)
        
        # Sample for efficiency
        sample_size = min(5000, len(X))
        X_sample = X.iloc[:sample_size]
        
        # Calculate correlation matrix
        corr_matrix = X_sample.corr().abs()
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    duplicates.append(cols[j])
        
        return list(set(duplicates))
    
    def _find_autocorrelated_features(self, X: pd.DataFrame) -> List[str]:
        """Identify features with extreme lag-1 autocorrelation."""
        autocorr_features = []
        
        for col in X.columns:
            try:
                autocorr = X[col].autocorr(lag=1)
                if not np.isnan(autocorr) and abs(autocorr) > self.autocorr_threshold:
                    autocorr_features.append(col)
            except Exception:
                continue
        
        return autocorr_features
    
    def _find_cumulative_features(self, X: pd.DataFrame, cols: List[str]) -> List[str]:
        """
        Identify cumulative features based on:
        1. Naming patterns (cumulative, sum, total, etc.)
        2. Monotonic behavior
        """
        cumulative = []
        cumulative_patterns = ['_C', '_cum', '_total', '_sum', 'CV', 'CPV', 'CCV']
        
        for col in cols:
            # Check naming pattern
            if any(pattern in col for pattern in cumulative_patterns):
                cumulative.append(col)
                continue
            
            # Check monotonicity (sample for efficiency)
            try:
                values = X[col].dropna().values[:1000]
                if len(values) > 100:
                    diffs = np.diff(values)
                    pos_ratio = np.mean(diffs > 0)
                    neg_ratio = np.mean(diffs < 0)
                    # If mostly increasing or decreasing, likely cumulative
                    if pos_ratio > 0.95 or neg_ratio > 0.95:
                        cumulative.append(col)
            except Exception:
                continue
        
        return list(set(cumulative))
    
    def _calculate_normalization_params(self, X: pd.DataFrame, cols: List[str]):
        """Calculate normalization parameters from training data."""
        for col in cols:
            if col not in X.columns:
                continue
            values = X[col].dropna()
            if len(values) == 0:
                continue
                
            if self.normalize_method == 'robust':
                self.normalization_params_[col] = {
                    'median': values.median(),
                    'iqr': values.quantile(0.75) - values.quantile(0.25) + 1e-10
                }
            elif self.normalize_method == 'standard':
                self.normalization_params_[col] = {
                    'mean': values.mean(),
                    'std': values.std() + 1e-10
                }
            else:  # minmax
                self.normalization_params_[col] = {
                    'min': values.min(),
                    'max': values.max() + 1e-10
                }
    
    def _causal_normalize(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        CAUSAL normalization using expanding window (only past data).
        """
        result = df.copy()
        
        for col in cols:
            if col not in result.columns:
                continue
            
            if self.normalize_method == 'robust':
                # Use expanding median and IQR
                expanding = result[col].expanding(min_periods=10)
                median = expanding.median()
                q75 = expanding.quantile(0.75)
                q25 = expanding.quantile(0.25)
                iqr = (q75 - q25).replace(0, 1e-10)
                result[col] = (result[col] - median) / iqr
            else:
                # Use expanding mean and std
                expanding = result[col].expanding(min_periods=10)
                mean = expanding.mean()
                std = expanding.std().replace(0, 1e-10)
                result[col] = (result[col] - mean) / std
        
        return result
    
    def _batch_normalize(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Batch normalization using pre-calculated parameters."""
        result = df.copy()
        
        for col in cols:
            if col not in result.columns or col not in self.normalization_params_:
                continue
            
            params = self.normalization_params_[col]
            
            if self.normalize_method == 'robust':
                result[col] = (result[col] - params['median']) / params['iqr']
            elif self.normalize_method == 'standard':
                result[col] = (result[col] - params['mean']) / params['std']
            else:
                result[col] = (result[col] - params['min']) / (params['max'] - params['min'])
        
        return result
    
    def get_sanitation_report(self) -> Dict:
        """Generate a report of sanitation actions taken."""
        return {
            'constant_features_removed': len(self.constant_features_),
            'duplicate_features_removed': len(self.duplicate_features_),
            'autocorrelated_features': len(self.autocorr_features_),
            'cumulative_features_differenced': len(self.cumulative_features_),
            'total_features_removed': len(self.constant_features_) + len(self.duplicate_features_)
        }