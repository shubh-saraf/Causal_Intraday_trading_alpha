"""
TargetCreator Module - Creates the prediction target for the model.

The target is the forward return of P3 at a specified horizon (default 30 bars).

Causality: The target is computed AFTER the current time - it's what we're
predicting. At inference time, we never have access to the target.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TargetCreator:
    """
    Creates forward-looking targets for model training.
    
    Target = P3[t+horizon] / P3[t] - 1 (forward return)
    """
    
    def __init__(self, horizon: int = 30, target_type: str = 'return'):
        """
        Initialize the target creator.
        
        Args:
            horizon: Number of bars to look ahead (minimum 30 per spec)
            target_type: 'return' for percentage return, 'direction' for +1/-1
        """
        if horizon < 30:
            logger.warning(f"Horizon {horizon} is less than required minimum of 30")
        self.horizon = horizon
        self.target_type = target_type
        
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the prediction target.
        
        Args:
            df: DataFrame with P3 price column
            
        Returns:
            DataFrame with target column added
        """
        result = df.copy()
        
        if 'P3' not in result.columns:
            raise ValueError("P3 column required for target creation")
        
        # Forward return: P3[t+horizon] / P3[t] - 1
        # NOTE: This uses FUTURE data (shift(-horizon)) which is ONLY used for training
        result['forward_price'] = result['P3'].shift(-self.horizon)
        result['target'] = (result['forward_price'] / result['P3']) - 1
        
        if self.target_type == 'direction':
            # Convert to direction: +1 if positive, -1 if negative, 0 if flat
            result['target_direction'] = np.sign(result['target'])
            result['target'] = result['target_direction']
        
        # Mark rows where target is not available (last 'horizon' rows)
        result['has_target'] = ~result['target'].isna()
        
        # Log statistics
        valid_targets = result[result['has_target']]['target']
        logger.info(f"Created target with horizon {self.horizon}")
        logger.info(f"Target stats - mean: {valid_targets.mean():.6f}, "
                   f"std: {valid_targets.std():.6f}, "
                   f"positive ratio: {(valid_targets > 0).mean():.2%}")
        
        return result
    
    def get_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get rows where target is valid (for training).
        
        Args:
            df: DataFrame with target column
            
        Returns:
            Filtered DataFrame with valid targets only
        """
        if 'has_target' not in df.columns:
            raise ValueError("Run create_target first")
        
        return df[df['has_target']].copy()
    
    def split_features_target(self, df: pd.DataFrame, 
                               feature_cols: list) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            
        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        valid_df = self.get_valid_rows(df)
        X = valid_df[feature_cols].copy()
        y = valid_df['target'].copy()
        
        return X, y