"""
CausalFeatureEngineer Module - Creates causal features from raw data.

All features are computed using ONLY past data (up to and including time t).
No lookahead bias is allowed.

Features created:
- Rolling statistics (mean, std, min, max, skew, kurtosis)
- Momentum indicators
- Technical indicators (RSI, volatility)
- Cross-feature interactions
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CausalFeatureEngineer:
    """
    Engineers causal features from raw data.
    
    CAUSALITY: All features at time t use only data from times <= t.
    """
    
    def __init__(self, 
                 windows: List[int] = [5, 10, 20, 50],
                 price_cols: List[str] = ['P1', 'P2', 'P3', 'P4']):
        """
        Initialize the feature engineer.
        
        Args:
            windows: List of rolling window sizes
            price_cols: Price columns to use for feature engineering
        """
        self.windows = windows
        self.price_cols = price_cols
        self.engineered_cols_: List[str] = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all causal features.
        
        Args:
            df: Input DataFrame with price data
            
        Returns:
            DataFrame with additional engineered features
        """
        result = df.copy()
        self.engineered_cols_ = []
        
        # Price-based features
        result = self._add_price_features(result)
        
        # Rolling features on P3 (main tradeable)
        result = self._add_rolling_features(result, 'P3')
        
        # Momentum features
        result = self._add_momentum_features(result)
        
        # Technical indicators
        result = self._add_technical_indicators(result)
        
        # Cross-price features
        result = self._add_cross_price_features(result)
        
        logger.info(f"Engineered {len(self.engineered_cols_)} new features")
        return result
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price-derived features.
        
        CAUSAL: All use current or past prices only.
        """
        result = df.copy()
        
        for col in self.price_cols:
            if col not in result.columns:
                continue
            
            # Price returns (CAUSAL: uses t and t-1)
            result[f'{col}_return'] = result[col].pct_change()
            self.engineered_cols_.append(f'{col}_return')
            
            # Log returns
            result[f'{col}_log_return'] = np.log(result[col] / result[col].shift(1))
            self.engineered_cols_.append(f'{col}_log_return')
        
        # Price spreads between P3 and others
        if 'P3' in result.columns:
            for col in ['P1', 'P2', 'P4']:
                if col in result.columns:
                    result[f'{col}_P3_spread'] = result[col] - result['P3']
                    self.engineered_cols_.append(f'{col}_P3_spread')
        
        return result
    
    def _add_rolling_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Add rolling window features.
        
        CAUSAL: Rolling windows look backward only.
        """
        result = df.copy()
        
        if col not in result.columns:
            return result
        
        for window in self.windows:
            # Rolling mean
            result[f'{col}_roll_mean_{window}'] = result[col].rolling(
                window=window, min_periods=1).mean()
            self.engineered_cols_.append(f'{col}_roll_mean_{window}')
            
            # Rolling std (volatility)
            result[f'{col}_roll_std_{window}'] = result[col].rolling(
                window=window, min_periods=2).std()
            self.engineered_cols_.append(f'{col}_roll_std_{window}')
            
            # Rolling min/max
            result[f'{col}_roll_min_{window}'] = result[col].rolling(
                window=window, min_periods=1).min()
            result[f'{col}_roll_max_{window}'] = result[col].rolling(
                window=window, min_periods=1).max()
            self.engineered_cols_.extend([f'{col}_roll_min_{window}', f'{col}_roll_max_{window}'])
            
            # Price relative to rolling range
            roll_range = result[f'{col}_roll_max_{window}'] - result[f'{col}_roll_min_{window}']
            result[f'{col}_roll_position_{window}'] = (
                (result[col] - result[f'{col}_roll_min_{window}']) / 
                roll_range.replace(0, 1e-10)
            )
            self.engineered_cols_.append(f'{col}_roll_position_{window}')
            
            # Rolling skew
            if window >= 10:
                result[f'{col}_roll_skew_{window}'] = result[col].rolling(
                    window=window, min_periods=3).skew()
                self.engineered_cols_.append(f'{col}_roll_skew_{window}')
        
        return result
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.
        
        CAUSAL: All based on past price changes.
        """
        result = df.copy()
        
        if 'P3' not in result.columns:
            return result
        
        # Price momentum (rate of change)
        for lag in [5, 10, 20, 30]:
            result[f'P3_momentum_{lag}'] = (
                result['P3'] / result['P3'].shift(lag) - 1
            )
            self.engineered_cols_.append(f'P3_momentum_{lag}')
        
        # Acceleration (change in momentum)
        result['P3_return'] = result['P3'].pct_change()
        result['P3_acceleration'] = result['P3_return'].diff()
        self.engineered_cols_.append('P3_acceleration')
        
        # Consecutive up/down moves
        returns = result['P3_return'].fillna(0)
        result['P3_consec_up'] = (returns > 0).rolling(10).sum()
        result['P3_consec_down'] = (returns < 0).rolling(10).sum()
        self.engineered_cols_.extend(['P3_consec_up', 'P3_consec_down'])
        
        return result
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis indicators.
        
        CAUSAL: All indicators use only past data.
        """
        result = df.copy()
        
        if 'P3' not in result.columns:
            return result
        
        # RSI (Relative Strength Index)
        for period in [10, 20]:
            delta = result['P3'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 1e-10)
            result[f'P3_rsi_{period}'] = 100 - (100 / (1 + rs))
            self.engineered_cols_.append(f'P3_rsi_{period}')
        
        # Bollinger Bands position
        for window in [20, 50]:
            mean = result['P3'].rolling(window).mean()
            std = result['P3'].rolling(window).std()
            result[f'P3_bb_position_{window}'] = (
                (result['P3'] - mean) / (2 * std.replace(0, 1e-10))
            )
            self.engineered_cols_.append(f'P3_bb_position_{window}')
        
        # MACD-like feature
        ema_fast = result['P3'].ewm(span=12, adjust=False).mean()
        ema_slow = result['P3'].ewm(span=26, adjust=False).mean()
        result['P3_macd'] = ema_fast - ema_slow
        result['P3_macd_signal'] = result['P3_macd'].ewm(span=9, adjust=False).mean()
        result['P3_macd_hist'] = result['P3_macd'] - result['P3_macd_signal']
        self.engineered_cols_.extend(['P3_macd', 'P3_macd_signal', 'P3_macd_hist'])
        
        # Volatility ratio
        short_vol = result['P3'].rolling(5).std()
        long_vol = result['P3'].rolling(20).std()
        result['P3_vol_ratio'] = short_vol / long_vol.replace(0, 1e-10)
        self.engineered_cols_.append('P3_vol_ratio')
        
        return result
    
    def _add_cross_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on relationships between price series.
        
        CAUSAL: Uses current and past values only.
        """
        result = df.copy()
        
        if 'P3' not in result.columns:
            return result
        
        # Rolling correlation with other prices
        for col in ['P1', 'P2', 'P4']:
            if col in result.columns:
                result[f'{col}_P3_corr_20'] = result[col].rolling(20).corr(result['P3'])
                self.engineered_cols_.append(f'{col}_P3_corr_20')
        
        # Price convergence/divergence
        for col in ['P1', 'P2', 'P4']:
            if col in result.columns:
                spread = result[col] - result['P3']
                result[f'{col}_P3_spread_zscore'] = (
                    (spread - spread.rolling(20).mean()) / 
                    spread.rolling(20).std().replace(0, 1e-10)
                )
                self.engineered_cols_.append(f'{col}_P3_spread_zscore')
        
        return result
    
    def get_engineered_columns(self) -> List[str]:
        """Return list of engineered column names."""
        return self.engineered_cols_.copy()