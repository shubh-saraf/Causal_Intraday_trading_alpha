"""
SignalGenerator Module - Converts model predictions to trading signals.

Signals:
- +1: Long position
- -1: Short position
-  0: Flat (no position)

Signal generation can use:
- Simple thresholds
- Confidence-based sizing
- Dead zone filtering
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Converts model predictions to discrete trading signals.
    """
    
    def __init__(self,
                 long_threshold: float = 0.0001,
                 short_threshold: float = -0.0001,
                 dead_zone: float = 0.00005,
                 min_confidence: float = 0.0,
                 min_holding_period: int = 30):
        """
        Initialize the signal generator.
        
        Args:
            long_threshold: Prediction threshold for long signal
            short_threshold: Prediction threshold for short signal
            dead_zone: Zone around zero where signal is flat
            min_confidence: Minimum prediction magnitude for signal
            min_holding_period: Minimum bars to hold position before allowing change
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.dead_zone = dead_zone
        self.min_confidence = min_confidence
        self.min_holding_period = min_holding_period
        
    def generate_signals(self, predictions: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from predictions with minimum holding period.
        
        Args:
            predictions: Array of model predictions (expected returns)
            
        Returns:
            Array of signals (+1, -1, 0)
        """
        # First pass: generate raw signals based on thresholds
        raw_signals = np.zeros(len(predictions))
        raw_signals[predictions > self.long_threshold] = 1
        raw_signals[predictions < self.short_threshold] = -1
        
        # Apply dead zone
        dead_zone_mask = np.abs(predictions) < self.dead_zone
        raw_signals[dead_zone_mask] = 0
        
        # Apply minimum confidence filter
        if self.min_confidence > 0:
            low_confidence_mask = np.abs(predictions) < self.min_confidence
            raw_signals[low_confidence_mask] = 0
        
        # Second pass: apply minimum holding period constraint
        if self.min_holding_period > 1:
            signals = self._apply_holding_period(raw_signals)
        else:
            signals = raw_signals
        
        logger.info(f"Generated signals: {np.sum(signals == 1)} long, "
                   f"{np.sum(signals == -1)} short, {np.sum(signals == 0)} flat")
        
        return signals.astype(int)
    
    def _apply_holding_period(self, raw_signals: np.ndarray) -> np.ndarray:
        """
        Apply minimum holding period constraint.
        
        Once a position is entered, it must be held for min_holding_period bars
        before any position change is allowed.
        """
        signals = raw_signals.copy()
        current_position = 0
        bars_in_position = 0
        
        for i in range(len(signals)):
            if current_position == 0:
                # Not in position, can enter if signal exists
                if raw_signals[i] != 0:
                    current_position = raw_signals[i]
                    bars_in_position = 1
                signals[i] = current_position
            else:
                # In position
                bars_in_position += 1
                
                if bars_in_position < self.min_holding_period:
                    # Must hold - keep current position
                    signals[i] = current_position
                else:
                    # Can change position
                    if raw_signals[i] != current_position and raw_signals[i] != 0:
                        # Signal wants to change to opposite
                        current_position = raw_signals[i]
                        bars_in_position = 1
                        signals[i] = current_position
                    elif raw_signals[i] == 0:
                        # Signal wants to go flat
                        current_position = 0
                        bars_in_position = 0
                        signals[i] = 0
                    else:
                        # Stay in position
                        signals[i] = current_position
        
        return signals
    
    def generate_signals_adaptive(self, predictions: np.ndarray,
                                   rolling_window: int = 100) -> np.ndarray:
        """
        Generate signals with adaptive thresholds based on recent predictions.
        
        Args:
            predictions: Array of predictions
            rolling_window: Window for calculating adaptive threshold
            
        Returns:
            Array of signals
        """
        signals = np.zeros(len(predictions))
        
        pred_series = pd.Series(predictions)
        rolling_std = pred_series.rolling(rolling_window, min_periods=10).std()
        
        for i in range(len(predictions)):
            if i < 10:
                # Not enough history, use simple threshold
                if predictions[i] > self.long_threshold:
                    signals[i] = 1
                elif predictions[i] < self.short_threshold:
                    signals[i] = -1
            else:
                # Adaptive threshold based on recent volatility
                std = rolling_std.iloc[i]
                if np.isnan(std) or std == 0:
                    std = 0.001
                
                adaptive_long = 0.5 * std
                adaptive_short = -0.5 * std
                
                if predictions[i] > adaptive_long:
                    signals[i] = 1
                elif predictions[i] < adaptive_short:
                    signals[i] = -1
        
        return signals.astype(int)