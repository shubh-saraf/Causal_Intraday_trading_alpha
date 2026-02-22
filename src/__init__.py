__init__.py
"""
Ensemble Trading System

A comprehensive ML-based trading system combining XGBoost and LightGBM
with hyperparameter tuning for intraday trading predictions.

Modules:
- data_loader: Load and validate trading data
- feature_engineer: Causal feature engineering
- feature_sanitizer: Feature cleaning and normalization
- feature_selector: Feature selection
- target_creator: Forward-looking target creation
- ensemble_model: XGBoost + LightGBM ensemble
- signal_generator: Trading signal generation
- execution_engine: Trade execution simulation
- performance_analyzer: Performance metrics calculation
"""

__version__ = '1.0.0'

from .data_loader import DataLoader
from .feature_engineer import CausalFeatureEngineer
from .feature_sanitizer import FeatureSanitizer
from .feature_selector import FeatureSelector
from .target_creator import TargetCreator
from .ensemble_model import EnsembleModel
from .signal_generator import SignalGenerator
from .execution_engine import ExecutionEngine
from .performance_analyzer import PerformanceAnalyzer