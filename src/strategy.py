#!/usr/bin/env python
"""
Causal Intraday Trading Strategy

Usage:
    python strategy.py --input day.csv --output trades_day.csv

This script implements a causal prediction system that:
1. Loads intraday data
2. Sanitizes features (removes constant, normalizes, etc.)
3. Engineers causal features (rolling windows, momentum, etc.)
4. Generates predictions using pre-trained LightGBM model
5. Converts predictions to trading signals (+1, -1, 0)
6. Executes trades with 0.01% transaction cost
7. Outputs a trade log with PnL tracking

CAUSALITY:
All operations at time t use only data from times <= t.
No lookahead bias is allowed at any stage.

Author: Trading System
Date: February 2026
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from feature_sanitizer import FeatureSanitizer
from feature_engineer import CausalFeatureEngineer
from target_creator import TargetCreator
from feature_selector import FeatureSelector
from trading_model import TradingModel
from ensemble_model import EnsembleModel
from signal_generator import SignalGenerator
from execution_engine import ExecutionEngine
from performance_analyzer import PerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model_if_needed(model_path: str, train_dir: str = "/home/ubuntu/dataset/train/", 
                         model_type: str = 'lightgbm', n_trials: int = 50):
    """
    Train the model if not already trained.
    
    Supports: 'lightgbm', 'xgboost', 'ensemble'
    
    Args:
        model_path: Path to save the model
        train_dir: Directory with training files
        model_type: Type of model ('lightgbm', 'xgboost', 'ensemble')
        n_trials: Number of Optuna trials for hyperparameter tuning
    """
    import gc
    model_path = Path(model_path)
    
    if model_path.exists():
        logger.info(f"Model found at {model_path}")
        return True
    
    logger.info(f"Training new {model_type.upper()} model...")
    
    # Get available days - use 20 files for training
    all_files = sorted(Path(train_dir).glob("*.csv"), key=lambda x: int(x.stem))
    train_files = all_files[:20]  # Use first 20 files
    
    if len(train_files) < 10:
        logger.error("Not enough training files")
        return False
    
    logger.info(f"Using {len(train_files)} files for training (out of {len(all_files)} available)")
    
    # Load and combine training data with 50% sampling for memory efficiency
    loader = DataLoader(train_dir)
    all_data = []
    
    for f in train_files:
        df = loader.load_file(str(f))
        df = df.sample(frac=0.5, random_state=42).sort_values('bar_index').reset_index(drop=True)
        all_data.append(df)
        del df
        gc.collect()
    
    combined = pd.concat(all_data, ignore_index=True)
    del all_data
    gc.collect()
    
    logger.info(f"Training data: {len(combined)} rows")
    
    # Get feature columns
    raw_feature_cols = loader.get_feature_columns(combined)
    
    # Sanitize
    sanitizer = FeatureSanitizer()
    combined, sanitized_cols = sanitizer.fit_transform(combined, raw_feature_cols, causal=True)
    gc.collect()
    
    # Engineer features
    engineer = CausalFeatureEngineer()
    combined = engineer.engineer_features(combined)
    engineered_cols = engineer.get_engineered_columns()
    gc.collect()
    
    # Combine features
    all_features = sanitized_cols + engineered_cols
    all_features = [c for c in all_features if c in combined.columns]
    
    # Create target
    target_creator = TargetCreator(horizon=30)
    combined = target_creator.create_target(combined)
    
    # Get valid data
    X, y = target_creator.split_features_target(combined, all_features)
    del combined
    gc.collect()
    
    # Select features - use 75 features (optimized)
    selector = FeatureSelector(n_features=75)
    X_selected = selector.fit_transform(X, y)
    del X
    gc.collect()
    
    # Save selected features
    selected_features = selector.selected_features_
    feature_path = model_path.parent / "selected_features.txt"
    with open(feature_path, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    # Save sanitizer params
    import pickle
    sanitizer_path = model_path.parent / "sanitizer.pkl"
    with open(sanitizer_path, 'wb') as f:
        pickle.dump(sanitizer, f)
    
    # Save optimized config (threshold multipliers)
    config_path = model_path.parent / "best_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump({
            'threshold_mult': 5.0,
            'dead_zone_mult': 0.3
        }, f)
    
    # Train model based on type
    if model_type == 'ensemble':
        model = EnsembleModel(ensemble_strategy='weighted_average', n_trials=n_trials)
        model.train(X_selected, y, tune=True)
        model.save(str(model_path))
        
        # Save tuning results
        outputs_dir = Path("/home/ubuntu/trading_system/outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        with open(outputs_dir / "hyperparameter_tuning_results.txt", 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of trials per model: {n_trials}\n\n")
            f.write("LightGBM Best Parameters:\n")
            for k, v in model.lgbm_params_.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nLightGBM Best Score: {model.tuning_results_.get('lgbm_best_score', 'N/A')}\n\n")
            f.write("XGBoost Best Parameters:\n")
            for k, v in model.xgb_params_.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nXGBoost Best Score: {model.tuning_results_.get('xgb_best_score', 'N/A')}\n\n")
            f.write(f"Ensemble Weights: XGBoost={model.ensemble_weights_[0]:.2f}, LightGBM={model.ensemble_weights_[1]:.2f}\n")
        
        # Save feature importance
        xgb_importance = model.get_xgb_feature_importance()
        lgbm_importance = model.get_lgbm_feature_importance()
        xgb_importance.to_csv(outputs_dir / "xgboost_feature_importance.csv", index=False)
        lgbm_importance.to_csv(outputs_dir / "lightgbm_feature_importance.csv", index=False)
        
        logger.info(f"Feature importance saved to outputs/")
        
    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, verbosity=0)
        model.fit(X_selected, y)
        with open(str(model_path), 'wb') as f:
            pickle.dump({'model': model, 'feature_names': list(X_selected.columns)}, f)
    else:  # lightgbm
        model = TradingModel(n_estimators=300, max_depth=7, learning_rate=0.03, 
                            min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8)
        model.train(X_selected, y, early_stopping_rounds=None)
        model.save(str(model_path))
    
    logger.info(f"Model trained and saved to {model_path}")
    return True


def run_strategy(input_path: str, output_path: str, model_dir: str = "/home/ubuntu/trading_system/models",
                 model_type: str = 'ensemble', n_trials: int = 50):
    """
    Run the trading strategy on a single day.
    
    CAUSAL PIPELINE:
    1. Load data for the day
    2. Sanitize features (causal normalization)
    3. Engineer features (rolling windows use past data only)
    4. Generate predictions using trained model
    5. Convert to signals
    6. Execute trades with transaction costs
    7. Save trade log
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save trade log
        model_dir: Directory containing model files
        model_type: Type of model ('lightgbm', 'xgboost', 'ensemble')
        n_trials: Number of Optuna trials for hyperparameter tuning
    """
    import pickle
    
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    # Determine model path based on type
    model_filename = f"{model_type}_model.pkl"
    model_path = model_dir / model_filename
    sanitizer_path = model_dir / "sanitizer.pkl"
    features_path = model_dir / "selected_features.txt"
    
    # Train model if needed
    if not model_path.exists():
        train_model_if_needed(str(model_path), model_type=model_type, n_trials=n_trials)
    
    # Load model based on type
    if model_type == 'ensemble':
        model = EnsembleModel()
        model.load(str(model_path))
    elif model_type == 'xgboost':
        with open(str(model_path), 'rb') as f:
            data = pickle.load(f)
        xgb_model = data['model']
        xgb_feature_names = data['feature_names']
    else:  # lightgbm
        model = TradingModel()
        model.load(str(model_path))
    
    # Load sanitizer
    if sanitizer_path.exists():
        with open(sanitizer_path, 'rb') as f:
            sanitizer = pickle.load(f)
    else:
        sanitizer = FeatureSanitizer()
        sanitizer.fitted_ = True  # Skip fitting
    
    # Get feature names based on model type
    if model_type == 'xgboost':
        model_feature_names = xgb_feature_names
    else:
        model_feature_names = model.feature_names_
    
    # Load selected features
    if features_path.exists():
        with open(features_path, 'r') as f:
            selected_features = [line.strip() for line in f if line.strip()]
    else:
        selected_features = model_feature_names
    
    logger.info(f"Processing: {input_path} (model: {model_type})")
    logger.info(f"Output: {output_path}")
    
    # === STEP 1: Load Data ===
    loader = DataLoader()
    df = loader.load_file(input_path)
    raw_feature_cols = loader.get_feature_columns(df)
    
    logger.info(f"Loaded {len(df)} bars, {len(raw_feature_cols)} raw features")
    
    # === STEP 2: Sanitize Features (CAUSAL) ===
    # Use expanding window normalization - only uses past data
    df, sanitized_cols = sanitizer.transform(df, raw_feature_cols, causal=True)
    
    # === STEP 3: Engineer Features (CAUSAL) ===
    engineer = CausalFeatureEngineer()
    df = engineer.engineer_features(df)
    
    # === STEP 4: Prepare Features for Prediction ===
    available_features = [f for f in selected_features if f in df.columns]
    # Add engineered features that model knows about
    for f in model_feature_names:
        if f in df.columns and f not in available_features:
            available_features.append(f)
    
    X = df[available_features].copy() if available_features else df[sanitized_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info(f"Using {len(X.columns)} features for prediction")
    
    # === STEP 5: Generate Predictions (CAUSAL) ===
    # Each prediction at time t uses only features available at t
    if model_type == 'xgboost':
        X_aligned = X.reindex(columns=xgb_feature_names, fill_value=0)
        predictions = xgb_model.predict(X_aligned)
    else:
        predictions = model.predict(X)
    df['prediction'] = predictions
    
    # === STEP 6: Generate Trading Signals ===
    # Log prediction statistics for debugging
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    pred_min = np.min(predictions)
    pred_max = np.max(predictions)
    pred_positive = np.sum(predictions > 0)
    pred_negative = np.sum(predictions < 0)
    
    logger.info(f"Prediction Stats: mean={pred_mean:.8f}, std={pred_std:.8f}")
    logger.info(f"Prediction Range: min={pred_min:.8f}, max={pred_max:.8f}")
    logger.info(f"Prediction Distribution: {pred_positive} positive, {pred_negative} negative")
    
    # Load optimized threshold config if available
    config_path = model_dir / "best_config.pkl"
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        threshold_mult = config.get('threshold_mult', 5.0)
        dead_zone_mult = config.get('dead_zone_mult', 0.3)
    else:
        # Use optimized defaults: higher thresholds = fewer trades = less transaction costs
        threshold_mult = 5.0
        dead_zone_mult = 0.3
    
    # Adaptive thresholds based on prediction distribution with OPTIMIZED multipliers
    # CRITICAL: Use minimum holding period of 30 bars to match prediction horizon
    signal_gen = SignalGenerator(
        long_threshold=threshold_mult * pred_std,   # Higher threshold = stronger conviction needed
        short_threshold=-threshold_mult * pred_std,
        dead_zone=dead_zone_mult * pred_std,
        min_holding_period=30  # Hold for at least 30 bars (matching prediction horizon)
    )
    logger.info(f"Signal thresholds: long={threshold_mult*pred_std:.8f}, short={-threshold_mult*pred_std:.8f}, dead_zone={dead_zone_mult*pred_std:.8f}")
    logger.info(f"Minimum holding period: 30 bars")
    signals = signal_gen.generate_signals(predictions)
    df['signal'] = signals
    
    # Log signal distribution
    n_long = np.sum(signals == 1)
    n_short = np.sum(signals == -1)
    n_flat = np.sum(signals == 0)
    logger.info(f"Signal Distribution: {n_long} long (+1), {n_short} short (-1), {n_flat} flat (0)")
    
    # === STEP 7: Execute Trades with Transaction Costs ===
    # Transaction cost: 0.01% (1 basis point)
    execution = ExecutionEngine(transaction_cost_bps=1.0)
    
    trade_log = execution.execute_series(
        df['bar_index'].values,
        df['signal'].values,
        df['P3'].values
    )
    
    # Close position at end of day
    execution.close_all_positions(
        int(df['bar_index'].iloc[-1]),
        float(df['P3'].iloc[-1])
    )
    trade_log = execution.get_trade_log()
    
    # === STEP 8: Save Trade Log ===
    trade_log.to_csv(output_path, index=False)
    logger.info(f"Trade log saved: {output_path}")
    
    # === STEP 9: Analyze and Report Performance ===
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(trade_log)
    report = analyzer.generate_report(trade_log)
    
    print(report)
    
    # Save detailed report
    report_path = Path(output_path).with_suffix('.report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    return trade_log, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Causal Intraday Trading Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy.py --input /home/ubuntu/dataset/train/1.csv --output trades_1.csv
  python strategy.py --input day.csv --output trades_day.csv --model ensemble
  python strategy.py --input day.csv --output trades_day.csv --model lightgbm
  
The output trade log contains:
  - timestamp: Bar index
  - signal: Trading signal (+1 long, -1 short, 0 flat)
  - price: P3 price at execution
  - position: Current position
  - pnl: Realized PnL
  - cumulative_pnl: Running total PnL
  - transaction_cost: Cost for this trade
"""
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path (day data)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output trade log CSV path'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='ensemble',
        choices=['lightgbm', 'xgboost', 'ensemble'],
        help='Model type to use (default: ensemble)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials for hyperparameter tuning (default: 50)'
    )
    
    parser.add_argument(
        '--model-dir',
        default='/home/ubuntu/trading_system/models',
        help='Directory for model files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run strategy
    trade_log, metrics = run_strategy(
        args.input,
        args.output,
        args.model_dir,
        args.model,
        args.n_trials
    )
    
    # Print summary
    print(f"\n=== Strategy Complete ===")
    print(f"Total PnL: {metrics.get('total_pnl', 0):.6f}")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.4f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Transaction Costs: {metrics.get('total_transaction_costs', 0):.6f}")


if __name__ == '__main__':
    main()