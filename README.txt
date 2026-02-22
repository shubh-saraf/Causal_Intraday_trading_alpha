================================================================================
ENSEMBLE TRADING SYSTEM
================================================================================

A causal, intraday ML trading system that predicts 30-bar forward returns of 
P3 using an XGBoost + LightGBM ensemble with Optuna hyperparameter tuning, 
then executes a fully iterative position strategy with realistic transaction 
costs (0.01%).

DATASET LINK: https://drive.google.com/drive/folders/1wABn0aZJNzdiKeHaBnTA1Xph4YVqb0Hw?usp=sharing

================================================================================
QUICK START
================================================================================


1. Install dependencies
   pip install -r requirements.txt

2. Run the strategy on a single trading day
   python src/strategy.py --input day.csv --output trades_day.csv

3. Or run the full multi-file train + test evaluation
   python src/run_full_evaluation.py

strategy.py handles everything end-to-end: if no trained model exists it 
trains one from the data directory, then produces a per-day trade log at the 
output path. The default model type is 'ensemble' (XGBoost + LightGBM).

NOTE: strategy.py defaults --model-dir to /home/ubuntu/trading_system/models
      Override with --model-dir ./models for local use.

================================================================================
DATASET FORMAT
================================================================================

Each CSV file (e.g., 1.csv, 2.csv, ...) represents ONE TRADING DAY of 
intraday observations (~20,000 rows).

REQUIRED COLUMNS:
-----------------
ts_ns      Nanosecond timestamp. Used for strict chronological ordering.
P3         Tradeable mid-price series. All execution and PnL is computed on P3.

OPTIONAL COLUMNS:
-----------------
P1, P2, P4           Alternative price proxies. Used for cross-price feature 
                     engineering only.
All other columns    Pre-engineered features provided with the dataset (see 
                     naming convention below).

FEATURE NAMING CONVENTION:
--------------------------
Feature names encode hierarchical information via underscore (_) delimiters:

    m0_BK_RC_5
    |  |  |  |
    |  |  |  +--- Lookback window or scale parameter
    |  |  +------ Sub-family (e.g., RC = rolling correlation)
    |  +--------- Group (e.g., BK = book-related)
    +------------ Family prefix (e.g., m0 = primary horizon)

KEY FAMILIES DISCOVERED VIA IMPORTANCE ANALYSIS:

Prefix          Likely Meaning                        Importance
------------------------------------------------------------------------
m0_BK_RC_*      Primary book rolling correlations     Dominant (12 of top-20)
m1_BK_RC_*      Secondary horizon book correlations   Strong
m0_P_S_*, m1_P_E  Price spread / execution features   Moderate
C_*             Cumulative counters (differenced)     Low-moderate

The system exploits this structure for group-level normalization and 
additionally engineers causal features (rolling statistics, momentum, RSI, 
MACD, cross-price correlations) from the price columns.

================================================================================
PROJECT STRUCTURE
================================================================================

trading_system_final/
├── README.md                    # Markdown version
├── README.txt                   # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── docs/
│   └── architecture.md          # Architecture documentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load and validate CSV data
│   ├── feature_engineer.py      # Causal feature engineering
│   ├── feature_sanitizer.py     # Remove constants, duplicates; normalize
│   ├── feature_selector.py      # Variance → correlation → importance
│   ├── target_creator.py        # 30-bar forward return target
│   ├── ensemble_model.py        # XGB + LGB + CatBoost ensemble
│   ├── signal_generator.py      # Predictions → {+1, -1, 0} signals
│   ├── execution_engine.py      # Trade execution with TC
│   ├── performance_analyzer.py  # Sharpe, PnL, drawdown, win rate
│   ├── strategy.py              # CLI entry point
│   ├── run_full_evaluation.py   # Multi-file train + test pipeline
│   ├── run_test_only.py         # Re-test with saved model
│   └── plot_predictions.py      # Interactive Plotly visualizations
├── examples/
│   └── example_usage.py         # Standalone usage examples
└── outputs/
    ├── aggregate_metrics.json   # Aggregate test results
    ├── test_results.json        # Per-file test results
    ├── ensemble_trades_*.csv    # Ensemble per-day trade logs
    ├── xgb_trades_*.csv         # XGBoost per-day trade logs
    ├── lgbm_trades_*.csv        # LightGBM per-day trade logs
    ├── trading_results/         # Additional trade logs and HTML charts
    └── prediction_plots/        # Interactive HTML plots (Plotly)

================================================================================
USAGE
================================================================================

SINGLE-DAY STRATEGY (PS DELIVERABLE B):
---------------------------------------
python src/strategy.py --input day.csv --output trades_day.csv

FLAGS:
  --input, -i          (required) Path to input CSV file
  --output, -o         (required) Path to save trade log CSV
  --model, -m          (default: ensemble) Model type: ensemble, xgboost, 
                       or lightgbm
  --n-trials           (default: 50) Optuna trials for hyperparameter tuning
  --model-dir          (default: /home/ubuntu/trading_system/models) 
                       Directory to save/load model files
  --verbose, -v        Enable debug logging

FULL MULTI-FILE EVALUATION:
---------------------------
python src/run_full_evaluation.py

Trains on 90 files, tests on 21 held-out files, generates trade logs for 
each model (ensemble, XGBoost, LightGBM), visualizations, and aggregate 
metrics. Saves trained models and the feature pipeline to outputs/.

RE-TEST WITH SAVED MODEL:
-------------------------
cd src && python run_test_only.py

Loads a previously trained model and feature pipeline from outputs/, re-runs 
testing on the 21 test files with the current ExecutionEngine settings. 
Useful for experimenting with execution parameters (e.g., loss-hold) without 
retraining.

GENERATE PREDICTION PLOTS:
--------------------------
cd src && python plot_predictions.py

Loads the saved model and pipeline, generates interactive Plotly HTML charts:
predicted P3 vs actual P3, delta-P3, return comparisons, scatter plots, 
direction accuracy, correlation heatmaps, and cumulative error.

PYTHON API:
-----------
from src import DataLoader, EnsembleModel, SignalGenerator, ExecutionEngine

# Load one day
loader = DataLoader()
df = loader.load_file('data/108.csv')

# Train ensemble (or load from pkl)
model = EnsembleModel(
    ensemble_strategy='weighted_average',
    n_trials=30,
    optimize_metric='direction_accuracy'
)
model.train(X_train, y_train, tune=True)

# Predict → Signal → Execute
predictions = model.predict(X_test)
signals = SignalGenerator(long_threshold=0.0001).generate_signals(predictions)
trade_log = ExecutionEngine(transaction_cost_bps=1.0).execute_series(
    timestamps, signals, prices
)

================================================================================
SYSTEM ARCHITECTURE
================================================================================

Data Input (day.csv)
        |
        v
+---------------+      +------------------+      +-------------------+
|  DataLoader   |----->| FeatureEngineer  |----->| FeatureSanitizer  |
| (load, sort)  |      | (causal only)    |      | (clean, normalize)|
+---------------+      +------------------+      +-------------------+
                                                           |
                                                           v
+---------------+      +------------------+      +-------------------+
|  Execution    |<-----| SignalGenerator  |<-----| FeatureSelector   |
|  Engine       |      | (+1 / -1 / 0)    |      | (top-50 features) |
| (TC + PnL)    |      +------------------+      +-------------------+
+---------------+                                          |
        |                                                  v
        v                                        +-------------------+
  Trade Log CSV                                  | EnsembleModel     |
                                                 | XGB + LGB (+Cat)  |
                                                 | (custom dir loss) |
                                                 +-------------------+

CAUSALITY GUARANTEES:
---------------------
Every stage enforces strict causality:

- Features: All rolling windows look backward only. No .shift(-n) in 
  feature code.
- Target: Forward return uses .shift(-30) — used only as a training label, 
  never as a feature.
- Normalization: Expanding-window robust normalization (at bar t, uses 
  only bars 0..t).
- Sanitizer/Selector: Fit on training data only; transform-only on test data.
- Signals: Based solely on current prediction, subject to holding period 
  constraints.
- Execution: Processes one bar at a time; position changes depend only on 
  current state.

================================================================================
CONFIGURATION REFERENCE
================================================================================

ENSEMBLEMODEL:
--------------
ensemble_strategy    (str, default: 'weighted_average') 
                     'average' or 'weighted_average'
n_trials             (int, default: 50) Optuna trials per model
optimize_metric      (str, default: 'direction_accuracy') 
                     'sharpe', 'win_rate', or 'direction_accuracy'
use_custom_loss      (bool, default: True) 
                     10x penalty for wrong-direction predictions (LGB)
include_catboost     (bool, default: True) 
                     Include CatBoost in ensemble (requires catboost package)

SIGNALGENERATOR:
----------------
long_threshold       (float, default: 0.0001) 
                     Minimum prediction for long signal
short_threshold      (float, default: -0.0001) 
                     Maximum prediction for short signal
dead_zone            (float, default: 0.00005) No-trade zone around zero
min_confidence       (float, default: 0.0) 
                     Minimum prediction magnitude for any signal
min_holding_period   (int, default: 30) 
                     Minimum bars before position change allowed

EXECUTIONENGINE:
----------------
transaction_cost_bps (float, default: 1.0) 
                     Transaction cost in basis points (1 bps = 0.01%)
max_loss_hold        (int, default: 30) 
                     Max bars to hold a losing position before force-close. 
                     Set to 0 to disable.

================================================================================
TRADE LOG OUTPUT FORMAT
================================================================================

The output CSV (from strategy.py or the evaluation scripts) contains one 
row per bar:

COLUMN              DESCRIPTION
------------------------------------------------------------------------
timestamp           Bar index
signal              Raw model signal (+1, -1, 0)
price               P3 price at this bar
position            Actual position after execution logic
entry_price         Price at which current position was entered
pnl                 Realized PnL (non-zero only on position close)
mtm_pnl             Mark-to-market (unrealized) PnL
transaction_cost    Cost incurred at this bar
cumulative_pnl      Running total PnL (net of costs)
held_loss           True if a signal change was suppressed by loss-hold

================================================================================
FEATURE SANITATION
================================================================================

The FeatureSanitizer automatically detects and handles:

ISSUE                                     COUNT FOUND    ACTION
------------------------------------------------------------------------
Constant features (near-zero variance)    67             Removed
Duplicate features (corr > 0.999)         Variable       Removed
Extreme autocorrelation (lag-1 > 0.99)    147            Flagged
Scale differences (13 orders of magnitude) All           Robust normalization
                                                         (median/IQR, expanding)
Cumulative features (monotonic)           27             Differenced

================================================================================
PERFORMANCE SUMMARY (21 OUT-OF-SAMPLE TEST DAYS)
================================================================================

METRIC                  ENSEMBLE     XGBOOST      LIGHTGBM
------------------------------------------------------------------------
Total PnL               +0.726       +1.542       +1.540
Avg Sharpe              0.002        0.162        0.042
Avg Win Rate            54.1%        57.8%        54.0%
Direction Accuracy      49.7%        50.1%        49.8%
Profitable Days         9/21         12/21        10/21
Avg Trades/Day          320          458          258

================================================================================
END OF README
================================================================================
