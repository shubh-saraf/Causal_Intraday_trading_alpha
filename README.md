# Ensemble Trading System

A causal, intraday ML trading system that predicts 30-bar forward returns of P3 using an XGBoost + LightGBM ensemble with Optuna hyperparameter tuning.

## 📊 Dataset Download

The full 10GB dataset (intraday CSV files) is hosted on Google Drive due to its size.

👉 [**Download Dataset from Google Drive**](https://drive.google.com/drive/folders/1wABn0aZJNzdiKeHaBnTA1Xph4YVqb0Hw?usp=sharing)

Once downloaded, place the CSV files in the `data/` directory.

## 🚀 Quick Start

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the strategy on a single trading day**:
    ```bash
    python src/strategy.py --input data/108.csv --output outputs/trades_108.csv
    ```

3.  **Run the full multi-file evaluation**:
    ```bash
    python src/run_full_evaluation.py
    ```

## 📂 Project Structure

- `src/`: Core logic (DataLoader, FeatureEngineer, EnsembleModel, etc.)
- `data/`: Placeholder for your downloaded CSV files (ignored by Git)
- `outputs/`: Trained models (`.pkl`) and evaluation results
- `setup.py`: Package installation
- `README.txt`: Legacy documentation

## 🛠 Features
- **Causal Engineering**: Strict look-back windows to prevent data leakage.
- **Ensemble Learning**: Combined power of XGBoost and LightGBM.
- **Realistic Execution**: Iterative position management with transaction costs (0.01%).
- **Interactive Plots**: Visualize predictions and returns with Plotly.
