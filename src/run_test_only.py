#!/usr/bin/env python3
"""
Test-Only Run — loads saved model + pipeline, runs testing on all test files
using the updated ExecutionEngine (loss-hold strategy), then regenerates
visualizations, report, and aggregate metrics.

Usage:
    cd src && python3 run_test_only.py
"""

import sys
import pickle
import logging
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from ensemble_model import EnsembleModel
from signal_generator import SignalGenerator
from run_full_evaluation import MultiFileEvaluator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = _PROJECT_ROOT / 'outputs'


def main():
    logger.info("=" * 60)
    logger.info("TEST-ONLY RUN (using saved model + new execution engine)")
    logger.info("=" * 60)

    # ── 1. Create evaluator (sets up data loader, file lists, etc.) ──
    evaluator = MultiFileEvaluator(
        n_train_files=90,
        n_test_files=21,
        n_optuna_trials=30,
    )

    # ── 2. Load saved ensemble model ─────────────────────────────────
    ens_path = OUTPUT_DIR / 'ensemble_model.pkl'
    if not ens_path.exists():
        logger.error(f"ensemble_model.pkl not found at {ens_path}")
        sys.exit(1)

    evaluator.ensemble_model = EnsembleModel()
    evaluator.ensemble_model.load(str(ens_path))
    logger.info(f"Loaded ensemble model: {len(evaluator.ensemble_model.feature_names_)} features, "
                f"CatBoost={evaluator.ensemble_model.include_catboost}")

    # ── 3. Load saved feature pipeline (sanitizer + selector) ────────
    pipeline_path = OUTPUT_DIR / 'feature_pipeline.pkl'
    if not pipeline_path.exists():
        logger.error(f"feature_pipeline.pkl not found at {pipeline_path}")
        logger.error("Run plot_predictions.py first (it auto-builds the pipeline) or run_full_evaluation.py")
        sys.exit(1)

    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)

    evaluator.sanitizer = pipeline['sanitizer']
    evaluator.selector = pipeline['selector']
    evaluator._raw_feature_cols = pipeline['raw_feature_cols']
    evaluator.selected_features = pipeline.get('selected_features',
                                                evaluator.selector.selected_features_)
    logger.info(f"Loaded pipeline: sanitizer (removed {len(evaluator.sanitizer.constant_features_)} const + "
                f"{len(evaluator.sanitizer.duplicate_features_)} dup), "
                f"selector ({len(evaluator.selector.selected_features_)} features)")

    # ── 4. Calibrate signal thresholds ───────────────────────────────
    #    Use same logic as training: 0.3×pred_std thresholds, min_hold=5
    #    We re-derive pred_std from the model on any training file
    import numpy as np
    train_files = evaluator.all_files[:evaluator.n_train_files]
    # Quick: use last training file to estimate pred_std
    logger.info("Calibrating signal thresholds from training predictions...")
    df_cal = evaluator.data_loader.load_file(str(train_files[-1]))
    X_cal, y_cal, _ = evaluator._prepare_test_features(df_cal)
    cal_preds = evaluator.ensemble_model.predict(X_cal)
    pred_std = np.std(cal_preds)

    evaluator.signal_generator = SignalGenerator(
        long_threshold=0.3 * pred_std,
        short_threshold=-0.3 * pred_std,
        dead_zone=0.1 * pred_std,
        min_holding_period=5
    )
    logger.info(f"Signal thresholds: long={0.3*pred_std:.8f}, "
                f"short={-0.3*pred_std:.8f}, dead_zone={0.1*pred_std:.8f}")

    # ── 5. Build dummy train_results + feature_importance ────────────
    evaluator.train_results = {
        'n_files': evaluator.n_train_files,
        'n_samples': 0,
        'n_features_original': len(evaluator._raw_feature_cols),
        'n_features_selected': len(evaluator.selected_features),
        'tuning_results': getattr(evaluator.ensemble_model, 'tuning_results_', {})
    }
    evaluator.tuning_results = evaluator.train_results['tuning_results']

    # Build feature importance df if model has the methods
    try:
        import pandas as pd
        xgb_imp = evaluator.ensemble_model.get_xgb_feature_importance()
        lgbm_imp = evaluator.ensemble_model.get_lgbm_feature_importance()
        evaluator.feature_importance_df = pd.merge(
            xgb_imp.rename(columns={'importance': 'xgb_importance'}),
            lgbm_imp.rename(columns={'importance': 'lgbm_importance'}),
            on='feature'
        )
        evaluator.feature_importance_df['avg_importance'] = (
            evaluator.feature_importance_df['xgb_importance'] +
            evaluator.feature_importance_df['lgbm_importance']
        ) / 2
        evaluator.feature_importance_df = evaluator.feature_importance_df.sort_values(
            'avg_importance', ascending=False
        )
    except Exception as e:
        logger.warning(f"Could not build feature importance: {e}")
        evaluator.feature_importance_df = None

    # ── 6. Run tests ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running tests with LOSS-HOLD execution engine (max_loss_hold=30)")
    logger.info("=" * 60)

    evaluator.test_on_multiple_files()

    # ── 7. Generate outputs ──────────────────────────────────────────
    evaluator.generate_visualizations()

    report = evaluator.generate_report()
    report_path = evaluator.output_dir / 'FINAL_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    import json
    agg_metrics = evaluator.calculate_aggregate_metrics()
    with open(evaluator.output_dir / 'aggregate_metrics.json', 'w') as f:
        json.dump(agg_metrics, f, indent=2)
    with open(evaluator.output_dir / 'test_results.json', 'w') as f:
        json.dump(evaluator.test_results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST-ONLY EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nTest Files: {len(evaluator.test_results)}")
    print(f"Execution Engine: Loss-Hold (max_loss_hold=30)")
    print()
    for mt in ['ensemble', 'xgboost', 'lightgbm']:
        m = agg_metrics.get(mt, {})
        print(f"  {mt.upper():12s}  PnL={m.get('total_pnl',0):.6f}  "
              f"Sharpe={m.get('avg_sharpe',0):.3f}  "
              f"WinRate={m.get('avg_win_rate',0):.2%}  "
              f"Profitable={m.get('profitable_files',0)}/{len(evaluator.test_results)}")
    if 'catboost' in agg_metrics:
        m = agg_metrics['catboost']
        print(f"  {'CATBOOST':12s}  PnL={m.get('total_pnl',0):.6f}  "
              f"Sharpe={m.get('avg_sharpe',0):.3f}  "
              f"WinRate={m.get('avg_win_rate',0):.2%}  "
              f"Profitable={m.get('profitable_files',0)}/{len(evaluator.test_results)}")
    print("=" * 60)


if __name__ == '__main__':
    main()