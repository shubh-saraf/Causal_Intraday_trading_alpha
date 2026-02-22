#!/usr/bin/env python3
"""
Plot Predicted P3 vs Actual P3 using saved pkl model files.

This script properly:
  1. Loads the saved FeatureSanitizer + FeatureSelector from feature_pipeline.pkl
     (fitted during training in run_full_evaluation.py)
  2. Runs the same pipeline on each test file:
     engineer → target → sanitize(transform, causal) → select(transform)
  3. Reconstructs predicted P3 prices from predicted returns
  4. Plots at DECISION TIME t (not t+30) to avoid visual shift

Reconstruction:
  target[t]  = (P3[t+30] - P3[t]) / P3[t]
  predicted_P3_at_t+30 = P3[t] * (1 + model_pred[t])
  actual_P3_at_t+30    = P3[t+30]
  Both plotted at x = t (decision bar) to avoid 30-bar shift.

Usage:
    cd src && python3 plot_predictions.py
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project paths ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

from data_loader import DataLoader
from feature_engineer import CausalFeatureEngineer
from feature_sanitizer import FeatureSanitizer
from feature_selector import FeatureSelector
from target_creator import TargetCreator
from ensemble_model import EnsembleModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

DATA_DIR      = _PROJECT_ROOT / 'data'
OUTPUT_DIR    = _PROJECT_ROOT / 'outputs'
PLOTS_DIR     = OUTPUT_DIR / 'prediction_plots'
HORIZON       = 30
N_TRAIN_FILES = 90
N_TEST_FILES  = 21


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def get_sorted_files(data_dir: Path) -> List[Path]:
    return sorted(
        [f for f in data_dir.glob('*.csv')],
        key=lambda x: int(x.stem) if x.stem.isdigit() else 0
    )


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns, excluding non-feature columns."""
    exclude = {
        'ts_ns', 'timestamp', 'target', 'P1', 'P2', 'P3', 'P4',
        'bar_index', 'day', 'forward_price', 'has_target', 'target_direction'
    }
    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
    ]


def rebuild_feature_pipeline(data_dir: Path, output_dir: Path,
                             n_train_files: int, horizon: int) -> dict:
    """
    Rebuild the sanitizer + selector from training data (no model training).
    Saves to feature_pipeline.pkl and returns the pipeline dict.
    """
    logger.info("=" * 50)
    logger.info("REBUILDING feature pipeline from training data...")
    logger.info("(This runs sanitizer.fit + selector.fit; no model training)")
    logger.info("=" * 50)

    loader = DataLoader(data_dir=str(data_dir))
    eng    = CausalFeatureEngineer()
    tgt    = TargetCreator(horizon=horizon)
    sanitizer = FeatureSanitizer()
    selector  = FeatureSelector(n_features=50)

    all_files   = get_sorted_files(data_dir)
    n_train     = min(n_train_files, int(len(all_files) * 0.8))
    train_files = all_files[:n_train]
    logger.info(f"  Training on {len(train_files)} files")

    # Pass 1: engineer + target
    raw_dfs = []
    for i, fp in enumerate(train_files):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  [Pass 1] {i+1}/{len(train_files)}")
        try:
            df = loader.load_file(str(fp))
            df = eng.engineer_features(df)
            df = tgt.create_target(df)
            df = df[df['has_target']].copy()
            raw_dfs.append(df)
        except Exception as e:
            logger.warning(f"  Skip {fp.name}: {e}")

    raw_feature_cols = get_feature_cols(raw_dfs[0])
    logger.info(f"  Raw features: {len(raw_feature_cols)}")

    # Fit sanitizer on combined training data
    combined = pd.concat(raw_dfs, ignore_index=True)
    sanitizer.fit(combined, raw_feature_cols)
    logger.info(f"  Sanitizer fitted: removed {len(sanitizer.constant_features_)} const + "
                f"{len(sanitizer.duplicate_features_)} dup")

    # Pass 2: sanitize per-file + combine
    all_X, all_y = [], []
    for df in raw_dfs:
        df_t, remaining_cols = sanitizer.transform(df, raw_feature_cols, causal=True)
        all_X.append(df_t[remaining_cols])
        all_y.append(df_t['target'])

    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    # Remove constants
    non_const = X_combined.nunique() > 1
    X_combined = X_combined.loc[:, non_const]

    # Fit selector
    selector.fit_transform(X_combined, y_combined)
    logger.info(f"  Selector fitted: {len(selector.selected_features_)} features")

    # Save
    pipeline = {
        'sanitizer': sanitizer,
        'selector': selector,
        'raw_feature_cols': raw_feature_cols,
        'selected_features': selector.selected_features_,
    }
    pipeline_path = output_dir / 'feature_pipeline.pkl'
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"  Pipeline saved to {pipeline_path}")

    # Free memory
    del raw_dfs, combined, all_X, all_y, X_combined, y_combined

    return pipeline


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load saved ensemble model ─────────────────────────────────
    logger.info("Loading saved ensemble model...")
    ens_path = OUTPUT_DIR / 'ensemble_model.pkl'
    if not ens_path.exists():
        logger.error(f"ensemble_model.pkl not found at {ens_path}")
        sys.exit(1)

    model = EnsembleModel()
    model.load(str(ens_path))
    has_catboost = model.include_catboost and model.catboost_model_ is not None
    logger.info(f"  Model features: {len(model.feature_names_)}, CatBoost: {has_catboost}")

    # ── 2. Load or rebuild feature pipeline (sanitizer + selector) ───
    pipeline_path = OUTPUT_DIR / 'feature_pipeline.pkl'
    if pipeline_path.exists():
        logger.info("Loading saved feature pipeline (sanitizer + selector)...")
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
    else:
        logger.warning("feature_pipeline.pkl NOT found — rebuilding from training data...")
        pipeline = rebuild_feature_pipeline(DATA_DIR, OUTPUT_DIR, N_TRAIN_FILES, HORIZON)

    sanitizer        = pipeline['sanitizer']
    selector         = pipeline['selector']
    raw_feature_cols = pipeline['raw_feature_cols']
    logger.info(f"  Sanitizer: removed {len(sanitizer.constant_features_)} const + "
                f"{len(sanitizer.duplicate_features_)} dup features")
    logger.info(f"  Selector: {len(selector.selected_features_)} features selected")
    logger.info(f"  Raw feature cols: {len(raw_feature_cols)}")

    # ── 3. File lists ────────────────────────────────────────────────
    all_files  = get_sorted_files(DATA_DIR)
    n_train    = min(N_TRAIN_FILES, int(len(all_files) * 0.8))
    n_test     = min(N_TEST_FILES, len(all_files) - n_train)
    test_files = all_files[n_train : n_train + n_test]
    logger.info(f"  Test files ({len(test_files)}): {[f.stem for f in test_files]}")

    # ── 4. Pipeline components ───────────────────────────────────────
    loader  = DataLoader(data_dir=str(DATA_DIR))
    eng     = CausalFeatureEngineer()
    tgt     = TargetCreator(horizon=HORIZON)

    # ── Model metadata ───────────────────────────────────────────────
    model_labels = ['Ensemble', 'XGBoost', 'LightGBM']
    model_colors = ['green', 'blue', 'orange']
    pred_p3_keys = ['ens_pred_P3', 'xgb_pred_P3', 'lgbm_pred_P3']
    ret_keys     = ['ens_ret', 'xgb_ret', 'lgbm_ret']
    delta_keys   = ['ens_delta_P3', 'xgb_delta_P3', 'lgbm_delta_P3']
    if has_catboost:
        model_labels.append('CatBoost')
        model_colors.append('purple')
        pred_p3_keys.append('cat_pred_P3')
        ret_keys.append('cat_ret')
        delta_keys.append('cat_delta_P3')
    n_models = len(model_labels)

    # ── 5. Generate predictions on test files ────────────────────────
    logger.info("Generating predictions on test files (with proper sanitization)...")
    all_dfs: List[pd.DataFrame] = []

    for fi, fpath in enumerate(test_files):
        file_id = fpath.stem
        logger.info(f"  [{fi+1}/{len(test_files)}] File {file_id}")
        try:
            df_raw = loader.load_file(str(fpath))
            df_eng = eng.engineer_features(df_raw.copy())

            # Full P3 series BEFORE target filtering
            full_p3 = df_eng['P3'].values.copy()

            df_tgt = tgt.create_target(df_eng)
            df_valid = df_tgt[df_tgt['has_target']].copy()

            # Sanitize (transform with already-fitted sanitizer)
            df_san, remaining_cols = sanitizer.transform(
                df_valid, raw_feature_cols, causal=True
            )
            X = df_san[remaining_cols]
            y = df_san['target'].values

            # Select (transform with already-fitted selector)
            X_sel = selector.transform(X)

            # Predictions (using properly normalized features)
            ens_ret  = model.predict(X_sel)
            indiv    = model.predict_individual(X_sel)
            xgb_ret  = indiv[0]
            lgbm_ret = indiv[1]
            cat_ret  = indiv[2] if len(indiv) == 3 else None

            n = len(ens_ret)

            # P3 at decision time t
            p3_at_t = df_valid['P3'].values[:n]

            # Actual P3 at t+30 (from the full unfiltered series)
            valid_idx = df_valid.index[:n].values
            fwd_idx   = np.clip(valid_idx + HORIZON, 0, len(full_p3) - 1)
            p3_actual_fwd = full_p3[fwd_idx]

            # Predicted P3 at t+30
            ens_pred_p3  = p3_at_t * (1 + ens_ret)
            xgb_pred_p3  = p3_at_t * (1 + xgb_ret)
            lgbm_pred_p3 = p3_at_t * (1 + lgbm_ret)

            # Price CHANGES (removes common baseline)
            actual_delta = p3_actual_fwd - p3_at_t
            ens_delta    = p3_at_t * ens_ret
            xgb_delta    = p3_at_t * xgb_ret
            lgbm_delta   = p3_at_t * lgbm_ret

            rec = {
                'file_id':          file_id,
                'bar_t':            valid_idx,
                'P3_at_t':          p3_at_t,
                'P3_actual_fwd':    p3_actual_fwd,
                'actual_delta_P3':  actual_delta,
                'target_return':    y[:n],
                'ens_ret':          ens_ret,
                'xgb_ret':          xgb_ret,
                'lgbm_ret':         lgbm_ret,
                'ens_pred_P3':      ens_pred_p3,
                'xgb_pred_P3':     xgb_pred_p3,
                'lgbm_pred_P3':    lgbm_pred_p3,
                'ens_delta_P3':     ens_delta,
                'xgb_delta_P3':     xgb_delta,
                'lgbm_delta_P3':    lgbm_delta,
            }
            if cat_ret is not None:
                rec['cat_ret']      = cat_ret
                rec['cat_pred_P3']  = p3_at_t * (1 + cat_ret)
                rec['cat_delta_P3'] = p3_at_t * cat_ret

            pred_df = pd.DataFrame(rec)
            pred_df.to_csv(PLOTS_DIR / f'predictions_{file_id}.csv', index=False)
            all_dfs.append(pred_df)

        except Exception as e:
            logger.error(f"  Error on file {file_id}: {e}")
            import traceback; traceback.print_exc()

    if not all_dfs:
        logger.error("No predictions. Exiting.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Quick sanity check
    logger.info("Sanity check — prediction variability:")
    for lbl, rk in zip(model_labels, ret_keys):
        s = combined[rk]
        logger.info(f"  {lbl:12s}  unique={s.nunique():>6}  "
                    f"std={s.std():.8f}  range=[{s.min():.8f}, {s.max():.8f}]")
    logger.info(f"  {'Actual':12s}  unique={combined['target_return'].nunique():>6}  "
                f"std={combined['target_return'].std():.8f}")

    logger.info(f"Total: {len(combined)} bars across {len(all_dfs)} files")

    # ==================================================================
    # PLOTS — all plotted at DECISION TIME t (x = bar_t)
    # ==================================================================

    # ── CHART 1: Predicted P3 vs Actual P3 (absolute price) ──────────
    logger.info("Chart 1: Per-file P3 price overlay (Plotly)...")

    for pred_df in all_dfs:
        fid   = pred_df['file_id'].iloc[0]
        bar_t = pred_df['bar_t'].values

        fig = make_subplots(
            rows=n_models, cols=1, shared_xaxes=True,
            subplot_titles=[f'{lbl}' for lbl in model_labels],
            vertical_spacing=0.04
        )

        p3_now    = pred_df['P3_at_t'].values
        p3_actual = pred_df['P3_actual_fwd'].values

        for i, (lbl, pcol, clr) in enumerate(zip(model_labels, pred_p3_keys, model_colors)):
            row = i + 1
            p3_pred = pred_df[pcol].values

            # Current P3 (for reference)
            fig.add_trace(go.Scatter(
                x=bar_t, y=p3_now,
                mode='lines', name='P3 now (at t)',
                line=dict(color='lightgrey', width=0.8),
                legendgroup='p3now', showlegend=(i == 0)
            ), row=row, col=1)

            # Actual P3 at t+30
            fig.add_trace(go.Scatter(
                x=bar_t, y=p3_actual,
                mode='lines', name='Actual P3 (at t+30)' if i == 0 else None,
                line=dict(color='black', width=1.2), opacity=0.7,
                showlegend=(i == 0), legendgroup='actual'
            ), row=row, col=1)

            # Predicted P3 at t+30
            fig.add_trace(go.Scatter(
                x=bar_t, y=p3_pred,
                mode='lines', name=f'{lbl} Pred P3(t+30)',
                line=dict(color=clr, width=1.2), opacity=0.8,
            ), row=row, col=1)

            fig.update_yaxes(title_text='P3 Price', row=row, col=1)

        fig.update_xaxes(title_text='Decision Bar (t)', row=n_models, col=1)
        fig.update_layout(
            title=f'Predicted P3(t+30) vs Actual P3(t+30) — File {fid}<br>'
                  f'<sub>Both plotted at decision time t. Grey=current P3, Black=actual future P3, Color=predicted future P3</sub>',
            height=350 * n_models,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
        )
        fig.write_html(str(PLOTS_DIR / f'pred_p3_vs_actual_p3_{fid}.html'))

    logger.info(f"  {len(all_dfs)} P3 HTML files saved")

    # ── CHART 2: ΔP3 — Price CHANGE predicted vs actual ──────────────
    logger.info("Chart 2: Per-file ΔP3 (price change) overlay (Plotly)...")

    for pred_df in all_dfs:
        fid   = pred_df['file_id'].iloc[0]
        bar_t = pred_df['bar_t'].values

        fig = make_subplots(
            rows=n_models, cols=1, shared_xaxes=True,
            subplot_titles=[f'{lbl} — Predicted ΔP3 vs Actual ΔP3' for lbl in model_labels],
            vertical_spacing=0.04
        )

        actual_delta = pred_df['actual_delta_P3'].values

        for i, (lbl, dcol, clr) in enumerate(zip(model_labels, delta_keys, model_colors)):
            row = i + 1
            pred_delta = pred_df[dcol].values

            fig.add_trace(go.Scatter(
                x=bar_t, y=actual_delta,
                mode='lines', name='Actual ΔP3' if i == 0 else None,
                line=dict(color='black', width=1), opacity=0.5,
                showlegend=(i == 0), legendgroup='actual_d'
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=bar_t, y=pred_delta,
                mode='lines', name=f'{lbl} Pred ΔP3',
                line=dict(color=clr, width=1), opacity=0.8,
            ), row=row, col=1)
            fig.add_hline(y=0, line_dash='dash', line_color='grey',
                          line_width=0.5, row=row, col=1)

            act_std = np.std(actual_delta)
            prd_std = np.std(pred_delta)
            ratio = act_std / max(prd_std, 1e-12)
            fig.add_annotation(
                text=f'Actual std={act_std:.4f}, Pred std={prd_std:.6f} (ratio={ratio:.1f}x)',
                xref='x domain', yref='y domain', x=0.01, y=0.95,
                showarrow=False, font=dict(size=10, color='grey'),
                row=row, col=1
            )
            fig.update_yaxes(title_text='ΔP3 (price change)', row=row, col=1)

        fig.update_xaxes(title_text='Decision Bar (t)', row=n_models, col=1)
        fig.update_layout(
            title=f'Predicted ΔP3 vs Actual ΔP3 (30-bar price change) — File {fid}<br>'
                  f'<sub>ΔP3 = P3(t+30) − P3(t).  Removes baseline; shows pure change.</sub>',
            height=350 * n_models,
            template='plotly_white',
            hovermode='x unified'
        )
        fig.write_html(str(PLOTS_DIR / f'delta_p3_{fid}.html'))

    logger.info(f"  {len(all_dfs)} ΔP3 HTML files saved")

    # ── CHART 3: Return comparison ────────────────────────────────────
    logger.info("Chart 3: Per-file return comparison (Plotly)...")

    for pred_df in all_dfs:
        fid   = pred_df['file_id'].iloc[0]
        bar_t = pred_df['bar_t'].values

        fig = make_subplots(
            rows=n_models, cols=1, shared_xaxes=True,
            subplot_titles=[f'{lbl} — Predicted vs Actual Return' for lbl in model_labels],
            vertical_spacing=0.04
        )
        actual_ret = pred_df['target_return'].values

        for i, (lbl, rcol, clr) in enumerate(zip(model_labels, ret_keys, model_colors)):
            row = i + 1
            fig.add_trace(go.Scatter(
                x=bar_t, y=actual_ret,
                mode='lines', name='Actual Return' if i == 0 else None,
                line=dict(color='black', width=0.8), opacity=0.5,
                showlegend=(i == 0), legendgroup='act'
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=bar_t, y=pred_df[rcol].values,
                mode='lines', name=f'{lbl} Predicted',
                line=dict(color=clr, width=0.8), opacity=0.7
            ), row=row, col=1)
            fig.add_hline(y=0, line_dash='dash', line_color='grey',
                          line_width=0.5, row=row, col=1)
            fig.update_yaxes(title_text='Return', row=row, col=1)

        fig.update_xaxes(title_text='Decision Bar (t)', row=n_models, col=1)
        fig.update_layout(
            title=f'Predicted vs Actual 30-bar Return — File {fid}',
            height=300 * n_models, template='plotly_white',
            hovermode='x unified'
        )
        fig.write_html(str(PLOTS_DIR / f'return_comparison_{fid}.html'))

    logger.info(f"  {len(all_dfs)} return HTML files saved")

    # ── CHART 4: Combined scatter ─────────────────────────────────────
    logger.info("Chart 4: Combined scatter...")

    fig = make_subplots(rows=1, cols=n_models,
                        subplot_titles=model_labels)
    for i, (lbl, rcol, clr) in enumerate(zip(model_labels, ret_keys, model_colors)):
        col = i + 1
        actual = combined['target_return']
        pred   = combined[rcol]

        # Subsample for plotly performance
        if len(actual) > 20000:
            idx = np.random.RandomState(42).choice(len(actual), 20000, replace=False)
            a_s, p_s = actual.iloc[idx], pred.iloc[idx]
        else:
            a_s, p_s = actual, pred

        fig.add_trace(go.Scattergl(
            x=a_s, y=p_s, mode='markers',
            marker=dict(color=clr, size=2, opacity=0.15), name=lbl
        ), row=1, col=col)

        lo = min(a_s.min(), p_s.min())
        hi = max(a_s.max(), p_s.max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode='lines',
            line=dict(color='red', dash='dash', width=1), showlegend=False
        ), row=1, col=col)

        r  = np.corrcoef(actual, pred)[0, 1]
        da = np.mean(np.sign(actual) == np.sign(pred)) * 100
        fig.layout.annotations[i].text = f'{lbl} (r={r:.4f}, DA={da:.1f}%)'
        fig.update_xaxes(title_text='Actual Return', row=1, col=col)
        fig.update_yaxes(title_text='Predicted Return', row=1, col=col)

    fig.update_layout(title='Predicted vs Actual Return (all files)',
                      height=500, template='plotly_white')
    fig.write_html(str(PLOTS_DIR / 'scatter_returns_all.html'))
    logger.info("  Scatter saved")

    # ── CHART 5: Direction accuracy bar chart ─────────────────────────
    logger.info("Chart 5: Direction accuracy...")
    file_ids = [df['file_id'].iloc[0] for df in all_dfs]

    fig = go.Figure()
    for lbl, rcol, clr in zip(model_labels, ret_keys, model_colors):
        das = []
        for pdf in all_dfs:
            da = np.mean(np.sign(pdf[rcol]) == np.sign(pdf['target_return'])) * 100
            das.append(da)
        fig.add_trace(go.Bar(x=file_ids, y=das, name=lbl,
                             marker_color=clr, opacity=0.7))

    fig.add_hline(y=50, line_dash='dash', line_color='red',
                  annotation_text='50% baseline')
    fig.update_layout(
        title='Direction Accuracy by File & Model',
        xaxis_title='File ID', yaxis_title='Direction Accuracy (%)',
        barmode='group', template='plotly_white', yaxis=dict(range=[0, 100])
    )
    fig.write_html(str(PLOTS_DIR / 'direction_accuracy.html'))
    logger.info("  Direction accuracy saved")

    # ── CHART 6: Correlation heatmap ──────────────────────────────────
    logger.info("Chart 6: Correlation heatmap...")
    corr_mx = []
    for pdf in all_dfs:
        row = []
        for rcol in ret_keys:
            r = np.corrcoef(pdf['target_return'], pdf[rcol])[0, 1]
            row.append(round(r, 4))
        corr_mx.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=corr_mx, x=model_labels, y=file_ids,
        colorscale='RdYlGn', zmid=0, zmin=-0.1, zmax=0.3,
        text=[[f'{v:.4f}' for v in row] for row in corr_mx],
        texttemplate='%{text}', textfont=dict(size=10),
        hovertemplate='File %{y}<br>Model: %{x}<br>r: %{z:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title='Prediction–Actual Return Correlation',
        height=max(400, len(file_ids) * 30), template='plotly_white'
    )
    fig.write_html(str(PLOTS_DIR / 'correlation_heatmap.html'))
    logger.info("  Heatmap saved")

    # ── CHART 7: Cumulative prediction error ──────────────────────────
    logger.info("Chart 7: Cumulative error...")
    fig = make_subplots(rows=1, cols=1)
    for lbl, pcol, clr in zip(model_labels, pred_p3_keys, model_colors):
        err = combined[pcol] - combined['P3_actual_fwd']
        cum_err = err.groupby(combined['file_id']).cumsum()
        fig.add_trace(go.Scatter(
            x=np.arange(len(cum_err)), y=cum_err,
            mode='lines', name=lbl, line=dict(color=clr, width=1), opacity=0.7
        ))

    pos = 0
    for pdf in all_dfs:
        pos += len(pdf)
        fig.add_vline(x=pos, line_dash='dot', line_color='grey',
                      line_width=0.5, opacity=0.4)

    fig.update_layout(
        title='Cumulative P3 Prediction Error',
        xaxis_title='Bar (sequential)', yaxis_title='Cumul. Error (price)',
        template='plotly_white', hovermode='x unified'
    )
    fig.write_html(str(PLOTS_DIR / 'cumulative_prediction_error.html'))
    logger.info("  Cumulative error saved")

    # ── Summary ──────────────────────────────────────────────────────
    n = len(all_dfs)
    logger.info("=" * 65)
    logger.info("All outputs → %s", PLOTS_DIR)
    logger.info("")
    logger.info("INTERACTIVE HTML (zoom, pan, hover in browser):")
    logger.info("  pred_p3_vs_actual_p3_{id}.html — P3 price (3 lines)   (%d)", n)
    logger.info("  delta_p3_{id}.html             — ΔP3 price change     (%d)", n)
    logger.info("  return_comparison_{id}.html     — %% return            (%d)", n)
    logger.info("  scatter_returns_all.html        — return scatter")
    logger.info("  direction_accuracy.html         — DA bar chart")
    logger.info("  correlation_heatmap.html        — r heatmap")
    logger.info("  cumulative_prediction_error.html— cumul. error")
    logger.info("")
    logger.info("DATA:")
    logger.info("  predictions_{id}.csv            — raw data (%d files)", n)
    logger.info("=" * 65)


if __name__ == '__main__':
    main()