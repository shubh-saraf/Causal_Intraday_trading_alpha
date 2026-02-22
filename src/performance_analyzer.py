"""
PerformanceAnalyzer Module - Calculates trading performance metrics.

Metrics:
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Trade statistics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes trading performance from trade logs.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self.metrics_: Dict = {}
        
    def analyze(self, trade_log: pd.DataFrame) -> Dict:
        """
        Analyze trading performance.
        
        Args:
            trade_log: DataFrame with trade log from ExecutionEngine
            
        Returns:
            Dictionary of performance metrics
        """
        if trade_log.empty:
            return {'error': 'Empty trade log'}
        
        metrics = {}
        
        # Basic metrics
        metrics['total_bars'] = len(trade_log)
        metrics['total_pnl'] = trade_log['cumulative_pnl'].iloc[-1]
        metrics['total_transaction_costs'] = trade_log['transaction_cost'].sum()
        
        # Return metrics
        initial_price = trade_log['price'].iloc[0]
        metrics['total_return_pct'] = (metrics['total_pnl'] / initial_price) * 100
        
        # Calculate returns per bar
        pnl_changes = trade_log['cumulative_pnl'].diff().fillna(0)
        returns = pnl_changes / initial_price
        
        # Sharpe Ratio (annualized, assuming ~252 trading days * ~400 bars/day)
        if returns.std() > 0:
            bars_per_year = 252 * 400  # Approximate
            sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
            metrics['sharpe_ratio'] = sharpe
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Maximum Drawdown
        cumulative = trade_log['cumulative_pnl']
        running_max = cumulative.cummax()
        drawdown = running_max - cumulative
        metrics['max_drawdown'] = drawdown.max()
        metrics['max_drawdown_pct'] = (metrics['max_drawdown'] / initial_price) * 100 if initial_price > 0 else 0
        
        # Trade statistics
        position_changes = trade_log['position'].diff().fillna(0)
        trades = trade_log[position_changes != 0]
        
        metrics['num_trades'] = len(trades)
        
        # Win rate
        realized_pnls = trade_log[trade_log['pnl'] != 0]['pnl']
        if len(realized_pnls) > 0:
            metrics['num_winning_trades'] = (realized_pnls > 0).sum()
            metrics['num_losing_trades'] = (realized_pnls < 0).sum()
            metrics['win_rate'] = metrics['num_winning_trades'] / len(realized_pnls)
            
            # Profit factor
            gross_profit = realized_pnls[realized_pnls > 0].sum()
            gross_loss = abs(realized_pnls[realized_pnls < 0].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average trade
            metrics['avg_winning_trade'] = realized_pnls[realized_pnls > 0].mean() if metrics['num_winning_trades'] > 0 else 0
            metrics['avg_losing_trade'] = realized_pnls[realized_pnls < 0].mean() if metrics['num_losing_trades'] > 0 else 0
        else:
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['num_winning_trades'] = 0
            metrics['num_losing_trades'] = 0
        
        # Position statistics
        metrics['time_in_market_pct'] = (trade_log['position'] != 0).mean() * 100
        metrics['time_long_pct'] = (trade_log['position'] == 1).mean() * 100
        metrics['time_short_pct'] = (trade_log['position'] == -1).mean() * 100
        
        self.metrics_ = metrics
        return metrics
    
    def generate_report(self, trade_log: pd.DataFrame) -> str:
        """
        Generate a text report of performance.
        
        Args:
            trade_log: DataFrame with trade log
            
        Returns:
            Formatted report string
        """
        metrics = self.analyze(trade_log)
        
        report = """
=== TRADING PERFORMANCE REPORT ===

## Summary
- Total Bars: {total_bars:,}
- Total PnL: {total_pnl:.4f}
- Total Return: {total_return_pct:.4f}%
- Transaction Costs: {total_transaction_costs:.4f}

## Risk Metrics
- Sharpe Ratio: {sharpe_ratio:.4f}
- Max Drawdown: {max_drawdown:.4f}
- Max Drawdown %: {max_drawdown_pct:.4f}%

## Trade Statistics
- Number of Trades: {num_trades}
- Winning Trades: {num_winning_trades}
- Losing Trades: {num_losing_trades}
- Win Rate: {win_rate:.2%}
- Profit Factor: {profit_factor:.4f}

## Position Analysis
- Time in Market: {time_in_market_pct:.2f}%
- Time Long: {time_long_pct:.2f}%
- Time Short: {time_short_pct:.2f}%

================================
""".format(**metrics)
        
        return report
    
    def save_report(self, trade_log: pd.DataFrame, filepath: str):
        """Save performance report to file."""
        report = self.generate_report(trade_log)
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {filepath}")