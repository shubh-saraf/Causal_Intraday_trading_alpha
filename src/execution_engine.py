"""
ExecutionEngine Module - Executes trades and tracks PnL.

Features:
- Position management (+1, -1, 0)
- Transaction cost accounting (0.01%)
- Realized and Mark-to-Market PnL tracking
- Trade logging
- Loss-hold strategy: losing positions are held up to max_loss_hold bars
  to allow recovery; profitable positions can be flipped immediately.

Causality: Execution happens AFTER signal generation at each bar.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: int
    signal: int          # raw signal from model
    price: float
    position: int        # actual position after execution logic
    entry_price: float
    realized_pnl: float
    mtm_pnl: float
    transaction_cost: float
    cumulative_pnl: float
    held_loss: bool = False   # True if signal was suppressed due to loss-hold


class ExecutionEngine:
    """
    Executes trading signals and tracks PnL.

    Loss-hold strategy
    ------------------
    When a signal requests a position change (flip or exit):
      • If the current position is **profitable** (unrealized PnL > 0):
        → Execute immediately.
      • If the current position is **at a loss** (unrealized PnL ≤ 0):
        → Hold the losing position; do NOT flip/exit yet.
        → Keep holding for up to ``max_loss_hold`` bars from entry.
        → During the hold, if the position becomes profitable at any bar
          AND a change signal is active → execute the change.
        → After ``max_loss_hold`` bars of holding the loser → force-close
          (book the loss) and execute whatever the current signal is.

    Transaction cost model: 0.01% per trade (entry/exit/flip).
    """

    def __init__(self,
                 transaction_cost_bps: float = 1.0,
                 max_loss_hold: int = 30):
        """
        Initialize the execution engine.

        Args:
            transaction_cost_bps: Transaction cost in basis points (default 1 = 0.01%)
            max_loss_hold: Max bars to hold a losing position before force-close.
                           Set to 0 to disable loss-hold (old behaviour).
        """
        self.transaction_cost_rate = transaction_cost_bps / 10000
        self.max_loss_hold = max_loss_hold

        # State variables
        self.position = 0
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        self.total_transaction_costs = 0.0
        self.bars_in_position = 0       # bars since current entry

        # Trade log
        self.trades: List[Trade] = []

    def reset(self):
        """Reset execution state."""
        self.position = 0
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        self.total_transaction_costs = 0.0
        self.bars_in_position = 0
        self.trades = []

    # ──────────────────────────────────────────────────────────────────
    #  Core execution with loss-hold logic
    # ──────────────────────────────────────────────────────────────────
    def execute(self, timestamp: int, signal: int, price: float) -> Trade:
        """
        Execute a trading signal with loss-hold strategy.

        Decision tree (when signal ≠ current position):

        1. No position → allow entry.
        2. In position, unrealized PnL > 0 → allow flip/exit (take profit).
        3. In position, unrealized PnL ≤ 0:
           a. bars_in_position < max_loss_hold → HOLD (suppress change).
           b. bars_in_position ≥ max_loss_hold → FORCE close/flip (book loss).

        If max_loss_hold == 0, the loss-hold logic is disabled and every
        signal is executed immediately (backward-compatible).

        Args:
            timestamp: Current bar index
            signal:    Raw model signal (+1, -1, 0)
            price:     Current P3 price

        Returns:
            Trade object with execution details
        """
        realized_pnl = 0.0
        transaction_cost = 0.0
        held_loss = False

        old_position = self.position
        desired_position = signal

        # Increment hold counter every bar we're in a position
        if old_position != 0:
            self.bars_in_position += 1

        # ── Decide whether the signal is allowed to execute ──────────
        execute_change = False

        if desired_position == old_position:
            # No change requested → nothing to decide
            execute_change = False
        elif old_position == 0:
            # Not in a position → allow entry unconditionally
            execute_change = True
        elif self.max_loss_hold == 0:
            # Loss-hold disabled → always execute
            execute_change = True
        else:
            # Currently in a position and signal wants to change
            unrealized_pnl = old_position * (price - self.entry_price)

            # Must account for transaction cost of the proposed change.
            # "Profitable" means: realized PnL AFTER paying exit/flip cost > 0
            if desired_position == 0:
                # Exit: pay 1× cost
                proposed_tc = abs(old_position) * price * self.transaction_cost_rate
            else:
                # Flip: pay 2× cost (exit old + enter new)
                proposed_tc = 2 * price * self.transaction_cost_rate

            net_after_costs = unrealized_pnl - proposed_tc

            if net_after_costs > 0:
                # Position is profitable AFTER costs → allow immediate change
                execute_change = True
            elif self.bars_in_position >= self.max_loss_hold:
                # Held the loser long enough → force-close
                execute_change = True
                logger.debug(f"Bar {timestamp}: force-closing losing position "
                             f"after {self.bars_in_position} bars "
                             f"(unrealized={unrealized_pnl:.6f})")
            else:
                # Losing position, not yet at max hold → suppress change
                execute_change = False
                held_loss = True

        # ── Execute the position change (or not) ─────────────────────
        if execute_change and desired_position != old_position:
            if old_position == 0:
                # Entry: 0 → ±1
                transaction_cost = abs(desired_position) * price * self.transaction_cost_rate
            elif desired_position == 0:
                # Exit: ±1 → 0
                transaction_cost = abs(old_position) * price * self.transaction_cost_rate
                realized_pnl = old_position * (price - self.entry_price)
            else:
                # Flip: +1 → −1 or −1 → +1  (double cost)
                transaction_cost = 2 * price * self.transaction_cost_rate
                realized_pnl = old_position * (price - self.entry_price)

            # Update position state
            self.position = desired_position
            if desired_position != 0:
                self.entry_price = price
                self.bars_in_position = 0      # reset counter on new entry
            else:
                self.entry_price = 0.0
                self.bars_in_position = 0

            self.total_transaction_costs += transaction_cost

        # ── Calculate MTM PnL (unrealized) ────────────────────────────
        if self.position != 0:
            mtm_pnl = self.position * (price - self.entry_price)
        else:
            mtm_pnl = 0.0

        # ── Update cumulative PnL ────────────────────────────────────
        self.cumulative_pnl += realized_pnl - transaction_cost

        # ── Log the trade record ─────────────────────────────────────
        trade = Trade(
            timestamp=timestamp,
            signal=signal,
            price=price,
            position=self.position,
            entry_price=self.entry_price,
            realized_pnl=realized_pnl,
            mtm_pnl=mtm_pnl,
            transaction_cost=transaction_cost,
            cumulative_pnl=self.cumulative_pnl,
            held_loss=held_loss,
        )

        self.trades.append(trade)
        return trade

    # ──────────────────────────────────────────────────────────────────
    #  Series execution
    # ──────────────────────────────────────────────────────────────────
    def execute_series(self, timestamps: np.ndarray, signals: np.ndarray,
                       prices: np.ndarray) -> pd.DataFrame:
        """
        Execute a series of signals.

        Args:
            timestamps: Array of timestamps
            signals:    Array of signals
            prices:     Array of P3 prices

        Returns:
            DataFrame with trade log
        """
        self.reset()

        for i in range(len(timestamps)):
            self.execute(int(timestamps[i]), int(signals[i]), float(prices[i]))

        return self.get_trade_log()

    # ──────────────────────────────────────────────────────────────────
    #  Trade log
    # ──────────────────────────────────────────────────────────────────
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get the trade log as DataFrame.

        Returns:
            DataFrame with columns: timestamp, signal, price, position,
            pnl, cumulative_pnl, transaction_cost, entry_price, mtm_pnl,
            held_loss
        """
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                'timestamp': t.timestamp,
                'signal': t.signal,
                'price': t.price,
                'position': t.position,
                'pnl': t.realized_pnl,
                'cumulative_pnl': t.cumulative_pnl,
                'transaction_cost': t.transaction_cost,
                'entry_price': t.entry_price,
                'mtm_pnl': t.mtm_pnl,
                'held_loss': t.held_loss,
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────
    #  End-of-day close
    # ──────────────────────────────────────────────────────────────────
    def close_all_positions(self, timestamp: int, price: float):
        """
        Close all open positions at end of day.

        Args:
            timestamp: Closing timestamp
            price:     Closing price
        """
        if self.position != 0:
            # Force close regardless of PnL (end of day)
            old_max = self.max_loss_hold
            self.max_loss_hold = 0              # temporarily disable loss-hold
            self.execute(timestamp, 0, price)
            self.max_loss_hold = old_max        # restore
            logger.info(f"Closed position at timestamp {timestamp}, price {price}")