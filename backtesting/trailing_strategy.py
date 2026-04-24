"""
`TrailingStrategy`: a `Strategy` helper with automatic ATR-based trailing
stop-loss for every open trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtesting import Strategy


class TrailingStrategy(Strategy):
    """
    A strategy with automatic trailing stop-loss, trailing the current
    price at distance of some multiple of average true range (ATR). Call
    `TrailingStrategy.set_trailing_sl()` to set said multiple
    (`6` by default). See [tutorials] for usage examples.

    [tutorials]: index.html#tutorials

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """

    __n_atr = 6.0
    __atr = None

    def init(self):
        super().init()
        self.set_atr_periods()

    def set_atr_periods(self, periods: int = 100):
        """
        Set the lookback period for computing ATR. The default value
        of 100 ensures a _stable_ ATR.
        """
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        atr = pd.Series(tr).rolling(periods).mean().bfill().values  # type: ignore[attr-defined]
        self.__atr = atr

    def set_trailing_sl(self, n_atr: float = 6):
        """
        Set the future trailing stop-loss as some multiple (`n_atr`)
        average true bar ranges away from the current price.
        """
        self.__n_atr = n_atr

    def set_trailing_pct(self, pct: float = 0.05):
        """
        Set the future trailing stop-loss as some percent (`0 < pct < 1`)
        below the current price (default 5% below).

        .. note:: Stop-loss set by `pct` is inexact
            Stop-loss set by `set_trailing_pct` is converted to units of ATR
            with `mean(Close * pct / atr)` and set with `set_trailing_sl`.
        """
        assert 0 < pct < 1, "Need pct= as rate, i.e. 5% == 0.05"
        pct_in_atr = float(np.mean(self.data.Close * pct / self.__atr))  # type: ignore
        self.set_trailing_sl(pct_in_atr)

    def next(self):
        super().next()
        # Can't use index=-1 because self.__atr is not an Indicator type
        assert self.__atr is not None, "call set_atr_periods() before next()"
        index = len(self.data) - 1
        for trade in self.trades:
            if trade.is_long:
                trade.sl = max(
                    trade.sl or -np.inf, self.data.Close[index] - self.__atr[index] * self.__n_atr
                )
            else:
                trade.sl = min(
                    trade.sl or np.inf, self.data.Close[index] + self.__atr[index] * self.__n_atr
                )


__all__ = ["TrailingStrategy"]
