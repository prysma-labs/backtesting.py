"""
`SignalStrategy`: a `Strategy` helper that operates on entry/exit signal vectors,
making the backtest behave like a vectorized backtest.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .backtesting import Strategy


class SignalStrategy(Strategy):
    """
    A simple helper strategy that operates on position entry/exit signals.
    This makes the backtest of the strategy simulate a [vectorized backtest].
    See [tutorials] for usage examples.

    [vectorized backtest]: https://www.google.com/search?q=vectorized+backtest
    [tutorials]: index.html#tutorials

    To use this helper strategy, subclass it, override its
    `backtesting.backtesting.Strategy.init` method,
    and set the signal vector by calling
    `backtesting.lib.SignalStrategy.set_signal` method from within it.

        class ExampleStrategy(SignalStrategy):
            def init(self):
                super().init()
                self.set_signal(sma1 > sma2, sma1 < sma2)

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """

    __entry_signal = (0,)
    __exit_signal = (False,)

    def set_signal(
        self,
        entry_size: Sequence[float],
        exit_portion: Sequence[float] | None = None,
        *,
        plot: bool = True,
    ):
        """
        Set entry/exit signal vectors (arrays).

        A long entry signal is considered present wherever `entry_size`
        is greater than zero, and a short signal wherever `entry_size`
        is less than zero, following `backtesting.backtesting.Order.size` semantics.

        If `exit_portion` is provided, a nonzero value closes portion the position
        (see `backtesting.backtesting.Trade.close()`) in the respective direction
        (positive values close long trades, negative short).

        If `plot` is `True`, the signal entry/exit indicators are plotted when
        `backtesting.backtesting.Backtest.plot` is called.
        """
        self.__entry_signal = self.I(  # type: ignore
            lambda: pd.Series(entry_size, dtype=float).replace(0, np.nan),
            name="entry size",
            plot=plot,
            overlay=False,
            scatter=True,
            color="black",
        )

        if exit_portion is not None:
            self.__exit_signal = self.I(  # type: ignore
                lambda: pd.Series(exit_portion, dtype=float).replace(0, np.nan),
                name="exit portion",
                plot=plot,
                overlay=False,
                scatter=True,
                color="black",
            )

    def next(self):
        super().next()

        exit_portion = self.__exit_signal[-1]
        if exit_portion > 0:
            for trade in self.trades:
                if trade.is_long:
                    trade.close(exit_portion)
        elif exit_portion < 0:
            for trade in self.trades:
                if trade.is_short:
                    trade.close(-exit_portion)

        entry_size = self.__entry_signal[-1]
        if entry_size > 0:
            self.buy(size=entry_size)
        elif entry_size < 0:
            self.sell(size=-entry_size)


__all__ = ["SignalStrategy"]
