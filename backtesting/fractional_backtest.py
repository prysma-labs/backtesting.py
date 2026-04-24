"""
`FractionalBacktest`: a `Backtest` variant that supports fractional share
trading by rescaling OHLC and Volume by a configurable `fractional_unit`.
"""

from __future__ import annotations

import warnings

import pandas as pd

from ._util import patch
from .backtesting import Backtest


class FractionalBacktest(Backtest):
    """
    A `backtesting.backtesting.Backtest` that supports fractional share trading
    by simple composition. It applies roughly the transformation:

        data = (data * fractional_unit).assign(Volume=data.Volume / fractional_unit)

    as left unchallenged in [this FAQ entry on GitHub](https://github.com/kernc/backtesting.py/issues/134),
    then passes `data`, `args*`, and `**kwargs` to its super.

    Parameter `fractional_unit` represents the smallest fraction of currency that can be traded
    and defaults to one [satoshi]. For μBTC trading, pass `fractional_unit=1/1e6`.
    Thus-transformed backtest does a whole-sized trading of `fractional_unit` units.

    [satoshi]: https://en.wikipedia.org/wiki/Bitcoin#Units_and_divisibility
    """

    def __init__(self, data, *args, fractional_unit=1 / 100e6, **kwargs):
        if "satoshi" in kwargs:
            warnings.warn(
                "Parameter `FractionalBacktest(..., satoshi=)` is deprecated. "
                "Use `FractionalBacktest(..., fractional_unit=)`.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            fractional_unit = 1 / kwargs.pop("satoshi")
        self._fractional_unit = fractional_unit
        self.__data: pd.DataFrame = data.copy(deep=False)  # Shallow copy
        for col in (
            "Open",
            "High",
            "Low",
            "Close",
        ):
            self.__data[col] = self.__data[col] * self._fractional_unit
        for col in ("Volume",):
            self.__data[col] = self.__data[col] / self._fractional_unit
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(action="ignore", message="frac")
            super().__init__(data, *args, **kwargs)

    def run(self, **kwargs) -> pd.Series:
        with patch(self, "_data", self.__data):
            result = super().run(**kwargs)

        trades: pd.DataFrame = result["_trades"]  # type: ignore[assignment]
        trades["Size"] *= self._fractional_unit
        trades[["EntryPrice", "ExitPrice", "TP", "SL"]] /= self._fractional_unit

        indicators = result["_strategy"]._indicators  # type: ignore[attr-defined]
        for indicator in indicators:
            if indicator._opts["overlay"]:
                indicator.setflags(write=True)
                indicator /= self._fractional_unit

        return result


__all__ = ["FractionalBacktest"]
