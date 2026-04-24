"""
Multi-dataset `Backtest` wrapper.

Run a `Strategy` across many DataFrames (typically different instruments
or time windows) in parallel on Modal and gather their stats into one
table.
"""

from __future__ import annotations

from itertools import count
from typing import Any, Optional

import cloudpickle
import modal
import pandas as pd

from ._util import _tqdm
from .backtesting import Backtest, Strategy
from .modal_runtime import _resolve_image, run_remote


def _run_one(args):
    df, strategy, bt_kwargs, run_kwargs = args
    stats = Backtest(df, strategy, **bt_kwargs).run(**run_kwargs)
    return stats.filter(regex="^[^_]")


class MultiBacktest:
    """
    Multi-dataset `backtesting.backtesting.Backtest` wrapper.

    Run supplied `backtesting.backtesting.Strategy` on several instruments,
    in parallel via Modal.  Used for comparing strategy runs across many
    instruments or classes of instruments. Example:

        from backtesting.test import EURUSD, BTCUSD, SmaCross
        btm = MultiBacktest([EURUSD, BTCUSD], SmaCross)
        stats_per_ticker: pd.DataFrame = btm.run(fast=10, slow=20)
        heatmap_per_ticker: pd.DataFrame = btm.optimize(...)

    Pass ``image=modal.Image(...)`` to override the default Modal image
    for this instance only. Otherwise the global `backtesting.configure`
    setting is used, falling back to `backtesting.DEFAULT_IMAGE` plus a
    best-effort auto-mount of the strategy's package directory.
    """

    def __init__(
        self,
        df_list,
        strategy_cls: type[Strategy],
        *,
        image: Optional[modal.Image] = None,
        **kwargs,
    ):
        self._dfs = df_list
        self._strategy = strategy_cls
        self._bt_kwargs = kwargs
        self._image = image

    def run(self, **kwargs):
        """
        Wraps `backtesting.backtesting.Backtest.run`. Returns `pd.DataFrame` with
        currency indexes in columns.
        """
        image = _resolve_image(self._strategy, self._image)
        payloads = [
            cloudpickle.dumps((_run_one, (df, self._strategy, self._bt_kwargs, kwargs)))
            for df in self._dfs
        ]

        all_results: list[pd.Series] = []
        with _tqdm(  # type: ignore[call-arg]
            total=len(payloads), desc=self.run.__qualname__, mininterval=2
        ) as pbar:
            for result_bytes in run_remote(image, payloads, desc="MultiBacktest.run"):
                all_results.append(cloudpickle.loads(result_bytes))
                pbar.update(1)
        return pd.DataFrame(all_results).transpose()

    def optimize(self, **kwargs) -> pd.DataFrame:
        """
        Wraps `backtesting.backtesting.Backtest.optimize`, but returns `pd.DataFrame` with
        currency indexes in columns.

            heamap: pd.DataFrame = btm.optimize(...)
            from backtesting.plot import plot_heatmaps
            plot_heatmaps(heatmap.mean(axis=1))
        """
        heatmaps = []
        for df in _tqdm(self._dfs, desc=self.__class__.__name__, mininterval=2):
            bt = Backtest(df, self._strategy, image=self._image, **self._bt_kwargs)
            _best_stats, heatmap = bt.optimize(  # type: ignore
                return_heatmap=True, return_optimization=False, **kwargs
            )
            heatmaps.append(heatmap)
        heatmap = pd.DataFrame(dict(zip(count(), heatmaps)))
        return heatmap


__all__ = ["MultiBacktest"]
