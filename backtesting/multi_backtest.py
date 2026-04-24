"""
Multi-dataset `Backtest` wrapper.

Run a `Strategy` across many DataFrames (typically different instruments
or time windows) in parallel and gather their stats into one table.
"""

from __future__ import annotations

import os
from itertools import chain, count

import pandas as pd

from ._util import SharedMemoryManager, _batch, _tqdm
from .backtesting import Backtest, Strategy


class MultiBacktest:
    """
    Multi-dataset `backtesting.backtesting.Backtest` wrapper.

    Run supplied `backtesting.backtesting.Strategy` on several instruments,
    in parallel.  Used for comparing strategy runs across many instruments
    or classes of instruments. Example:

        from backtesting.test import EURUSD, BTCUSD, SmaCross
        btm = MultiBacktest([EURUSD, BTCUSD], SmaCross)
        stats_per_ticker: pd.DataFrame = btm.run(fast=10, slow=20)
        heatmap_per_ticker: pd.DataFrame = btm.optimize(...)
    """

    def __init__(self, df_list, strategy_cls: type[Strategy], **kwargs):
        self._dfs = df_list
        self._strategy = strategy_cls
        self._bt_kwargs = kwargs

    # Cap on DataFrames whose POSIX shared-memory segments are held open
    # simultaneously when using a process-based pool. macOS POSIX shm limits
    # are very tight (see `sysctl kern.sysv.shmseg|shmmni`) and leaked
    # segments persist across crashes until reboot, so we keep peak usage
    # well bounded. Override via env var BACKTESTING_SHM_BATCH if needed.
    _SHM_BATCH_CAP = int(os.environ.get("BACKTESTING_SHM_BATCH", "24"))

    def run(self, **kwargs):
        """
        Wraps `backtesting.backtesting.Backtest.run`. Returns `pd.DataFrame` with
        currency indexes in columns.
        """
        # When the pool is thread-based (e.g. macOS spawn fallback to
        # multiprocessing.dummy.Pool), skip POSIX shared memory entirely --
        # threads share memory natively, and `shm_open` on macOS hits hard
        # system limits well before our DataFrame counts. When the pool is
        # process-based, allocate shm in bounded chunks instead of all at
        # once to keep peak FD usage low. The original implementation
        # allocated one segment per column per DataFrame upfront which
        # exhausts macOS POSIX shm pools and leaks segments on crash.
        from multiprocessing.pool import ThreadPool

        from . import Pool

        with (
            Pool() as pool,
            _tqdm(  # type: ignore[call-arg]
                total=len(self._dfs), desc=self.run.__qualname__, mininterval=2
            ) as pbar,
        ):
            if isinstance(pool, ThreadPool):
                results_iter = pool.imap(
                    self._thread_task_run,
                    ((df, self._strategy, self._bt_kwargs, kwargs) for df in self._dfs),
                )
                all_results = []
                for stats in results_iter:
                    all_results.append(stats)
                    pbar.update(1)
            else:
                cap = max(1, self._SHM_BATCH_CAP)
                chunks = [self._dfs[i : i + cap] for i in range(0, len(self._dfs), cap)]
                all_results = []
                for chunk in chunks:
                    with SharedMemoryManager() as smm:
                        shm = [smm.df2shm(df) for df in chunk]
                        chunk_results = list(
                            pool.imap(
                                self._mp_task_run,
                                (
                                    (sub_shm, self._strategy, self._bt_kwargs, kwargs)
                                    for sub_shm in _batch(shm)
                                ),
                            )
                        )
                        all_results.extend(chain(*chunk_results))
                        pbar.update(len(chunk))
        df = pd.DataFrame(all_results).transpose()
        return df

    @staticmethod
    def _thread_task_run(args):
        df, strategy, bt_kwargs, run_kwargs = args
        stats = Backtest(df, strategy, **bt_kwargs).run(**run_kwargs)
        return stats.filter(regex="^[^_]")

    @staticmethod
    def _mp_task_run(args):
        data_shm, strategy, bt_kwargs, run_kwargs = args
        dfs, shms = zip(*(SharedMemoryManager.shm2df(i) for i in data_shm), strict=True)
        try:
            return [
                stats.filter(regex="^[^_]")
                for stats in (Backtest(df, strategy, **bt_kwargs).run(**run_kwargs) for df in dfs)
            ]
        finally:
            for shmem in chain(*shms):
                shmem.close()

    def optimize(self, **kwargs) -> pd.DataFrame:
        """
        Wraps `backtesting.backtesting.Backtest.optimize`, but returns `pd.DataFrame` with
        currency indexes in columns.

            heamap: pd.DataFrame = btm.optimize(...)
            from backtesting.plot import plot_heatmaps
            plot_heatmaps(heatmap.mean(axis=1))
        """
        heatmaps = []
        # Simple loop since bt.optimize already does its own multiprocessing
        for df in _tqdm(self._dfs, desc=self.__class__.__name__, mininterval=2):
            bt = Backtest(df, self._strategy, **self._bt_kwargs)
            _best_stats, heatmap = bt.optimize(  # type: ignore
                return_heatmap=True, return_optimization=False, **kwargs
            )
            heatmaps.append(heatmap)
        heatmap = pd.DataFrame(dict(zip(count(), heatmaps)))
        return heatmap


__all__ = ["MultiBacktest"]
