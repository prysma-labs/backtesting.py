"""Overridable `Pool` factory used by parallel optimization / multi-backtests.

Users may rebind `backtesting.Pool` to a callable returning their own pool
implementation (e.g. `multiprocessing.Pool` from a specific context) when
they need real process-based parallelism on Windows / macOS where the
default start method is `spawn`.
"""

from __future__ import annotations

from ._util import try_


def Pool(processes=None, initializer=None, initargs=()):
    import multiprocessing as mp
    import sys
    if sys.platform.startswith('linux') and mp.get_start_method(allow_none=True) != 'fork':
        try_(lambda: mp.set_start_method('fork'))
    if mp.get_start_method() == 'spawn':
        import warnings
        warnings.warn(
            "If you want to use multi-process optimization with "
            "`multiprocessing.get_start_method() == 'spawn'` (e.g. on Windows),"
            "set `backtesting.Pool = multiprocessing.Pool` (or of the desired context) "
            "and hide `bt.optimize()` call behind a `if __name__ == '__main__'` guard. "
            "Currently using thread-based paralellism, "
            "which might be slightly slower for non-numpy / non-GIL-releasing code. "
            "See https://github.com/kernc/backtesting.py/issues/1256",
            category=RuntimeWarning, stacklevel=3)
        from multiprocessing.dummy import Pool as _Pool
        return _Pool(processes, initializer, initargs)
    return mp.Pool(processes, initializer, initargs)


__all__ = ["Pool"]
