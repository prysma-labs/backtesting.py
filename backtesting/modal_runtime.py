"""Modal-backed execution runtime for `MultiBacktest` and `Backtest.optimize`.

Replaces the previous local `multiprocessing.Pool` / `ThreadPool` /
POSIX-shared-memory pipeline. All parallel backtests run inside ephemeral
Modal containers built from `DEFAULT_IMAGE` (overridable via `configure(...)`
or per-instance `image=` kwargs).

Public surface:
    DEFAULT_IMAGE: modal.Image
    configure(*, image=None, secrets=None, cpu=None, memory=None,
              min_containers=0, max_containers=None, timeout=3600)

Internal:
    _resolve_image(strategy_cls, override) -> modal.Image
    run_remote(image, payloads, *, desc=None) -> Iterator[bytes]
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cloudpickle
import modal

DEFAULT_IMAGE: modal.Image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "backtesting",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "cloudpickle",
)


@dataclass
class _Config:
    image: modal.Image | None = None
    secrets: list[modal.Secret] = field(default_factory=list)
    cpu: float | None = None
    memory: int | None = None
    min_containers: int = 0
    max_containers: int | None = None
    timeout: int = 3600


_CONFIG = _Config()


def configure(
    *,
    image: modal.Image | None = None,
    secrets: list[modal.Secret] | None = None,
    cpu: float | None = None,
    memory: int | None = None,
    min_containers: int = 0,
    max_containers: int | None = None,
    timeout: int = 3600,
) -> None:
    """Override process-wide defaults for the Modal execution context.

    Precedence at call time: per-instance `image=` > `configure(image=)` >
    `DEFAULT_IMAGE` + auto-mount of the strategy module's package.
    """
    _CONFIG.image = image
    _CONFIG.secrets = list(secrets) if secrets else []
    _CONFIG.cpu = cpu
    _CONFIG.memory = memory
    _CONFIG.min_containers = min_containers
    _CONFIG.max_containers = max_containers
    _CONFIG.timeout = timeout


def _auto_mount(strategy_cls: type, base_image: modal.Image) -> modal.Image:
    """Mount the user's strategy package dir into the image when feasible.

    Only fires when no explicit image was supplied. Walks the strategy
    class's MRO to find the first ancestor whose source lives in the
    user's project tree (skipping stdlib `abc` for `type()`-built dynamic
    subclasses, the `backtesting` package itself, and site-packages).
    """
    backtesting_root = Path(__file__).resolve().parent
    stdlib_names = getattr(sys, "stdlib_module_names", frozenset())
    mod_path: Path | None = None
    for cls in strategy_cls.__mro__:
        mod_name = cls.__module__
        if mod_name == "builtins" or mod_name.split(".", 1)[0] in stdlib_names:
            continue
        try:
            mod = sys.modules.get(mod_name)
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue
            candidate = Path(mod_file).resolve()
        except Exception:
            continue
        if "site-packages" in candidate.parts:
            continue
        try:
            candidate.relative_to(backtesting_root)
            continue
        except ValueError:
            pass
        mod_path = candidate
        break

    if mod_path is None:
        return base_image

    top = mod_path.parent
    while (top.parent / "__init__.py").exists():
        top = top.parent

    if not (top / "__init__.py").exists():
        return base_image

    return base_image.add_local_dir(str(top), f"/root/{top.name}", copy=True)


def _resolve_image(strategy_cls: type, override: modal.Image | None = None) -> modal.Image:
    if override is not None:
        return override
    if _CONFIG.image is not None:
        return _CONFIG.image
    return _auto_mount(strategy_cls, DEFAULT_IMAGE)


def _worker(payload: bytes) -> bytes:
    """Generic remote bouncer: cloudpickle.loads -> call -> cloudpickle.dumps.

    Ensures `/root` is on sys.path so packages added via auto-mount
    (`add_local_dir(..., "/root/<pkg>", ...)`) are importable when
    cloudpickle rehydrates strategy classes by reference.
    """
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    fn, args = cloudpickle.loads(payload)
    return cloudpickle.dumps(fn(args))


def run_remote(
    image: modal.Image,
    payloads: list[bytes],
    *,
    desc: str | None = None,
) -> Iterator[bytes]:
    """Dispatch cloudpickled (callable, args) payloads to a Modal worker pool.

    Yields raw cloudpickled result bytes in input order. Caller is
    responsible for `cloudpickle.loads` on each item.
    """
    app = modal.App(desc or "backtesting-job")

    function_kwargs: dict[str, Any] = {
        "image": image,
        "timeout": _CONFIG.timeout,
        "min_containers": _CONFIG.min_containers,
    }
    if _CONFIG.cpu is not None:
        function_kwargs["cpu"] = _CONFIG.cpu
    if _CONFIG.memory is not None:
        function_kwargs["memory"] = _CONFIG.memory
    if _CONFIG.max_containers is not None:
        function_kwargs["max_containers"] = _CONFIG.max_containers
    if _CONFIG.secrets:
        function_kwargs["secrets"] = _CONFIG.secrets

    remote_worker = app.function(**function_kwargs)(_worker)

    label = desc or "backtesting-job"
    print(
        f"[backtesting] Fanning out {len(payloads)} parallel invocation(s) "
        f"to Modal (app={label!r}, timeout={_CONFIG.timeout}s)",
        flush=True,
    )
    with app.run():
        yield from remote_worker.map(payloads)


__all__ = ["DEFAULT_IMAGE", "configure", "run_remote"]
