from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cloudpickle
import modal

DEFAULT_IMAGE: modal.Image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "numpy",
    "pandas",
    "bokeh",
    "scipy",
    "scikit-learn",
    "joblib",
    "cloudpickle",
)


def _worker(payload: bytes) -> bytes:
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    fn, args = cloudpickle.loads(payload)
    return cloudpickle.dumps(fn(args))


class RemoteExecutor:
    def __init__(
        self,
        *,
        image: modal.Image | None = None,
        secrets: list[modal.Secret] | None = None,
        cpu: float | None = None,
        memory: int | None = None,
        min_containers: int = 0,
        max_containers: int | None = None,
        timeout: int = 3600,
    ):
        self.configure(
            image=image,
            secrets=secrets,
            cpu=cpu,
            memory=memory,
            min_containers=min_containers,
            max_containers=max_containers,
            timeout=timeout,
        )

    def configure(
        self,
        *,
        image: modal.Image | None = None,
        secrets: list[modal.Secret] | None = None,
        cpu: float | None = None,
        memory: int | None = None,
        min_containers: int = 0,
        max_containers: int | None = None,
        timeout: int = 3600,
    ) -> None:
        self.image = image
        self.secrets: list[modal.Secret] = list(secrets) if secrets else []
        self.cpu = cpu
        self.memory = memory
        self.min_containers = min_containers
        self.max_containers = max_containers
        self.timeout = timeout

    def resolve_image(self, strategy_cls: type, override: modal.Image | None = None) -> modal.Image:
        if override is not None:
            return override
        if self.image is not None:
            return self.image
        return self._auto_mount(strategy_cls, DEFAULT_IMAGE)

    @staticmethod
    def _auto_mount(strategy_cls: type, base_image: modal.Image) -> modal.Image:
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

    def run(
        self,
        image: modal.Image,
        payloads: list[bytes],
        *,
        desc: str,
    ) -> Iterator[bytes]:
        app = modal.App(desc)

        function_kwargs: dict[str, Any] = {
            "image": image,
            "timeout": self.timeout,
            "min_containers": self.min_containers,
        }
        if self.cpu is not None:
            function_kwargs["cpu"] = self.cpu
        if self.memory is not None:
            function_kwargs["memory"] = self.memory
        if self.max_containers is not None:
            function_kwargs["max_containers"] = self.max_containers
        if self.secrets:
            function_kwargs["secrets"] = self.secrets

        remote_worker = app.function(**function_kwargs)(_worker)

        print(
            f"[backtesting] Fanning out {len(payloads)} parallel invocation(s) "
            f"to Modal (app={desc!r}, timeout={self.timeout}s)",
            flush=True,
        )
        with app.run():
            yield from remote_worker.map(payloads)


_DEFAULT_EXECUTOR = RemoteExecutor()


def default_executor() -> RemoteExecutor:
    return _DEFAULT_EXECUTOR


__all__ = ["DEFAULT_IMAGE", "RemoteExecutor", "default_executor"]
