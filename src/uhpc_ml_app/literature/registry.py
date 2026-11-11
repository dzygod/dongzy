"""全局 Adapter 注册表。

使用方法::

    from uhpc_ml_app.literature import get_adapter
    adapter = get_adapter("wos_ris")
    for rec in adapter.ingest("savedrecs.ris"):
        ...

新 adapter 在 ``adapters/__init__.py`` 里通过 :func:`register_adapter` 注册；
或在外部调用同名函数以便于下游扩展。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uhpc_ml_app.literature.base import SourceAdapter

_REGISTRY: dict[str, SourceAdapter] = {}


def register_adapter(adapter: SourceAdapter) -> None:
    """把 adapter 登记到全局注册表；重复注册会覆盖。"""
    if not adapter.name:
        raise ValueError("adapter.name must be a non-empty string")
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> SourceAdapter:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"unknown adapter {name!r}; available: {available}")
    return _REGISTRY[name]


def list_adapters() -> list[str]:
    return sorted(_REGISTRY.keys())


def _bootstrap_default_adapters() -> None:
    """在首次 ``import uhpc_ml_app.literature`` 时把内置 adapter 全部登记。"""
    from uhpc_ml_app.literature.adapters.cnki_refworks import CnkiRefWorksAdapter
    from uhpc_ml_app.literature.adapters.crossref import CrossrefAdapter
    from uhpc_ml_app.literature.adapters.openalex import OpenAlexAdapter
    from uhpc_ml_app.literature.adapters.scopus_csv import ScopusCsvAdapter
    from uhpc_ml_app.literature.adapters.wos_ris import WosRisAdapter

    for cls in (
        OpenAlexAdapter,
        CrossrefAdapter,
        WosRisAdapter,
        ScopusCsvAdapter,
        CnkiRefWorksAdapter,
    ):
        register_adapter(cls())


_bootstrap_default_adapters()


__all__ = ["get_adapter", "list_adapters", "register_adapter"]
