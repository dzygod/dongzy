"""UHPC-NC 修复界面粘结数据库的文献采集 / 抽取 / 归一化子包。

协议参见 ``docs/data_protocol.md``。
"""

from uhpc_ml_app.literature.base import LiteratureRecord, SourceAdapter
from uhpc_ml_app.literature.registry import get_adapter, list_adapters, register_adapter

__all__ = [
    "LiteratureRecord",
    "SourceAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
