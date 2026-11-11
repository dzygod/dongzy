"""``SourceAdapter`` 抽象基类与 ``LiteratureRecord`` 数据模型。

每个 adapter 的职责是：
* 从自己专有的来源（API、RIS 文件、Scopus CSV、CNKI RefWorks、PDF 表格…）读取 0 或多条记录；
* 把每条记录转换成统一的 ``LiteratureRecord``；
* 不做去重、不做数值归一化——那些属于 pipeline 的下游职责。

这样 pipeline 只需要拿一个 ``list[LiteratureRecord]`` 即可统一去重、单位换算、写入 staging schema。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

# 跟 ``docs/data_protocol.md`` 第 4.1 节保持一致。
SOURCES = (
    "wos",
    "scopus",
    "cnki",
    "openalex",
    "crossref",
    "manual",
    "pdf_grobid",
    "pdf_pdfplumber",
)

EXTRACTION_METHODS = (
    "abstract_regex",
    "table_parser",
    "pdf_grobid_table",
    "pdf_pdfplumber_table",
    "manual_entry",
)


@dataclass(slots=True)
class LiteratureRecord:
    """单篇文献 × 单个试件组合的一行观测。

    ``None`` 表示该字段在来源中缺失；下游 pipeline 负责判断能否进入 CLEAN_SCHEMA。
    """

    # --- 标识 / 来源 ---
    record_id: str
    source: str
    doi: str | None = None
    first_author: str | None = None
    year: int | None = None
    title: str | None = None
    journal: str | None = None

    # --- 试件元信息 ---
    specimen_type: str | None = None
    interface_prep: str | None = None

    # --- 数值特征（建模关心的列）---
    roughness_sa: float | None = None
    water_binder_ratio: float | None = None
    fiber_volume_ratio: float | None = None
    nc_strength_grade: float | None = None
    curing_age_days: float | None = None
    tensile_bond_strength: float | None = None
    shear_bond_strength: float | None = None

    # --- 审计 / 追溯 ---
    extraction_method: str = "abstract_regex"
    confidence: int = 1
    notes: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of {SOURCES}")
        if self.extraction_method not in EXTRACTION_METHODS:
            raise ValueError(
                f"unknown extraction_method {self.extraction_method!r}; expected one of "
                f"{EXTRACTION_METHODS}"
            )
        if not 0 <= self.confidence <= 3:
            raise ValueError(f"confidence must be in 0..3, got {self.confidence}")
        if self.doi:
            self.doi = normalize_doi(self.doi)

    def to_dict(self) -> dict[str, Any]:
        """扁平化 dict；``raw_payload`` 保留为嵌套 dict 便于 JSON 落地。"""
        return asdict(self)


def normalize_doi(doi: str) -> str:
    """标准化 DOI：去前缀、strip、转小写。"""
    s = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s.strip("/")


class SourceAdapter(ABC):
    """每个文献来源都要实现的抽象适配器。

    name :
        全局唯一的标识符；用于 :func:`uhpc_ml_app.literature.registry.get_adapter`。
    source :
        写入 ``LiteratureRecord.source`` 的枚举值，一般与 ``name`` 相同但不要求。
    """

    name: str
    source: str

    @abstractmethod
    def ingest(self, query_or_path: str, **options: Any) -> Iterable[LiteratureRecord]:
        """生成器：从该来源读取 0 或多条记录。

        参数的语义按 adapter 不同：
        * API adapter（OpenAlex / Crossref）：``query_or_path`` 是检索关键字；
        * 文件 adapter（WoS RIS / Scopus CSV / CNKI RefWorks）：``query_or_path`` 是文件路径。
        """


__all__ = [
    "EXTRACTION_METHODS",
    "LiteratureRecord",
    "SOURCES",
    "SourceAdapter",
    "normalize_doi",
]
