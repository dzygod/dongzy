"""PDF 表格抽取抽象。

设计要点：
* 不直接产出 :class:`LiteratureRecord`；PDF 抽取的产物是"表格二维数组"，而一篇 PDF 里可能有
  多个表格，哪个是 bond-strength 表、哪些列对应 `Ra`/`Rz`/`tensile`，由下游 mapping 决定；
* ``extract(pdf_path) -> list[ExtractedTable]``；
* ``ExtractedTable.confidence`` 来自具体实现（pdfplumber 保守给 2，GROBID 给 3）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExtractedTable:
    """PDF 中抽取到的一个表格。

    ``rows[0]`` 通常是表头；但实现可能返回无表头的纯数值表，由调用方判断。
    """

    source_pdf: str
    page: int
    rows: list[list[str]]
    caption: str | None = None
    confidence: int = 2
    extraction_method: str = "pdf_pdfplumber_table"
    raw: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.rows or all(not any(cell.strip() for cell in row) for row in self.rows)


class PdfExtractor(ABC):
    """所有 PDF 抽取器的抽象基类。"""

    name: str
    extraction_method: str

    @abstractmethod
    def extract(self, pdf_path: str | Path) -> list[ExtractedTable]:  # pragma: no cover - abstract
        ...


__all__ = ["ExtractedTable", "PdfExtractor"]
