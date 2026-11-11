"""基于 pdfplumber 的离线表格抽取。

``pdfplumber`` 是可选依赖（``pip install .[pdf]``）；若未安装则在 ``extract`` 调用时
抛 ``RuntimeError``，这样单测与 CI 默认不强依赖它。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from uhpc_ml_app.literature.pdf.base import ExtractedTable, PdfExtractor

if TYPE_CHECKING:  # pragma: no cover - 仅给类型检查用
    import pdfplumber as _pdfplumber  # noqa: F401


def _load_pdfplumber() -> Any | None:
    try:
        import pdfplumber  # type: ignore[import-not-found]

        return pdfplumber
    except ImportError:
        return None


class PdfPlumberExtractor(PdfExtractor):
    name = "pdfplumber"
    extraction_method = "pdf_pdfplumber_table"

    def __init__(self, min_rows: int = 2) -> None:
        self._min_rows = min_rows

    def extract(self, pdf_path: str | Path) -> list[ExtractedTable]:
        pdfplumber = _load_pdfplumber()
        if pdfplumber is None:
            raise RuntimeError(
                "pdfplumber is not installed; install the [pdf] extra to use PdfPlumberExtractor"
            )

        path = Path(pdf_path)
        tables: list[ExtractedTable] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for tbl in page.extract_tables() or []:
                    cleaned = _clean_table(tbl)
                    if len(cleaned) < self._min_rows:
                        continue
                    tables.append(
                        ExtractedTable(
                            source_pdf=str(path),
                            page=page_index,
                            rows=cleaned,
                            confidence=2,
                            extraction_method=self.extraction_method,
                        )
                    )
        return tables


def _clean_table(tbl: list[list[str | None]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in tbl:
        cleaned.append([(cell or "").strip() for cell in row])
    # 去掉整行空白
    return [r for r in cleaned if any(cell for cell in r)]


__all__ = ["PdfPlumberExtractor"]
