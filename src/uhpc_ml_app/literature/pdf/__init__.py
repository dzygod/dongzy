"""PDF 表格抽取骨架。

- :class:`PdfExtractor` 是 ABC；
- :class:`PdfPlumberExtractor` 纯 Python 可在 CI 跑；
- :class:`GrobidExtractor` 需要本地/远程启动 GROBID 容器，通过 ``GROBID_URL`` 环境变量控制，
  CI 默认不启动。
"""

from uhpc_ml_app.literature.pdf.base import ExtractedTable, PdfExtractor
from uhpc_ml_app.literature.pdf.grobid_extractor import GrobidExtractor
from uhpc_ml_app.literature.pdf.pdfplumber_extractor import PdfPlumberExtractor

__all__ = [
    "ExtractedTable",
    "GrobidExtractor",
    "PdfExtractor",
    "PdfPlumberExtractor",
]
