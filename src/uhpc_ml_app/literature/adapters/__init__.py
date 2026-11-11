"""内置 :class:`SourceAdapter` 实现集合。"""

from uhpc_ml_app.literature.adapters.cnki_refworks import CnkiRefWorksAdapter
from uhpc_ml_app.literature.adapters.crossref import CrossrefAdapter
from uhpc_ml_app.literature.adapters.openalex import OpenAlexAdapter
from uhpc_ml_app.literature.adapters.scopus_csv import ScopusCsvAdapter
from uhpc_ml_app.literature.adapters.wos_ris import WosRisAdapter

__all__ = [
    "CnkiRefWorksAdapter",
    "CrossrefAdapter",
    "OpenAlexAdapter",
    "ScopusCsvAdapter",
    "WosRisAdapter",
]
