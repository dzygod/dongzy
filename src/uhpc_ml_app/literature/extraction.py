"""从摘要 / 正文文本回归抽取数值特征的正则工具。

原先写在 ``uhpc_ml_app.literature_crawler`` 里；PR#5 把它搬出来独立，使得任何 adapter
都可以在拿到 abstract/full_text 后调用同一套抽取逻辑。

所有公开的抽取函数返回 ``float | None``；``None`` 表示"没抽到"，不是 0。
"""

from __future__ import annotations

import re
from typing import Any

_NC_COMPRESSIVE_PATTERN = re.compile(
    r"(?:nc compressive strength|concrete compressive strength|compressive strength)"
    r"\s*(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*MPa",
    re.IGNORECASE,
)

_NC_GRADE_PATTERN = re.compile(r"\bC\s*([0-9]{2})\b", re.IGNORECASE)

# 公共连接词：允许 "was / = / : / of / reached / with / about / approximately" 出现在字段名与数字之间。
_JOINER = r"(?:\s*(?:was|were|=|:|of|reached|with|about|approximately|approx\.?|around))*\s*"

_FIELD_PATTERNS: dict[str, str] = {
    "roughness_sa": r"\bSa\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
    "water_binder_ratio": (
        r"(?:water[-\s]?binder(?:\s+ratio)?|w/b)" + _JOINER + r"([0-9]+(?:\.[0-9]+)?)"
    ),
    "fiber_volume_ratio": (
        r"(?:fiber volume fraction|fiber volume ratio)" + _JOINER + r"([0-9]+(?:\.[0-9]+)?)"
    ),
    "curing_age_days": (
        r"(?:curing age|age)" + _JOINER + r"([0-9]+(?:\.[0-9]+)?)\s*(?:d|day|days)\b"
    ),
    "tensile_bond_strength": (r"(?:tensile bond strength)" + _JOINER + r"([0-9]+(?:\.[0-9]+)?)"),
    "shear_bond_strength": (
        r"(?:shear bond strength|抗剪粘结强度)" + _JOINER + r"([0-9]+(?:\.[0-9]+)?)"
    ),
}

NUMERIC_FIELDS = (
    "roughness_sa",
    "water_binder_ratio",
    "fiber_volume_ratio",
    "nc_strength_grade",
    "curing_age_days",
    "tensile_bond_strength",
    "shear_bond_strength",
)


def _first_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def extract_nc_strength_grade(text: str) -> float | None:
    m = _NC_GRADE_PATTERN.search(text)
    if m:
        return float(m.group(1))
    return _first_float(text, _NC_COMPRESSIVE_PATTERN.pattern)


def extract_numeric_features(text: str) -> dict[str, Any]:
    """一次性抽取 8 列数值特征；适配器把摘要/正文喂进来即可。

    返回值声明为 ``dict[str, Any]`` 以便直接 ``**numeric`` 展开到
    :class:`LiteratureRecord` 构造参数（混合 str/float/None 的关键字形参）。
    """
    if not text:
        return dict.fromkeys(NUMERIC_FIELDS)
    content = text.replace("\n", " ")
    out: dict[str, Any] = {}
    for field_name, pattern in _FIELD_PATTERNS.items():
        out[field_name] = _first_float(content, pattern)
    out["nc_strength_grade"] = extract_nc_strength_grade(content)
    return out


__all__ = [
    "NUMERIC_FIELDS",
    "extract_nc_strength_grade",
    "extract_numeric_features",
]
