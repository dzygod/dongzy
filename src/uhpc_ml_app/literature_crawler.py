from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import requests

DEFAULT_QUERY = (
    "UHPC NC repair interface bond roughness surface preparation "
    "water binder ratio fiber volume curing age tensile shear"
)

# 与 literature_sample.csv 对齐的建模列顺序
MODEL_FIELDNAMES = [
    "roughness_sa",
    "water_binder_ratio",
    "fiber_volume_ratio",
    "nc_strength_grade",
    "curing_age_days",
    "interface_prep",
    "tensile_bond_strength",
    "shear_bond_strength",
]


def fetch_openalex(query: str, per_page: int = 100, pages: int = 2) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    base_url = "https://api.openalex.org/works"
    for page in range(1, pages + 1):
        params = {
            "search": query,
            "per-page": per_page,
            "page": page,
            "filter": "type:article|dissertation",
        }
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for work in data.get("results", []):
            items.append(
                {
                    "title": work.get("display_name", ""),
                    "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
                    "year": work.get("publication_year"),
                    "journal": (work.get("primary_location") or {})
                    .get("source", {})
                    .get("display_name", ""),
                    "source": "openalex",
                    "url": work.get("id", ""),
                    "abstract": _rebuild_openalex_abstract(work.get("abstract_inverted_index")),
                }
            )
    return items


def _rebuild_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    token_positions: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for pos in positions:
            token_positions.append((pos, token))
    token_positions.sort(key=lambda x: x[0])
    return " ".join(token for _, token in token_positions)


def fetch_crossref(query: str, rows: int = 200) -> list[dict[str, Any]]:
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [normalize_crossref_item(i) for i in data.get("message", {}).get("items", [])]


def normalize_crossref_item(item: dict[str, Any]) -> dict[str, Any]:
    title_list = item.get("title", []) or [""]
    journal_list = item.get("container-title", []) or [""]
    year = _parse_crossref_year(item)
    return {
        "title": title_list[0],
        "doi": item.get("DOI", "") or "",
        "year": year,
        "journal": journal_list[0],
        "source": "crossref",
        "url": item.get("URL", "") or "",
        "abstract": item.get("abstract", "") or "",
    }


def _parse_crossref_year(item: dict[str, Any]) -> int | None:
    candidates = ["published-print", "published-online", "issued", "created"]
    for key in candidates:
        obj = item.get(key, {})
        parts = obj.get("date-parts", [])
        if parts and parts[0] and parts[0][0] is not None:
            return int(parts[0][0])
    return None


def is_relevant_paper(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    must_keywords = ["uhpc", "bond"]
    optional_keywords = [
        "roughness",
        "ra",
        "sa",
        "interface",
        "surface",
        "water-binder",
        "fiber",
        "curing",
        "shear",
        "tensile",
        "repair",
        "nc",
    ]
    return all(k in text for k in must_keywords) and any(k in text for k in optional_keywords)


def extract_structured_features(text: str) -> dict[str, Any]:
    content = text.replace("\n", " ")
    nc_strength = _extract_nc_strength_grade(content)
    return {
        "roughness_sa": _extract_first_float(content, r"\bSa\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)"),
        "water_binder_ratio": _extract_first_float(
            content, r"(?:water[-\s]?binder ratio|w/b)\s*(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"
        ),
        "fiber_volume_ratio": _extract_first_float(
            content,
            r"(?:fiber volume fraction|fiber volume ratio)\s*(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
        ),
        "nc_strength_grade": nc_strength,
        "curing_age_days": _extract_first_float(
            content, r"(?:curing age|age)\s*(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:d|day|days)\b"
        ),
        "interface_prep": _extract_interface_prep(content),
        "tensile_bond_strength": _extract_first_float(
            content, r"(?:tensile bond strength)\s*(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"
        ),
        "shear_bond_strength": _extract_first_float(
            content, r"(?:shear bond strength)\s*(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?)"
        ),
    }


_INTERFACE_PREP_PATTERNS: dict[str, str] = {
    "C": r"untreated|as[\s-]?cast|smooth|成膜|未处理|光滑",
    "SB": r"sand[\s-]?blast\w*|喷砂",
    "HW": r"hydro[\s-]?demolit\w*|water[\s-]?jet|高压水射流|水射流",
    "CH": r"chipp\w*|凿毛|机械凿毛",
    "M": r"mill\w*|铣刨|机械铣刨",
}


def _extract_interface_prep(text: str) -> str | None:
    """从摘要/正文识别界面处理方式，返回标准编号枚举值；识别不到返回 None。"""
    lowered = text.lower()
    for value, pattern in _INTERFACE_PREP_PATTERNS.items():
        if re.search(pattern, lowered):
            return value
    return None


def _extract_first_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _extract_nc_strength_grade(text: str) -> float | None:
    # 优先匹配常见写法：C40、C50 等
    m = re.search(r"\bC\s*([0-9]{2})\b", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # 备选：NC compressive strength = 50 MPa
    return _extract_first_float(
        text,
        r"(?:nc compressive strength|concrete compressive strength|compressive strength)\s*(?:was|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*MPa",
    )


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in records:
        doi = (r.get("doi") or "").strip().lower()
        title = (r.get("title") or "").strip().lower()
        key = f"doi:{doi}" if doi else f"title:{title}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def _count_filled_fields(row: dict[str, Any]) -> int:
    return sum(1 for v in row.values() if v is not None and v != "")


def build_literature_database(
    query: str,
    output_csv: Path,
    pages: int = 2,
    min_filled: int = 0,
) -> None:
    all_records = fetch_openalex(query=query, pages=pages) + fetch_crossref(query=query, rows=200)
    all_records = deduplicate_records(all_records)

    filtered = [
        r for r in all_records if is_relevant_paper(r.get("title", ""), r.get("abstract", ""))
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        # 只输出建模所需列，保证与 sample.csv 完全同构，可直接被系统读取
        writer = csv.DictWriter(f, fieldnames=MODEL_FIELDNAMES)
        writer.writeheader()
        written_count = 0
        for rec in filtered:
            features = extract_structured_features(rec.get("abstract", ""))
            row = {k: features.get(k) for k in MODEL_FIELDNAMES}
            if _count_filled_fields(row) >= min_filled:
                writer.writerow(row)
                written_count += 1

    manifest = {
        "query": query,
        "total_fetched": len(all_records),
        "total_relevant": len(filtered),
        "total_written": written_count,
        "min_filled": min_filled,
        "output_csv": str(output_csv),
        "note": (
            "WoS/Scopus/Engineering Village/CNKI 多为受限数据库，"
            "建议通过机构授权 API 或人工补录后合并本表。"
        ),
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="UHPC-NC 文献整合数据库采集器（合规 API 版）")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="检索关键词")
    parser.add_argument("--output", default="data/raw/literature_real.csv", help="输出 CSV 路径")
    parser.add_argument("--pages", type=int, default=2, help="OpenAlex 页数")
    parser.add_argument("--min-filled", type=int, default=0, help="最少非空特征字段数（0-8）")
    args = parser.parse_args()
    if args.min_filled < 0 or args.min_filled > len(MODEL_FIELDNAMES):
        raise ValueError(f"--min-filled 必须在 0 到 {len(MODEL_FIELDNAMES)} 之间")
    build_literature_database(
        args.query,
        Path(args.output),
        pages=args.pages,
        min_filled=args.min_filled,
    )
    print(f"文献数据采集完成: {args.output}")


if __name__ == "__main__":
    main()
