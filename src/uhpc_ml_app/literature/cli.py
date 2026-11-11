"""literature-ingest CLI。

示例::

    python -m uhpc_ml_app.literature.cli \
        --plan wos_ris:data/raw/wos_export.ris \
        --plan scopus_csv:data/raw/scopus_export.csv \
        --plan cnki_refworks:data/raw/cnki_export.ris \
        --plan openalex:"UHPC NC bond roughness" \
        --output-dir data/literature/run-2026-04-22
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uhpc_ml_app.literature import list_adapters
from uhpc_ml_app.literature.pipeline import run_ingest


def _parse_plan(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(f"plan must be 'adapter_name:query_or_path', got {raw!r}")
    adapter_name, rest = raw.split(":", 1)
    if adapter_name not in list_adapters():
        raise argparse.ArgumentTypeError(
            f"unknown adapter {adapter_name!r}; available: {', '.join(list_adapters())}"
        )
    return adapter_name, rest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="literature-ingest",
        description="从多来源 adapter 采集 → staging → clean 的流水线",
    )
    parser.add_argument(
        "--plan",
        action="append",
        required=True,
        type=_parse_plan,
        help="格式 adapter_name:query_or_path；可重复",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录（staging parquet、prisma_log、ingest_run.json）",
    )
    args = parser.parse_args(argv)

    manifest = run_ingest(args.plan, output_dir=args.output_dir)
    print(
        "ingest 完成: "
        f"{manifest['total_kept_after_dedup']}/{manifest['total_raw']} 条进入 staging, "
        f"{manifest['clean_rows']} 条进入 clean"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 启动
    raise SystemExit(main())
