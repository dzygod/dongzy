# UHPC-NC 修复界面粘结数据库——文献筛选与数据提取协议

> 本文档是 `simpledong81/dongzy` 仓库 PR#5 的配套协议文件，对标 **PRISMA 2020** 报告规范与系统综述数据提取的常规做法，用于约束"从文献到宽表"这一条流水线的全部决策。
>
> 若协议与代码不一致，**以协议为准**；代码需整改以满足协议。

## 1. 研究问题 (PICO)

| 维度 | 内容 |
|---|---|
| **P**opulation | UHPC 作为修复层、普通混凝土（NC）作为基层的复合构件 |
| **I**ntervention / 输入 | 界面粗糙度（三维 `Sa` 或界面处理方式 `interface_prep`）；UHPC 配合比（`w/b`、纤维体积率 `fiber_volume_ratio`）；NC 强度等级；养护龄期 `curing_age_days`；界面处理工艺；试件几何 |
| **C**omparator | 光滑界面 / 低粗糙度界面；不同 UHPC 配方；不同龄期 |
| **O**utcome | 界面**拉伸粘结强度** `tensile_bond_strength` (MPa)；**抗剪粘结强度** `shear_bond_strength` (MPa) |
| **S**tudy types | 试验性研究；含量化数据的学位论文章节；排除纯数值模拟 |

## 2. 文献检索策略

### 2.1 数据库覆盖

| 来源 | 形态 | 采集方式 | 覆盖期 | 适配器 |
|---|---|---|---|---|
| Web of Science | 关键词 → RIS 导出 | `wos_ris` adapter | 2000–至今 | `adapters.wos_ris.WosRisAdapter` |
| Scopus | 关键词 → CSV 导出 | `scopus_csv` adapter | 2000–至今 | `adapters.scopus_csv.ScopusCsvAdapter` |
| Engineering Village (Compendex/Inspec) | 关键词 → RIS 导出 | `wos_ris` adapter（兼容 RIS） | 2005–至今 | `adapters.wos_ris.WosRisAdapter` |
| CNKI | 主题词 → RefWorks/NoteExpress 导出 | `cnki_refworks` adapter | 2005–至今 | `adapters.cnki_refworks.CnkiRefWorksAdapter` |
| OpenAlex | 公共 API | `openalex` adapter | 全库 | `adapters.openalex.OpenAlexAdapter` |
| Crossref | 公共 API | `crossref` adapter | 全库 | `adapters.crossref.CrossrefAdapter` |

> **注**：WoS/Scopus/EV/CNKI 需要订阅访问，代码不尝试直接爬网，而是读取用户手动导出的离线文件。OpenAlex/Crossref 公共 API 免登录。

### 2.2 检索式（英文主库）

```
(TS=("ultra high performance concrete" OR "UHPC") AND
 TS=("normal concrete" OR "NC" OR "ordinary Portland") AND
 TS=("bond" OR "adhesion" OR "interface") AND
 TS=("roughness" OR "Sa" OR "surface preparation" OR "sandblast*" OR "hydrodemolition" OR "chipp*" OR "mill*" OR "untreated"))
```

CNKI 使用等价的中文主题词：`(UHPC OR 超高性能混凝土) AND (界面 OR 修复) AND (粘结 OR 剪切 OR 拉伸) AND (粗糙度 OR 界面处理)`

### 2.3 时间窗与语言

- **时间窗**：2000-01 至执行检索日（首次 PR#5 执行日应记录在 `prisma_log.csv`）。
- **语言**：英文、中文；其他语言暂排除。

## 3. 筛选标准

### 3.1 纳入标准 (inclusion)

1. 研究对象明确含 UHPC/NC 界面；
2. 至少报告一项粘结强度结果（拉伸 or 抗剪）；
3. 至少报告界面粗糙度指标（三维 `Sa`），或以"喷砂 sandblast / 高压水射流 hydrodemolition / 凿毛 chipping / 铣刨 milling / 未处理 untreated"等可映射到离散粗糙度的处理方法；
4. 报告 UHPC 配比的 `w/b` 或 `fiber_volume_ratio` 中至少一项；
5. 报告养护龄期；
6. 单位可换算到 `MPa`、`μm`、`day`、`%` 的 SI 派生单位。

### 3.2 排除标准 (exclusion)

- **E1**：纯数值模拟、无实测数据；
- **E2**：只报告整体强度、未报告界面粘结强度；
- **E3**：界面修补用的非 UHPC 材料（普通砂浆/ECC/聚合物砂浆等）；
- **E4**：数据以图线呈现但无数字表，且经 2 次独立尝试仍无法从图中数位化；
- **E5**：综述、评述、勘误、撤稿。

排除原因必须记录到 `prisma_log.csv` 的 `exclusion_reason` 列（E1–E5）。

### 3.3 去重

- 以 `normalize(DOI)` 作为主键去重；无 DOI 的记录用 `(first_author_lastname, publication_year, title_normalized[:80])` 作为备用键；
- 同一论文多条记录（OpenAlex + Crossref）保留 `source_priority` 最高的一条，优先级 **WoS > Scopus > CNKI > Crossref > OpenAlex**；
- 重复记录写入 `deduplication_log.csv`。

## 4. 数据提取规则

### 4.1 宽表字段（参见 <ref_file file="src/uhpc_ml_app/literature/schemas.py" />）

| 字段 | 单位 | 说明 |
|---|---|---|
| `record_id` | – | 主键，形如 `{source}:{doi-or-hash}:{row_idx}` |
| `source` | – | `wos` / `scopus` / `cnki` / `openalex` / `crossref` / `manual` / `pdf_grobid` / `pdf_pdfplumber` |
| `doi` | – | 小写、去掉 `https://doi.org/` 前缀后的纯 DOI 串 |
| `first_author` | – | 通讯作者 lastname |
| `year` | – | 出版年 (`int`) |
| `title` | – | 原始题目 |
| `journal` | – | 期刊名 |
| `specimen_type` | – | `slant_shear` / `splitting_tensile` / `direct_tensile` / `pull_off` / `push_out` / `other` |
| `interface_prep` | – | 编号 `C`（未处理）/ `SB`（喷砂）/ `HW`（高压水射流）/ `CH`（凿毛）/ `M`（铣刨），其余归基准（建模输入，one-hot 编码） |
| `roughness_sa` | μm | 三维面粗糙度 Sa（可选数值输入，与 `interface_prep` 二选一或同时提供） |
| `water_binder_ratio` | – | UHPC 的水胶比，范围 0.08–0.60 |
| `fiber_volume_ratio` | % (体积) | 0–6，超过 6% 标记为异常值需复核 |
| `nc_strength_grade` | MPa | 普通混凝土抗压强度等级（转为数值） |
| `curing_age_days` | d | 0.5–400 |
| `tensile_bond_strength` | MPa | 0–20 |
| `shear_bond_strength` | MPa | 0–40 |
| `extraction_method` | – | `abstract_regex` / `table_parser` / `pdf_grobid_table` / `pdf_pdfplumber_table` / `manual_entry` |
| `confidence` | 0/1/2/3 | 见 §4.3 |
| `notes` | – | 附加说明（如图数字化、单位换算记录） |
| `raw_payload_json` | – | 保留原始响应/行原文，便于审计 |

### 4.2 单位换算约定

| 原始 | 目标 | 换算 |
|---|---|---|
| kgf/cm²、MPa、psi → MPa | `1 kgf/cm² ≈ 0.0980665 MPa`；`1 psi ≈ 0.00689476 MPa` |
| mm Sa | μm | ×1000 |
| C25/C30/C40 … | MPa (28 d 立方体) | 查 GB/T 50010 附表 |
| months → days | | ×30 |

换算方式记录于 `notes` 列。

### 4.3 置信度评分

| 分数 | 条件 |
|---|---|
| 3 | 全部关键字段直接来自论文表格；`extraction_method` ∈ {`table_parser`, `pdf_grobid_table`, `manual_entry`} |
| 2 | 关键字段之一来自 `pdf_pdfplumber_table`（自动表格抽取，但已人工核对） |
| 1 | 存在从正文/图线数字化；或单位换算不含糊；`extraction_method` ∈ {`abstract_regex`} |
| 0 | 未通过 `CLEAN_SCHEMA`（仅进入 staging 表，不进入分析） |

默认阈值：进入 `literature_clean.csv` 的记录必须 `confidence >= 2`。

## 5. 管道与代码映射

```
docs/data_protocol.md  ←  本文档（协议）
src/uhpc_ml_app/literature/
├── schemas.py                     # pandera 宽表 schema（兼容 confidence & raw_payload）
├── base.py                        # SourceAdapter ABC、LiteratureRecord dataclass
├── registry.py                    # get_adapter(name) / list_adapters()
├── adapters/
│   ├── openalex.py                # API 适配器
│   ├── crossref.py                # API 适配器
│   ├── wos_ris.py                 # WoS / EV RIS 文件
│   ├── scopus_csv.py              # Scopus CSV
│   └── cnki_refworks.py           # CNKI RefWorks (.ris)
├── pdf/
│   ├── base.py                    # PdfExtractor ABC
│   ├── pdfplumber_extractor.py    # 表格抽取
│   └── grobid_extractor.py        # GROBID 服务客户端（可选）
└── cli.py                         # literature-ingest CLI
```

### 5.1 运行流程

```
           ┌────────────┐  ┌────────────┐  ┌────────────┐
user 导出→ │  RIS/CSV   │  │ OpenAlex   │  │ Crossref   │
           └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
                 ▼               ▼              ▼
           SourceAdapter.ingest() → LiteratureRecord[]
                              │
                              ▼
                      deduplication & priority
                              │
                              ▼
                  staging_records.parquet (raw + metadata)
                              │
   ┌──────────────────────────┴──────────────────────────┐
   ▼                                                     ▼
abstract_regex 抽取数值                        PdfExtractor（手动触发）
   │                                                     │
   └──────────────────────────┬──────────────────────────┘
                              ▼
                     LITERATURE_STAGING_SCHEMA
                  （宽表，可能含 NaN / 低置信度）
                              │
              confidence >= 2 & 单位归一化
                              ▼
                     LITERATURE_CLEAN_SCHEMA
              （进入建模的 data/clean/literature.csv）
```

### 5.2 PDF 管道说明

- **pdfplumber** 路径：完全离线、纯 Python，提取论文内表格。CI 会跑一个小测（用 `reportlab` 生成的合成 PDF）。
- **GROBID** 路径：需要用户本地或服务器起 `lfoppiano/grobid` 容器，通过 `GROBID_URL` 环境变量指向；CI 默认**不启动** GROBID 服务，只测试 HTTP 客户端的拼 URL / 解析逻辑（用 `responses` mock）。

## 6. 审计日志

每次执行 `literature-ingest` 都必须生成：

- `prisma_log.csv`：每条记录的 ID、来源、是否纳入、排除原因（E1–E5）；
- `deduplication_log.csv`：被丢弃的重复记录及原因；
- `ingest_run.json`：执行时间、各适配器的计数、协议版本号。

这三个文件与 `literature_clean.csv` 一起构成**可审计**的最小证据集。

## 7. 版本

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1 | 2026-04-22 | PR#5 首次落地；PRISMA 流程、6 个 source、PDF 双路径、confidence 0–3 评分 |

---

*本协议未盖过的边界情况，请在 GitHub Issue 记录并附原始例子，由协议维护者合并入 v0.2+。*
