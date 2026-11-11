# UHPC-NC 修复界面粘结性能预测应用

本项目面向论文主题"多因素耦合下界面粗糙度对 UHPC-NC 修复界面粘结性能演化规律及机理研究"，提供一个可复现的 **"文献数据库 → 建模 → 可解释性 → 工程推荐"** 全链路：

- **PR#5**：PRISMA 2020 文献整合数据库（OpenAlex / Crossref / WoS RIS / Scopus CSV / CNKI RefWorks / PDF pdfplumber / PDF GROBID 六路适配器 + 两段式 pandera schema + 去重 + `ingest_run.json` / `prisma_log.csv` 审计）
- **PR#2**：RF / XGBoost / BPNN + 三基线（均值 / 线性 / `τ = k·√fc`）；BPNN 已改为 `Pipeline(StandardScaler + MLPRegressor)` 保证与树模型公平；`RepeatedKFold(5, 10)` + R² 均值 ± 95 % CI；per-target 指标（拉伸 / 抗剪分开）；跨模型 Permutation Importance
- **PR#3**：双目标 SHAP（TreeExplainer / KernelExplainer 自动派发）+ **依赖曲线最长正效应平台段阈值** + `w/b × 养护龄期` 3×3 条件化推荐 + 降级显式审计（`explanation_method="correlation_fallback"`）
- **PR#4**：PEP 621 打包、GitHub Actions CI（Python 3.10/3.12 矩阵）、两段式 pandera schema（INPUT + CLEAN）、pre-commit（ruff + mypy）、Dockerfile、可选 MLflow
- **PR#6**：`USER-GUIDE.md` 与上述实现对齐
- **PR#7**：Streamlit 面板对接以上全部新产物（6 个 Tab + SHAP 降级告警 + 条件化推荐表）

> ⚠️ **重要说明**：`data/raw/literature_sample.csv` 为**合成示例数据**（数值按随机分布生成、界面处理方式循环取值，仅用于走通程序流程与单测），**禁止将基于该数据集的输出直接用于论文正文、答辩或工程决策**。论文级结果必须基于按 [`docs/data_protocol.md`](docs/data_protocol.md)（PRISMA 流程）整理出的真实文献数据库重新训练。

## 1. 安装

项目采用 `pyproject.toml`（PEP 621），推荐 **editable install**：

```bash
cd ./dongzy
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"          # 含 ruff / mypy / pytest / pandas-stubs
pip install -e ".[dev,pdf]"      # 额外安装 pdfplumber，用于离线 PDF 表格抽取
pre-commit install                # 安装 ruff / mypy pre-commit 钩子
```

> Python 3.10 亦受支持（`requires-python = ">=3.10"`），CI 矩阵覆盖 3.10 / 3.12。

## 2. 运行测试 + 静态检查

```bash
ruff check .
ruff format --check .
mypy src
pytest -q           # 当前 62 条单测
```

CI 流程（lint + mypy + pytest，Python 3.10/3.12 矩阵）见 `.github/workflows/ci.yml.disabled`，本地通过即可直接推 PR。

## 3. 建模 CLI（`uhpc-ml`）

`pip install -e .` 会安装 console script。最小化运行（使用合成示例数据验证管道）：

```bash
uhpc-ml --input data/raw/literature_sample.csv --output outputs/run-sample --seed 42
```

小样本数据可启用**数据增强**（KNN 插值 + 高斯噪声，仅作用于训练集，测试集保持真实）：

```bash
uhpc-ml --input data/raw/literature_sample.csv --output outputs/run-aug --seed 42 --augment
```

真实文献数据集走通后（见 §5）：

```bash
uhpc-ml --input outputs/run-$(date +%F)/literature_clean.csv --output outputs/run-$(date +%F) --seed 42
```

产物清单见 §6。也可按老习惯以模块方式运行：

```bash
python -m uhpc_ml_app.app --input <csv> --output <dir>
```

## 4. Streamlit 可视化面板

```bash
streamlit run streamlit_app.py
```

面板侧栏上传 CSV + 选 `seed` + 点击"开始训练与分析"后，主区按 **6 个 Tab** 展示产物：

| Tab | 对应产物 |
|---|---|
| ① 概览 | `best_model_info.json`（最优模型 / `model_selection_metric` / CV R² 95 % CI / SHAP 降级告警） |
| ② 精度对比 | `metrics_summary.csv` + `metrics_per_target.csv` + `cv_metrics.csv` + `baseline_metrics.csv` |
| ③ 特征重要性 | `permutation_importance.csv` 跨模型透视 + 每模型条形图 |
| ④ 预测散点 | `scatter_{model}_{target}.png` 自动分组 |
| ⑤ SHAP 可解释性 | `shap_summary_{target}.png` + `shap_dependence_{target}_{feature}.png` + `explanation_method` 标注 |
| ⑥ 工程推荐区间 | `roughness_recommendations.json` 全局 + `roughness_conditional_recommendations.csv` 条件化 |

所有 CSV / JSON / PNG 均提供单独下载按钮，底部"一键下载全部产物 (ZIP)"打包 `outputs/` 目录。

## 5. 真实文献数据库（PRISMA CLI：`literature-ingest`）

**推荐主入口**。从多路导出（OpenAlex / Crossref / WoS RIS / Scopus CSV / CNKI RefWorks / 本地 PDF）一次走完 STAGING → CLEAN：

```bash
literature-ingest \
  --plan openalex:"UHPC NC bond roughness" \
  --plan crossref:"UHPC NC interface bond strength" \
  --plan wos_ris:data/raw/wos_export.ris \
  --plan scopus_csv:data/raw/scopus_export.csv \
  --plan cnki_refworks:data/raw/cnki_refworks.txt \
  --plan pdf_pdfplumber:data/raw/pdfs/ \
  --output-dir outputs/run-$(date +%F)
```

输出目录一览（详见 [`docs/data_protocol.md`](docs/data_protocol.md) 与 [`USER-GUIDE.md`](USER-GUIDE.md)）：

```
outputs/run-YYYY-MM-DD/
├── ingest_run.json                       # 运行元数据（每路 adapter 纳入/排除/去重统计）
├── prisma_log.csv                        # PRISMA 2020 审计（identified / screened / eligible / included）
├── deduplication_log.csv                 # 去重审计（同 DOI / 近似标题 + 年份）
├── staging.parquet                       # STAGING 宽表（20 列 + source + confidence 0–3）
└── literature_clean.csv                  # CLEAN 宽表（物理上下界裁剪后，直接喂给 `uhpc-ml`）
```

**兼容保留**：旧单路抓取脚本 `python -m uhpc_ml_app.literature_crawler` 仍可用，但已不再是推荐入口（不带 PRISMA 审计与去重）。

## 6. 建模产物（`outputs/run-*/`）

| 文件 | 内容 |
|---|---|
| `metrics_summary.csv` | 每模型 overall train / test 指标 **+ CV R² 均值与 95 % CI** |
| `metrics_per_target.csv` | 每模型 × 每目标（tensile / shear）× train / test 的 R²/RMSE/MAE |
| `cv_metrics.csv` | `RepeatedKFold(5, 10)` 详细 CV 记录 |
| `baseline_metrics.csv` | 均值 / 线性 / `τ = k·√fc` 基线对照 |
| `permutation_importance.csv` | 跨模型统一口径的 Permutation Importance（R² 下降量） |
| `permutation_importance_{model}.png` | 每模型 PI 条形图 |
| `scatter_{model}_{target}.png` | 实测-预测散点图 |
| `shap_summary_{target}.png` | 双目标 SHAP summary beeswarm |
| `shap_dependence_{target}_{feature}.png` | SHAP 依赖曲线（用于阈值推导） |
| `roughness_recommendations.json` | 每 `(target, feature)` 全局推荐区间 + 条件化 + `explanation_method` + `failed_targets` |
| `roughness_conditional_recommendations.csv` | `w/b × 养护龄期` 3×3 桶的条件化阈值表 |
| `best_model_info.json` | 顶层摘要（供 Streamlit 概览 Tab 使用） |
| `ingest_run.json`（若来自 §5） | PRISMA 摄入审计 |

## 7. 输入数据字段规范

必需列（CSV / Parquet）：

- 连续数值特征（4 列必填）：`water_binder_ratio`（水胶比）、`fiber_volume_ratio`（纤维体积率）、`nc_strength_grade`（NC 混凝土强度等级）、`curing_age_days`（养护龄期）
- 分类特征（2 列必填，建模前 one-hot）：
  - `interface_prep`：界面处理方式，取值 `C` 未处理 / `SB` 喷砂 / `HW` 高压水射流 / `CH` 凿毛/机械凿毛 / `M` 机械铣刨，缺失或其他值归入基准类别；
  - `specimen_type`：试件/试验类型，取值 `pull_off` / `slant_shear` / `splitting` / `bi_shear` / `direct_shear` / `direct_tension` / `bending` / `mixed`，缺失或其他值归入基准类别
- 界面粗糙度（可选稀疏数值特征）：`roughness_mm` 宏观纹理深度（mm），文献中仅少数记录给出数值，缺失时用哨兵值 `-1` 标记（不做中位数填补、不参与 IQR 裁剪）；`roughness_sa`（三维面粗糙度 Sa，μm）因已纳入文献中近似为常数、无信息量，不再作为建模特征（列仍保留供后续补充实测数据）
- 输出目标（至少填其一）：`tensile_bond_strength`（拉伸）、`shear_bond_strength`（抗剪）；允许样本只含拉伸或只含抗剪，系统将分别独立建模

可选列（强烈建议保留，便于溯源与复现）：`doi`、`source`、`confidence`、`reference`。

各列的物理上下界见 [`src/uhpc_ml_app/schemas.py`](src/uhpc_ml_app/schemas.py) 与 [`src/uhpc_ml_app/literature/schemas.py`](src/uhpc_ml_app/literature/schemas.py)；CLEAN schema 超界会被裁剪或记为 NaN，杜绝把野值喂进模型。

## 8. Docker 运行

```bash
docker build -t uhpc-ml-app .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
  uhpc-ml-app uhpc-ml --input data/raw/literature_sample.csv --output outputs/run-docker
```

Streamlit：

```bash
docker run --rm -p 8501:8501 uhpc-ml-app streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 9. 可选：MLflow 实验跟踪

设置 `MLFLOW_TRACKING_URI` 后，`uhpc-ml` 会把 run 参数 / 指标 / 产物同步到 MLflow。tracking server 不可达时会**静默降级**（只打 warning），不阻塞主管道：

```bash
export MLFLOW_TRACKING_URI=file:./mlruns   # 或 http://<mlflow-host>:5000
uhpc-ml --input outputs/run-*/literature_clean.csv --output outputs/run-latest
```

## 10. 论文复现（小论文 1）

论文核心结果（抗剪为主，pooled OOF R² ≈ 0.81）由 [`scripts/reproduce_paper1.py`](scripts/reproduce_paper1.py) 复现：

```bash
# 真实数据（需按 §5 整理，或向通讯作者合理请求，见 §11）
python scripts/reproduce_paper1.py --csv <path>/uhpc_nc_bond_literature_data_v6.csv --out outputs/run-paper1

# 合成示例（仅验证脚本可运行，禁止用于论文结论）
python scripts/reproduce_paper1.py
```

脚本固定 XGBoost 超参（max_depth=5 / lr=0.02 / n_estimators=300 / subsample=0.8 / colsample_bytree=0.8）、5 折 × 30 次重复 CV、pooled OOF R² 口径，与论文一致；输出 `metrics_table.csv` / `metrics_summary.json` / 散点 / SHAP / 特征重要性图。

> 论文建模特征为 `interface_prep` + `specimen_type` 的 one-hot 与 `roughness_mm` 稀疏特征。
> 通用 `uhpc-ml` CLI / Streamlit 面板中基于 `roughness_sa` 的"粗糙度推荐区间"模块为早期
> 探索性功能，论文结论以本复现脚本为准。

## 11. License

本仓库**仅开源代码与可复现流水线**，采用 [MIT License](./LICENSE)。

**说明**：

- **代码与流水线**（`src/`、`tests/`、脚本与配置）：MIT 许可，可自由使用、修改与再分发，需保留版权声明。
- **整合文献数据库条目**：版权归原文献作者及出版商；本仓库不附带完整文献整合表。真实数据需按 [`docs/data_protocol.md`](docs/data_protocol.md) 自行整理，或向通讯作者合理请求。
- **合成示例**（`data/raw/literature_sample.csv`）：仅用于跑通流程与单元测试，**不得**作为论文结果依据。
