# USER GUIDE

本文档详解本仓库核心模块的实现细节与命令行用法，对齐论文研究流程。本文档状态：**已对齐 PR#1~#5 的当前主干实现**（PRISMA + SourceAdapter + 双目标 SHAP + Repeated K-Fold + 基线 + pandera + CI）。

---

> ⚠️ **样本数据警告**
>
> 本仓库自带 `data/raw/literature_sample.csv` 为**合成示例数据**（数值按随机分布生成、界面处理方式循环取值），仅用于走通程序流程与单元测试。**切勿将其产出直接用于论文正文、答辩或工程决策。** 论文真实结果必须基于按 [`docs/data_protocol.md`](docs/data_protocol.md)（PRISMA 协议）整理的文献数据库。

## 1. 整体流程图（从数据到结论）

```
 ┌────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │ A. 文献整合数据库         │  →   │ B. ML 建模与交叉验证        │  →   │ C. 可解释性与工程推荐        │
 │ literature-ingest        │      │ uhpc-ml-app (run_pipeline) │      │ 双目标 SHAP + 依赖曲线     │
 │ (PRISMA, 6 来源, pandera)│      │ RF/XGB/BPNN + 3 基线       │      │ 最长正效应平台 + 条件化     │
 │ STAGING → CLEAN          │      │ RepeatedKFold 5×10 + 95%CI │      │ 失败显式降级                │
 └────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
```

对应论文研究内容四个关键动作：

1. **文献数据整合与预处理**（PRISMA 检索 → 宽表 → 两段式 pandera 校验，见 §2 & §10）
2. **RF / XGBoost / BP 三模型构建 + 基线对照**（`StandardScaler` 公平化、GridSearchCV 调参、Repeated K-Fold 95 % CI 选模型，见 §3）
3. **精度指标对比 + 可视化验证 + 跨模型重要性**（overall / per-target / CV / Permutation Importance，见 §3 & §4）
4. **SHAP 解释 + 粗糙度工程推荐区间**（双目标、TreeExplainer/KernelExplainer 自动派发、依赖曲线最长正效应平台段、w/b × 龄期条件化，见 §4）

---

## 2. `data_pipeline.py` + `schemas.py` 详解

### 2.1 入口：`load_and_clean_data(csv_path)`

承担建模前的全部数据准备工作。

#### 步骤 A：pandera INPUT_SCHEMA 入口校验（PR#4 新增）

- [`src/uhpc_ml_app/schemas.py`](src/uhpc_ml_app/schemas.py) 定义 `INPUT_SCHEMA`（原始 CSV 入口校验，宽松）与 `CLEAN_SCHEMA`（清洗后物理上下界，严格）。
- 入口校验通过 `pa.errors.SchemaError` / `SchemaErrors` 快速失败，避免后续训练得到错误结论。

#### 步骤 B：字段与类型

- 连续数值特征（4 列必填）：`water_binder_ratio`、`fiber_volume_ratio`、`nc_strength_grade`、`curing_age_days`；
- 界面粗糙度（**二选一或同时提供**）：`roughness_sa`（三维面粗糙度 Sa，μm）与 `interface_prep`（界面处理方式，分类）；二者至少提供其一；
- 输出目标列：`tensile_bond_strength`（拉伸）、`shear_bond_strength`（抗剪）；**每行至少填一个目标**，允许缺失另一个（系统将按目标独立训练）；
- 使用 `pd.to_numeric(errors="coerce")` 统一成数值，不能转换的字符串被设为 `NaN`；`interface_prep` 转大写编号（如 `sb` → `SB`）、去空白、占位符归一为 `None`，随后 one-hot 展开为 `interface_prep__<value>` 0/1 列（`C`/`SB`/`HW`/`CH`/`M` 共 5 列）。

#### 步骤 C：缺失值处理

- 删除"整行全空"记录；
- **仅对输入特征列**用每列中位数填补缺失（目标列保留 NaN，不参与填补）；

#### 步骤 D：异常值剔除（IQR）

- `_clip_outliers_iqr`：`low = Q1 − 1.5·IQR`，`high = Q3 + 1.5·IQR`；越界样本删除。
- 策略简单可解释，适合小样本。

#### 步骤 E：CLEAN_SCHEMA 出口校验（PR#4 新增）

- 清洗后 DataFrame 再经 `CLEAN_SCHEMA.validate(lazy=True)` 检查物理上下界：
  - Sa 0–50 μm，w/b 0.08–0.60，纤维率 0–6 %，NC 10–120 MPa，养护龄期 0.5–400 d；
  - 拉伸 0–20 MPa，抗剪 0–40 MPa；`interface_prep__*` 为 0/1 one-hot 列。
- 违反上下界的记录不在主流程里自动剔除，会在 lazy 模式下聚合抛出，便于一次性定位问题。

---

## 3. `modeling.py` + `baselines.py` + `config.py` 详解

### 3.1 `TrainConfig` dataclass（PR#2/#4 新增）

[`src/uhpc_ml_app/config.py`](src/uhpc_ml_app/config.py) 控制运行时行为：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `test_size` | 0.2 | 8:2 切分比例 |
| `cv_folds` | 5 | GridSearchCV 折数 |
| `repeated_cv_n_splits` | 5 | RepeatedKFold n_splits |
| `repeated_cv_n_repeats` | 10 | RepeatedKFold n_repeats（共 50 次折叠） |
| `permutation_importance_n_repeats` | 20 | PI 重复次数 |
| `include_baselines` | `True` | 是否加入三种基线 |
| `model_selection_metric` | `"cv_r2_mean"` | 选最优模型的依据（CV R² 均值） |
| `mlp_max_iter` | 2 000 | BPNN 最大迭代 |

CI 使用 `TrainConfig.for_testing()` 压缩 CV 与 PI 重复次数，避免长时间阻塞。

### 3.2 指标：`_evaluate`（保留）

统一返回 R²（越大越好）、RMSE、MAE（两者越小越好）。**PR#2 新增**按目标独立返回 per-target 指标（拉伸 / 抗剪分开），避免 uniform_average 掩盖单目标问题。

### 3.3 模型工厂：`_build_models`（PR#2 重写）

三类候选模型：

1. **`random_forest`**：`RandomForestRegressor`（原生多输出，不再套 `MultiOutputRegressor`）；网格含 `n_estimators`、`max_depth`、`min_samples_leaf`、`max_features`。
2. **`xgboost`**：`MultiOutputRegressor(XGBRegressor(...))`；若环境缺 `libomp` 自动回退 `GradientBoostingRegressor`；网格含 `n_estimators`、`max_depth`、`learning_rate`、`reg_alpha`、`reg_lambda`。
3. **`bp_neural_network`**：**`Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(..., early_stopping=True))])`**。接入 `StandardScaler` 是 PR#2 的关键修正，**修复了老实现 BPNN 不标准化与树模型硬比、系统性低估**的缺陷。

### 3.4 `train_and_compare_models` 核心流程（PR#2 重写）

**步骤 A：切分**：`train_test_split(test_size=0.2, random_state=42)`。

**步骤 B：GridSearchCV**：`cv=5, scoring="r2"`。

**步骤 C：双指标评估**：训练集 / 测试集 + per-target 指标（`metrics_per_target.csv`）。

**步骤 D：Repeated K-Fold 95 % CI**（PR#2 新增）：对每个模型跑 `RepeatedKFold(5, 10)` 共 50 次折叠，报 R² 均值 ± 95 % CI 到 `cv_metrics.csv`。**在 N ≈ 100 小样本下，单次 8:2 方差大到模型排名会翻转，CV 均值才是稳健依据。**

**步骤 E：最优模型选择**：`model_selection_metric="cv_r2_mean"` → 取 CV R² 均值最高者为 `best_model`。（不再用单次测试集 R²。）

**步骤 F：基线对照**（PR#2 新增）：[`src/uhpc_ml_app/baselines.py`](src/uhpc_ml_app/baselines.py) 实现三种基线：
- `baseline_mean` — `MeanPredictor`，零学习下界（R² 理论上界 0）；
- `baseline_linear` — `LinearRegression`，线性可分上限；
- `baseline_sqrt_fc` — `EmpiricalSqrtFcBaseline(fc_column="nc_strength_grade")`，经验公式 τ = k·√f_c 的最小二乘拟合。

三种基线结果写入 `baseline_metrics.csv`，用于回答"ML 相对经验公式是否带来实质增益"。

**步骤 G：Permutation Importance 跨模型重要性**（PR#2 新增）：`sklearn.inspection.permutation_importance` 对三个候选模型分别在两个目标上计算"R² 下降量"作为统一口径的重要性，解决"RF/XGBoost/BPNN 内置重要性量纲不可比"问题；输出 `permutation_importance_by_model`，写 `permutation_importance.csv` 与每模型 per-target 条形图。

**步骤 H：返回结果字典**，包含：

- `metrics`（三模型训练/测试 overall）
- `metrics_per_target`（拉伸/抗剪分列）
- `cv_metrics`（RepeatedKFold 均值 + 95 % CI）
- `baseline_metrics`（三基线对照）
- `permutation_importance_by_model`
- `feature_importance_by_model`（模型内置重要性，仅保留作参考）
- `predictions_by_model`
- `best_model`、`best_model_name`、`model_selection_metric`
- `x_train`、`y_train`、`y_test_df`

---

## 4. `reporting.py` + `shap_analysis.py` 详解

### 4.1 `save_metrics`（扩展）

导出 `metrics_summary.csv`、`metrics_per_target.csv`、`cv_metrics.csv`、`baseline_metrics.csv` 便于直接写入论文表格。

### 4.2 `save_prediction_scatter`

生成"实测-预测"散点图，红色虚线 `y=x` 为理想预测。文件命名：`scatter_{model}_{target}.png`。

### 4.3 `run_dual_target_shap`（PR#3 完全重写）

老版本是单目标 SHAP 柱状图 + SHAP 失败时**静默回退**到相关系数。现在（[`src/uhpc_ml_app/shap_analysis.py`](src/uhpc_ml_app/shap_analysis.py)）：

1. **按模型拓扑自动派发 explainer**：
   - `RandomForestRegressor`（原生多输出树）→ `TreeExplainer`，shape `(n, p, n_targets)` 切片取每个目标；
   - `MultiOutputRegressor(XGBRegressor)` → 对每个 `estimators_[t]` 分别跑 `TreeExplainer`；
   - `Pipeline([StandardScaler, MLPRegressor])` → `KernelExplainer`（K-means 压缩背景 + 评估样本子采样）；
   - TreeExplainer 失败（如 `shap 0.49.1` + `xgboost 3.2.0` 的 `base_score` 序列化 bug）→ 自动回退 `KernelExplainer` 而不是回退到相关系数。
2. **双目标独立运行**：分别对拉伸、抗剪输出 `shap_summary_{target}.png` + 每个粗糙度特征的 `shap_dependence_{target}_{feature}.png`。
3. **显式降级审计**：SHAP 真失败时在 `best_model_info.json` 写入 `explanation_method="correlation_fallback"` 与 `failed_targets` 字段，论文审稿人一眼可判断该结果是否为真实 SHAP。

### 4.4 粗糙度推荐区间 `_recommend_threshold_from_dependence`（PR#3 重写）

**老实现**："roughness_ra 10–90% 分 5 桶取均值最高" → 不是 SHAP 驱动、不体现非线性、不体现工况。

**新实现**：

1. 在依赖曲线 `(x_i, φ_i)` 上按 x 升序排序，滑动均值平滑 `w = max(6, n/6)`；
2. 找出**最长连续 SHAP ≥ 0 的平台段** `[x_low, x_high] = argmax (b−a) s.t. φ̄(x) ≥ 0 ∀ x ∈ [a, b]`；
3. 全负情形：`reason="no_positive_region_report_peak"`，报告最大点附近窗口并标注为"负效应被最小化"；样本 < 6：`reason="too_few_samples"`，`low=None, high=None`，**禁止**出现"0–0"假区间。

`_conditional_recommendations` 按 `water_binder_ratio` 与 `curing_age_days` 各分 3 桶（低/中/高、短/中/长），对 9 个桶独立调用阈值提取，结果写入 `roughness_conditional_recommendations.csv`；全局结果写入 `roughness_recommendations.json`。

### 4.5 产物速查表

| 产物 | 来源函数 | 论文用途 |
|---|---|---|
| `metrics_summary.csv` | `save_metrics` | 表：三模型 overall 精度 |
| `metrics_per_target.csv` | `save_metrics` | 表：拉伸 vs 抗剪 per-target 精度 |
| `cv_metrics.csv` | `save_metrics` | 表：RepeatedKFold CV 均值 ± 95% CI |
| `baseline_metrics.csv` | `save_metrics` | 表：ML vs 三基线对照 |
| `permutation_importance.csv` + 条形图 | `save_permutation_importance` | 图：跨模型重要性 |
| `scatter_{model}_{target}.png` | `save_prediction_scatter` | 图：预测-实测 |
| `shap_summary_{target}.png` | `run_dual_target_shap` | 图：SHAP 总结（双目标） |
| `shap_dependence_{target}_{feature}.png` | `run_dual_target_shap` | 图：SHAP 依赖曲线 |
| `roughness_recommendations.json` | `run_dual_target_shap` | 表：Sa 全局推荐区间 |
| `roughness_conditional_recommendations.csv` | `run_dual_target_shap` | 表：条件化推荐（w/b × 龄期 3×3） |
| `best_model_info.json` | 顶层 `run_pipeline` | 元数据：最优模型名、选模型指标、`explanation_method`、`failed_targets` |

---

## 5. 你在写论文时可如何引用这些结果

- **模型精度对比**：`metrics_summary.csv` + `metrics_per_target.csv` + `cv_metrics.csv`（三张表合起来写第 4 章，能同时给出 overall、per-target、CV 95 % CI）；
- **ML 相对基线的增益**：`baseline_metrics.csv`（回答"ML 是否真的优于经验公式 τ=k√fc"）；
- **跨模型特征重要性**：`permutation_importance.csv` + 条形图（统一口径，不再用各模型内置重要性跨模型比较）；
- **可靠性与泛化能力展示**：`scatter_{model}_{target}.png` 六张散点；
- **粗糙度核心影响作用**：`shap_summary_{target}.png`（双目标）+ `best_model_info.json` 中的 `shap_importance_by_target`；
- **工程推荐区间**：`roughness_recommendations.json`（全局）+ `roughness_conditional_recommendations.csv`（条件化）+ `shap_dependence_*.png`（图示）。

---

## 6. PR 路线图完成状态（对齐本文各节）

| 原路线图条目 | 状态 | PR# | 实现位置 |
|---|---|---|---|
| BPNN 公平化（`StandardScaler`）| ✅ 已完成 | PR#2 | §3.3、`modeling.py:243` |
| 多输出分目标管道、per-target 指标 | ✅ 已完成 | PR#2 | §3.4 Step C、`metrics_per_target.csv` |
| Repeated K-Fold 95 % CI 不确定性评估 | ✅ 已完成 | PR#2 | §3.4 Step D、`cv_metrics.csv` |
| 基线模型（mean / linear / τ=k√fc） | ✅ 已完成 | PR#2 | §3.4 Step F、`baselines.py` |
| 双目标 SHAP + 依赖曲线阈值 + 条件化推荐 | ✅ 已完成 | PR#3 | §4.3–4.4、`shap_analysis.py` |
| 工程化骨架（pyproject / CI / pandera / pre-commit / Dockerfile / MLflow） | ✅ 已完成 | PR#4 | §9 |
| 真实文献数据库（PRISMA + SourceAdapter + PDF 骨架） | ✅ 已完成 | PR#5 | §10、`docs/data_protocol.md`、`src/uhpc_ml_app/literature/` |
| 用真实文献数据做外部验证 | ⏳ 待真实语料注入 | — | §10.5 |
| 最优模型持久化（`joblib.dump`）+ schema 导出 | ⏳ 后续小 PR | — | `reporting.py` |

---

## 7. 文献数据库采集入口迁移

> **重要变更**：PR#5 起，推荐使用新的 `literature-ingest` CLI（基于 `uhpc_ml_app.literature` 子包，支持 6 类来源 + PDF）。旧的 `python -m uhpc_ml_app.literature_crawler` 保留但**不再主推**，仅处理 OpenAlex/Crossref 摘要级抽取。

### 7.1 为什么不直接"全站爬取" WoS/Scopus/CNKI？

- 这些平台通常有登录、授权、反爬与版权条款；
- 建议优先使用官方 API 或机构授权导出（WoS RIS、Scopus CSV、CNKI RefWorks）；
- 本项目提供开放 API（OpenAlex/Crossref）+ 授权导出文件 + PDF 表格抽取的组合方案，合规且可复现。

### 7.2 新主入口 `literature-ingest`（PR#5）

完整用法详见 §10；快速示例：

```bash
literature-ingest \
  --plan wos_ris:data/raw/wos_export.ris \
  --plan scopus_csv:data/raw/scopus_export.csv \
  --plan cnki_refworks:data/raw/cnki_refworks.txt \
  --plan openalex:"UHPC NC bond roughness" \
  --plan crossref:"UHPC NC interface bond strength" \
  --output-dir data/literature/run-$(date +%Y-%m-%d)
```

### 7.3 旧入口（仅兼容保留）

```bash
# 仅从 OpenAlex/Crossref 摘要抽数字，召回率低，生产请改用 literature-ingest
PYTHONPATH=src python -m uhpc_ml_app.literature_crawler \
  --query "UHPC NC repair interface bond roughness surface preparation" \
  --output data/raw/literature_real.csv \
  --pages 3
```

### 7.4 数据质量建议

- 对自动抽取出的数值进行人工复核（摘要文本常不完整，全文更可靠，**优先使用 §10 的 PDF 管道**）；
- 将学校账号可导出的 WoS/Scopus/CNKI 结果补录到同一 `literature-ingest` 流水线（各有对应 adapter）；
- 复核后的数据（`confidence ≥ 2`）作为最终训练集输入主建模流程。

---

## 8. "模型解释性分析"章节模板（占位，禁止直接复制数值）

> ⚠️ **重要提示（请勿忽略）**
>
> 原版本在本节给出了具体数值（养护龄期 0.0927、Ra 0.0706、推荐区间 3.35–3.83 等），这些数值来源于合成示例数据 `literature_sample.csv`，**不具备任何工程或学术意义**。已按评审建议改为占位模板，以避免被误写入论文正文或答辩材料。
>
> **正式使用前必须：**
>
> 1. 用按 [`docs/data_protocol.md`](docs/data_protocol.md)（PRISMA）整理的真实文献数据库重新训练（`literature-ingest` → `uhpc-ml-app`）；
> 2. 从当次训练产物 `best_model_info.json` 与 `shap_importance_by_target`、`roughness_recommendations.json` 中取数替换所有 `<占位>`；
> 3. 基于 SHAP 依赖图（`shap_dependence_{target}_{feature}.png`）而非朴素分桶均值来陈述"阈值区间"与"正负作用方向"；
> 4. 若 `explanation_method="correlation_fallback"` 出现，**不得**把该目标的"重要性"写为 SHAP 结论。

以下为供论文"模型解释性分析"小节参考的**填写模板**：

基于最优模型（`<模型名：RF / XGBoost / BP>`）的 SHAP 解释结果可知，不同输入特征对 UHPC-NC 修复界面 `<目标：拉伸 / 抗剪>` 粘结强度预测的贡献存在明显差异。按平均绝对 SHAP 值排序，特征重要性由高到低依次为：`<特征1>`（`<SHAP均值1>`）、`<特征2>`（`<SHAP均值2>`）、…、`<特征n>`（`<SHAP均值n>`）。该结果表明，在本研究数据集条件下，`<简要说明主导因素>`，界面粗糙度 `<Sa / 界面处理方式>` 排序为第 `<位次>` 位，贡献占比约 `<百分比>`，`<说明其是否为核心影响因素及耦合关系>`。

进一步结合 SHAP 依赖图可知，`<Sa>` 在 `<下界>`–`<上界>` 范围内对应**最长连续正效应平台段**（即 SHAP ≥ 0 的最长区间），可作为界面处理工艺的推荐控制区间。在 `水胶比 <桶>` 与 `养护龄期 <桶>` 等耦合条件下，该推荐区间可能进一步收窄/平移（详见 `roughness_conditional_recommendations.csv` 中对应 3 × 3 分桶结果）。

> 使用步骤：训练完成后打开 `outputs/best_model_info.json` → 按 `shap_importance_by_target.<target>` 字段填入各特征均值；打开 `outputs/roughness_recommendations.json` → 按 `<feature>.<target>` 字段填入阈值区间（同时读取 `reason` 字段以确认推导来源）；打开 `outputs/roughness_conditional_recommendations.csv` → 按 (`w/b 桶`, `龄期桶`) 行填入条件化推荐。结合 `outputs/shap_summary_<target>.png` 与 `outputs/shap_dependence_<target>_<feature>.png` 陈述正负作用方向与非线性拐点。

### 8.1 常见错误（切勿出现）

- 直接复制本模板中的 `<占位>` 字样作为数值；
- 使用合成数据 `literature_sample.csv` 跑出的数字写入论文；
- 使用 SHAP 失败回退路径（`explanation_method="correlation_fallback"`）得到的"重要性"作为 SHAP 结论；
- 使用老式"5 桶均值取最大"得到的阈值区间（PR#3 起已替换为 SHAP 依赖曲线的最长正效应平台段，老方法的结果不可信）。

### 8.2 图注版（对应 `shap_summary_{target}.png`）

图 X 展示了最优模型（`<模型名>`）在 `<目标：拉伸 / 抗剪>` 粘结强度预测下的 SHAP 总结图。各特征按平均绝对 SHAP 值由高到低排序，依次为 `<特征1>`、`<特征2>`、…、`<特征n>`。横坐标为 SHAP 值（MPa 量纲），表示该特征对预测结果的正负贡献；点的颜色表示特征取值高低（红高蓝低）。结果表明，样本中 `<主导因素>` 的整体贡献最高，粗糙度参数（`Sa` / 界面处理方式）进入前 `<位次>` 位。基于该模型的 SHAP 依赖曲线最长正效应平台段，推荐界面粗糙度 `Sa` 控制在 `<SHAP 阈值下界>`–`<SHAP 阈值上界>` μm 范围内（来源：`roughness_recommendations.json` 的 `<feature>.<target>.low`/`.high`）。

---

## 9. 工程化基础设施（PR#4）

本仓库采用 PEP 621 打包 + GitHub Actions CI + pandera schema + pre-commit 钩子 + Dockerfile + 可选 MLflow，保证**任何人 clone 下来都能一键复现**。

### 9.1 安装与开发环境

```bash
# 需要 Python 3.10+
python -m venv .venv && source .venv/bin/activate

# 开发 + 测试 + PDF 抽取依赖全装
pip install -e ".[dev,pdf]"

# 本地 pre-commit 钩子（ruff + ruff-format + mypy + 基础卫生检查）
pre-commit install
```

### 9.2 GitHub Actions CI

工作流 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：在每次 push / PR 时，在 Python **3.10** 与 **3.12** 矩阵下自动跑：

```yaml
- ruff check .
- ruff format --check .
- mypy src
- pytest -q      # 54 个测试 < 60 s
```

所有检查通过才能合入 main；Devin Review 机器人还会额外做一次代码级 review。

### 9.3 两段式 pandera schema

- 通用数据管道：[`src/uhpc_ml_app/schemas.py`](src/uhpc_ml_app/schemas.py) `INPUT_SCHEMA`（入口宽松） + `CLEAN_SCHEMA`（清洗后严格）；
- 文献管道：[`src/uhpc_ml_app/literature/schemas.py`](src/uhpc_ml_app/literature/schemas.py) `STAGING_SCHEMA`（允许 NaN） + `CLEAN_SCHEMA`（物理上下界 + `confidence ≥ 2` + DOI 或 (first_author, year) 必须之一）。

### 9.4 pre-commit 钩子

[`.pre-commit-config.yaml`](.pre-commit-config.yaml) 在 commit 前自动跑 ruff / ruff-format / mypy / 大文件 / 尾空格 / YAML 与合并冲突检查。钩子触发的自动修改（如 ruff format）会写回你的工作区，**再 commit 一次**即可。

### 9.5 Dockerfile（可选）

[`Dockerfile`](Dockerfile) 多阶段构建：builder 装 `pip install -e ".[dev]"` 到 `/opt/venv`，runtime 只拷贝虚拟环境 + 源码。路径与 Python 版本解耦（PR#4 修复 Devin Review 指出的硬编码路径 bug）：

```bash
docker build -t uhpc-ml-app .
docker run --rm -it -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs uhpc-ml-app \
  uhpc-ml-app --input data/raw/literature_sample.csv --output outputs/docker-run/
```

### 9.6 可选 MLflow 实验跟踪

[`src/uhpc_ml_app/mlflow_integration.py`](src/uhpc_ml_app/mlflow_integration.py) 在 `MLFLOW_TRACKING_URI` 环境变量设置时自动把 `TrainConfig`、`metrics_summary`、`cv_metrics`、`baseline_metrics`、关键产物路径 log 到 MLflow；未设置 URI 或连接失败时**完全静默退化**，不会拖崩主管道（PR#4 修复 Devin Review 指出的无异常保护 bug）。

```bash
export MLFLOW_TRACKING_URI="http://your-mlflow:5000"
uhpc-ml-app --input ... --output ...
# 结果会出现在 MLflow UI
```

---

## 10. PRISMA 文献整合数据库（PR#5）

### 10.1 协议与宽表

- 完整协议：[`docs/data_protocol.md`](docs/data_protocol.md)（v0.1，纳入/排除 I1–I4 + E1–E5、6 类来源、去重规则、宽表字段、单位换算、confidence 评分、审计日志）。
- 宽表字段（20 列）定义：[`src/uhpc_ml_app/literature/base.py`](src/uhpc_ml_app/literature/base.py) 的 `LiteratureRecord` dataclass（含 `normalize_doi` 静态方法）。
- 两段式 schema：[`src/uhpc_ml_app/literature/schemas.py`](src/uhpc_ml_app/literature/schemas.py) 的 `STAGING_SCHEMA`（宽松，保留 NaN，枚举约束）与 `CLEAN_SCHEMA`（严格，物理上下界 + `confidence ≥ 2` + DataFrame-level `必须有 DOI 或 (first_author, year)` 检查）。

### 10.2 来源适配器

| adapter 名 | 输入类型 | 实现 | 适用场景 |
|---|---|---|---|
| `openalex` | 关键词查询 | [`src/uhpc_ml_app/literature/adapters/openalex.py`](src/uhpc_ml_app/literature/adapters/openalex.py) | 免费 API，inverted abstract 重建 |
| `crossref` | 关键词查询 | [`src/uhpc_ml_app/literature/adapters/crossref.py`](src/uhpc_ml_app/literature/adapters/crossref.py) | 免费 API，JATS 标签剥离 |
| `wos_ris` | 本地 `.ris` 文件 | [`src/uhpc_ml_app/literature/adapters/wos_ris.py`](src/uhpc_ml_app/literature/adapters/wos_ris.py) | WoS / EV 机构导出 |
| `scopus_csv` | 本地 `.csv` 文件 | [`src/uhpc_ml_app/literature/adapters/scopus_csv.py`](src/uhpc_ml_app/literature/adapters/scopus_csv.py) | Scopus "Export → CSV" |
| `cnki_refworks` | 本地 RefWorks/RIS 文件 | [`src/uhpc_ml_app/literature/adapters/cnki_refworks.py`](src/uhpc_ml_app/literature/adapters/cnki_refworks.py) | CNKI 导出 |

可选：PDF 管道（`uhpc_ml_app.literature.pdf`）—— `pdfplumber_extractor`（本地离线）与 `grobid_extractor`（远程 `GROBID_URL`）。CI 不启 GROBID 服务，单测使用 fake session 跑。

### 10.3 去重规则

**SOURCE_PRIORITY**（值越小优先级越高，见 [`src/uhpc_ml_app/literature/pipeline.py`](src/uhpc_ml_app/literature/pipeline.py)）：

```
manual < pdf_grobid < pdf_pdfplumber < wos < scopus < cnki < crossref < openalex
```

- 主键：`normalize_doi(doi)`（去除 URL 前缀、小写化）；
- 退化键：`(first_author, year, title[:80])`；
- 冲突记录写入 `deduplication_log.csv`，保留最高优先级的条目。

### 10.4 端到端命令

```bash
# 1. 准备你的导出文件（各自 adapter 独立）
ls data/raw/
# wos_export.ris  scopus_export.csv  cnki_refworks.txt  papers/*.pdf

# 2. 运行 literature-ingest
literature-ingest \
  --plan wos_ris:data/raw/wos_export.ris \
  --plan scopus_csv:data/raw/scopus_export.csv \
  --plan cnki_refworks:data/raw/cnki_refworks.txt \
  --plan openalex:"UHPC NC bond roughness surface preparation" \
  --plan crossref:"UHPC NC interface bond strength" \
  --output-dir data/literature/run-$(date +%Y-%m-%d)

# 终端打印:
# ingest 完成: 142/387 条入 staging, 89 条入 clean

# 3. 查看产物
ls data/literature/run-<ts>/
# staging.parquet
# literature_clean.csv            ← 喂给 uhpc-ml-app 的训练数据
# prisma_log.csv                  ← PRISMA 流程图各节点数量
# deduplication_log.csv           ← 去重被丢弃记录
# ingest_run.json                 ← manifest（protocol_version, 各 adapter 命中数）
# clean_schema_error.txt          ← 可选，CLEAN schema 违反记录

# 4. 用 literature_clean.csv 作为输入跑建模
uhpc-ml-app --input data/literature/run-<ts>/literature_clean.csv \
            --output outputs/modeling-$(date +%Y-%m-%d)/
```

### 10.5 下一步（待真实语料）

- 收集 2–3 篇含清晰 bond-strength 表格的代表性 PDF → 喂给 `pdf_pdfplumber` / `pdf_grobid` adapter 完善领域映射 DSL；
- 用真实 `literature_clean.csv` 跑一遍端到端 `uhpc-ml-app`，从产物回填论文中的数字类占位符（见 §5 产物速查表）；
- 把结果做外部验证（留出一部分最新文献作为 hold-out），回填到论文结论章节。

---

## 附：54 个单元测试覆盖

```
tests/
├── test_data_pipeline.py       # INPUT/CLEAN schema + IQR + 中位数填补
├── test_schemas.py             # pandera 边界值
├── test_modeling.py            # 旧行为兼容
├── test_modeling_pr2.py        # RepeatedKFold / per-target / PI / 基线
├── test_baselines.py           # MeanPredictor / LinearRegression / τ=k√fc
├── test_shap_dual_target.py    # 双目标 SHAP + 依赖曲线阈值 + 条件化
├── test_pipeline_app.py        # 顶层 run_pipeline 端到端
├── test_web_export.py          # Streamlit 数据导出
├── test_literature_crawler.py  # 兼容测试（旧模块）
└── literature/
    ├── test_base_and_registry.py
    ├── test_extraction.py      # 正则抽取
    ├── test_file_adapters.py   # WoS/Scopus/CNKI 文件解析
    ├── test_api_adapters.py    # OpenAlex/Crossref fake session
    ├── test_pdf_extractors.py  # pdfplumber + GROBID TEI-XML
    └── test_pipeline.py        # deduplicate + run_ingest
```

全部 54 个测试在 Python 3.10 与 3.12 下均通过；`pytest -q` < 60 s。

---

**文档版本**：v2.0（对齐 PR#1~#5 主干实现；2026-04-22）
