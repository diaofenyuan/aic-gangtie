# 煤气发电量预测与发电优化

这是面向“全球校园人工智能算法精英大赛·AI+钢铁·煤气发电预测与发电优化”的离线工程。项目已接入初赛正式数据，默认入口使用隔离的初赛配置；合成数据仅用于验证代码链路。

> **重要：合成数据只能测试读取、清洗、特征、训练、验证、预测、文件生成和优化流程，不能用于判断真实准确率、模型优劣、经济收益或竞赛成绩。** 正式模型必须通过训练期原始标签验证和门控后才会写入 `models/`。

## 环境与安装

目标环境为 Python 3.10，代码不依赖 GPU，同时兼容 Windows 和 Linux。核心流程使用 NumPy、Pandas、SciPy/HiGHS、scikit-learn 和 PyYAML。

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

也可以用 `uv` 在项目内创建固定的 Python 3.10 环境：

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Windows
uv pip install --python .venv/bin/python -r requirements.txt           # Linux
```

LightGBM、CatBoost 和 Optuna 是高精度流程的可选依赖，默认基线不要求安装：

```bash
python -m pip install ".[lightgbm]"
python -m pip install ".[catboost]"
python -m pip install ".[high-accuracy]"
```

正式离线环境应提前下载对应 Python 3.10 wheel。若国内网络下载受限，可在联网准备环境中使用清华或阿里云镜像，然后把 wheel 和项目一起带入离线环境。`doctor` 会报告 LightGBM、CatBoost 和 Optuna 是否可用。

## 一键运行

无需安装本项目包也能从仓库根目录运行：

```bash
python run.py
```

该命令默认读取 `config/official_preliminary.yaml`，初赛会依次执行训练期高精度搜索/门控、原始标签滚动验证，并为评分期生成 192 个短周期预测起点；未通过门控时自动使用最后值回退。`input.csv` 和 `s_result.csv` 写完后记录 SHA-256 冻结清单。运行流程不读取未来标签，也不计算测试集成绩。终端默认只显示中文运行摘要，完整结果保存在本次输出目录的 `运行结果.json`，详细报告保存在 `reports` 中。

训练目录和评分目录已经隔离；也可以显式指定初赛正式配置来执行单个阶段：

```bash
.venv/Scripts/python.exe run.py --config config/official_preliminary.yaml audit-data
```

`paths.data` 指向项目内的 `data/preliminary/train`；`paths.scoring_data` 指向隔离存放的 `data/preliminary/scoring`，只在预测期作为逐起点历史输入。评分目录不会进入模型拟合、训练期模型选择或本地评分。该配置只允许初赛相关命令，并明确禁止联网推理、外部数据和外部预训练权重。

每次命令的结果都会写入独立目录。运行期间先使用 `outputs` 下的临时目录，任务结束后按结束时刻重命名，目录名只由“结束时间 + 预测结果”组成，例如：

```text
outputs/2026-07-31_02-10-30预测结果/
├── s_result.csv
├── input.csv
├── submission_freeze.json        # 提交文件 SHA-256 冻结清单
├── 运行结果.json                 # 终端摘要对应的完整结构化结果
├── teamname_gas_predict_prelim.zip # ZIP 根目录仅包含 input.csv 和 s_result.csv
└── reports/
```

线程数和进度显示可以在命令行临时覆盖：

```bash
python run.py --workers 8
python run.py --no-progress
python run.py --no-progress --json
python run.py --config config/default.yaml --workers 16 demo
```

默认终端输出适合人工阅读；需要完整 JSON 标准输出供脚本处理时使用 `--json`，通常与 `--no-progress` 一起使用。

`demo` 会把合成数据放入 `data/synthetic`，与 `data` 根目录中的正式数据隔离。

各阶段也可以独立执行：

```bash
python run.py --config config/default.yaml generate-synthetic
python run.py --config config/default.yaml train
python run.py --config config/official_preliminary.yaml doctor
python run.py --config config/official_preliminary.yaml tune
python run.py --config config/default.yaml validate
python run.py --config config/default.yaml audit-data
python run.py --config config/default.yaml benchmark
python run.py --config config/default.yaml discover-relations
python run.py --config config/default.yaml backtest
python run.py --config config/default.yaml predict
python run.py --config config/default.yaml optimize
python run.py --config config/default.yaml validate-submission
python -m pytest
```

安装为可编辑包后，也可使用完全相同的模块入口：

```bash
python -m src.cli --config config/default.yaml audit-data
python -m src.cli --config config/default.yaml benchmark
```

排行榜中的 99.9 分若按百分制 `100 × (1-MAPE)` 解释，确实对应极低误差，但工程不会据此推断作弊或承诺 99% 以上准确率。审计只验证稳定工况、时间错位、确定性关系、合法计划量和未来泄漏等可检验假设。

## 目录结构

```text
.
├── config/default.yaml            # 字段映射、特征、模型、验证和优化参数
├── config/feature_availability.yaml # 字段业务可用时间与正式白名单
├── src/gas_power/
│   ├── availability.py            # 保守字段白名单和可用时间门控
│   ├── time_semantics.py          # 预测起点、目标时刻和列偏移校验
│   ├── audit.py                   # 泄漏、滞后相关和重采样审计
│   ├── benchmark.py               # 统一时间折的强基线矩阵
│   ├── relations.py               # 确定性关系和固定时延发现
│   ├── scoring.py                 # 可配置评分公式
│   ├── postprocessing.py          # 物理约束后处理及前后对比
│   ├── gpu_gate.py                # 残差模型的时间折稳定性门控
│   ├── submission.py              # 提交文件联合校验与追踪清单
│   ├── data.py                    # 多表读取、重采样、对齐和因果清洗
│   ├── features.py                # 先 shift 后 rolling 的因果特征
│   ├── metrics.py                 # 官方 MAPE 与 1-MAPE
│   ├── validation.py              # expanding/rolling 验证和泄漏检查
│   ├── models/                    # 基线、融合、LightGBM/CatBoost 接口
│   ├── optimization.py            # SciPy/HiGHS 混合整数调度
│   ├── outputs.py                 # 官方结果格式和 UTF-8 校验
│   ├── synthetic.py               # 仅用于流程测试的缺陷合成数据
│   ├── pipeline.py                # 端到端业务流水线
│   └── cli.py                     # CLI 子命令
├── tests/                         # 单元测试和临时目录端到端测试
├── data/                          # 正式输入数据；合成演示数据位于 data/synthetic
├── cache/                         # 清洗数据、特征和质量报告
├── models/                        # 模型和训练元数据
├── outputs/                       # 按运行结束时间归档的提交文件、验证明细和报告
└── logs/                          # 命令日志
```

## 正式数据接入

1. 将官方文件放到独立数据目录，不要覆盖原始文件。
2. 复制 `config/default.yaml`，修改 `paths.data` 及 `data.tables`。
3. 每个标准字段可用 `sources` 指定一个或多个原字段，也可用 `patterns` 正则匹配多座炉/多用户字段；多个来源通过 `combine: sum|mean|first` 合并。
4. 逐字段确认单位和 15 分钟聚合口径，通过字段级 `aggregation` 配置 `mean|sum|last` 等规则。
5. 在 `data.roles` 中指定预测目标、三类煤气产量、优先用户需求、气柜和发电耗气字段。未知字段没有写死在 Python 代码里。
6. 逐字段更新 `config/feature_availability.yaml`。无法确认业务发布时间的字段保持 `available_at_origin: false`，不得因为相关性高而放入白名单。
7. 先运行 `audit-data` 和 `train`，检查本次运行的 `outputs/<完成时间>预测结果/reports/data_audit.*` 与 `cache/data_quality.json`，确认缺失、重复、异常、时间断点和重采样边界后再进行验证。

对于 PDF 中不带时间戳的月度 `price.xlsx`，将对应表设为 `time_series: false`，避免参与 15 分钟对齐；待官方表结构到达后再把月份、时段和单价映射到优化时域。

初赛正式数据已经通过 `config/official_preliminary.yaml` 接入；`config/default.yaml` 继续保留为合成流程和通用基线配置。正式配置按字段字典汇总多座高炉、热风炉、煤气用户和气柜，不会将评分集作为训练输入。

清洗保留所有有限原始观测；滚动 IQR 只生成 `feat_outlier__*` 和稳健缩放特征，不会平滑或替换标签。缺失值最多因果前向填补 8 点，整列缺失填 0 并保留 `feat_missing__*` 标记。项目刻意不使用后向填补或双向插值，因为全表预处理时它们可能把未来观测带到预测起点之前。训练、验证和 MAPE 始终使用未修改的 `Pre_load.csv` 原始标签。

## 预测实现

已实现以下基线：

- 最后值保持；
- 最近 2/4/8 点均值和窗口中位数；
- 最近 15/30/60 分钟线性趋势与阻尼趋势；
- 昨日同时刻；
- 上周同时刻；
- YAML 权重归一化融合。

`benchmark` 在同一组严格时间折上比较 12 个配置化基线，输出每个目标和步长的 MAPE、1-MAPE、MAE、RMSE、偏差、样本数、近零样本数和最佳策略。`benchmark_alignment_diagnostics.csv` 会比较标签整体平移的影响；非零偏移均标记为诊断，不参与模型选择。

`forecast.model.type` 可切换为 `lightgbm` 或 `catboost`。两者统一支持：

- `strategy: direct`：每个目标、每个步长独立建模；
- `strategy: global`：每个目标一个模型，将预测步长作为 `feat_horizon_steps`；
- `target_mode: delta`：默认预测未来相对当前负荷的变化量；
- `target_mode: absolute`：直接预测未来绝对负荷。

`ForecastModel` 是统一接口。高精度配置下，LightGBM/CatBoost direct 残差模型和组件重建模型共用 `fit/predict` 契约。

`residual_lightgbm` 与 `residual_catboost` 实现“基线预测 + 残差预测”。残差模型支持 direct/global 两种多步策略；最终 `train/predict` 前必须存在当前运行目录的 `outputs/<完成时间>预测结果/reports/residual_gate.json`，且全部时间折达到配置改善阈值。验证标签不能用于单样本融合权重。

物理后处理支持非负、容量、可配置逐步爬坡及 `generator_1 <= generator_all`。回测会把约束前后指标分别写入本次运行的 `outputs/<完成时间>预测结果/reports/postprocessing_metrics.csv`，不默认假设约束一定改善准确率。

`tune` 只读取训练目录。`config/official_preliminary.yaml` 是高分档：30 次 Optuna trial、最近 4 折粗筛、复核前 5 组参数，树数范围为 150–750；`config/official_preliminary_fast.yaml` 保留 8 次 trial、2 折粗筛和前 2 组复核，用于快速排错。候选包含直接残差模型、机组分解模型和“煤气产量/优先消耗/气柜变化 -> 可用煤气 -> 发电量”两阶段模型。最终从近期验证和跨月份验证各均匀选取 4 折，按目标×步长独立检查平均改善、至少 5/8 折不退化及最差折退化不超过 1 个百分点；合格候选使用非负 NNLS 和留一折预测融合，不合格的单列独立回退 `LastValueModel`，不再由整模门槛否决所有步长。

`scoring` 不是验证集。它只能在 `predict` 阶段按起点拼接训练尾部 672 点历史；评分期后续行不会进入拟合、OOF、早停、特征选择或融合调权。`reports/high_accuracy_selection.json`、模型元数据和 SHA-256 清单记录这一边界。

## 防止未来信息泄露

- 禁止随机划分，验证仅使用 expanding 或 rolling window。
- 所有滞后使用 `shift(lag)`；所有滚动均值、标准差、最小值、最大值和斜率先执行 `shift(1)`。
- 训练样本仅保留“目标时刻不晚于训练终点”的行。
- 每个基线预测都在当前起点显式截断历史。
- 自动检查会扰动截止点之后全部数值，并要求截止点之前特征、当前起点预测逐元素不变。
- `outputs/<完成时间>预测结果/validation_splits.json` 保存每折训练和验证边界，便于审计。
- 每个字段在 `feature_availability.yaml` 中记录来源文件、时间戳含义、采集时间、业务事件时间、最小滞后、标签/计划属性及短长周期许可。未知字段默认拒绝。
- 自动校验 `feature_available_time <= forecast_origin_time` 和 `target_time = forecast_origin_time + horizon`。
- `audit-data` 检查目标直入、可疑字段名、±96 步相关曲线、平移/缩放/线性组合、跨表时延、未来合并、填补、中心滚动、双向插值、全量预处理、验证窗口重叠和四种重采样边界。

审计输出：

- `outputs/<完成时间>预测结果/reports/data_audit.json`、`data_audit.md`；
- `outputs/<完成时间>预测结果/reports/lag_correlation.csv`：正偏移代表当前字段与未来标签的相关性，统一标红且只用于诊断；
- `outputs/<完成时间>预测结果/reports/suspicious_features.csv`：字段名、仿射复制、多变量组合和未来高相关风险；
- `outputs/<完成时间>预测结果/reports/relation_coefficients.csv`、`setpoint_analysis.csv`、`delay_relations.csv`：确定性关系、设定值和跨表固定延迟。

高相关不等于违规，也不等于可用。计划量只有在业务发布时间早于预测起点且官方允许后才可修改白名单；关系发现模块本身不会自动修改白名单。

官方 MAPE 不增加 epsilon。真实值接近零时会单独统计并告警；严格为零时结果可能为 `inf` 或 `nan`，不会被悄悄改写。验证输出包括两个目标、每个预测步长、目标汇总、步长汇总和整体指标，同时输出误差最大的时间点及煤气平衡、气柜变化和机组工况。

`backtest` 同时执行 expanding、固定长度 rolling 和连续两天模拟评测，报告每折均值/标准差/最差值、稳定/启停/爬坡/高缺失/月切换工况、步长误差曲线和相对最后值增益。模型选择优先参考最差折和跨月份稳定性。

评分模块支持原始 `1-MAPE`、百分制 `100 × (1-MAPE)`、目标权重、步长权重以及可选数据处理分组合。默认沿用 PDF 可确认的原始 `1-MAPE`；排行榜是否仅做百分制展示、目标/步长权重及是否组合数据处理分仍待官方确认。

## 优化模型

`HighsDispatchOptimizer` 使用 SciPy/HiGHS 的混合整数规划，当前包含：

- 三类煤气物料平衡、发电耗气、放散和供气不足松弛量；
- 三类气柜安全上下限；
- 三级词典序求解：先最小化生产用户供气不足，再固定最优保供结果并最小化放散，最后固定前两级结果并最大化峰谷收益、控制启停成本；
- 4 套 50MW 与 2 套 120MW 机组，其中 `generator_1` 是 4 套 50MW 机组负荷合计，`generator_all` 是全部 6 套机组负荷合计；
- 二进制启停、60% 最小稳定负荷、100% 容量上限；
- 额定容量 10%/分钟爬坡约束；
- 煤气消耗量与发电负荷线性换算；
- 分时电价收益、放散、供气不足和启动惩罚；
- 基于历史可用数据分别预测三类煤气发生量、优先生产需求和历史运行方式下的发电基准，不直接把昨日同期当作唯一资源边界。

若资源边界下仍不能实现零供气不足、求解失败或约束残差超过容差，则不生成越界的 `opt_result.csv`，完整诊断写入本次运行的 `outputs/<完成时间>预测结果/optimization_diagnostics.json`。成功时还会输出 `generator_1_plan_mw`、`generator_all_plan_mw`、单机计划和气柜/放散/供气不足明细，并报告电价加权负荷比；只有基准放散量可得时才计算题面定义的完整相对收益提升率。

## 输出文件

- `outputs/<完成时间>预测结果/s_result.csv`：`datetime` + 两个目标的 t+15 至 t+120，共 17 列；
- `outputs/<完成时间>预测结果/input.csv`：192 个预测起点、训练期有效的官方原始字段及经过训练期常数/重复裁剪的 `feat_` 因果派生特征；
- `outputs/<完成时间>预测结果/teamname_gas_predict_prelim.zip`：ZIP 根目录仅包含原名 `input.csv` 和 `s_result.csv`；
- `outputs/<完成时间>预测结果/submission_freeze.json`：两个提交文件的 SHA-256 与字节数；
- `outputs/<完成时间>预测结果/reports/resource_boundary_forecast.csv`：三类煤气发生量、优先需求、储气前可用量、历史发电基准和电价的 96 步预测；
- `outputs/<完成时间>预测结果/reports/inference_runtime.json`：单样本和总推理耗时，并检查题面建议的单样本 30 秒、总计 30 分钟限制；
- `outputs/<完成时间>预测结果/submission_manifest.json`：源码/config 哈希、模型训练区间、字段白名单、文件哈希与校验结果。

写出前后都会检查时间戳是否为预测起点、列名及顺序、步长完整性、目标互换、缺行/重复/错位、意外索引列、数值类型、缺失/无穷值、容量边界和 UTF-8 编码。`input.csv` 剔除训练期全缺失原始字段，缺失和异常值只用历史观测因果修复，所有派生字段统一使用 `feat_` 前缀。ZIP 打包前会再次校验冻结哈希。

提交压缩包按初赛要求固定打包 `input.csv` 和 `s_result.csv`，压缩包名称由 `submission.archive_name` 配置且不增加额外目录层级。

## 等待正式数据或规则确认

以下内容不能从赛题 PDF 唯一确定，当前只在 `config/default.yaml` 以 `value + status` 形式提供合成流程占位值：

- 三类气柜各自容量及“约 20 万 m³”是单柜还是场景总容量；
- 各字段物理单位、流量是瞬时值/区间均值/区间体积及 15 分钟聚合方式；
- 三类煤气热值、机组效率、煤气量到 MW 的换算关系；
- 启停成本、放散成本、供气不足成本和收益换算口径；
- 官方逐月尖峰平谷时段和电价；
- 机组初始启停状态、初始负荷、最小开停机时间；
- 多个煤气用户的真实优先级及混合煤气配比约束；
- 完整字段字典和最终提交模板；
- 每个源文件时间戳代表区间起点还是终点，`label/closed` 的官方重采样口径；
- 测试起点是否包含该时刻目标实测值，以及计划字段的实际发布时间；
- 榜单是原始 `1-MAPE` 还是百分制展示，两个目标和各步长是否等权；
- 是否存在独立数据处理得分及其组合权重；
- 优化输出的精确字段名、单位及预测资源边界接口。

当前只处理初赛短周期预测，不运行发电优化。初赛提交压缩包固定只包含 `input.csv` 与 `s_result.csv`。

建议首批执行顺序：

```bash
.venv/Scripts/python.exe run.py --config config/official_preliminary.yaml audit-data
.venv/Scripts/python.exe run.py --config config/official_preliminary.yaml benchmark
.venv/Scripts/python.exe run.py --config config/official_preliminary.yaml discover-relations
.venv/Scripts/python.exe run.py --config config/official_preliminary.yaml backtest
.venv/Scripts/python.exe run.py --config config/official_preliminary.yaml train
.venv/Scripts/python.exe run.py --config config/official_preliminary.yaml predict
```

## 当前验收说明

2026-07-31 已在本机隔离的 CPython 3.10 环境完成以下高精度链路验收：

```text
python run.py --config config/default.yaml generate-synthetic 通过，2016 个干净时间点
python run.py --config config/default.yaml audit-data         通过，78 个正式特征；识别 3 个未来红色风险字段
python run.py --config config/default.yaml benchmark          通过，12 个基线、442368 个预测对
python run.py --config config/default.yaml discover-relations 通过，5 组线性关系、39 个延迟候选
python run.py --config config/default.yaml backtest            通过，3 类时间验证、110592 个预测对
python run.py --config config/default.yaml train               通过，WeightedEnsembleModel
python run.py --config config/default.yaml validate            通过，36864 个预测对；泄漏检查通过
python run.py --config config/default.yaml predict             通过，生成短长周期结果
python run.py --config config/default.yaml optimize            通过，HiGHS Optimal；最大约束违反量 2.27e-9
python run.py --config config/default.yaml validate-submission 通过，联合提交校验和 manifest 校验通过
python -m compileall -q src tests run.py                       通过
python -m pytest                                               45 passed
ruff check src tests run.py                                    All checks passed
```

端到端流程生成 2016 个合成时间点。所有合成分数只写入机器报告，不在 README 中解释为正式成绩。近零标签按官方未平滑 MAPE 单独告警；本轮基线和回测均未达到配置中的 99.5/99.9 可达性阈值。扩展/滚动回测均覆盖稳定、爬坡、启停和高缺失工况，官方两天回测覆盖月份切换和周末代理变量。残差门控因未在所有时间折稳定改善而拒绝。

- `s_result.csv`：4 行、17 列；
- `l_result.csv`：4 行、193 列；
- `opt_result.csv`：96 行、4 列；
- 两个预测目标及 96 个长周期步长完整，预测起点无重复；
- 提交文件及生成的诊断 CSV 均通过 UTF-8 严格解码；
- `input.csv` 仅含白名单输入和工程特征；
- HiGHS 返回 `Optimal`，最大约束违反量为 `2.27e-9`，合成流程中供气不足和放散均为 0。

以上合成流程仅是代码、约束和文件格式验收，不代表正式成绩。正式初赛应先执行 `doctor`，再执行 `tune`；只有 `high_accuracy_selection.json` 显示门控通过时才使用机器学习融合，否则使用可审计的最后值回退。一次官方数据单折冒烟中，30 棵树的直接 LightGBM 原始标签 MAPE 为 6.0753%，同折最后值为 6.3520%；该结果不是完整 10 折/8 折选择结论，也不代表排行榜成绩。
