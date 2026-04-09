# CHANGELOG — AutoModel 变更日志

> **维护规则**: 每次涉及代码更新时，**必须**在此文件顶部追加一条变更记录。  
> **格式规范**: 遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) + [语义化版本](https://semver.org/lang/zh-CN/)  
> **责任**: 无论是人工修改还是 AI 辅助修改，均需同步更新本文件。

---

## [v2.9.4] - 2026-04-02

### 📋 接口注释对齐（无运行逻辑改动）

继续审阅翻译官相关模块后，发现 `knowledge_graph_interface.py` 中仍保留“翻译官默认持有原始数据库连接”的早期注释，与当前项目真实主线（主程序先将 CSV / CK 数据加载为 DataFrame，再交由 `KnowledgeGraphTranslator` 做语义映射）不一致，容易误导后续维护者。

### Changed（注释修正）

#### knowledge_graph_interface.py
| 改动项 | 旧表述 | 新表述 | 原因 |
|--------|--------|--------|------|
| 接口定位 | 默认面向数据库直连 | 面向能力契约，数据来源可为 DataFrame / DB 连接等 | 与当前主线一致 |
| `get_entity_dataframe()` 说明 | 从数据库加载该表 | 从底层数据载体中取表，当前主线通常是预加载 DataFrame | 避免误解 |
| `get_relationship_keys()` 说明 | 简单的关系名→外键名 | 当前主线实际返回更丰富的关系元数据对象 | 与真实实现对齐 |

#### data_translator.py
| 改动项 | 旧表述 | 新表述 | 原因 |
|--------|--------|--------|------|
| 模块说明 | 以内存模拟数据库为主 | 以外部 DataFrame 注入为主，模拟内存数据为兜底 | 与当前生产主线一致 |

### 影响评估
- **运行逻辑**: **零影响**。仅修正注释与说明文字
- **维护成本**: **降低**。后续阅读接口时不再被“数据库直连”旧表述误导

---

## [v2.9.3] - 2026-04-02

### 🔍 变更背景
继续审计训练主流程后发现，`v2.9.1` 虽已让普通种群评估跳过无用的 full fit，但**每代精英**仍会再次调用完整 `evaluate()`，重复执行 3-fold CV 后才做 full fit + SHAP；同时最终返回的冠军仍是“最后一代冠军”，而非整个演化过程中的**全局最佳冠军**。

### Changed（修改）

#### architect.py
| 改动项 | 旧逻辑 | 新逻辑 | 原因 |
|--------|--------|--------|------|
| 精英终训路径 | 精英复评复用 `evaluate()`，重复跑 3-fold CV + full fit + SHAP | 新增 `finalize_chromosome()`，复用本代已算好的 CV 指标，仅执行 full fit + SHAP | 去掉精英阶段重复 CV |
| 评估耗时记录 | `evaluation_time_ms` 混合记录 | 新增 `cv_evaluation_time_ms`、`final_training_time_ms` | 区分搜索评估耗时和精英终训耗时 |

#### main.py
| 改动项 | 旧逻辑 | 新逻辑 | 原因 |
|--------|--------|--------|------|
| 精英评估调用 | 每代精英再次 `evaluate(..., train_final_model=True)` | 每代精英改为 `finalize_chromosome(...)` | 避免重复 3-fold CV |
| 最终冠军选择 | 直接覆盖为最后一代冠军 | 仅当综合得分刷新时更新为全局最佳冠军 | 防止最后一代回退导致保存次优模型 |

### 影响评估
- **AUC 搜索效果**: **零影响**。搜索选择仍完全由普通评估阶段的 OOF AUC 驱动
- **最终模型质量**: **更稳**。最终保存对象从“最后一代冠军”修正为“全局最佳冠军”
- **训练耗时**: 在 `v2.9.1` 基础上继续下降，去掉每代精英重复的 3-fold CV

---

## [v2.9.2] - 2026-03-11

### 📋 适应度函数设计决策（无代码改动，仅文档澄清）

经代码审查与业务验证，确认**适应度函数应使用且仅使用 AUC，无需添加其他指标**。理由：

1. **KS/Precision/F1 本质上是 AUC 的阈值依赖产物**：在相同的 OOF 概率分布下，AUC 已包含全部排序信息；Precision/F1 随阈值变化，不稳定
2. **样本不平衡下的 Precision 失真**：正样本占比 4.48% 时，Precision=0.076 并不代表模型差，数学上等价于 1:1 平衡集下 Precision≈0.64，与人工建模持平
3. **对抗机制（Saboteur）代码已完备，但权重全为 0**：`economics`/`causal`/`synthesis` 三个评分项接口预留，当前均设为 weight=0，不影响搜索，无需改动
4. **单目标 AUC 使遗传搜索更稳定**：多目标加权需人工调参权重，引入额外超参风险，性价比不高

**结论**：`main.py` 中 `fitness_weights = {'auc': 1.0, 'economics': 0.0, 'causal': 0.0, 'synthesis': 0.0}` 配置正确，**无需修改代码**。

### Changed（文档修正）

#### PROJECT_CONTEXT.md
| 改动项 | 旧表述 | 新表述 | 原因 |
|--------|--------|--------|------|
| §二点三 评分指标 | `AUC（主指标）` | `AUC（唯一评分依据，weight=1.0）` | 措辞"主指标"暗示还有副指标参与打分，产生误导 |
| 变更日志表格 | 缺少 v2.9.1/v2.9.2 记录 | 补充两条记录 | 保持文档与 CHANGELOG 同步 |

---

## [v2.9.1] - 2026-03-09

### 🔍 变更背景
通过代码审计发现 `FitnessEvaluator.evaluate()` 在**每次**调用时都无条件执行 `model_pipeline.fit(X, y)` 全量训练（[architect.py L873](src/architect.py#L873)），但在遗传搜索阶段，驱动选择的 AUC 完全由 `cross_val_predict` 的 OOF 概率决定（[architect.py L810](src/architect.py#L810)），全量 fit 产出的 pipeline 对搜索过程**零贡献**。

以默认配置 5代×6个体 计算：
- 30 次普通评估的 full fit 全部被丢弃（浪费 30 次全量训练）
- 5 次精英重评估中 CV 部分也是重复计算（浪费 15 次 fold 训练）
- 折算后浪费约 **38%** 的总训练时间

### Changed（修改）

#### architect.py
| 改动项 | 旧逻辑 | 新逻辑 | 原因 |
|--------|--------|--------|------|
| `evaluate()` 签名 | `(chromosome, calculate_shap=False)` | `(chromosome, calculate_shap=False, train_final_model=True)` | 新增参数控制是否执行全量 fit |
| `evaluate()` 全量训练 | 无条件执行 `model_pipeline.fit(X, y)` | 仅当 `train_final_model=True` 时执行 full fit + SHAP | 搜索阶段只需 CV 出 AUC，无需全量训练 |

#### main.py
| 改动项 | 旧逻辑 | 新逻辑 | 原因 |
|--------|--------|--------|------|
| 普通评估调用 | `evaluate(chromosome)` | `evaluate(chromosome, train_final_model=False)` | 搜索阶段跳过无用的 full fit |
| 精英评估调用 | `evaluate(elite, calculate_shap=True)` | `evaluate(elite, calculate_shap=True, train_final_model=True)` | 冠军需要保存完整 pipeline，必须做 full fit |

### 影响评估
- **AUC 搜索效果**: **零影响**。搜索选择完全由 `cross_val_predict` OOF 概率驱动，full fit 从未参与评分
- **最终模型质量**: **零影响**。冠军的精英评估仍执行完整的 full fit + SHAP
- **训练耗时**: 预计节省 **~38%**（40W 数据规模下约节省数分钟）

---

## [v2.9] - 2026-02-28

### 🔍 变更背景
对特征选择策略做深度分析后发现**根本性架构缺陷**：系统按星型模型（1:N 流水表）设计，但实际 3 张表均为 **月快照宽表**，bill_no 在每张表中唯一（1:1 关系）。导致：
- `GROUP BY bill_no` + `AVG/SUM/MAX/MIN(col)` 在 1 行数据上运算 = 原字段值（无意义）
- `COUNT(*)` 恒等于 1（无意义）
- LLM 生成的跨表聚合窗口建议全部失效
- 规则引擎的窗口特征（DAYS列）产出为 0
- `build_features()` 对跨表 LATEST 基因只填 NaN 而非真实 JOIN 取值

### Added（新增）

#### architect.py — 3 个新方法/重写
| 方法 | 作用 |
|------|------|
| `_detect_one_to_one_tables()` | 按 `unique_keys / len(df) > 0.95` 判断副表是否为 1:1 快照表 |
| `_machine_screen_secondary_table()` | 对 1:1 副表：LEFT JOIN 主表后用 LGBM 预筛选，返回 top-K 特征的 LATEST 基因 |

### Changed（修改）

#### architect.py
| 改动项 | 旧逻辑 | 新逻辑 | 原因 |
|--------|--------|--------|------|
| `generate_initial_pool()` | 所有副表统一走 LLM 聚合建议 + 规则引擎 | 自动检测 1:1 vs 1:N；1:1 走 LGBM 筛选 + LATEST；1:N 保留原有策略 | 月快照表聚合无意义，应直接取字段值 |
| `build_features()` LATEST 处理 | 跨表 LATEST 仅在 base_df 中查列，找不到则填 NaN | 跨表 LATEST 通过 `merge(sec_df[[join_key, field]], on=join_key)` 实际 JOIN 取值 | 修复副表 LATEST 特征全为 NaN 的严重 bug |

#### sql_generator.py
| 改动项 | 状态 | 说明 |
|--------|------|------|
| `LATEST → MAX(col)` on 1:1 table | **保留不变** | 对 1:1 表 `MAX(col)` = 原值，功能正确；改为直接列引用属优化项但非必要，留作后续版本 |

### 根因分析详情

```
┌──────────────────────────────────────────────────────────────┐
│ 数据结构 (实际)                                               │
│ ┌──────────┐  1:1  ┌───────────┐  1:1  ┌──────────┐          │
│ │ utrm(主) ├───────┤ base_info ├───────┤  upay    │          │
│ │ 120列    │  key  │   33列    │  key  │  80列    │          │
│ └──────────┘bill_no└───────────┘bill_no└──────────┘          │
│ 每表每 bill_no 仅 1 行（月快照）                              │
│                                                              │
│ 原系统假设:                                                   │
│   bill_no ─── 1:N ───> 副表（多笔流水）                       │
│   → AVG/SUM/COUNT/MAX/MIN 有统计意义                          │
│                                                              │
│ 实际:                                                        │
│   bill_no ─── 1:1 ───> 副表（单行快照）                       │
│   → AVG = MAX = MIN = SUM = 原值, COUNT = 1                  │
│   → 跨表聚合全部退化为恒等映射                                │
└──────────────────────────────────────────────────────────────┘
```

### V2.9 修复策略

```
1:1 快照表路径（新增）:
  detect_one_to_one → LEFT JOIN 主表 + LGBM 排名 → top-K LATEST 基因
  build_features: LATEST 跨表 → 真实 merge 取值

1:N 流水表路径（保留）:
  LLM 跨表聚合建议 + 规则引擎窗口特征 → AVG/SUM/COUNT 基因  
  build_features: 聚合逻辑不变
```

### 预期效果
| 指标 | V2.8 | V2.9（预期） | 说明 |
|------|------|-------------|------|
| 有效跨表特征数 | ~0（聚合退化为原值） | 10~30（副表 LGBM 筛选后精选） | 修复核心缺陷 |
| 特征多样性 | 仅主表 LATEST 有意义 | 主表 + 副表均有效 | 3 张表 ~233 列全部可参与评选 |
| 初始基因池质量 | LLM 聚合建议失效 | LGBM 数据驱动筛选 | 消除"创意"建议的无效计算 |
| AUC | 基准 | 预期提升 0.02~0.05 | 更多有效特征参与 GA 搜索 |

---

## [v2.8] - 2026-02-28

### 🔍 变更背景
基于 40W×3 表（换机预测）生产数据首次全量运行的结果分析，发现以下问题：
- SHAP 在全量 40W 数据上 OOM（`Unable to allocate 521 MiB` + `std::bad_alloc`）
- 模型验证阶段在全量数据上重建特征矩阵导致内存耗尽，`evaluation_report.json` 未生成
- 使用固定阈值 0.5 导致 Recall=0.004, F1=0.008（严重不合理）
- 5 折 CV + 双重 CV 调用导致一代评估耗时过长
- LGBM 未启用 `is_unbalance` 导致对正样本（~3%）欠拟合
- 特征比例配置注释与实际值不匹配，演化参数过于激进

### Changed（修改）

#### architect.py
| 改动项 | 旧值 | 新值 | 原因 |
|--------|------|------|------|
| LGBM 预筛选参数 | `n_estimators=60, max_depth=6` | `n_estimators=100, max_depth=8` | 40W 数据量下提升特征重要度排序准确性 |
| CV 折数 | `n_splits=5` | `n_splits=3` | 40W×3折已足够可靠，减少 40% 评估耗时 |
| CV 策略 | `cross_val_score` + 再调 `cross_val_predict`（双重 CV） | 仅 `cross_val_predict` 一次获取 OOF 概率 | 消除双重计算，节省 ~50% 评估时间 |
| 分类阈值 | 固定 `0.5` | KS 最优阈值（自动搜索） | 不平衡数据下 0.5 阈值使 Recall/F1 趋近于零 |
| 模型基因 | 单一配置 `{n_estimators:100, lr:0.1}` | 3 个差异化配置，均含 `is_unbalance:True` + 正则化 | 增加超参多样性，启用类别不平衡处理 |
| SHAP 分析 | 在全量数据上运行 | 采样 8000 行后运行 | 修复 OOM，40W 全量 SHAP 消耗 >500MiB |
| 交叉特征限制 | 硬编码 `min=5, max=50` | 动态适配 `min=max(5, 5%池), max=max(50, 60%池)` | 适应不同基因池规模 |

#### main.py
| 改动项 | 旧值 | 新值 | 原因 |
|--------|------|------|------|
| `lgbm_top_k_single_ratio` | `0.5`（注释写 25%） | `0.35` | 修正注释与值不匹配，35% 平衡覆盖与噪声 |
| `lgbm_top_k_multi_ratio` | `0.5`（注释写 15%） | `0.25` | 多表场景副表特征占主导时适当收窄主表 |
| `min_features_ratio` | `0.5` | `0.15` | 原值导致每个染色体包含 50%+ 特征，搜索空间过窄 |
| `max_features_ratio` | `0.8` | `0.45` | 降低单个染色体最大特征数，防止过拟合 |
| `max_features_floor` | `20` | `15` | 与新比例配合 |
| 验证阶段 | 全量 40W 数据验证 | 采样 50000 行验证 | 修复验证阶段 OOM，确保 evaluation_report.json 生成 |
| 评估报告 | 不含最优阈值 | 新增 `最优阈值` 字段 | 方便部署时使用正确阈值 |

#### llm_interface.py
| 改动项 | 旧值 | 新值 | 原因 |
|--------|------|------|------|
| API 调用超时 | 未设置（依赖 SDK 默认值） | `timeout=120` 秒 | 上次运行出现 `Request timed out`，显式设置避免挂起 |

### 预期效果
| 指标 | V2.7（上次运行） | V2.8（预期） | 说明 |
|------|------------------|-------------|------|
| AUC | 0.7092 | ≥0.70 | AUC 应保持或略有提升 |
| KS | 0.2916 | ≥0.28 | 应保持 |
| Recall | 0.004 (💀) | 0.3~0.6 | KS 最优阈值 + is_unbalance 大幅改善 |
| F1 | 0.008 (💀) | 0.3~0.5 | 同上 |
| 单代评估耗时 | ~长 | 降低约 60% | 3 折 + 消除双重 CV |
| SHAP | OOM 崩溃 | 正常完成 | 采样 8000 行 |
| evaluation_report.json | 未生成 | 正常生成 | 验证采样 50000 行 |

---

## [v2.7] - 2026-02-04

### Added
- 扩展评估指标：KS、Precision、Recall、F1、混淆矩阵
- 自动生成 `evaluation_report.json`
- 敏捷平台兼容版模型 (`automl_pipeline_compatible.pkl`)

### Changed
- 前端简化为运行后展示 AUC

---

## [v2.6] - 2026-02-04

### Changed
- 前端简化为"运行后展示 AUC"
- 加强平台脚本生成与日志稳定性
- 子进程输出重定向到 `automl.log`

---

## [v2.5] - 2026-02-03

### Added
- 新增 Streamlit 可视化前端 (`web_ui.py`)

---

## [v2.4] - 2026-02-03

### Added
- 新增部署架构图、平台环境信息
- 敏捷平台兼容性问题清单
- 数据集构造说明

---

## [v2.3] - 2026-02-02

### Changed
- 明确系统为"规则驱动的 AutoML 原型系统"
- 引入文档维护规则

---

## [v2.2] - 2026-02-02

### Changed
- 代码文件迁移至 `src/` 目录
- 删除冗余文件

---

## [v2.1] - 2026-02-02

### Added
- 初始化 PROJECT_CONTEXT.md
- 基于当前代码自动生成

---

## 维护规范

### 变更记录模板
```markdown
## [vX.Y] - YYYY-MM-DD

### 🔍 变更背景
简述触发本次变更的原因（bug、需求、性能问题等）

### Added（新增）
- 新功能描述

### Changed（修改）
- 修改内容 | 旧值 → 新值 | 原因

### Fixed（修复）
- 修复的bug描述

### Removed（移除）
- 移除的功能描述

### 预期效果
简述预期的改善效果
```

### 规则
1. **强制同步**: 每次代码变更必须同步更新本文件
2. **日期标记**: 使用 ISO 8601 日期格式 (YYYY-MM-DD)
3. **版本号**: 遵循语义化版本 MAJOR.MINOR（重大变更.次要更新）
4. **溯源性**: 变更背景必须写明触发原因
5. **量化**: 尽量包含旧值→新值的对比表格
6. **文件关联**: 标记受影响的文件名
