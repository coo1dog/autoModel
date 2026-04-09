# PROJECT_CONTEXT.md — AutoModel 项目上下文文档

> **自动生成时间**: 2026-02-02  
> **当前修订时间**: 2026-04-02  
> **文档目的**: 作为本项目的**最高事实来源**，用于 AI 协作、项目交接与长期维护  
> **当前版本**: v2.9.3（精英终训去重CV + 最终冠军改为全局最佳）  
> **变更日志**: 详细变更记录请查看 [CHANGELOG.md](CHANGELOG.md)

---

## 变更日志（Change Log）

| 日期 | 版本 | 变更类型 | 内容摘要 | 影响范围 |
|------|------|---------|---------|---------|
| 2026-02-02 | v2.1 | 初始化 | 基于当前代码自动生成 PROJECT_CONTEXT.md | 全局 |
| 2026-02-02 | v2.2 | 目录重构 | 代码文件迁移至 `src/` 目录，删除冗余文件 | 目录结构 |
| 2026-02-02 | v2.3 | 定位修订 | 明确系统为"规则驱动的 AutoML 原型系统"，引入文档维护规则 | 项目定位、AI 协作 |
| 2026-02-03 | v2.4 | 架构补充 | 新增部署架构图、平台环境信息、敏捷平台兼容性问题清单、数据集构造说明 | 部署运维 |
| 2026-02-03 | v2.5 | 演示增强 | 新增 Streamlit 可视化前端 (`web_ui.py`)，支持动态参数配置与实时演示 | 演示汇报 |
| 2026-02-04 | v2.6 | 稳定性修订 | 前端简化为"运行后展示 AUC"，加强平台脚本生成与日志稳定性 | 演示汇报、产出物 |
| 2026-02-28 | v2.8 | **策略&性能优化** | 40W数据策略全面审查: SHAP采样防OOM, KS最优阈值替代0.5, is_unbalance启用, 3折CV, 验证采样 | architect.py, main.py, llm_interface.py |
| 2026-02-28 | v2.9 | **架构修复** | 修复1:1月快照表特征选择缺陷: 新增表关系自动检测, 1:1副表LGBM筛选+LATEST基因, 跨表LATEST真实JOIN取值 | architect.py |
| 2026-03-09 | v2.9.1 | **性能优化** | 搜索阶段跳过无用的full fit，仅冠军精英保留全量训练+SHAP，节省~38%训练耗时 | architect.py, main.py |
| 2026-03-11 | v2.9.2 | **文档澄清** | 确认适应度函数以AUC为唯一评分依据（weight=1.0），其余指标仅供观测，无需修改代码 | PROJECT_CONTEXT.md, CHANGELOG.md |
| 2026-04-02 | v2.9.3 | **流程修正** | 精英终训复用本代CV指标，仅做full fit + SHAP；最终冠军改为全局最佳而非最后一代冠军 | architect.py, main.py, CHANGELOG.md |

---

## 一、项目概览

| 项目属性 | 信息 | 置信度 |
|---------|------|--------|
| **项目名称** | AutoModel（对抗式共演化自动建模原型） | 【确定】 |
| **核心目标** | 自动化机器学习（AutoML），通过遗传算法完成特征工程与模型训练 | 【确定】 |
| **应用场景** | 多业务场景数据挖掘建模（当前已验证：用户换机预测） | 【确定】 |
| **开发语言** | Python 3.10+, Streamlit | 【确定】 |

### 项目定位【确定】

这是一个以**规则驱动 + 配置化**为核心的 **AutoML 原型系统**，  
用于在明确业务 Schema 前提下，自动完成特征构建、模型训练与部署产物生成。
**特别地，v2.5 版本新增了 Streamlit 前端，专为演示汇报场景设计，展示系统“自主研发”的动态过程。**

**核心设计原则：代码不写死，业务可配置，演示可交互**
- 系统根据**传入的数据表**和**知识图谱配置**（`schema_config.json`）动态适配
- 系统并非通用 AutoML 平台，而是**工程导向的自动建模工具**
- 多业务场景能力依赖于：数据表结构 + `schema_config.json` 的明确配置
- 当前已完整跑通并验证的业务场景为：**用户换机预测**

> ⚠️ **重要说明**  
> 不同业务场景对特征工程、评估指标、约束条件要求不同，实际可复用程度需经具体场景验证。

**系统能力**：
1. 自动从原始数据推断业务语义（借助 LLM + 外部 Schema 配置）
2. 自动生成和筛选特征（LGBM 预筛选 + LLM 启发建议）
3. 使用遗传算法演化出最优建模策略
4. 输出可部署的模型（`.pkl`）、建宽表 SQL、敏捷平台发布代码
5. **(New) 可视化演示：展示进化监控曲线、冠军基因解析、代码生成成果**

---

## 二、目录结构【确定】

```
AutoModel/
├── src/                              # ✅ 源代码目录（上传到平台时只需上传此目录）
│   ├── main.py                       # 主入口（兼容 CLI 和 Streamlit 调用）
│   ├── web_ui.py                     # (New) Streamlit 可视化前端
│   ├── create_platform_script.py     # (New) 手动生成敏捷平台脚本的兜底工具
│   ├── architect.py                  # 架构师模块
│   ├── saboteur.py                   # 破坏者模块
│   ├── core_structures.py            # 基因/染色体数据结构
│   ├── data_translator.py            # 数据翻译官实现
│   ├── knowledge_graph_interface.py  # 翻译官抽象接口
│   ├── semantic_inference.py         # 语义推断模块
│   ├── schema_config.py              # Schema 配置加载器
│   ├── llm_interface.py              # LLM 统一接口
│   ├── sql_generator.py              # SQL 自动生成器
│   ├── clickhouse_loader.py          # ClickHouse 数据加载
│   ├── predict.py                    # 预测脚本
│   ├── logger_config.py              # 日志配置
│   └── requirements.txt              # Python 依赖 (新增 streamlit, plotly)
│
├── dataset_termchange/               # 示例数据集（换机预测）
│   ├── schema_config.json            # 业务 Schema 配置
│   ├── ziduan.txt                    # 字段说明文档
│   └── *.csv                         # 数据文件
│
├── saved_model/                      # ❌ 运行产出目录（不需要上传）
│   ├── automl_pipeline_compatible.pkl
│   ├── production_query.sql
│   ├── production_query_short.sql
│   ├── platform_inference_template.py
│   └── ...

├── start_ui.bat                       # (New) Windows 一键启动前端脚本
│
└── PROJECT_CONTEXT.md                # 本文档
```

### 部署说明

**上传到多维分析平台时**：只需上传 `src/` 文件夹内容即可。

---

## 二点一、前端界面功能现状（Streamlit）【v2.6】

**目标**：演示参数可控、过程可追溯、产出可交付。

**当前实现**：
- 支持参数输入（世代、种群、数据目录、主表、目标列）并启动/停止任务。
- 运行中仅展示日志流，**AUC 曲线与每代记录在任务结束后统一展示**。
- 展示冠军基因、冠军特征列表与特征重要度（Top 50）。
- 展示可复制的 SQL 与平台发布脚本。

**稳定性策略**：
- 关闭实时图表刷新，避免 Streamlit 在长时间循环中卡顿。
- 子进程输出重定向到 `automl.log`，避免 PIPE 阻塞。
- 平台脚本生成失败时，可使用 `src/create_platform_script.py` 手动兜底生成。

---

## 二点二、代码审查结论（截至 2026-02-04）

**清理/确认事项**：
- `web_ui.py` 已移除不必要的实时图表刷新逻辑，避免重复渲染导致错误。
- 子进程日志输出统一写入 `automl.log`，避免编码与管道阻塞问题。
- 平台脚本生成流程已加入兜底与错误日志，确保失败可定位。

**保留的工具脚本**：
- `src/create_platform_script.py`：手动生成平台脚本，作为自动生成失败的兜底手段。
- `start_ui.bat`：Windows 一键启动前端。

---

## 二点三、自动建模实现逻辑与复用边界【v2.6】

### A. 自动建模主流程（代码依据）
1. **数据加载**：
   - CSV 或 ClickHouse 输入 → DataFrame 字典。入口：[src/main.py](src/main.py)
2. **语义推断**：
   - 优先使用 `schema_config.json`；缺失表/字段可自动补全。入口：[src/semantic_inference.py](src/semantic_inference.py)
3. **翻译官桥接**：
   - 将物理表/列映射为标准实体与字段，提供统一 DataFrame 访问。入口：[src/data_translator.py](src/data_translator.py)
4. **特征基因生成（混合模式）**：
   - **主表 LGBM 预筛选**：从主表选择 Top-K 字段 → `LATEST` 基因。
   - **规则窗口特征**：仅在副表存在 `DAYS` 类时间列时生成窗口特征。
   - **LLM 跨表特征**：用于“候选特征建议”，随后会做关系与字段校验，不合法会被丢弃。
   入口：[src/architect.py](src/architect.py)
5. **演化与评估**：
   - 遗传算法选择/交叉/变异。
   - Pipeline：数值缺失填充 + 类别 One‑Hot + LGBMClassifier。
   - 评分指标：**AUC（唯一评分依据，weight=1.0）**。KS/Precision/Recall/F1 在精英评估时计算并记录到报告，但不影响遗传搜索选择。
6. **冠军产物生成**：
   - 保存模型（原版 + 平台兼容版）与冠军染色体 JSON。
   - 生成 SQL（长列名 + 短列名）与特征列表。
   - 生成平台发布脚本（失败时可用兜底脚本）。

### B. LLM 的使用边界
- **LLM 仅用于“建议”**：
  - Schema 推断（无外部配置时）。
  - 跨表特征候选生成。
- **最终进入建模的特征必须通过代码校验**：
  - 必须存在于真实字段中。
  - 必须满足关系映射与实体规则。

### C. 自动生成 SQL 与平台脚本的可复用性
- **可复用前提**：
  1. `schema_config.json` 中能正确描述表、主键与关系。
  2. 目标列存在且为二分类标签。
  3. 表字段命名与训练时一致，避免字段漂移。
  4. 若要严格“最新一条记录”语义，需要在数据层保证 1:1 或扩展 SQL（当前 LATEST 使用 MAX）。

- **适用场景**：
  - 多业务“主表 + 副表聚合”结构都可以适配。
  - 只要关系配置准确，SQL 与平台脚本都可自动生成。

- **易失败环节**：
  - 关系配置缺失或主键错误 → 跨表特征会大面积失效。
  - 类别字段分布极端/缺失 → 重要度解释难。
  - 新业务无 `DAYS`/时间字段 → 窗口特征为 0。


## 三、核心设计理念

### 3.1 对抗式共演化结构（预留）【确定】

系统在架构层面预留对抗式评估结构：
- **Architect（架构师）**：生成特征、训练模型、主导演化
- **Saboteur（破坏者）**：对模型进行经济性、因果合理性等批判性评估

> ⚠️ **当前状态说明**  
> - `saboteur.py` 仅作为扩展接口存在，代码框架已实现  
> - 权重初始化为 0，不参与当前演化过程  
> - **当前版本不计划启用对抗机制**

### 3.2 遗传算法隐喻【确定】

项目将建模过程类比为生物进化：
| 生物学概念 | 项目对应 |
|-----------|---------|
| 基因（Gene） | 单个特征提取/变换/模型选择操作 |
| 染色体（Chromosome） | 一个完整的建模流程（特征+模型） |
| 种群（Population） | 多个候选建模策略 |
| 适应度（Fitness） | 模型 AUC（当前仅使用 AUC） |
| 交叉/变异 | 策略组合与随机调整 |
| 精英学习 | 使用 SHAP 对冠军策略进行精炼优化 |

---

## 四、技术栈

| 类别 | 技术/库 | 用途 |
|-----|--------|------|
| **机器学习** | scikit-learn, LightGBM | 模型训练、Pipeline 构建 |
| **特征解释** | SHAP | 特征重要性分析、精英学习 |
| **数据处理** | pandas, numpy | DataFrame 操作 |
| **大语言模型** | OpenAI SDK (兼容接口) | 语义推断、基因生成、因果批判 |
| **模型持久化** | joblib | 保存训练好的 Pipeline |
| **数据库** | ClickHouse (可选) | 生产环境数据加载 |

### LLM 配置【确定】

从 [src/llm_interface.py](src/llm_interface.py) 可见：
- 使用内部部署的 Qwen3-32B 模型
- 端点：`http://10.79.231.133:6000/xyfx-chat/v1`

### LLM 使用边界说明【重要】

LLM **仅用于**：
- 语义解释（字段含义推断）
- 候选特征建议（启发式头脑风暴）
- 因果性批判（预留功能）

**所有进入建模流程的逻辑，必须满足**：
- 代码可执行
- Schema 约束校验
- 显式规则检查

> 🚫 **LLM 不具备最终建模决策权**

---

## 四点一、关键参数矩阵【v2.9 确定】

> 以下为 40W×3 表数据场景的生产参数配置，已在代码中实装。

### A. 演化引擎参数 (main.py `EVOLUTION_CONFIG`)
| 参数 | 值 | 说明 |
|------|-----|------|
| `generations` | CLI `--generations`（默认 2） | 进化世代数，生产建议 5~8 |
| `population_size` | CLI `--population`（默认 2） | 种群规模，生产建议 6~10 |
| `min_features_ratio` | 0.15 | 单个染色体最少使用基因池 15% 的特征 |
| `max_features_ratio` | 0.45 | 单个染色体最多使用基因池 45% 的特征 |
| `max_features_floor` | 15 | 特征下限保底 |

### B. 特征预筛选参数 (main.py `FEATURE_GEN_CONFIG`)
| 参数 | 值 | 说明 |
|------|-----|------|
| `lgbm_top_k_single_ratio` | 0.35 | 单表场景选前 35% 特征 |
| `lgbm_top_k_multi_ratio` | 0.25 | 多表场景选前 25% 特征 |
| LGBM 预筛选 | `n_estimators=100, max_depth=8` | 比默认更强以提升排序准确性 |

### B2. 1:1 快照表检测与副表筛选【v2.9 新增】
| 参数 | 值 | 说明 |
|------|-----|------|
| 1:1 判定阈值 | `unique_keys / len(df) > 0.95` | 副表中 join_key 唯一率超过 95% 即判定为 1:1 快照表 |
| 副表 LGBM 筛选 | `n_estimators=100, max_depth=6, is_unbalance=True` | 对 1:1 副表 LEFT JOIN 主表后做特征重要度排名 |
| 副表 top-K 比例 | `single_ratio`（同 B 节） | 从副表列中选取前 K 列作为 LATEST 基因 |
| 跨表 LATEST JOIN | `merge(sec_df[[key, field]], on=key, how='left')` | 1:1 副表的 LATEST 基因通过真实 JOIN 取值（非 NaN 占位） |

> **策略路由**: 1:1 快照表 → LGBM 筛选 + LATEST 基因（跳过 LLM）；1:N 流水表 → LLM 聚合 + 规则引擎窗口特征（原流程）

### C. 评估器参数 (architect.py `FitnessEvaluator`)
| 参数 | 值 | 说明 |
|------|-----|------|
| CV 折数 | 3 | 40W 数据 3 折即可靠 |
| 分类阈值 | KS 最优阈值（自动搜索） | 替代固定 0.5 |
| SHAP 采样量 | 8000 行 | 防止 OOM |

### D. 模型基因默认超参 (architect.py `GeneGenerator`)
| 配置编号 | n_estimators | learning_rate | num_leaves | 特殊参数 |
|----------|-------------|---------------|------------|---------|
| 配置 1 | 200 | 0.05 | 31 | `reg_alpha=0.1, reg_lambda=0.1` |
| 配置 2 | 150 | 0.08 | 63 | `min_child_samples=50` |
| 配置 3 | 300 | 0.03 | 31 | `reg_alpha=0.3, reg_lambda=0.3, min_child_samples=100` |

> **所有配置均启用 `is_unbalance=True`**，适配正样本占比 ~3% 的不平衡场景

### E. 验证与安全参数 (main.py)
| 参数 | 值 | 说明 |
|------|-----|------|
| 验证采样 | 50000 行 | 防止验证阶段 OOM |
| LLM 超时 | 120 秒 | 防止 API 调用挂起 |

---

## 五、核心模块职责

### 5.1 系统入口模块

| 文件 | 职责 | 置信度 |
|-----|------|--------|
| [src/main.py](src/main.py) | **唯一入口**，包含完整流程（训练+SQL生成+平台脚本生成） | 【确定】 |
| [src/predict.py](src/predict.py) | 加载已训练模型，对新数据进行预测 | 【确定】 |

### 5.2 核心逻辑模块

| 文件 | 职责 | 关键类/函数 | 置信度 |
|-----|------|------------|--------|
| [src/core_structures.py](src/core_structures.py) | 定义"DNA"数据结构 | `FeatureGene`, `ModelGene`, `ModelingChromosome` | 【确定】 |
| [src/architect.py](src/architect.py) | "架构师"模块，负责基因生成、特征工程、模型评估、遗传演化 | `GeneGenerator`, `FeatureEngine`, `FitnessEvaluator`, `EvolutionaryEngine` | 【确定】 |
| [src/saboteur.py](src/saboteur.py) | "破坏者"模块（当前权重为0，未启用） | `EconomicsAttacker`, `CausalAttacker`, `SynthesisAttacker` | 【确定】 |

### 5.3 数据与语义模块

| 文件 | 职责 | 置信度 |
|-----|------|--------|
| [src/semantic_inference.py](src/semantic_inference.py) | 语义推断，从原始表结构推断业务实体和关系 | 【确定】 |
| [src/data_translator.py](src/data_translator.py) | "翻译官"实现，将物理表/列名转换为标准业务名称 | 【确定】 |
| [src/knowledge_graph_interface.py](src/knowledge_graph_interface.py) | "翻译官"抽象接口定义 | 【确定】 |
| [src/schema_config.py](src/schema_config.py) | 外部 Schema 配置加载器（避免 LLM 幻觉） | 【确定】 |

### 5.4 LLM 与外部接口

| 文件 | 职责 | 置信度 |
|-----|------|--------|
| [src/llm_interface.py](src/llm_interface.py) | 统一的 LLM 调用接口，支持语义推断、基因生成、因果批判 | 【确定】 |
| [src/clickhouse_loader.py](src/clickhouse_loader.py) | ClickHouse 数据库加载器 | 【确定】 |

### 5.5 生产部署模块

| 文件 | 职责 | 置信度 |
|-----|------|--------|
| [src/sql_generator.py](src/sql_generator.py) | 从训练结果自动生成生产 SQL（长列名+短列名双版本） | 【确定】 |

### 5.6 辅助模块

| 文件 | 职责 | 置信度 |
|-----|------|--------|
| [src/logger_config.py](src/logger_config.py) | 日志配置（控制台 + 文件输出） | 【确定】 |

---

## 六、工作流程

### 6.1 训练流程【确定】

```
1. 加载数据 (CSV/ClickHouse)
        ↓
2. 语义推断 (LLM + schema_config.json)
   - 识别业务实体和关系
   - 建立物理名↔标准名映射
        ↓
3. 初始化基因池 (GeneGenerator)
   - LGBM 预筛选主表重要特征
   - LLM 头脑风暴生成跨表聚合特征建议
        ↓
4. 遗传演化循环 (EvolutionaryEngine)
   ├─ 4.1 评估种群 (FitnessEvaluator)
   │       - 构建特征矩阵
   │       - 3-折交叉验证（v2.8）训练 LightGBM，计算 OOF AUC
   │       - 使用 KS 最优阈值计算 Precision/Recall/F1（v2.8）
   ├─ 4.2 破坏者批判 (Attackers) [当前权重=0，未启用]
   ├─ 4.3 计算综合得分 (当前 = AUC)
   ├─ 4.4 选择、交叉、变异
   └─ 4.5 精英学习 (复用本代OOF指标，仅做 full fit + SHAP；最终冠军按全局最佳保存)
        ↓
5. 保存最优模型
   - automl_pipeline.joblib (本地用)
   - automl_pipeline_compatible.pkl (敏捷平台用，protocol=2)
   - champion_chromosome.json (最优策略 DNA + 元数据)
        ↓
6. 生成生产部署文件
   - production_query.sql (长列名版)
   - production_query_short.sql (短列名版，推荐使用)
   - feature_alias_map.json (短列名→长列名映射)
   - table_mapping_template.json (表名占位符映射)
   - expected_feature_columns.json (特征白名单)
   - platform_inference_template.py (敏捷平台发布代码)
```

### 6.2 预测流程【确定】

```
1. 加载 automl_pipeline_compatible.pkl + champion_chromosome.json
        ↓
2. 加载新数据（从敏捷平台数据源读取）
        ↓
3. 根据特征白名单过滤和对齐列（自动将短字段名映射回长字段名）
        ↓
4. Pipeline.predict_proba() 输出预测概率
```

---

## 七、输出文件说明

### 7.1 saved_model 目录结构

| 文件 | 用途 | 是否必需 | 生成方式 |
|------|------|---------|---------|
| **automl_pipeline_compatible.pkl** | 敏捷平台用的模型 | ✅ 必需 | 自动 |
| **production_query_short.sql** | DM 建宽表 SQL（短列名 f001~fxxx） | ✅ 推荐 | 自动 |
| **platform_inference_template.py** | 敏捷平台发布代码 | ✅ 必需 | 自动 |
| production_query.sql | DM 建宽表 SQL（长列名） | 可选 | 自动 |
| automl_pipeline.joblib | 本地高版本 sklearn 用的模型 | 可选 | 自动 |
| champion_chromosome.json | 染色体 JSON（用于 SQL 生成） | 中间产物 | 自动 |
| expected_feature_columns.json | 特征白名单 | 中间产物 | 自动 |
| feature_alias_map.json | 短列名→长列名映射 | 中间产物 | 自动 |
| table_mapping_template.json | 表名占位符映射说明 | 中间产物 | 自动 |

### 7.2 你真正需要的 3 个文件

| 序号 | 文件 | 用途 | 粘贴到哪里 |
|------|------|------|-----------|
| 1 | `automl_pipeline_compatible.pkl` | 模型文件 | 上传到敏捷平台 |
| 2 | `production_query_short.sql` | 建宽表 SQL（短列名版） | DM 数据开发平台 |
| 3 | `platform_inference_template.py` | 发布代码 | 敏捷挖掘平台 |

### 7.3 SQL 文件使用说明

生成的 SQL 包含以下占位符，使用前需替换：

| 占位符 | 说明 | 示例 |
|--------|------|------|
| `{{TARGET_TABLE}}` | 你要创建的目标宽表名 | `dm_terminal_change_features_202512` |
| `{{MainTable_xxx}}` | 主表的生产表名 | `a_utrm_use_m` |
| `{{Table_xxx}}` | 副表的生产表名 | `a_upay_user_attr_m` |

---

## 八、已验证业务场景：换机预测【确定】

> 💡 以下是当前已完整跑通并验证的业务场景。  
> 作为系统能力示例，非唯一适用场景。新场景需逐一验证。

根据 `dataset_termchange/schema_config.json`：

**已验证业务**：浙江移动用户换机预测

**数据表**：
| 表名 | 业务含义 |
|-----|---------|
| temp_a_utrm_use_3m_inline_encrypt_202508 | **主表**：用户终端使用月视图（当前设备、流量、换机历史） |
| temp_user_base_info_m_bak_encrypt_202508 | 用户画像表（家庭结构、收入、职业等） |
| temp_a_upay_user_attr_m_bak_encrypt_202508 | 消费行为表（话费、流量费、充值、积分等） |

**目标变量**：`flag`（1=换机，0=未换机）

**关联键**：`bill_no`（用户号码）

---

## 九、使用方式

### 9.1 在平台上运行【确定】

#### 方式一：命令行模式（推荐生产使用）
```bash
cd src
python main.py --csv_dir ../dataset_termchange --main_table temp_a_utrm_use_3m_inline_encrypt_202508 --target_column flag
```

#### 方式二：可视化演示模式（推荐演示汇报使用）
```bash
# 在项目根目录运行
streamlit run src/web_ui.py
```
> **提示**：需要额外安装 `streamlit` 和 `plotly`。此模式下，主要参数（世代、种群）可通过网页界面动态配置。

### 9.2 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_source` | 数据源类型：`csv` 或 `clickhouse` | `csv` |
| `--csv_dir` | CSV 数据目录路径 | `dataset_termchange` |
| `--main_table` | 主表名（包含目标变量） | - |
| `--target_column` | 目标变量列名 | `flag` |
| `--output_dir` | 输出目录 | `./saved_model` |
| `--generations` | 进化世代数 | `2` |
| `--population` | 种群规模 | `2` |
| `--ck_host` | ClickHouse 主机地址 | `localhost` |

### 9.3 本地手动运行与修改指南（脱离前端界面）

如果您在本地开发或不想使用 Web 界面，完全可以通过**修改命令行参数**或**修改默认配置代码**来运行系统。

#### 方法 A：使用命令行参数（推荐，不改代码）
通过传递参数覆盖默认设置，适合临时切换不同的数据集。

**示例 1：运行默认数据集（换机预测）**
```bash
python main.py
```

**示例 2：切换到新数据集（如 dataset_hcdr）**
```bash
python main.py \
  --csv_dir dataset_hcdr \
  --main_table application_train \
  --target_column TARGET \
  --generations 5 \
  --population 10
```

#### 方法 B：修改默认配置代码（固定配置）
如果您想永久修改默认的数据集路径，避免每次都敲命令，可以直接修改 `src/main.py` 顶部的配置区。

**文件位置**: `src/main.py` (约 560 行)

```python
# --- 默认配置区域（CSV模式使用） ---
# 修改这里的值即可
DEFAULT_DATA_FOLDER = Path(r"dataset_hcdr")          # <--- 改为新文件夹名
DEFAULT_TARGET_TABLE = "application_train"           # <--- 改为主表文件名(无后缀)
DEFAULT_TARGET_COLUMN = "TARGET"                     # <--- 改为目标列名
DEFAULT_SCHEMA_CONFIG = DEFAULT_DATA_FOLDER / "schema_config.json"
```

修改后，直接运行 `python main.py` 即可生效。

---

## 十、项目亮点【确定】| `--ck_port` | ClickHouse 端口 | `9000` |
| `--ck_database` | ClickHouse 数据库名 | `default` |
| `--aux_tables` | 副表名（逗号分隔） | - |

---

## 十、项目亮点【确定】

1. **规则驱动的 AutoML 原型**：代码不写死，根据 Schema 配置动态适配
2. **LLM 增强的语义与特征启发**：利用大语言模型提供智能建议（非决策）
3. **遗传算法搜索最优策略**：自动演化最优特征组合
4. **SHAP 精英学习**：对冠军策略进行特征重要性精炼
5. **端到端自动化产出**：从原始数据到生产 SQL + 发布代码
6. **平台适配友好**：自动生成本地版和敏捷平台兼容版模型
7. **短字段名 SQL 支持**：自动生成 f001~fxxx 短列名版 SQL
8. **扩展接口预留清晰**：对抗式评估框架已实现，可按需启用

---

## 十一、依赖安装

```bash
cd src
pip install -r requirements.txt
```

依赖列表：
- scikit-learn
- pandas
- numpy
- lightgbm
- shap
- openai
- joblib
- clickhouse-driver（可选，用于 ClickHouse 数据加载）

---

## 十二、待确认事项

| 事项 | 状态 | 说明 |
|-----|------|------|
| Saboteur 启用计划 | 🔒 暂不启用 | 框架已实现，权重=0 |
| 新业务场景适配 | ⚠️ 需逐一验证 | 准备 Schema + 数据表后需实际测试 |

---


## 十三、AI 协作使用约束（必须遵守）

1. **本文档是最高事实来源**  
   - 若代码与本文档冲突，应先质疑代码，而不是修改文档

2. **未写明的能力一律视为不存在**  
   - 禁止 AI 自行补全模块、能力、流程

3. **禁止 AI**：
   - 擅自引入新的 AutoML 框架
   - 将本项目表述为“通用平台”
   - 启用 saboteur 对抗机制（除非明确指示）

4. **对“推断”标注的内容**  
   - 任何使用前必须重新确认

---

## 十四、文档维护规则（必须遵守）

**当以下情况发生时，必须同步更新本文件并记录 Change Log**：

| 变更类型 | 是否需更新文档 |
|---------|-------------|
| 新增/删除核心模块 | ✅ 必须 |
| 工作流程结构性变化 | ✅ 必须 |
| 输出产物形式/用途变化 | ✅ 必须 |
| LLM 使用边界变化 | ✅ 必须 |
| 项目定位/适用范围变化 | ✅ 必须 |
| 纯重构/性能优化 | ❌ 可不更新 |
| 变量调整/代码风格 | ❌ 可不更新 |

---

## 十五、部署架构与工作流程【确定】

### 15.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AutoModel 部署架构                              │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│ 1. DM 数据开发平台        │
│    ├─ 编写 SQL 构造标签   │
│    ├─ 导出训练数据 CSV    │
│    └─ (未来) ETL→多维CK库 │
└───────────┬──────────────┘
            │ 人工上传 CSV
            ▼
┌──────────────────────────┐
│ 2. 多维分析平台主机       │   ← 有 LLM (Qwen3-32B)
│    ├─ 运行 AutoModel 训练 │
│    └─ 产出：              │
│       • pkl 模型          │
│       • 建宽表 SQL        │
│       • 发布代码模板      │
└───────────┬──────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌────────────┐  ┌─────────────────────┐
│ 3. DM 平台  │  │ 4. 敏捷挖掘平台      │  ← 无 LLM，无外网
│  执行 SQL   │  │    ├─ 上传 pkl 模型  │
│  建宽表     │  │    ├─ 粘贴发布代码   │
└─────┬──────┘  │    └─ 执行预测       │
      │         └─────────────────────┘
      │ 宽表数据            ▲
      └─────────────────────┘
```

### 15.2 平台能力对比

| 平台 | LLM | 外网 | ClickHouse | 主要用途 |
|-----|-----|-----|-----------|---------|
| **DM 数据开发平台** | ❌ | ❌ | ✅ | SQL 编写、数据 ETL |
| **多维分析平台主机** | ✅ Qwen3-32B | ✅ | ✅ (需 ETL) | **AutoModel 训练** |
| **敏捷挖掘平台** | ❌ | ❌ | ❌ | 模型发布、批量预测 |

### 15.3 当前数据传输方式

| 环节 | 当前方式 | 未来规划 |
|-----|---------|---------|
| 训练数据→多维主机 | 人工上传 CSV | ETL 到多维 CK 库 |
| 宽表数据→敏捷平台 | 通过数据源 ID 读取 | 保持不变 |

---

## 十六、多维分析平台主机环境【确定】

> 以下为已验证的环境配置，可作为兼容性参考基准。

| 包名 | 版本 | 状态 |
|-----|------|------|
| Python | 3.10.15 | ✅ OK |
| pandas | 2.2.3 | ✅ OK |
| numpy | 1.26.4 | ✅ OK |
| scikit-learn | 1.6.1 | ✅ OK |
| lightgbm | 3.3.5 | ✅ OK |
| shap | 0.46.0 | ✅ OK |
| openai | 1.73.0 | ✅ OK |
| clickhouse-driver | 已安装 | ✅ OK |

---

## 十七、敏捷挖掘平台兼容性问题清单【确定】

> 以下是训练阶段正常、但发布到敏捷平台时遇到的 pkl 兼容性问题及修复方案。  
> 所有修复已内置于 `platform_inference_template.py` 中。

| 序号 | 问题描述 | 错误信息特征 | 修复方式 |
|-----|---------|------------|---------|
| 1 | `_RemainderColsList` 缺失 | `AttributeError: module 'sklearn.compose._column_transformer' has no attribute '_RemainderColsList'` | 注入空类补丁 |
| 2 | `sklearn.preprocessing._label` 模块缺失 | `ModuleNotFoundError: No module named 'sklearn.preprocessing._label'` | 创建模块并挂载 LabelEncoder |
| 3 | `OneHotEncoder._legacy_mode` 属性缺失 | `AttributeError: 'OneHotEncoder' object has no attribute '_legacy_mode'` | `setattr(OneHotEncoder, '_legacy_mode', False)` |
| 4 | `OneHotEncoder.sparse` 属性缺失 | `AttributeError: 'OneHotEncoder' object has no attribute 'sparse'` | 从 `sparse_output` 复制属性到 `sparse` |
| 5 | `SimpleImputer` 序列化不兼容 | 加载时报 unpickle 错误 | 保存时用 `CompatibleImputer` 替换 + `protocol=2` |
| 6 | Encoder 输入类型不匹配 | `ValueError: could not convert string to float` | 预测前 `df.astype(str)` |

### 遇到新兼容性问题时的处理流程

1. 截图或复制完整错误信息（含 Traceback）
2. 分析是哪个 sklearn 组件的版本差异
3. 在 `platform_inference_template.py` 中添加对应补丁
4. 更新本文档的兼容性问题清单

### 常见 UI 问题排查

| 现象 | 可能原因 | 解决方案 |
|-----|--------|---------|
| **日志显示乱码** (如 `璇涔夋帹鏂`) | Windows 控制台编码默认为 GBK，与 Python 输出的 UTF-8 冲突 | 1. 尝试重启 Web UI <br> 2. 代码已强制注入 `PYTHONIOENCODING=utf-8`，请确保从 `web_ui.py` 启动而非直接运行脚本 |
| **点击启动没反应** | 上次运行的后台进程未完全关闭，锁定了日志文件 | 1. 点击 UI 上的 [停止] 按钮 <br> 2. 手动删除 `automl.log` <br> 3. 重启 Web UI |
| **找不到 main.py** | 启动路径不对 | 确保在项目根目录 `f:\AutoModel` 下运行 `streamlit run src/web_ui.py` |

---

## 十八、训练数据集构造说明【确定】

### 18.1 当前验证场景：换机预测

**数据表结构**：

| 表名 | 角色 | flag 字段 | 说明 |
|-----|------|----------|------|
| `temp_a_utrm_use_3m_inline_encrypt_202508` | **主表** | ✅ 有 | 用户终端使用月视图 |
| `temp_user_base_info_m_bak_encrypt_202508` | 副表 | ❌ 无 | 用户画像表 |
| `temp_a_upay_user_attr_m_bak_encrypt_202508` | 副表 | ❌ 无 | 消费行为表 |

**关联关系**：三表通过 `bill_no`（用户号码）字段关联

### 18.2 flag 标签构造逻辑

- **位置**：仅在主表最后一列
- **含义**：`1` = 换机，`0` = 未换机
- **构造方式**：在 DM 平台用 SQL 判断用户在下一账期是否更换终端，将结果追加为 `flag` 字段

### 18.3 数据集文件清单

```
dataset_termchange/
├── schema_config.json                              # 必需：知识图谱配置
├── temp_a_utrm_use_3m_inline_encrypt_202508.csv    # 主表（含 flag）
├── temp_user_base_info_m_bak_encrypt_202508.csv    # 副表：用户画像
├── temp_a_upay_user_attr_m_bak_encrypt_202508.csv  # 副表：消费行为
└── ziduan.txt                                      # 字段说明文档（用于生成 schema_config.json）
```

**文件说明**：
- `ziduan.txt`：人工整理的字段说明文档，用于辅助生成 `schema_config.json` 知识图谱配置
- `schema_config.json`：系统实际使用的知识图谱配置文件（基于 `ziduan.txt` 生成）

---

## 十九、ClickHouse 配置参考【暂存】

> 以下配置为多维分析平台 CK 库的历史配置，当前未启用。  
> 已保存至 `ck_config_example.json`，不写入代码，避免硬编码。

| 配置项 | 值 | 状态 |
|-------|-----|------|
| 服务器地址 | `10.179.75.162` | ⏸️ 待启用 |
| 端口号 | `9000` | ⏸️ 待启用 |
| 数据库名 | `default` | ⏸️ 待启用 |
| 用户名 | `chtest` | ⏸️ 待启用 |
| 密码 | `8675301...e063f9` | ⏸️ 待启用（已加密存储） |

**使用方式**：
```bash
# 当 CK 库配置好后，使用以下命令运行
python main.py --data_source clickhouse \
  --ck_host 10.179.75.162 \
  --ck_port 9000 \
  --ck_database default \
  --ck_user chtest \
  --ck_password "8675301adbe66cf0f090190fba81b0cdb29d3f405e82c02a3403d34442e063f9" \
  --main_table your_table \
  --aux_tables "table1,table2"
```

**注意**：未来 CK 配置可能变更，使用前请确认最新配置。

---

## 待优化 Backlog

> 以下为已识别但暂不实现的优化项，按优先级排列。每项实现后移入 CHANGELOG。

### P1 — 小数据量 + 正样本极少场景适配

**背景**：当前系统针对 40W 级数据 + 正样本 ~3% 优化。如果某些业务数据量很小、正样本极少，现有策略可能不够稳健。当前人工建模时通过手动控制正负样本比例解决，不繁琐，暂不自动化。

| 数据规模 | 正样本数 | 当前系统表现 | 需要加的功能 |
|---------|---------|------------|-------------|
| >10W，正样本 >3000 | 充足 | ✅ `is_unbalance` + KS 阈值够用 | 不需要 |
| 1~10W，正样本 500~3000 | 偏少 | ⚠️ 3 折 CV 每折正样本可能只有 100+ | 自动切换 5 折 CV |
| <1W，正样本 <500 | 极少 | ❌ LGBM 预筛选不稳定，CV 可能失败 | SMOTE 过采样 + 留一法 CV |

**实现思路**：
1. 在 `main.py` 启动时检测正样本数量和占比
2. 根据正样本数自动选择策略：
   - 正样本 ≥3000：当前默认策略（3 折 CV + is_unbalance）
   - 正样本 500~3000：切换 5 折 CV，保持 is_unbalance
   - 正样本 <500：启用 SMOTE 过采样 + 5 折 CV，或使用 `scale_pos_weight` 精确计算
3. 可选：支持 CLI 参数 `--sample_strategy auto|none|smote` 手动覆盖

**当前替代方案**：人工建数据集时手动控制 flag 正负样本数量（已验证可行）。

---

### P1 — 适应度函数可配置化（多业务指标适配）

**背景**：当前进化算法的适应度函数 = **纯 AUC**（`fitness_weights['auc'] = 1.0`，其余三项经济/因果/泛化惩罚权重均为 0）。Precision/Recall/F1 仅在评估报告中输出，不参与种群选择排名。这导致：

1. **进化方向单一**：只选择"排序能力强"的个体，不关心实际分类决策质量
2. **不同业务需要不同目标**：换机预测（营销场景）侧重 Precision/F1，风控侧重 Recall/KS，信用评分侧重 AUC —— 当前无法切换
3. **不平衡场景下的阈值问题**：KS 最优阈值在正样本仅 ~4.5% 时会将大量负样本预测为正，造成 Precision 极低（如 0.076），但 AUC 本身不反映这个问题

**注意**：AUC 本身不会导致"无脑预测为负"（AUC 衡量的是排序而非阈值决策），但只用 AUC 做适应度确实会忽略分类质量。

**拟实现方案**：
1. 新增 CLI 参数 `--fitness_metric`，支持以下选项：
   - `auc`（默认，当前行为）
   - `f1`（以 OOF F1-最优阈值下的 F1 为适应度）
   - `ks`（以 KS 统计量为适应度）
   - `auc_f1`（AUC × 0.6 + F1 × 0.4 复合得分）
   - `precision@k`（Top-K% 样本的 Precision，适合营销场景）
2. `FitnessEvaluator.evaluate()` 返回所有候选指标，`main.py` 根据 `--fitness_metric` 选择用哪个做综合得分
3. 阈值策略配套扩展：除 KS 最优阈值外，增加 F1 最优阈值搜索

| 业务场景 | 推荐 fitness_metric | 理由 |
|---------|-------------------|------|
| 换机预测（精准营销） | `f1` 或 `precision@k` | 减少无效推荐，控制营销成本 |
| 欺诈/风控 | `ks` 或 `auc` | 不遗漏高风险用户 |
| 流失预警 | `auc_f1` | 平衡排序和分类 |
| 信用评分 | `auc` | 纯排序能力 |

**当前状态**：方案待讨论确认后实施。

---

*文档结束。如有疑问，请联系项目维护者。*
