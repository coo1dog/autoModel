"""
对抗性共演化系统 - 架构师模块

这是本项目的核心模块，包含四个核心类：
- GeneGenerator: 基因生成器
- FeatureEngine: V3真实特征引擎  
- FitnessEvaluator: 真实评估器
- EvolutionaryEngine: 演化引擎

V1.0 架构约定：
- 实现完整的机器学习流水线
- 支持跨表聚合特征工程
- 真实的模型训练和评估
- 遗传算法优化
"""

import time
import random
import json
import warnings
from typing import Optional, List, Dict, Any, Tuple
import joblib

import numpy as np
import pandas as pd
import shap
# 统一禁止进入建模特征的字段（防泄漏）
# 增加常见主键/ID 列，避免误入建模
EXCLUDED_COLUMNS = set({
    "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"
})  # 可按需扩展，比如 {"duration", "y"}
# 新增：兜底所需的 sklearn 组件
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 导入我们项目中的其他模块
from core_structures import ModelingGene, FeatureGene, TransformGene, ModelGene, FilterGene, ModelingChromosome
from knowledge_graph_interface import KnowledgeGraphInterface
from llm_interface import llm_generate_cross_table_genes

import logging

# ... (other imports)

class GeneGenerator:
    """
    负责调用 LLM (占位符) 来动态生成初始基因池。
    """
    def __init__(self, translator: KnowledgeGraphInterface, target_variable: str, feature_config: Optional[Dict[str, Any]] = None):
        self.translator = translator
        self.target_variable = target_variable
        self.feature_config = feature_config or {}

    def _machine_screen_features(self, entity_name: str, top_k: Optional[int] = None) -> List[FeatureGene]:
        """
        使用一个轻量的 LGBM 模型对主表做预筛选，返回 Top-K 的 LATEST 特征。
        - 若训练失败或数据不满足条件，则降级为“从该表采样少量列”的兜底策略。
        - 只生成 LATEST 特征（不做任何跨表聚合）。
        """
        try:
            df = self.translator.get_entity_dataframe(entity_name)
        except Exception as e:
            logging.error(f"[架构师-基因][机器筛选-错误] 读取实体 {entity_name} 失败: {e}")
            return []

        if df is None or getattr(df, "empty", True):
            logging.warning(f"[架构师-基因][机器筛选-兜底] 实体 {entity_name} 数据为空")
            return []

        # 解析目标字段
        tgt_field = None
        if isinstance(self.target_variable, str) and '.' in self.target_variable:
            _, tgt_field = self.target_variable.split('.', 1)
        # 兜底：若目标列不在表中，则不做机器筛选
        if not tgt_field or tgt_field not in df.columns:
            logging.info(f"[架构师-基因][机器筛选-提示] 目标列 {tgt_field} 不在实体 {entity_name} 中，跳过机器筛选")
            return []

        y = df[tgt_field]
        X = df.drop(columns=[tgt_field])

        # 额外移除 ID/主键型列，避免被误选为特征
        try:
            id_like_cols = [c for c in X.columns if 'id' in c.lower() or 'sk_id' in c.lower()]
            if id_like_cols:
                X = X.drop(columns=id_like_cols, errors='ignore')
        except Exception:
            pass

        # 基本健壮性检查
        if len(X) < 50:
            logging.warning(f"[架构师-基因][机器筛选-兜底] 样本过少({len(X)}行)，跳过机器筛选")
            return []
        if X.shape[1] == 0:
            logging.warning("[架构师-基因][机器筛选-兜底] 无可用特征列")
            return []
        if getattr(y, 'nunique', lambda: 0)() < 2:
            logging.warning("[架构师-基因][机器筛选-兜底] 目标变量仅一个类别，跳过机器筛选")
            return []

        # 处理类别型特征
        try:
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                # 填充 NaN 以便 LabelEncoder 处理
                placeholder = '---missing---'
                X[col] = X[col].astype(str).fillna(placeholder)
                X[col] = le.fit_transform(X[col])
        except Exception as e:
            logging.warning(f"[架构师-基因][机器筛选-提示] 类别型转换失败: {e}")

        # 动态 top_k：可由配置覆盖；否则根据比例计算
        try:
            std_schema = self.translator.get_standard_schema() or {}
            num_tables = len(std_schema)
        except Exception:
            num_tables = 1
        
        if top_k is None:
            cfg_main = (self.feature_config or {}).get('main_table', {})
            total_features = X.shape[1]
            
            if num_tables <= 1:
                # 单表场景，使用 ratio 计算
                ratio = float(cfg_main.get('lgbm_top_k_single_ratio', 0.25))
                top_k = int(total_features * ratio)
            else:
                # 多表场景，使用 ratio 计算
                ratio = float(cfg_main.get('lgbm_top_k_multi_ratio', 0.15))
                top_k = int(total_features * ratio)
        
        # 确保 top_k 在合理范围内 [5, total_features]
        top_k = max(5, min(top_k, X.shape[1]))

        # 训练 LGBM 获取重要性
        try:
            lgbm = LGBMClassifier(random_state=42, verbose=-1, n_estimators=100, max_depth=8)
            lgbm.fit(X, y)
            importances = pd.Series(getattr(lgbm, 'feature_importances_', np.zeros(X.shape[1])), index=X.columns)
            if float(importances.sum()) == 0.0:
                logging.warning("[架构师-基因][机器筛选-兜底] 特征重要性全为0，跳过机器筛选")
                return []
            selected = importances.nlargest(top_k).index.tolist()
            logging.info(f"[架构师-基因][机器筛选] 主表Top-{len(selected)}: {selected[:10]}...")
            return [FeatureGene(op='LATEST', path=f"{entity_name}.{c}") for c in selected if c not in EXCLUDED_COLUMNS and c != tgt_field]
        except Exception as e:
            logging.warning(f"[架构师-基因][机器筛选-兜底] LGBM训练失败: {e}")
            return []

    def _generate_rule_based_window_genes(self, secondary_schema: Dict[str, List[str]]) -> List[FeatureGene]:
        """
        [V1.9 新增] 基于规则确定性地生成时间窗口特征。
        """
        logging.info("[架构师-基因] 启动基于规则的时间窗口基因生成...")
        rule_based_genes = []
        
        # 预定义的时间窗口和操作
        WINDOWS = [30, 90, 365]
        AGG_OPS = ['AVG', 'SUM', 'MAX', 'MIN']
        
        for entity_name, fields in secondary_schema.items():
            # 启发式寻找时间列 (简化逻辑)
            time_col = next((f for f in fields if 'DAYS' in f), None)
            if not time_col:
                logging.debug(f"  - 在表 {entity_name} 中未找到时间列，跳过窗口特征生成。")
                continue

            # 获取实体DataFrame以判断列类型
            try:
                entity_df = self.translator.get_entity_dataframe(entity_name)
                if entity_df is None or entity_df.empty:
                    continue
            except Exception:
                continue

            # 寻找适合聚合的数值列
            numeric_cols = [
                f for f in entity_df.select_dtypes(include=np.number).columns
                if f not in EXCLUDED_COLUMNS and 'SK_ID' not in f and f != time_col
            ]
            
            logging.info(f"  - 在表 {entity_name} 中找到时间列 '{time_col}' 和 {len(numeric_cols)} 个数值列。")
            
            # 对每个数值列、每个窗口、每个操作生成基因
            for col in numeric_cols:
                for op in AGG_OPS:
                    for window in WINDOWS:
                        # 注意：这里的实现简化了，我们假设 FeatureEngine 知道如何处理基于'DAYS'的窗口
                        rule_based_genes.append(FeatureGene(op=op, path=f"{entity_name}.{col}", window=window))

        logging.info(f"[架构师-基因] [OK] 基于规则成功生成 {len(rule_based_genes)} 个时间窗口基因。")
        return rule_based_genes

    def _detect_one_to_one_tables(self, primary_entity: str, relationships: Dict) -> set:
        """
        [V2.9 新增] 检测哪些副表与主表是 1:1 关系（月快照表）。
        判定标准：副表的 primary_key / join_key 与主表相同（如都是 bill_no），
        且副表数据行数与主表相近（±20%）。
        """
        one_to_one_entities = set()
        try:
            main_df = self.translator.get_entity_dataframe(primary_entity)
            if main_df is None or main_df.empty:
                return one_to_one_entities
            main_rows = len(main_df)

            for rel_name, rel_info in relationships.items():
                from_entity = rel_info.get('from_entity', '')
                to_entity = rel_info.get('to_entity', '')
                # 副表（from_entity）指向主表（to_entity）
                sec_entity = from_entity if to_entity == primary_entity else (
                    to_entity if from_entity == primary_entity else None)
                if not sec_entity or sec_entity == primary_entity:
                    continue
                try:
                    sec_df = self.translator.get_entity_dataframe(sec_entity)
                    if sec_df is None or sec_df.empty:
                        continue
                    # 检查主键/关联键
                    fk = rel_info.get('from_key') or rel_info.get('to_key')
                    if fk and fk in sec_df.columns:
                        unique_keys = sec_df[fk].nunique()
                        # 如果唯一键数 ≈ 行数，说明每个key只有1行
                        ratio = unique_keys / len(sec_df) if len(sec_df) > 0 else 0
                        if ratio > 0.95:  # 95%以上的key唯一 → 1:1
                            one_to_one_entities.add(sec_entity)
                            logging.info(f"[架构师-基因] 检测到 1:1 快照表: {sec_entity} (唯一率={ratio:.2%}, 行数={len(sec_df)})")
                        else:
                            logging.info(f"[架构师-基因] 检测到 1:N 流水表: {sec_entity} (唯一率={ratio:.2%}, 行数={len(sec_df)})")
                except Exception as e:
                    logging.warning(f"[架构师-基因] 检测表关系失败 {sec_entity}: {e}")
        except Exception as e:
            logging.warning(f"[架构师-基因] 1:1检测整体异常: {e}")
        return one_to_one_entities

    def _machine_screen_secondary_table(self, primary_entity: str, sec_entity: str,
                                         target_field: str, top_k_ratio: float = 0.35) -> List[FeatureGene]:
        """
        [V2.9 新增] 对 1:1 副表做 LGBM 预筛选。
        思路：将副表 LEFT JOIN 到主表，用主表的 y 训练轻量 LGBM，筛选副表中有预测力的字段。
        """
        try:
            main_df = self.translator.get_entity_dataframe(primary_entity)
            sec_df = self.translator.get_entity_dataframe(sec_entity)
            if main_df is None or sec_df is None or main_df.empty or sec_df.empty:
                return []
            if target_field not in main_df.columns:
                return []

            # 查找关联键
            relationships = self.translator.get_relationship_keys() or {}
            join_key = None
            for _, rel in relationships.items():
                if rel.get('from_entity') == sec_entity and rel.get('to_entity') == primary_entity:
                    join_key = rel.get('from_key') or rel.get('to_key')
                    break
                elif rel.get('to_entity') == sec_entity and rel.get('from_entity') == primary_entity:
                    join_key = rel.get('to_key') or rel.get('from_key')
                    break
            if not join_key:
                logging.warning(f"[架构师-基因] 未找到 {sec_entity} 的关联键，跳过")
                return []

            # 合并
            y = main_df[target_field]
            sec_cols_only = [c for c in sec_df.columns
                            if c != join_key and c != target_field
                            and c not in EXCLUDED_COLUMNS
                            and 'id' not in c.lower()[-3:]  # 排除xxx_id尾部字段
                            and c not in {'user_id', 'p_mon', 'city_id', 'county_id', 'p_city'}]
            if not sec_cols_only:
                return []

            merged = main_df[[join_key, target_field]].merge(
                sec_df[[join_key] + sec_cols_only], on=join_key, how='left')
            X = merged[sec_cols_only]
            y_merged = merged[target_field]

            # 类别型转码
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = X[col].astype(str).fillna('---missing---')
                X[col] = le.fit_transform(X[col])

            top_k = max(5, int(len(sec_cols_only) * top_k_ratio))

            lgbm = LGBMClassifier(random_state=42, verbose=-1, n_estimators=100, max_depth=8)
            lgbm.fit(X, y_merged)
            importances = pd.Series(lgbm.feature_importances_, index=X.columns)
            if importances.sum() == 0:
                return []
            selected = importances.nlargest(top_k).index.tolist()
            logging.info(f"[架构师-基因] 副表 {sec_entity} LGBM筛选 Top-{len(selected)}: {selected[:8]}...")
            return [FeatureGene(op='LATEST', path=f"{sec_entity}.{c}") for c in selected]
        except Exception as e:
            logging.warning(f"[架构师-基因] 副表 {sec_entity} LGBM筛选失败: {e}")
            return []

    def generate_initial_pool(self) -> List[ModelingGene]:
        """
        [V2.9 - 智能模式] 生成初始基因池：
        1) 检测副表与主表的关系类型（1:1 快照 vs 1:N 流水）
        2) 主表：LGBM 预筛选 → LATEST 特征
        3) 1:1 副表：同样 LGBM 预筛选 → LATEST 特征（跳过无意义的聚合）
        4) 1:N 副表：规则窗口 + LLM 跨表聚合特征
        5) 兜底 + 模型基因
        """
        logging.info("[架构师-基因] 启动基因生成：[V2.9 智能模式] 自动检测表关系类型")

        gene_pool: List[ModelingGene] = []

        # 1) 主表名称/目标字段解析
        primary_entity, target_field = self.target_variable.split('.', 1)
        
        # 2) 主表 LGBM 预筛选
        base_features = self._machine_screen_features(primary_entity)
        gene_pool.extend(base_features)
        logging.info(f"[架构师-基因] 预筛选得到主表基础特征数: {len(base_features)}")

        # 3) 追加跨表特征
        try:
            standard_schema = self.translator.get_standard_schema() or {}
            relationships = self.translator.get_relationship_keys() or {}
            secondary_schema = {k: v for k, v in standard_schema.items() if k != primary_entity}
            
            if secondary_schema:
                # [V2.9 核心] 检测哪些副表是 1:1 快照表
                one_to_one_entities = self._detect_one_to_one_tables(primary_entity, relationships)
                
                # 分离 1:1 快照副表 和 1:N 流水副表
                snapshot_entities = {k: v for k, v in secondary_schema.items() if k in one_to_one_entities}
                timeseries_entities = {k: v for k, v in secondary_schema.items() if k not in one_to_one_entities}
                
                logging.info(f"[架构师-基因] 副表分类: 1:1快照={list(snapshot_entities.keys())}, 1:N流水={list(timeseries_entities.keys())}")

                # 3a) [V2.9] 对 1:1 快照副表：用 LGBM 预筛选，直接生成 LATEST 基因
                for sec_entity in snapshot_entities:
                    sec_ratio = float((self.feature_config or {}).get('main_table', {}).get('lgbm_top_k_multi_ratio', 0.25))
                    sec_features = self._machine_screen_secondary_table(
                        primary_entity, sec_entity, target_field, top_k_ratio=sec_ratio)
                    gene_pool.extend(sec_features)
                    logging.info(f"[架构师-基因] 1:1副表 {sec_entity} 筛选得到 {len(sec_features)} 个LATEST特征")

                # 3b) 对 1:N 流水副表：保留原有的规则窗口 + LLM 策略
                if timeseries_entities:
                    rule_based_window_genes = self._generate_rule_based_window_genes(timeseries_entities)
                    gene_pool.extend(rule_based_window_genes)

                    primary_key = 'bill_no'
                    primary_entity_schema = self.translator.inferred_schema.get(primary_entity, {})
                    if isinstance(primary_entity_schema, dict) and primary_entity_schema.get('primary_key'):
                        primary_key = primary_entity_schema.get('primary_key')
                    logging.info(f"[架构师-基因] (LLM跨表) 1:N流水副表数={len(timeseries_entities)}")
                    genes_json = llm_generate_cross_table_genes(
                        secondary_schema=timeseries_entities,
                        primary_entity_name=primary_entity,
                        primary_key_name=primary_key,
                        target_variable=self.target_variable
                    )
                    
                    llm_gene_count = 0
                    if isinstance(genes_json, list):
                        for g in genes_json:
                            try:
                                op = g.get('op')
                                path = g.get('path')
                                if not op or not path or '.' not in path: continue
                                entity, field = path.split('.', 1)
                                if entity == primary_entity or entity not in standard_schema or field in EXCLUDED_COLUMNS: continue
                                has_relation = any(
                                    rel.get('from_entity') == entity and rel.get('to_entity') == primary_entity
                                    for _, rel in relationships.items()
                                )
                                if not has_relation: continue
                                gene_pool.append(FeatureGene(op=op, path=path, window=g.get('window')))
                                llm_gene_count += 1
                            except Exception:
                                continue
                    logging.info(f"[架构师-基因] [OK] 已从LLM加载 {llm_gene_count} 个跨表特征基因。")
                else:
                    logging.info("[架构师-基因] 所有副表均为1:1快照表，跳过LLM聚合特征生成（无意义）")
            else:
                logging.info("[架构师-基因] 单表场景：不调用规则或LLM生成器。")
        except Exception as e:
            logging.warning(f"[架构师-基因][警告] 跨表基因生成异常: {e}")


        # 4) 如果当前仍无任何特征，则从主表自动造一批 LATEST 特征作为兜底
        if not any(isinstance(g, FeatureGene) for g in gene_pool):
            logging.warning("[架构师-基因][兜底] LGBM和LLM均未产出特征，自动采样主表LATEST特征...")
            try:
                main_df = self.translator.get_entity_dataframe(primary_entity)
                if main_df is not None and not main_df.empty:
                    cols_to_exclude = {target_field} | EXCLUDED_COLUMNS
                    available_cols = [col for col in main_df.columns if col not in cols_to_exclude]
                    for col in available_cols[:15]:
                        gene_pool.append(FeatureGene(op="LATEST", path=f"{primary_entity}.{col}"))
            except Exception as e:
                logging.error(f"[架构师-基因][兜底-异常] 自动构造 LATEST 特征失败: {e}")

        # 5) 无论如何，确保有多个模型基因
        model_configs = [
            {"alg": "LGBMClassifier", "params": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42, "verbose": -1, "objective": "binary", "is_unbalance": True, "reg_alpha": 0.1, "reg_lambda": 0.1}},
            {"alg": "LGBMClassifier", "params": {"n_estimators": 150, "learning_rate": 0.08, "num_leaves": 63, "random_state": 42, "verbose": -1, "objective": "binary", "is_unbalance": True, "min_child_samples": 50}},
            {"alg": "LGBMClassifier", "params": {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "random_state": 42, "verbose": -1, "objective": "binary", "is_unbalance": True, "reg_alpha": 0.3, "reg_lambda": 0.3, "min_child_samples": 100}},
        ]
        for config in model_configs:
            gene_pool.append(ModelGene(alg=config["alg"], params=config["params"]))

        # [V1.7 调试] 将生成的特征基因保存到文件
        try:
            feature_genes_to_log = [g for g in gene_pool if isinstance(g, FeatureGene)]
            log_content = "--- Generated Feature Genes (Screening Mode) ---\n"
            log_content += f"Total Feature Genes: {len(feature_genes_to_log)}\n\n"
            
            main_table_genes = [f"  - {g.op}: {g.path}" for g in feature_genes_to_log if g.path.startswith(primary_entity)]
            cross_table_genes = [f"  - {g.op}: {g.path} (window: {g.window})" for g in feature_genes_to_log if not g.path.startswith(primary_entity)]

            log_content += f"Main Table Genes ({len(main_table_genes)}):\n"
            log_content += "\n".join(main_table_genes)
            log_content += f"\n\nCross-Table Genes ({len(cross_table_genes)}):\n"
            log_content += "\n".join(cross_table_genes)

            with open("generated_genes_log.txt", "w", encoding="utf-8") as f:
                f.write(log_content)
            logging.info("[架构师-基因] [OK] 已将生成的特征基因记录到 generated_genes_log.txt")
        except Exception as log_e:
            logging.warning(f"[架构师-基因][警告] 记录生成的基因失败: {log_e}")

        # 6) 去重
        seen = set()
        deduped: List[ModelingGene] = []
        for g in gene_pool:
            if isinstance(g, FeatureGene):
                key = ("F", g.op, g.path, g.window)
            elif isinstance(g, ModelGene):
                key = ("M", g.alg, tuple(sorted(g.params.items())) if isinstance(g.params, dict) else None)
            else:
                key = ("O", repr(g))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(g)

        logging.info(f"[架构师-基因] 成功生成 {len([x for x in deduped if isinstance(x, FeatureGene)])} 个特征基因 + {len([x for x in deduped if isinstance(x, ModelGene)])} 个模型基因。")
        return deduped


class FeatureEngine:
    """
    V3 真实特征工程引擎。
    负责将一个"染色体"翻译成一个可训练的 (X, y) 矩阵。
    """
    def __init__(self, translator: KnowledgeGraphInterface, standard_target_variable: str):
        self.translator = translator
        self.relationships = translator.get_relationship_keys()
        # (V1.1 新增) 保存标准目标变量信息
        self.target_entity, self.target_field = standard_target_variable.split('.')

    def build_features(self, chromosome: ModelingChromosome) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """
        V3 核心逻辑：
        1. (暂不实现) 处理 FilterGene
        2. 获取基础实体 (UserProfile)
        3. 处理 FeatureGene (LATEST 和 跨表聚合)
        4. (暂不实现) 处理 TransformGene
        5. 返回 X, y, 和特征类型列表
        """
        
        # 1) 取目标实体
        base_df = self.translator.get_entity_dataframe(self.target_entity)
        base_df = base_df.reset_index(drop=True)
        if base_df is None or base_df.empty:
            # 直接兜底到“单表全列基线”
            return self._build_baseline_X(pd.DataFrame(), self.target_field)

        # y (V1.3 修正: 使用与 baseline 一致的鲁棒查找逻辑)
        y_col_name = None
        normalized_target = self.target_field.lower().replace('_', '')
        if self.target_field in base_df.columns:
            y_col_name = self.target_field
        else:
            for c in base_df.columns:
                if c.lower().replace('_', '') == normalized_target:
                    y_col_name = c
                    break
        
        if y_col_name:
            y = base_df[y_col_name]
        else:
            y = pd.Series(0, index=base_df.index, dtype=int)


        # 2) X 初始化
        # 收集所有新生成的特征列，最后一次性合并，避免DataFrame碎片化
        new_feature_series_list = []

        # 3) 解析 FeatureGene（允许 LATEST 直接从当前目标实体取值）
        feature_genes = [g for g in chromosome.genes if isinstance(g, FeatureGene)]
        if not feature_genes:
            # 没有任何特征基因：自动采样少量LATEST特征，避免全列兜底
            logging.info(f"[特征引擎-提示] 染色体无特征基因，自动采样主表LATEST特征...")
            safe_cols = [
                c for c in base_df.columns
                if c not in EXCLUDED_COLUMNS and c != self.target_field
            ]
            sampled = safe_cols[: min(16, len(safe_cols))]
            for c in sampled:
                new_feature_series_list.append(base_df[c].rename(f"LATEST_{self.target_entity}_{c}"))
            logging.info(f"[特征引擎-提示] 已自动采样 {len(new_feature_series_list)} 个LATEST特征")
            
            if not new_feature_series_list:
                return self._build_baseline_X(base_df, self.target_field)
            
            X = pd.concat(new_feature_series_list, axis=1)
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = [c for c in X.columns if c not in numeric_features]
            return X, y, numeric_features, categorical_features

        for gene in feature_genes:
            try:
                entity_name, field_name = gene.path.split('.', 1)
            except Exception:
                continue

            feature_name = f"{gene.op}_{entity_name}_{field_name}"
            if gene.window:
                feature_name += f"_{gene.window}d"
            
            # 检查是否已存在，避免重复添加
            if any(s.name == feature_name for s in new_feature_series_list):
                continue

            if gene.op == 'LATEST':
                feature_generated = False
                if entity_name == self.target_entity and field_name in base_df.columns:
                    if field_name not in EXCLUDED_COLUMNS:
                        new_feature_series_list.append(base_df[field_name].rename(feature_name))
                        feature_generated = True
                elif field_name in base_df.columns:
                    if field_name not in EXCLUDED_COLUMNS:
                        logging.warning(f"[特征引擎-兜底] 实体名不匹配({entity_name}!={self.target_entity})，但字段 {field_name} 存在，使用该字段")
                        new_feature_series_list.append(base_df[field_name].rename(feature_name))
                        feature_generated = True
                
                # [V2.9 关键新增] 对跨表 LATEST 基因：通过 JOIN 副表获取字段值
                if not feature_generated and entity_name != self.target_entity:
                    try:
                        sec_df = self.translator.get_entity_dataframe(entity_name)
                        if sec_df is not None and not sec_df.empty and field_name in sec_df.columns:
                            # 查找关联键
                            join_key = None
                            for _, rel in (self.relationships or {}).items():
                                if rel.get('from_entity') == entity_name and rel.get('to_entity') == self.target_entity:
                                    join_key = rel.get('from_key') or rel.get('to_key')
                                    break
                                elif rel.get('to_entity') == entity_name and rel.get('from_entity') == self.target_entity:
                                    join_key = rel.get('to_key') or rel.get('from_key')
                                    break
                            if join_key and join_key in sec_df.columns and join_key in base_df.columns:
                                merged = base_df[[join_key]].merge(
                                    sec_df[[join_key, field_name]].drop_duplicates(subset=[join_key]),
                                    on=join_key, how='left')
                                new_feature_series_list.append(merged[field_name].rename(feature_name))
                                feature_generated = True
                                logging.info(f"[特征引擎] [OK] 跨表LATEST: {feature_name} (通过 {join_key} JOIN)")
                    except Exception as e:
                        logging.warning(f"[特征引擎-警告] 跨表LATEST JOIN失败 {gene.path}: {e}")
                
                if not feature_generated:
                    logging.warning(f"[特征引擎-警告] 无法找到LATEST特征 '{gene.path}'，将创建全为NaN的占位符列。")
                    placeholder_series = pd.Series(np.nan, index=base_df.index, name=feature_name)
                    new_feature_series_list.append(placeholder_series)
                
                continue
            elif gene.op in ['AVG', 'COUNT', 'SUM', 'MAX', 'MIN']:
                relation = None
                matched_key = None

                try:
                    for rel_key, rel_info in (self.relationships or {}).items():
                        if rel_info.get('from_entity') == entity_name and rel_info.get('to_entity') == self.target_entity:
                            relation = rel_info
                            matched_key = rel_key
                            break
                except Exception:
                    pass

                if relation is None:
                    possible_relation_keys = [
                        f"{entity_name}_to_{self.target_entity}",
                        f"{entity_name}_to_LoanApplication",
                        f"{entity_name}_to_UserProfile"
                    ]
                    for rel_key in possible_relation_keys:
                        if rel_key in self.relationships:
                            relation = self.relationships[rel_key]
                            matched_key = rel_key
                            break

                if not relation:
                    logging.warning(f"[特征引擎-警告] 找不到 {entity_name} -> {self.target_entity} 的关系映射，跳过 {gene.path}")
                    continue

                logging.info(f"[特征引擎] 使用关系: {matched_key} 构建跨表特征 {feature_name}")
                
                fk_from = relation['from_key']
                pk_to = relation['to_key']
                from_entity = relation.get('from_entity', entity_name)
                
                fact_df = self.translator.get_entity_dataframe(from_entity)
                if fact_df is None or fact_df.empty:
                    logging.warning(f"[特征引擎-警告] 无法获取实体 {from_entity} 的数据")
                    continue
                
                if field_name not in fact_df.columns:
                    logging.warning(f"[特征引擎-警告] 字段 {field_name} 不在 {from_entity} 中")
                    continue
                
                if fk_from not in fact_df.columns:
                    logging.warning(f"[特征引擎-警告] 外键 {fk_from} 不在 {from_entity} 中")
                    continue
                
                if pk_to not in base_df.columns:
                    logging.warning(f"[特征引擎-警告] 主键 {pk_to} 不在目标表 {self.target_entity} 中")
                    continue

                agg_op = gene.op.lower()
                if agg_op == 'avg': agg_op = 'mean'
                if agg_op == 'count':
                    agg_op = 'count'

                # [V1.9 新增] 对时间窗口基因的特殊处理
                fact_df_filtered = fact_df
                if gene.window is not None:
                    # 启发式寻找时间列
                    time_col = next((f for f in fact_df.columns if 'DAYS' in f), None)
                    if time_col:
                        logging.info(f"    - 应用时间窗口: {gene.window} 天, 基于列: {time_col}")
                        # 假设 'DAYS_' 列是负数，代表过去的天数
                        fact_df_filtered = fact_df[fact_df[time_col] >= -gene.window]
                    else:
                        logging.warning(f"    - 基因请求了窗口但未在 {from_entity} 中找到'DAYS_'时间列，将进行全局聚合。")
                
                try:
                    agg_series = fact_df_filtered.groupby(fk_from)[field_name].agg(agg_op)
                    agg_series.name = feature_name
                    agg_df = agg_series.reset_index()
                    agg_df.columns = [pk_to, feature_name]
                    merged = base_df[[pk_to]].merge(agg_df, on=pk_to, how='left')
                    new_feature_series = merged[feature_name].rename(feature_name)
                    new_feature_series_list.append(new_feature_series)
                    logging.info(f"[特征引擎] [OK] 成功构建跨表特征 {feature_name} ({len(agg_df)} 聚合值)")
                except Exception as merge_error:
                    logging.warning(f"[特征引擎-警告] 聚合或合并失败 {feature_name}: {merge_error}")
                    continue
        
        # 4) 清理 & 判空
        if not new_feature_series_list:
            logging.info(f"[特征引擎-提示] 基因未产出任何特征，回退到全列基线")
            return self._build_baseline_X(base_df, self.target_field)

        # 一次性合并所有新特征
        X = pd.concat(new_feature_series_list, axis=1)
        X = X.dropna(axis=1, how='all') # 再次清理全NaN列

        if X is None or X.shape[1] == 0:
            logging.warning(f"[特征引擎-警告] 无法采样任何特征，回退到全列基线")
            return self._build_baseline_X(base_df, self.target_field)

        # 5) 列类型拆分
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        # [V1.9 修正] 强制将所有分类特征转为字符串，避免混合类型导致编码器错误
        for col in categorical_features:
            X[col] = X[col].astype(str)

        return X, y, numeric_features, categorical_features

    def _build_baseline_X(self, df: pd.DataFrame, target_col: str):
        """
        单表全列基线特征（除目标列外全部进 X）。
        注意：这里返回的是 DataFrame/Series；具体缺失值补齐/标准化/OneHot 交给 Evaluator 的 Pipeline 做。
        """
        if df is None or df.empty:
            # 返回空壳，Evaluator 会判空
            return pd.DataFrame(), pd.Series(dtype=int), [], []

        df = df.copy()
        
        # y 强制为 0/1（保留原有映射）
        # V1.2 修正：更鲁棒的大小写不敏感查找（处理下划线）
        y_col_name = None
        normalized_target = target_col.lower().replace('_', '')
        if target_col in df.columns:
            y_col_name = target_col
        else:
            for c in df.columns:
                if c.lower().replace('_', '') == normalized_target:
                    y_col_name = c
                    break
        
        if y_col_name:
            y = df[y_col_name]
        else:
            # 如果真的找不到，则创建全为0的列作为兜底
            y = pd.Series(0, index=df.index)

        # y 强制为 0/1（保留原有映射）
        if y.dtype == object:
            y = y.map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).fillna(y).astype(str)
            y = y.astype('category').cat.codes
        y = y.astype(int)

        # 一次性剔除目标列 + 禁用列
        to_drop = {y_col_name, target_col} | (EXCLUDED_COLUMNS & set(df.columns))
        X = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        logging.info(f"[特征引擎-兜底] 使用全列表基线(已排除 {sorted([c for c in to_drop if c is not None])}): "
              f"num={len(numeric_features)}, cat={len(categorical_features)}, X形状={X.shape}")
        return X, y, numeric_features, categorical_features


class FitnessEvaluator:
    """
    V1.0 真实适应度评估器。
    调用特征引擎，并使用 Sklearn Pipeline 真实地训练和评估模型。
    """
    def __init__(self, feature_engine: FeatureEngine):
        self.feature_engine = feature_engine

    def _prepare_evaluation_context(
        self, chromosome: ModelingChromosome
    ) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], Pipeline]:
        """Build features and the reusable sklearn pipeline for a chromosome."""
        X, y, num_features_orig, cat_features_orig = self.feature_engine.build_features(chromosome)

        if y.nunique() < 2:
            raise ValueError("Target variable has less than 2 classes.")

        # Double-check excluded columns before the pipeline is constructed.
        for bad_col in list(EXCLUDED_COLUMNS):
            if hasattr(X, "columns") and bad_col in X.columns:
                logging.warning(f"[评估器] 发现被禁用列残留，已移除: {bad_col}")
                X = X.drop(columns=[bad_col])
                if bad_col in num_features_orig:
                    num_features_orig.remove(bad_col)
                if bad_col in cat_features_orig:
                    cat_features_orig.remove(bad_col)

        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features_orig),
                ('cat', categorical_transformer, cat_features_orig)
            ],
            remainder='passthrough'
        )

        model_gene = next((g for g in chromosome.genes if isinstance(g, ModelGene)), None)
        if model_gene:
            if model_gene.alg == 'LGBMClassifier':
                model_template = LGBMClassifier(**model_gene.params)
            else:
                logging.warning(f"[警告] 未知的模型算法: {model_gene.alg}，使用默认的LGBMClassifier。")
                model_template = LGBMClassifier(random_state=42, verbose=-1)
        else:
            logging.warning(f"[警告] 染色体中没有模型基因，使用默认的LGBMClassifier。")
            model_template = LGBMClassifier(random_state=42, verbose=-1)

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_template)
        ])
        return X, y, num_features_orig, cat_features_orig, model_pipeline

    def _fit_pipeline_and_collect_artifacts(
        self,
        model_pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        calculate_shap: bool = False
    ) -> Dict[str, Any]:
        """Train the final pipeline and optionally compute SHAP on sampled rows."""
        artifacts: Dict[str, Any] = {}
        logging.info("[评估器] 正在训练最终模型和预处理器...")
        final_model = None

        model_pipeline.fit(X, y)
        artifacts['pipeline'] = model_pipeline
        logging.info("[评估器] [OK] 最终模型和预处理器训练完成。")

        if not calculate_shap:
            return artifacts

        logging.info("[SHAP] 启动 SHAP 分析...")
        try:
            SHAP_SAMPLE_SIZE = 8000
            if len(X) > SHAP_SAMPLE_SIZE:
                shap_sample_idx = np.random.RandomState(42).choice(len(X), SHAP_SAMPLE_SIZE, replace=False)
                X_shap = X.iloc[shap_sample_idx]
                logging.info(f"[SHAP] 数据量({len(X)})超过阈值，已采样 {SHAP_SAMPLE_SIZE} 行用于SHAP分析")
            else:
                X_shap = X

            X_processed = model_pipeline.named_steps['preprocessor'].transform(X_shap)
            final_model = model_pipeline.named_steps['classifier']

            if isinstance(final_model, (LGBMClassifier, RandomForestClassifier)):
                explainer = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(X_processed)

                shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
                mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
                feature_names_out = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
                shap_dict = dict(zip(feature_names_out, mean_abs_shap))
                artifacts['shap_values'] = shap_dict
                logging.info(f"[SHAP] [OK] 分析完成，获得 {len(shap_dict)} 个特征的 SHAP 值。")
            else:
                logging.warning(f"[SHAP] 警告: 模型 {type(final_model).__name__} 不是受支持的 Tree 模型，跳过 SHAP 分析。")
        except Exception as shap_e:
            logging.error(f"[评估器] 错误: SHAP分析失败: {shap_e}")

        return artifacts

    def finalize_chromosome(
        self,
        chromosome: ModelingChromosome,
        base_result: Optional[Dict[str, Any]] = None,
        calculate_shap: bool = False
    ) -> Dict[str, Any]:
        """
        Reuse existing CV metrics and only run final fit/SHAP for the elite chromosome.
        This avoids re-running 3-fold CV during elite refinement.
        """
        start_time = time.time()
        result = dict(base_result or {})

        try:
            X, y, num_features_orig, cat_features_orig, model_pipeline = self._prepare_evaluation_context(chromosome)
            artifacts = self._fit_pipeline_and_collect_artifacts(
                model_pipeline=model_pipeline,
                X=X,
                y=y,
                calculate_shap=calculate_shap
            )
            result.update(artifacts)
            result['feature_count'] = len(num_features_orig) + len(cat_features_orig)

            cv_time_ms = float((base_result or {}).get('cv_evaluation_time_ms', (base_result or {}).get('evaluation_time_ms', 0.0)))
            final_fit_time_ms = (time.time() - start_time) * 1000
            result['cv_evaluation_time_ms'] = cv_time_ms
            result['final_training_time_ms'] = final_fit_time_ms
            result['evaluation_time_ms'] = cv_time_ms + final_fit_time_ms
            return result
        except Exception as e:
            logging.error(f"[评估器] 错误: 精英终训失败: {e}")
            result['final_model_error'] = str(e)
            result['final_training_time_ms'] = (time.time() - start_time) * 1000
            result['evaluation_time_ms'] = float(result.get('cv_evaluation_time_ms', result.get('evaluation_time_ms', 0.0))) + result['final_training_time_ms']
            return result

    def evaluate(self, chromosome: ModelingChromosome, calculate_shap: bool = False, train_final_model: bool = True) -> Dict[str, Any]:
        """
        [V1.1] 执行完整的"评估"流程 (使用3-折交叉验证)，并返回一个包含"分数"和"成本"的字典。
        [V1.4] 新增：可选择性地计算并返回 SHAP 值。
        [V1.5] 修正：对齐基线脚本的预处理和评估流程。
        [V2.9.1] 新增 train_final_model 参数：为 False 时跳过全量 fit，仅做 CV 出 AUC，节省 ~38% 训练时间。
        """
        start_time = time.time()
        
        try:
            X, y, num_features_orig, cat_features_orig, model_pipeline = self._prepare_evaluation_context(chromosome)

            # 5. [V2.8] 3-折交叉验证 (单次CV同时获取AUC和概率，避免双重计算)
            logging.debug("[调试] 开始3-折交叉验证...")
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # [V2.8 优化] 使用 cross_val_predict 一次获取OOF概率，同时计算AUC和扩展指标
            from sklearn.model_selection import cross_val_predict
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
                y_proba = cross_val_predict(model_pipeline, X, y, cv=kfold, method='predict_proba')[:, 1]
            
            auc_mean = float(roc_auc_score(y, y_proba))
            # 用bootstrap估算std
            auc_std = 0.0
            try:
                boot_aucs = []
                rng = np.random.RandomState(42)
                for _ in range(5):
                    idx = rng.choice(len(y), len(y), replace=True)
                    if len(np.unique(y.iloc[idx] if hasattr(y, 'iloc') else y[idx])) > 1:
                        boot_aucs.append(roc_auc_score(y.iloc[idx] if hasattr(y, 'iloc') else y[idx], y_proba[idx]))
                auc_std = float(np.std(boot_aucs)) if boot_aucs else 0.0
            except Exception:
                pass
            logging.debug(f"[调试] 交叉验证完成, OOF AUC={auc_mean:.4f}, Std≈{auc_std:.4f}")

            # [V2.8] 扩展评估指标：KS / 最优阈值Precision / Recall / F1
            extended_metrics = {}
            try:
                # KS 值计算
                pos_proba = np.sort(y_proba[y == 1])
                neg_proba = np.sort(y_proba[y == 0])
                all_thresholds = np.sort(np.unique(y_proba))
                ks_stat = 0.0
                best_threshold = 0.5
                for t in all_thresholds:
                    tpr = np.mean(pos_proba >= t)
                    fpr = np.mean(neg_proba >= t)
                    current_ks = abs(tpr - fpr)
                    if current_ks > ks_stat:
                        ks_stat = current_ks
                        best_threshold = float(t)
                extended_metrics['ks'] = round(float(ks_stat), 4)
                extended_metrics['best_threshold'] = round(best_threshold, 4)

                # [V2.8 关键改进] 使用KS最优阈值而非固定0.5
                y_pred = (y_proba >= best_threshold).astype(int)
                logging.info(f"  [阈值] 使用KS最优阈值: {best_threshold:.4f} (替代固定0.5)")

                extended_metrics['precision'] = round(float(precision_score(y, y_pred, zero_division=0)), 4)
                extended_metrics['recall'] = round(float(recall_score(y, y_pred, zero_division=0)), 4)
                extended_metrics['f1'] = round(float(f1_score(y, y_pred, zero_division=0)), 4)

                tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                extended_metrics['confusion_matrix'] = {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}

                logging.info(f"  [指标] KS={extended_metrics['ks']:.4f}, Precision={extended_metrics['precision']:.4f}, Recall={extended_metrics['recall']:.4f}, F1={extended_metrics['f1']:.4f}")
            except Exception as e_ext:
                logging.warning(f"[评估器] 扩展指标计算失败(不影响主流程): {e_ext}")

            result = {
                'auc': auc_mean,
                'auc_mean': auc_mean,
                'auc_std': auc_std,
                'feature_count': len(num_features_orig) + len(cat_features_orig),
                'cv_evaluation_time_ms': 0.0,
                **extended_metrics
            }
            result['cv_evaluation_time_ms'] = (time.time() - start_time) * 1000

            # [V2.9.1 优化] 仅当 train_final_model=True 时才在全量数据上训练最终模型
            # 普通评估（搜索阶段）只需 CV 出 AUC 即可，无需 full fit，节省 ~38% 训练时间
            if train_final_model:
                try:
                    result.update(
                        self._fit_pipeline_and_collect_artifacts(
                            model_pipeline=model_pipeline,
                            X=X,
                            y=y,
                            calculate_shap=calculate_shap
                        )
                    )
                except Exception as final_model_e:
                    logging.error(f"[评估器] 错误: 最终模型训练失败: {final_model_e}")
                    result['final_model_error'] = str(final_model_e)
            else:
                logging.debug("[评估器] 搜索阶段评估，跳过全量训练 (train_final_model=False)")

            result['evaluation_time_ms'] = (time.time() - start_time) * 1000
            return result


        except Exception as e:
            logging.error(f"[评估错误] {str(e)}")
            evaluation_time_ms = (time.time() - start_time) * 1000
            return {'auc': 0.0, 'evaluation_time_ms': evaluation_time_ms, 'error': str(e)}


class EvolutionaryEngine:
    """
    实现遗传算法（选择、交叉、变异）的纯逻辑。
    """
    def __init__(self, gene_pool: List[ModelingGene]):
        self.gene_pool = gene_pool
        self.feature_genes = [g for g in gene_pool if isinstance(g, FeatureGene)]
        self.model_genes = [g for g in gene_pool if isinstance(g, ModelGene)]
        # (未来可添加 Transform/Filter 基因)

    def _get_gene_key(self, gene: ModelingGene) -> Any:
        """为 ModelingGene 生成一个可哈希的键"""
        if isinstance(gene, FeatureGene):
            return ("F", gene.op, gene.path, gene.window)
        elif isinstance(gene, ModelGene):
            # 将字典转换为可哈希的元组，确保可哈希性
            return ("M", gene.alg, tuple(sorted(gene.params.items())) if isinstance(gene.params, dict) else None)
        elif isinstance(gene, TransformGene): # 假设 TransformGene 也有可能出现，虽然目前没有用到
            return ("T", gene.op, tuple(sorted(gene.inputs)) if isinstance(gene.inputs, list) else gene.inputs)
        elif isinstance(gene, FilterGene): # 假设 FilterGene 也有可能出现，虽然目前没有用到
            return ("L", gene.condition)
        else:
            return ("O", repr(gene)) # 其他未知基因类型，用repr兜底

    def initialize_population(self, size: int, config: Optional[Dict[str, Any]] = None) -> List[ModelingChromosome]:
        """V1.3 智能初始化 (动态特征数量，可配置)：确保染色体可被评估
        config: {
          'min_features_ratio': float,  # 默认0.15
          'max_features_ratio': float,  # 默认0.40
          'max_features_floor': int     # 默认8
        }
        """
        population = []
        if not self.feature_genes or not self.model_genes:
            raise ValueError("基因池中缺少必要的 FeatureGene 或 ModelGene！")

        total_features = len(self.feature_genes)
        # 根据配置/默认参数，动态决定初始特征数量范围
        cfg = config or {}
        min_ratio = float(cfg.get('min_features_ratio', 0.15))
        max_ratio = float(cfg.get('max_features_ratio', 0.40))
        max_floor = int(cfg.get('max_features_floor', 8))

        min_count = max(1, int(total_features * min_ratio))
        percentage_based_max = min(total_features, max(min_count, int(total_features * max_ratio)))
        max_count = min(total_features, max(percentage_based_max, max_floor))
        if min_count > max_count:
            min_count = max_count
        logging.info(f"[演化引擎] 动态初始化特征数范围: [{min_count}, {max_count}] (总特征池: {total_features})")

        for _ in range(size):
            genes = []
            # 从动态范围中随机选择特征数量
            if min_count >= max_count:
                num_features = min_count
            else:
                num_features = random.randint(min_count, max_count)
            if num_features > 0:
                genes.extend(random.sample(self.feature_genes, num_features))
            genes.append(random.choice(self.model_genes))
            population.append(ModelingChromosome(genes=genes))
        return population

    def select(self, population: List[ModelingChromosome], scores: List[float]) -> List[ModelingChromosome]:
        """V1.0 锦标赛选择"""
        selected = []
        pop_size = len(population)
        for _ in range(pop_size):
            i, j = random.sample(range(pop_size), 2)
            winner_idx = i if scores[i] > scores[j] else j
            selected.append(population[winner_idx])
        return selected

    def crossover(self, parent1: ModelingChromosome, parent2: ModelingChromosome) -> ModelingChromosome:
        """
        [V1.7] 基于集合的交叉 (Set-based Crossover)：
        1. 找出父母双方共有的特征基因 (交集)。
        2. 找出父母双方独有的特征基因 (对称差集)。
        3. 子代继承交集基因，并从对称差集中随机选择一部分基因进行补充。
        4. 继承父1的模型基因。
        """
        # 继承父1的模型，若不存在则随机兜底
        mg = [g for g in parent1.genes if isinstance(g, ModelGene)]
        model_gene = mg[0] if mg else (random.choice(self.model_genes) if self.model_genes else None)

        p1_feature_genes = {self._get_gene_key(g): g for g in parent1.genes if isinstance(g, FeatureGene)}
        p2_feature_genes = {self._get_gene_key(g): g for g in parent2.genes if isinstance(g, FeatureGene)}

        # 1. 交集：父母双方都拥有的基因
        common_gene_keys = p1_feature_genes.keys() & p2_feature_genes.keys()
        child_feature_genes = [p1_feature_genes[key] for key in common_gene_keys]

        # 2. 对称差集：父母双方独有的基因
        unique_gene_keys = p1_feature_genes.keys() ^ p2_feature_genes.keys()
        unique_genes = [p1_feature_genes[key] for key in unique_gene_keys if key in p1_feature_genes] + \
                       [p2_feature_genes[key] for key in unique_gene_keys if key in p2_feature_genes]

        # 3. 从独有基因中随机选择一部分进行补充
        # 随机选择补充的数量，例如在 20% 到 80% 之间
        if unique_genes:
            num_to_add = random.randint(int(len(unique_genes) * 0.2), int(len(unique_genes) * 0.8))
            child_feature_genes.extend(random.sample(unique_genes, min(num_to_add, len(unique_genes))))
        
        # [V2.8] 子代特征数量限制：动态适配基因池大小
        pool_size = len(self.feature_genes) if self.feature_genes else 50
        min_features = max(5, int(pool_size * 0.05))  # 至少5个或池的5%
        max_features = max(50, int(pool_size * 0.6))   # 至多池的60%
        
        if len(child_feature_genes) < min_features and self.feature_genes:
            # 如果特征太少，从基因池中随机补充
            missing_count = min_features - len(child_feature_genes)
            current_keys = {self._get_gene_key(g) for g in child_feature_genes}
            potential_additions = [g for g in self.feature_genes if self._get_gene_key(g) not in current_keys]
            child_feature_genes.extend(random.sample(potential_additions, min(missing_count, len(potential_additions))))
        
        if len(child_feature_genes) > max_features:
            # 如果特征太多，随机删除一部分
            child_feature_genes = random.sample(child_feature_genes, max_features)

        genes = child_feature_genes + ([model_gene] if model_gene is not None else [])
        return ModelingChromosome(genes=genes)

    def mutate(self, chromosome: ModelingChromosome) -> ModelingChromosome:
        """V1.1 变异 (增加/删除/替换)"""
        if random.random() < 0.3:
            genes = chromosome.genes[:]  # 操作副本
            mutation_roll = random.random()

            # 20% 概率增加一个特征
            if mutation_roll < 0.2 and self.feature_genes:
                current_genes_keys = {self._get_gene_key(g) for g in genes}
                potential_additions = [g for g in self.feature_genes if self._get_gene_key(g) not in current_genes_keys]
                if potential_additions:
                    genes.append(random.choice(potential_additions))
                    # 重建 genes 列表，确保添加后没有重复
                    genes = list({self._get_gene_key(g): g for g in genes}.values())
                    logging.debug("[变异] 增加基因")
                    return ModelingChromosome(genes=genes)

            # 20% 概率删除一个特征
            elif mutation_roll < 0.4:
                feature_genes_in_chromo = [g for g in genes if isinstance(g, FeatureGene)]
                if len(feature_genes_in_chromo) > 1:  # 至少保留一个特征
                    gene_to_remove = random.choice(feature_genes_in_chromo)
                    genes.remove(gene_to_remove)
                    # 重建 genes 列表，确保删除后没有其他重复
                    genes = list({self._get_gene_key(g): g for g in genes}.values())
                    logging.debug("[变异] 删除基因")
                    return ModelingChromosome(genes=genes)

            # 60% 概率替换一个基因 (原逻辑)
            else:
                if not genes: return chromosome
                idx_to_mutate = random.randrange(len(genes))
                gene_to_mutate = genes[idx_to_mutate]

                if isinstance(gene_to_mutate, FeatureGene) and self.feature_genes:
                    # 确保替换的基因是新的
                    current_genes_keys = {self._get_gene_key(g) for g in genes}
                    potential_replacements = [g for g in self.feature_genes if self._get_gene_key(g) not in current_genes_keys]
                    if potential_replacements:
                        genes[idx_to_mutate] = random.choice(potential_replacements)
                        # 重建 genes 列表，确保替换后去重
                        genes = list({self._get_gene_key(g): g for g in genes}.values())
                    logging.debug("[变异] 替换特征基因")
                elif isinstance(gene_to_mutate, ModelGene) and self.model_genes:
                    genes[idx_to_mutate] = random.choice(self.model_genes)
                    logging.debug("[变异] 替换模型基因")
                return ModelingChromosome(genes=genes)

        return chromosome

    def refine_chromosome(self, chromosome: ModelingChromosome, shap_values: Dict[str, float]) -> ModelingChromosome:
        """
        [V1.4] 使用 SHAP 值来“精炼”一个染色体。
        它会找到贡献度最低的特征，并用基因池中的一个新特征替换它。
        """
        if not shap_values:
            return chromosome

        # 1. 找到 SHAP 值最低的特征名
        # SHAP value 的 key 可能是 'num__LATEST_entity_field' 或 'cat__...'
        min_shap_feature = min(shap_values, key=shap_values.get)
        
        # 2. 从染色体中找到对应的基因
        target_gene_to_remove = None
        original_genes = [g for g in chromosome.genes if isinstance(g, FeatureGene)]

        for gene in original_genes:
            # 构建一个可能的特征名来进行模糊匹配
            # gene.path = "application_train.DAYS_BIRTH" -> "application_train_DAYS_BIRTH"
            gene_feature_part = gene.path.replace('.', '_')
            if gene_feature_part in min_shap_feature:
                target_gene_to_remove = gene
                break
        
        if not target_gene_to_remove:
            logging.warning(f"[精炼] 警告: 无法从染色体中匹配到SHAP值最低的特征 {min_shap_feature}，跳过精炼。")
            return chromosome

        # 3. 替换基因
        new_genes_list = [g for g in chromosome.genes if self._get_gene_key(g) != self._get_gene_key(target_gene_to_remove)]
        
        # 寻找一个不在当前染色体中的新基因
        current_genes_keys = {self._get_gene_key(g) for g in new_genes_list} # 使用新列表的键
        potential_additions = [g for g in self.feature_genes if self._get_gene_key(g) not in current_genes_keys]

        if not potential_additions:
            logging.warning("[精炼] 警告: 基因池中没有可用的新基因来替换，跳过精炼。")
            return chromosome
            
        new_gene = random.choice(potential_additions)
        new_genes_list.append(new_gene)
        
        logging.info(f"[精炼] [OK] 已将低贡献特征 {target_gene_to_remove.path} 替换为新特征 {new_gene.path}。")
        
        return ModelingChromosome(genes=new_genes_list)


if __name__ == "__main__":
    # 这是一个集成测试，它需要我们 V1.0 的所有依赖
    import semantic_inference
    from data_translator import KnowledgeGraphTranslator
    
    logging.info("--- \"架构师\"模块 (architect.py) V1.0 独立集成测试 ---")
    
    # 1. (模拟) 运行"感知"
    class MockDB: pass
    schema_map = semantic_inference.run_semantic_inference(MockDB())
    
    # 2. (真实) 实例化"翻译官"
    TARGET_VARIABLE = "UserProfile.IsDefault" # 定义标准目标
    translator = KnowledgeGraphTranslator(
        inferred_schema=schema_map,
        physical_target_table='tbl_user_01',
        physical_target_column='is_default'
    )
    
    # 3. (真实) 实例化"基因生成器"
    gene_gen = GeneGenerator(translator, target_variable=TARGET_VARIABLE)
    gene_pool = gene_gen.generate_initial_pool()
    
    # 4. (真实) 实例化"演化引擎"
    evo_engine = EvolutionaryEngine(gene_pool)
    population = evo_engine.initialize_population(size=10) # 创建10个个体
    
    # 5. (真实) 实例化"特征引擎"和"评估器"
    feature_engine = FeatureEngine(translator, standard_target_variable=TARGET_VARIABLE)
    fitness_evaluator = FitnessEvaluator(feature_engine)
    
    # 6. (关键测试) 评估一个"染色体"
    logging.info("\n--- 正在测试评估一个随机染色体 ---")
    test_chromosome = population[0]
    logging.info(f"测试染色体: {test_chromosome.genes}")
    
    evaluation_result = fitness_evaluator.evaluate(test_chromosome)
    
    logging.info("\n--- 评估结果 ---")
    logging.info(json.dumps(evaluation_result, indent=2))
    
    assert evaluation_result['auc'] > 0.0 # 验证模型至少运行成功了
    assert evaluation_result['evaluation_time_ms'] > 0 # 验证时间被测量了
    
    logging.info("\n--- \"架构师\"模块 V1.0 独立测试完毕 ---")
