"""
对抗性共演化系统 - 主控制单元

这是系统的"大脑"，负责组装并运行完整的"对抗性共演化"系统。
它将所有模块整合在一起，实现最终的"综合得分对抗循环"。

V1.0 架构约定：
- 组装所有V1.0模块（感知、翻译、架构师、破坏者）
- 实现完整的对抗性共演化循环
- 动态调整适应度函数权重
- 提供详细的演化过程日志
"""

import re
import json
import time
import random
import numpy as np
import joblib
import dataclasses # 新增
from typing import List, Dict, Any
from pathlib import Path  # 新增
import pandas as pd       # 新增
from schema_config import SchemaConfig  # 已有: 外部Schema配置支持
# 导入我们项目中的所有 V1.0 模块
from data_translator import KnowledgeGraphTranslator
from core_structures import ModelingChromosome
from architect import GeneGenerator, FeatureEngine, FitnessEvaluator, EvolutionaryEngine
from saboteur import EconomicsAttacker, CausalAttacker, SynthesisAttacker


import logging

class ControlUnit:
    """
    V1.1 "总指挥部"。
    负责组装并运行完整的"对抗性共演化"系统。
    """
    
    def __init__(self, physical_target_table: str, physical_target_column: str, dataframes: Dict[str, pd.DataFrame] = None, schema_config: SchemaConfig = None, feature_gen_config: Dict[str, Any] = None):
        logging.info("[控制单元] 正在启动...")
        
        self.physical_target_table = physical_target_table
        self.physical_target_column = physical_target_column
        self.dataframes = dataframes or {}
        self.schema_config = schema_config

        # 新增: 运行语义推断以获取 schema_map
        try:
            from semantic_inference import run_semantic_inference
            # 关闭字段自动补全，避免为主表构建无关字段映射
            self.schema_map = run_semantic_inference(self.dataframes, schema_config=self.schema_config, autofill_fields=False)
        except Exception as e:
            logging.warning(f"[控制单元][警告] 语义推断失败: {e}")
            self.schema_map = {}

        # --- 1. (感知) 运行"语义推断" ---
        # schema_map = semantic_inference.run_semantic_inference(dataframes=dataframes)
        
        # --- 2. (V1.2 修正) 解耦：先查找标准目标，再实例化翻译官 ---
        target_info = KnowledgeGraphTranslator.get_standard_target_info(
            self.schema_map, physical_target_table, physical_target_column
        )
        self.target_variable = f"{target_info['entity']}.{target_info['field']}"

        # --- 3. (数据) 实例化"翻译官" ---
        self.translator = KnowledgeGraphTranslator(
            inferred_schema=self.schema_map,
            physical_target_table=physical_target_table,
            physical_target_column=physical_target_column,
            dataframes=dataframes,
            disable_entity_fallback=True
        )
        
        # --- 4. (创造) 实例化"架构师"的所有组件 ---
        logging.info("[控制单元] 正在初始化\"架构师\"...")
        # 特征生成配置（外参化风格）：仅控制主表 LGBM 预筛选 Top-K
        self.feature_gen_config = feature_gen_config or {}
        self.gene_generator = GeneGenerator(self.translator, self.target_variable, feature_config=self.feature_gen_config)
        self.gene_pool = self.gene_generator.generate_initial_pool()
        
        self.evo_engine = EvolutionaryEngine(self.gene_pool)
        
        # (V1.1 修改) 传入目标变量
        self.feature_engine = FeatureEngine(self.translator, self.target_variable)
        
        self.fitness_evaluator = FitnessEvaluator(self.feature_engine)
        
        # --- 5. (批判) 实例化"破坏者"的所有组件 ---
        logging.info("[控制单元] 正在初始化\"破坏者\"...")
        self.attackers = {
            "economics": EconomicsAttacker(self.translator, self.target_variable),
            "causal": CausalAttacker(self.translator, self.target_variable),
            "synthesis": SynthesisAttacker(self.translator, self.target_variable)
        }
        # --- 6. (控制) 初始化"适应度函数"的权重 ---
        self.fitness_weights = {
            'auc': 1.0, # AUC 的基础权重
            'economics': 0.0, # 经济惩罚的初始权重 (负数)
            'causal': 0.0, # 因果惩罚的初始权重
            'synthesis': 0.0 # 泛化惩罚的初始权重
        }
        
        logging.info("[控制单元] 系统初始化完毕，准备就绪。")

    def run(self, generations: int, population_size: int, challenge_interval: int, evo_config: Dict[str, Any]):
        """
        运行"对抗性共演化"V1.0 主循环
        """
        logging.info("\n--- V1.0 \"对抗性共演化\"开始 ---")

        # 1. 初始化种群（从外部传入 evo_config）
        population = self.evo_engine.initialize_population(population_size, config=evo_config)
        
        # V2.1 修复：在循环外声明，以保存最后一代的冠军
        champion_chromosome = None
        champion_eval_results = None

        for gen in range(generations):
            logging.info(f"\n--- 世代 {gen+1}/{generations} ---")

            evaluation_results = []  # 存储评估器返回的"评估字典"
            penalty_scores = []      # 存储破坏者返回的"惩罚字典"
            comprehensive_scores = []  # 存储最终的"综合得分"

            # 2. (核心) 评估与批判阶段
            logging.info(f"正在评估和批判 {len(population)} 个个体...")
            start_gen_time = time.time()

            for chromosome in population:
                eval_result = self.fitness_evaluator.evaluate(chromosome)
                evaluation_results.append(eval_result)
                penalties = {}
                for attacker_name, attacker_instance in self.attackers.items():
                    if self.fitness_weights.get(attacker_name, 0.0) != 0.0:
                        penalties[attacker_name] = attacker_instance.challenge(chromosome, eval_result)
                    else:
                        penalties[attacker_name] = 0.0
                penalty_scores.append(penalties)
                final_score = (
                    eval_result.get('auc', 0.0) * self.fitness_weights['auc']
                    + penalties.get('economics', 0.0) * self.fitness_weights['economics']
                    + penalties.get('causal', 0.0) * self.fitness_weights['causal']
                    + penalties.get('synthesis', 0.0) * self.fitness_weights['synthesis']
                )
                comprehensive_scores.append(final_score)

            gen_time = time.time() - start_gen_time
            logging.info(f"世代评估完成，耗时: {gen_time:.2f} 秒")

            all_base_aucs = [r.get('auc', 0.0) for r in evaluation_results]
            logging.info(f"  [统计] 平均基础AUC: {np.mean(all_base_aucs):.4f} (最高: {max(all_base_aucs):.4f})")
            logging.info(f"  [统计] 平均综合得分: {np.mean(comprehensive_scores):.4f} (最高: {max(comprehensive_scores):.4f})")

            selected = self.evo_engine.select(population, comprehensive_scores)
            
            next_population = []
            while len(next_population) < (population_size - 1):
                p1, p2 = random.sample(selected, 2)
                child = self.evo_engine.crossover(p1, p2)
                child = self.evo_engine.mutate(child)
                next_population.append(child)

            best_individual_index = np.argmax(comprehensive_scores)
            elite_chromosome = population[best_individual_index]

            logging.info(f"\n[精英学习] 对本代冠军 (AUC: {all_base_aucs[best_individual_index]:.4f}) 进行 SHAP 精炼...")
            # V2.1 修复：在评估精英时就获取其包含完整Pipeline的评估结果
            elite_eval_result = self.fitness_evaluator.evaluate(elite_chromosome, calculate_shap=True)
            
            # V2.1 修复：将本次循环找到的精英及其评估结果（包含已训练的Pipeline）存为最终冠军候选
            champion_chromosome = elite_chromosome
            champion_eval_results = elite_eval_result
            
            shap_values = elite_eval_result.get('shap_values')
            if shap_values:
                refined_elite = self.evo_engine.refine_chromosome(elite_chromosome, shap_values)
                next_population.append(refined_elite)
            else:
                logging.warning("[精英学习] 警告: 未能获取 SHAP 值，直接保留原版精英。")
                next_population.append(elite_chromosome)

            population = next_population

            if (gen + 1) % challenge_interval == 0:
                logging.info(f"\n[控制单元] 触发\"权重动态更新\"！(对抗压力增加)")
                self.fitness_weights['causal'] = round(self.fitness_weights['causal'] * 1.2, 2)
                self.fitness_weights['economics'] = round(self.fitness_weights['economics'] * 1.1, 2)
                logging.info(f"[控制单元] 新权重: Causal={self.fitness_weights['causal']:.2f}, Economics={self.fitness_weights['economics']:.2f}")

        logging.info("\n--- V1.0 \"对抗性共演化\"结束 ---")

        # 7. V2.1 修复：返回在最后一次循环中保存的、正确配对的冠军染色体和评估结果
        logging.info(f"[控制单元] 最终冠军 (综合得分: {max(comprehensive_scores):.4f}):")
        logging.info(f"  - 基础 AUC: {champion_eval_results.get('auc', 0.0):.4f}")
        logging.info(f"  - KS 值: {champion_eval_results.get('ks', 'N/A')}")
        logging.info(f"  - Precision: {champion_eval_results.get('precision', 'N/A')}")
        logging.info(f"  - Recall: {champion_eval_results.get('recall', 'N/A')}")
        logging.info(f"  - F1: {champion_eval_results.get('f1', 'N/A')}")
        logging.info(f"  - 评估耗时: {champion_eval_results.get('evaluation_time_ms', 0.0):.0f} ms")
        logging.info(f"  - 特征数量: {champion_eval_results.get('feature_count', 0)}")
        logging.info(f"  - 最终基因:")
        logging.info(json.dumps(champion_chromosome, default=lambda o: o.__dict__, indent=4, ensure_ascii=False))
        
        return champion_chromosome, champion_eval_results


def save_champion_model(
    champion: ModelingChromosome, 
    eval_results: Dict[str, Any], 
    save_dir: Path,
    pipeline_filename: str = "automl_pipeline.joblib",
    chromosome_filename: str = "champion_chromosome.json",
    metadata: Dict[str, Any] = None
):
    """
    将最终的冠军Pipeline和染色体保存到磁盘。
    [V2.0 升级版] 自动生成"本地满血版"和"平台兼容版"，确保多环境无缝切换。
    [V2.1 新增] 支持保存元数据（主键、主表名等），供SQL生成器自动识别。
    """
    logging.info(f"\n--- 正在保存最终冠军工件到 '{save_dir}' ---")
    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存完整的 Pipeline (本地高版本专用)
        pipeline = eval_results.get('pipeline')
        if pipeline:
            # A. 保存原版 (给您本地用)
            pipeline_path = save_dir / pipeline_filename
            joblib.dump(pipeline, pipeline_path)
            logging.info(f"[OK] 原版模型已保存: {pipeline_path}")

            # --- B. [新增] 自动生成兼容版 (给平台用) ---
            try:
                from sklearn.base import BaseEstimator, TransformerMixin
                import numpy as np
                
                # 定义兼容替身 (内部类，防止污染全局)
                class CompatibleImputer(BaseEstimator, TransformerMixin):
                    def __init__(self, strategy='median'):
                        self.strategy = strategy
                        self.statistics_ = np.array([0]) 
                    def fit(self, X, y=None): return self
                    def transform(self, X): return X

                # 临时替换 Pipeline 的步骤
                original_steps = list(pipeline.steps) # 备份原步骤
                compatible_steps = []
                
                for name, step in original_steps:
                    # 检查是否是 SimpleImputer
                    if 'SimpleImputer' in str(type(step)):
                        # 替换为替身
                        compatible_steps.append((name, CompatibleImputer()))
                    else:
                        compatible_steps.append((name, step))
                
                # 应用替换并保存
                pipeline.steps = compatible_steps
                compatible_path = save_dir / "automl_pipeline_compatible.pkl"
                
                # 关键：使用 protocol=2 保存
                joblib.dump(pipeline, compatible_path, protocol=2)
                logging.info(f"[OK] ✅ 敏捷平台兼容版已自动生成: {compatible_path}")
                
                # 恢复原样 (以免影响内存中的对象)
                pipeline.steps = original_steps
                
            except Exception as e_compat:
                logging.warning(f"[警告] 自动生成兼容版失败: {e_compat}")
            # ------------------------------------------

        else:
            logging.warning("[警告] 评估结果中未找到'pipeline'对象，无法保存。")

        # 2. 保存染色体 (含元数据)
        chromosome_path = save_dir / chromosome_filename
        serializable_genes = []
        for gene in champion.genes:
            gene_dict = dataclasses.asdict(gene)
            gene_dict['_type'] = gene.__class__.__name__
            serializable_genes.append(gene_dict)
        
        # [V2.1] 构建带元数据的染色体结构
        serializable_chromosome = {
            "meta": metadata or {},
            "genes": serializable_genes
        }

        with open(chromosome_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chromosome, f, indent=4, ensure_ascii=False)
        logging.info(f"[OK] 冠军染色体已保存到: {chromosome_path}")
        if metadata:
            logging.info(f"    - 元数据: 主键={metadata.get('join_key')}, 主表={metadata.get('main_table')}")

    except Exception as e:
        logging.error(f"[错误] 保存工件时发生错误: {e}")


def generate_platform_inference_script(save_dir: Path, model_path: Path, features_path: Path, join_key: str = "bill_no", ds_id: str = "694b64f7adbe1d000b1651ef") -> None:
    """生成敏捷平台可粘贴的预测脚本（内联特征白名单）。

    Args:
        save_dir: 输出目录（通常为 saved_model）。
        model_path: 平台可见的 pkl 路径占位符。
        features_path: expected_feature_columns.json 路径。
        join_key: 业务主键（从知识图谱自动识别或默认）。
        ds_id: 平台数据源ID占位符。
    """
    try:
        logging.info(f"[平台脚本] 开始生成: save_dir={save_dir}, features_path={features_path}, model_path={model_path}")
        save_dir.mkdir(parents=True, exist_ok=True)
        features = []
        if features_path.exists():
            with open(features_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", data)
        
        # 即使 features 为空也继续生成，但在脚本中注释说明
        if not features:
            logging.warning("[平台脚本] features 列表在文件中未找到，将生成空列表模板")
            feature_lines = "    # WARNING: AutoModel未能自动提取特征列表，请手动填入\n"
        else:
            feature_lines = ",\n".join(f'    "{c}"' for c in features)

        template = f"""# ============================================================
# 【Cell 0】 数据源配置（发布界面必需）
# ============================================================
import aurai.data_source as ds
# 根据dsId获取数据源信息
dsId="{ds_id}"
dsCycle="yyyyMMdd"
ds.describeDS(dsId)


# ============================================================
# 【Cell 1】 基础配置与环境补丁
# ============================================================
import pandas as pd
import numpy as np
import pickle
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # 兼容旧版 sklearn
import sys
import warnings
import os

# ------------------------------------------------------------
# 1. 配置区域
# ------------------------------------------------------------
# 模型文件路径
MODEL_FILE = '{model_path}'

# 业务主键列 (从知识图谱自动识别)
ID_COLUMN = '{join_key}'

# ------------------------------------------------------------
# 2. 特征白名单 (自动生成，无需修改)
# ------------------------------------------------------------
FEATURE_WHITELIST = [
{feature_lines}
]

# ------------------------------------------------------------
# 2.1 短字段名映射（用于DM字段名长度受限场景）
# ------------------------------------------------------------
# 若上游宽表SQL输出列为 f001~f0xx，这里会在预测前自动 rename 回长特征名。
ALIAS_TO_FEATURE = {{
    f"f{{i:03d}}": FEATURE_WHITELIST[i - 1]
    for i in range(1, len(FEATURE_WHITELIST) + 1)
}}

# ------------------------------------------------------------
# 3. 兼容性补丁 (必须执行)
# ------------------------------------------------------------
# 【新增补丁】解决 sklearn 版本不一致导致的 _RemainderColsList 缺失问题
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        print("检测到平台 sklearn 版本较新，正在注入 _RemainderColsList 补丁...")
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except ImportError:
    pass

# 【新增补丁2】解决 sklearn.preprocessing._label 缺失问题
try:
    # 1. 获取真正的 LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    import types
    
    # 2. 检查或创建 sklearn.preprocessing._label 模块
    if 'sklearn.preprocessing._label' not in sys.modules:
        _label_mod = types.ModuleType('sklearn.preprocessing._label')
        sys.modules['sklearn.preprocessing._label'] = _label_mod
    
    # 3. 强制挂载 LabelEncoder 到该模块
    sys.modules['sklearn.preprocessing._label'].LabelEncoder = LabelEncoder
    print("已注入 sklearn.preprocessing._label.LabelEncoder 补丁")

except Exception as e:
    print(f"补丁2注入警告: {{e}}")

# 【新增补丁3】解决 OneHotEncoder._legacy_mode 缺失问题
try:
    from sklearn.preprocessing import OneHotEncoder
    if not hasattr(OneHotEncoder, '_legacy_mode'):
        print("检测到平台 sklearn 版本较新，正在注入 OneHotEncoder._legacy_mode 补丁...")
        setattr(OneHotEncoder, '_legacy_mode', False)
except ImportError:
    pass

class CompatibleImputer:
    def __init__(self, strategy='mean', fill_value=None, missing_values=np.nan):
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.statistics_ = None
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        for i, stat in enumerate(self.statistics_):
            mask = np.isnan(X[:, i])
            X[mask, i] = stat
        return X
    def fit_transform(self, X, y=None): return self.transform(X)

sys.modules['__main__'].CompatibleImputer = CompatibleImputer
warnings.filterwarnings('ignore')
print("【Cell 1】环境配置完成")


# ============================================================
# 【Cell 2】 加载模型
# ============================================================
model = joblib.load(MODEL_FILE)
print("模型加载完成")

# 【关键补丁】修复已加载模型中的 OneHotEncoder.sparse 属性
# 遍历 Pipeline 中的所有 Transformer，给 OneHotEncoder 实例添加缺失的 sparse 属性
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    def patch_onehot_encoders(obj):
        \"\"\"递归修补对象中的所有 OneHotEncoder 实例\"\"\"
        if isinstance(obj, OneHotEncoder):
            if not hasattr(obj, 'sparse'):
                obj.sparse = getattr(obj, 'sparse_output', True)
                print(f"已修补 OneHotEncoder 实例: sparse={{obj.sparse}}")
        elif isinstance(obj, Pipeline):
            for name, step in obj.steps:
                patch_onehot_encoders(step)
        elif isinstance(obj, ColumnTransformer):
            for name, transformer, columns in obj.transformers_:
                patch_onehot_encoders(transformer)
    
    patch_onehot_encoders(model)
    print("已完成模型中所有 OneHotEncoder 的补丁注入")
except Exception as e:
    print(f"OneHotEncoder 补丁注入失败: {{e}}")


# ============================================================
# 【Cell 3】 执行预测
# ============================================================

def preprocess_and_filter(df, feature_whitelist):
    \"\"\"根据白名单严格过滤和对齐特征列\"\"\"
    # 兼容短字段名：若存在 f001... 列，先映射回长特征名
    if any(col in df.columns for col in ALIAS_TO_FEATURE.keys()):
        df = df.rename(columns=ALIAS_TO_FEATURE)

    # 只保留白名单中的列，缺失列用 NaN 填充
    df_clean = df.reindex(columns=feature_whitelist)
    
    # 强制将每一列都转为 numeric，无法转换的变为 NaN
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 将所有的 NaN 填充为 0
    df_clean = df_clean.fillna(0)
    
    # 【关键修复】转为字符串类型，避免模型内部 Encoder 的类型冲突
    # 模型中的分类编码器期望字符串输入，如果传入 float 会导致类型转换错误
    df_clean = df_clean.astype(str)
    
    return df_clean

# 初始化数据读取器
print("正在连接数据源...")
reader = ds.load_block(dsId)
i = 0
onceReadLines = 50000

print("开始批量预测...")

while True:
    try:
        # 读取数据块
        df_chunk = reader.get_chunk(onceReadLines)
        if df_chunk.empty: break
        
        # 准备输出数据 (ID列)
        if ID_COLUMN in df_chunk.columns:
            result_ids = df_chunk[ID_COLUMN].values
        else:
            # 如果没有ID列，生成索引作为ID
            result_ids = df_chunk.index.values
            
        # 特征工程 (自动过滤 city_id 等无关列)
        X_test = preprocess_and_filter(df_chunk, FEATURE_WHITELIST)
        
        # 预测
        try:
            preds = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            preds = model.predict(X_test)
            
        # 构造结果集
        predDF = pd.DataFrame({{
            ID_COLUMN: result_ids,
            'p_value': preds
        }})
        
        # 写入文件
        if i == 0:
            predDF.to_csv("predict_result.csv", index=False)
        else:
            predDF.to_csv("predict_result.csv", index=False, header=False, mode='a')
            print("add " + str(i) + " data to csv")
        
        i += 1
            
    except StopIteration:
        break
    except Exception as e:
        print(f"错误: {{e}}")
        import traceback
        traceback.print_exc()
        break

reader.close()
print(f"循环预测完成，共处理 {{i}} 批次")


# ============================================================
# 【Cell 4】 读取预测结果
# ============================================================
predDF = pd.read_csv('predict_result.csv')
print(predDF.shape)


# ============================================================
# 【Cell 5】 保存预测结果
# ============================================================
predDF.to_csv('predict_result.csv', index=False)
"""

        out_path = save_dir / "platform_inference_template.py"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(template)
        try:
            file_size = out_path.stat().st_size
            logging.info(f"[平台脚本] 已生成: {out_path} (size={file_size} bytes)")
        except Exception:
            logging.info(f"[平台脚本] 已生成: {out_path}")
    except Exception as e:
        logging.warning(f"[平台脚本] 生成失败: {e}")
        import traceback
        logging.warning(traceback.format_exc())


import logging
from logger_config import setup_logging
from pathlib import Path
import argparse

# --- 默认配置区域（CSV模式使用） ---
DEFAULT_DATA_FOLDER = Path(r"dataset")
DEFAULT_TARGET_TABLE = "temp_a_utrm_use_3m_inline_encrypt_202511"
DEFAULT_TARGET_COLUMN = "flag"
DEFAULT_SCHEMA_CONFIG = DEFAULT_DATA_FOLDER / "schema_config.json"

# 输出文件路径和名称
SAVE_DIRECTORY = Path("./saved_model")
LOG_FILE_PATH = "automl.log"
CHAMPION_CHROMOSOME_FILENAME = "champion_chromosome.json"
PIPELINE_FILENAME = "automl_pipeline.joblib"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='AutoML自动建模系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  CSV模式（默认，本地测试）:
    python main.py
    python main.py --csv_dir dataset_termchange --main_table temp_xxx --target_column flag

  ClickHouse模式（生产环境）:
    python main.py --data_source clickhouse \\
      --ck_host 192.168.1.100 \\
      --ck_port 9000 \\
      --ck_database prod_db \\
      --ck_user admin \\
      --ck_password ****** \\
      --main_table temp_terminal_202512 \\
      --aux_tables "temp_user_info,temp_payment" \\
      --target_column flag \\
      --schema_config /path/to/schema_config.json
        """
    )
    
    # 数据源选择
    parser.add_argument('--data_source', type=str, default='csv',
                        choices=['csv', 'clickhouse'],
                        help='数据源类型: csv（默认）或 clickhouse')
    
    # CSV模式参数
    parser.add_argument('--csv_dir', type=str, default=str(DEFAULT_DATA_FOLDER),
                        help='CSV数据目录路径（CSV模式使用）')
    
    # ClickHouse模式参数
    parser.add_argument('--ck_host', type=str, default='localhost',
                        help='ClickHouse服务器地址')
    parser.add_argument('--ck_port', type=int, default=9000,
                        help='ClickHouse端口号')
    parser.add_argument('--ck_database', type=str, default='default',
                        help='ClickHouse数据库名')
    parser.add_argument('--ck_user', type=str, default='default',
                        help='ClickHouse用户名')
    parser.add_argument('--ck_password', type=str, default='',
                        help='ClickHouse密码')
    
    # 表配置（两种模式共用）
    parser.add_argument('--main_table', type=str, default=DEFAULT_TARGET_TABLE,
                        help='主表名（包含目标变量的表）')
    parser.add_argument('--aux_tables', type=str, default='',
                        help='副表名，逗号分隔（ClickHouse模式必填）')
    parser.add_argument('--target_column', type=str, default=DEFAULT_TARGET_COLUMN,
                        help='目标变量列名')
    
    # Schema配置
    parser.add_argument('--schema_config', type=str, default='',
                        help='知识图谱JSON文件路径')
    
    # 其他配置
    parser.add_argument('--limit', type=int, default=None,
                        help='限制读取行数（测试用）')
    parser.add_argument('--output_dir', type=str, default=str(SAVE_DIRECTORY),
                        help='输出目录')
    
    # 演化参数
    parser.add_argument('--generations', type=int, default=2,
                        help='进化世代数')
    parser.add_argument('--population', type=int, default=2,
                        help='每代种群规模')

    return parser.parse_args()


def load_data_from_csv(csv_dir: Path) -> Dict[str, pd.DataFrame]:
    """从CSV目录加载数据，自动检测分隔符和编码"""
    all_dataframes = {}
    
    if not csv_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {csv_dir}")
    
    for csv_path in csv_dir.glob("*.csv"):
        try:
            # 自动检测分隔符和编码：先读前几行判断
            with open(csv_path, 'rb') as f:
                raw_head = f.read(4096)
            
            # 检测编码
            encoding = 'utf-8'
            try:
                raw_head.decode('utf-8')
            except UnicodeDecodeError:
                encoding = 'gbk'
            
            # 检测分隔符：取第一行，看哪个分隔符出现最多
            first_line = raw_head.decode(encoding, errors='ignore').split('\n')[0]
            sep = ','
            for candidate in ['$', '\t', '|', ';']:
                if first_line.count(candidate) > first_line.count(sep):
                    sep = candidate
            
            df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
            df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
            all_dataframes[csv_path.stem] = df
            logging.info(f"[数据] 已加载 {csv_path.stem} shape={df.shape} (编码={encoding}, 分隔符='{sep}')")
        except Exception as e:
            logging.warning(f"[数据][警告] 加载 {csv_path.name} 失败: {e}")
    
    if not all_dataframes:
        raise ValueError(f"未在 {csv_dir} 找到任何CSV文件")
    
    return all_dataframes


def load_data_from_clickhouse(args) -> Dict[str, pd.DataFrame]:
    """从ClickHouse加载数据"""
    from clickhouse_loader import load_tables_from_clickhouse
    
    # 解析副表
    aux_tables = [t.strip() for t in args.aux_tables.split(',') if t.strip()]
    
    if not aux_tables:
        logging.warning("[数据] 未指定副表，将只加载主表")
    
    config = {
        'host': args.ck_host,
        'port': args.ck_port,
        'database': args.ck_database,
        'user': args.ck_user,
        'password': args.ck_password,
        'main_table': args.main_table,
        'aux_tables': aux_tables,
        'limit': args.limit
    }
    
    logging.info(f"[数据] 正在从ClickHouse加载数据...")
    logging.info(f"  - 主机: {args.ck_host}:{args.ck_port}")
    logging.info(f"  - 数据库: {args.ck_database}")
    logging.info(f"  - 主表: {args.main_table}")
    logging.info(f"  - 副表: {aux_tables}")
    
    return load_tables_from_clickhouse(config)


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 配置日志
    setup_logging(log_file=LOG_FILE_PATH)
    
    logging.info("=" * 60)
    logging.info("AutoML 自动建模系统启动")
    logging.info(f"数据源模式: {args.data_source.upper()}")
    logging.info("=" * 60)

    # 根据数据源加载数据
    all_dataframes: Dict[str, pd.DataFrame] = {}
    
    if args.data_source == 'csv':
        # CSV模式
        DATA_FOLDER_PATH = Path(args.csv_dir)
        all_dataframes = load_data_from_csv(DATA_FOLDER_PATH)
        
        # Schema配置路径（如未指定，默认在数据目录下）
        if args.schema_config:
            SCHEMA_CONFIG_PATH = Path(args.schema_config)
        else:
            SCHEMA_CONFIG_PATH = DATA_FOLDER_PATH / "schema_config.json"
    
    elif args.data_source == 'clickhouse':
        # ClickHouse模式
        all_dataframes = load_data_from_clickhouse(args)
        
        # Schema配置路径（ClickHouse模式必须指定）
        if args.schema_config:
            SCHEMA_CONFIG_PATH = Path(args.schema_config)
        else:
            logging.warning("[主控] ClickHouse模式建议指定 --schema_config 参数")
            SCHEMA_CONFIG_PATH = Path("schema_config.json")
    
    # 设置目标表和列
    PHYSICAL_TARGET_TABLE = args.main_table
    PHYSICAL_TARGET_COLUMN = args.target_column
    SAVE_DIRECTORY = Path(args.output_dir)

    # 加载外部schema配置
    schema_config = None
    if SCHEMA_CONFIG_PATH.exists():
        try:
            schema_config = SchemaConfig(SCHEMA_CONFIG_PATH)
            logging.info(f"[主控] [OK] 已加载外部Schema: {SCHEMA_CONFIG_PATH}")
        except Exception as e:
            logging.warning(f"[主控][警告] 外部Schema加载失败: {e}")
    else:
        logging.info(f"[主控] 未找到外部Schema，回退到LLM推断: {SCHEMA_CONFIG_PATH}")

    # 新增: 启动控制单元
    # (V2.8 优化) 特征生成配置，使用比例以获得灵活性
    FEATURE_GEN_CONFIG = {
        "main_table": {
            "lgbm_top_k_single_ratio": 0.35,  # 单表场景下，筛选前 35% 的特征
            "lgbm_top_k_multi_ratio": 0.25    # 多表场景下，筛选前 25% 的特征
        }
    }
    control_unit = ControlUnit(
        physical_target_table=PHYSICAL_TARGET_TABLE,
        physical_target_column=PHYSICAL_TARGET_COLUMN,
        dataframes=all_dataframes,
        schema_config=schema_config,
        feature_gen_config=FEATURE_GEN_CONFIG
    )
    # [V2.8 优化] 演化配置：针对40W数据量优化
    EVOLUTION_CONFIG = {
        "generations": args.generations,
        "population_size": args.population,
        "challenge_interval": 2,
        "evo_feature_config": {
            "min_features_ratio": 0.15,   # 单个染色体最少使用基因池的15%特征
            "max_features_ratio": 0.45,   # 单个染色体最多使用基因池的45%特征
            "max_features_floor": 15      # 特征数量下限保底
        }
    }
    # 运行演化并获取最终冠军
    champion_chromosome, champion_eval_results = control_unit.run(
        generations=EVOLUTION_CONFIG["generations"],
        population_size=EVOLUTION_CONFIG["population_size"],
        challenge_interval=EVOLUTION_CONFIG["challenge_interval"],
        evo_config=EVOLUTION_CONFIG["evo_feature_config"]
    )

    # [V2.1] 自动从 schema_config 获取主键
    inferred_join_key = "bill_no"  # 默认兜底
    if schema_config and schema_config.schema:
        # 尝试从主表配置中读取 primary_key
        main_table_info = schema_config.get_table_info(PHYSICAL_TARGET_TABLE)
        if main_table_info and main_table_info.get('primary_key'):
            inferred_join_key = main_table_info.get('primary_key')
            logging.info(f"[元数据] 从知识图谱自动识别主键: {inferred_join_key}")
    
    # 构建元数据
    chromosome_metadata = {
        "join_key": inferred_join_key,
        "main_table": PHYSICAL_TARGET_TABLE,
        "target_column": PHYSICAL_TARGET_COLUMN,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存最终模型 (含元数据)
    save_champion_model(
        champion=champion_chromosome,
        eval_results=champion_eval_results,
        save_dir=SAVE_DIRECTORY,
        pipeline_filename=PIPELINE_FILENAME,
        chromosome_filename=CHAMPION_CHROMOSOME_FILENAME,
        metadata=chromosome_metadata
    )

    # --- 生成生产环境SQL (自动使用元数据中的主键) ---
    logging.info("\n--- 生成生产环境SQL脚本 ---")
    features_path = None
    try:
        from sql_generator import generate_production_sql
        
        sql_path, mapping_path, features_path = generate_production_sql(
            chromosome_path=SAVE_DIRECTORY / CHAMPION_CHROMOSOME_FILENAME,
            output_dir=SAVE_DIRECTORY
            # join_key 现在会自动从染色体元数据中读取
        )
        
        if sql_path:
            logging.info(f"[OK] 生产SQL已生成，请查看: {SAVE_DIRECTORY}")
            logging.info(f"  - SQL脚本: {sql_path.name}")
            logging.info(f"  - 表映射模板: {mapping_path.name}")
            logging.info(f"  - 特征列表: {features_path.name}")
        else:
            logging.warning("[警告] generate_production_sql 返回空值。这通常是因为基因列表中没有有效的特征基因。")
        
    except Exception as e:
        logging.warning(f"[警告] SQL生成失败（不影响模型使用）: {e}")
        import traceback
        logging.warning(traceback.format_exc())

    # --- 无论SQL是否生成成功，都尝试生成敏捷平台发布脚本 ---
    logging.info("--- 开始生成敏捷发布脚本 ---")
    try:
        # 若 SQL 生成阶段没返回 features_path，则使用默认输出路径兜底
        if not features_path:
            features_path = SAVE_DIRECTORY / "expected_feature_columns.json"

        generate_platform_inference_script(
            save_dir=SAVE_DIRECTORY,
            model_path=SAVE_DIRECTORY / "automl_pipeline_compatible.pkl",
            features_path=features_path,
            join_key=chromosome_metadata.get("join_key", "bill_no"),
            ds_id="YOUR_DS_ID"
        )
    except Exception as e_platform:
        logging.warning(f"[警告] 平台脚本生成失败: {e_platform}")
        import traceback
        logging.warning(traceback.format_exc())

    # --- 模型保存与加载验证 ---
    logging.info("\n--- 验证模型保存与加载 ---")
    try:
        # 1. 重新加载工件
        reloaded_pipeline = joblib.load(SAVE_DIRECTORY / PIPELINE_FILENAME)
        from predict import load_chromosome
        reloaded_champion_chromosome = load_chromosome(SAVE_DIRECTORY / CHAMPION_CHROMOSOME_FILENAME)

        # 2. 重新构建训练集的特征 (不经过预处理)
        re_feature_engine = FeatureEngine(control_unit.translator, control_unit.target_variable)
        X_train_reloaded, y_train_reloaded, _, _ = \
            re_feature_engine.build_features(reloaded_champion_chromosome)

        # [V2.8 优化] 对大数据集采样验证，防止内存溢出
        VALIDATION_SAMPLE = 50000
        if len(X_train_reloaded) > VALIDATION_SAMPLE:
            sample_idx = np.random.RandomState(42).choice(len(X_train_reloaded), VALIDATION_SAMPLE, replace=False)
            X_val_sample = X_train_reloaded.iloc[sample_idx]
            y_val_sample = y_train_reloaded.iloc[sample_idx] if hasattr(y_train_reloaded, 'iloc') else y_train_reloaded[sample_idx]
            logging.info(f"[验证] 数据量({len(X_train_reloaded)})较大，已采样 {VALIDATION_SAMPLE} 行进行验证")
        else:
            X_val_sample = X_train_reloaded
            y_val_sample = y_train_reloaded

        # 3. 使用加载的 Pipeline 直接进行预测 (会自动处理 transform 和 predict)
        predictions = reloaded_pipeline.predict_proba(X_val_sample)[:, 1]
        
        # 4. 计算AUC
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_val_sample, predictions)
        logging.info(f"[OK] [验证] 重新加载Pipeline后在训练集上的 AUC: {auc_score:.4f}")

    except Exception as e:
        logging.error(f"[验证] 模型加载或测试失败: {e}", exc_info=True)

    # --- [V2.7] 自动生成模型评估报告 ---
    logging.info("\n--- 生成模型评估报告 ---")
    try:
        report = {
            "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "数据集": args.csv_dir if args.data_source == 'csv' else f"{args.ck_database}.{args.main_table}",
            "主表": PHYSICAL_TARGET_TABLE,
            "目标列": PHYSICAL_TARGET_COLUMN,
            "演化世代": EVOLUTION_CONFIG["generations"],
            "种群规模": EVOLUTION_CONFIG["population_size"],
            "模型指标": {
                "AUC": champion_eval_results.get('auc', 'N/A'),
                "AUC_std": champion_eval_results.get('auc_std', 'N/A'),
                "KS": champion_eval_results.get('ks', 'N/A'),
                "最优阈值": champion_eval_results.get('best_threshold', 'N/A'),
                "Precision": champion_eval_results.get('precision', 'N/A'),
                "Recall": champion_eval_results.get('recall', 'N/A'),
                "F1": champion_eval_results.get('f1', 'N/A'),
                "混淆矩阵": champion_eval_results.get('confusion_matrix', 'N/A'),
            },
            "特征数量": champion_eval_results.get('feature_count', 0),
            "评估耗时_ms": round(champion_eval_results.get('evaluation_time_ms', 0), 1),
        }
        report_path = SAVE_DIRECTORY / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False, default=str)
        logging.info(f"[OK] 评估报告已保存: {report_path}")
    except Exception as e_report:
        logging.warning(f"[警告] 评估报告生成失败: {e_report}")

    logging.info("\n--- 自动化建模流程结束 ---")
    logging.info(f"冠军工件已保存在: '{SAVE_DIRECTORY}'")