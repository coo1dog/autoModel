"""
模型预测脚本

该脚本负责加载经过 'main.py' 训练和保存的冠军模型，
并使用它对新的、未见过的数据集进行预测。

核心步骤:
1. 加载配置文件和命令行参数。
2. 加载所有相关的数据表。
3. 初始化数据翻译器 (KnowledgeGraphTranslator) 和特征引擎 (FeatureEngine)。
4. 加载保存的冠军染色体 (champion_chromosome.json)。
5. 使用特征引擎和染色体为新数据构建特征矩阵。
6. 加载保存的模型 (model.joblib)。
7. 执行预测并保存结果。
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import joblib
from typing import Dict, Any

# 导入项目模块
from schema_config import SchemaConfig
from data_translator import KnowledgeGraphTranslator
from core_structures import ModelingChromosome, ModelingGene, FeatureGene, ModelGene
from architect import FeatureEngine

def load_chromosome(path: Path) -> ModelingChromosome:
    """从JSON文件加载染色体并重建对象。"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    genes = []
    for gene_data in data.get('genes', []):
        gene_type = gene_data.get('_type')
        if gene_type == 'FeatureGene':
            genes.append(FeatureGene(op=gene_data['op'], path=gene_data['path'], window=gene_data['window']))
        elif gene_type == 'ModelGene':
            genes.append(ModelGene(alg=gene_data['alg'], params=gene_data['params']))
        elif gene_type == 'TransformGene':
            genes.append(TransformGene(op=gene_data['op'], inputs=gene_data['inputs']))
        elif gene_type == 'FilterGene':
            genes.append(FilterGene(condition=gene_data['condition']))
        # 可以根据需要扩展其他基因类型
    
    return ModelingChromosome(genes=genes)

import logging
# ... (other imports)

def predict(
    model_dir: Path,
    data_dir: Path,
    output_path: Path,
    physical_target_table: str,
):
    """
    执行预测流程。

    :param model_dir: 包含 automl_pipeline.joblib 和 champion_chromosome.json 的目录。
    :param data_dir: 包含新数据的CSV文件目录。
    :param output_path: 预测结果的输出路径 (CSV文件)。
    :param physical_target_table: 物理主表的文件名 (不含.csv)。
    """
    logging.info("--- 开始执行预测 ---")

    # --- 1. 加载工件 (Pipeline和染色体) ---
    pipeline_path = model_dir / "automl_pipeline.joblib"
    chromosome_path = model_dir / "champion_chromosome.json"
    schema_config_path = data_dir / "schema_config.json"

    if not pipeline_path.exists() or not chromosome_path.exists():
        raise FileNotFoundError(f"在 '{model_dir}' 中找不到必要的 automl_pipeline.joblib 或 champion_chromosome.json 文件。")

    logging.info(f"正在加载模型 Pipeline: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    logging.info(f"正在加载染色体: {chromosome_path}")
    champion_chromosome = load_chromosome(chromosome_path)

    # --- 2. 加载新数据 ---
    logging.info(f"正在从 '{data_dir}' 加载新数据...")
    all_dataframes: Dict[str, pd.DataFrame] = {}
    if data_dir.exists():
        for csv_path in data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
                all_dataframes[csv_path.stem] = df
                logging.info(f"  已加载 {csv_path.stem} (shape: {df.shape})")
            except Exception as e:
                logging.warning(f"  [警告] 加载 {csv_path.name} 失败: {e}")
    else:
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # --- 3. 初始化系统组件 (与训练时类似) ---
    # 加载 Schema
    schema_config = None
    if schema_config_path.exists():
        try:
            schema_config = SchemaConfig(schema_config_path)
            logging.info(f"已加载外部Schema: {schema_config_path}")
        except Exception as e:
            logging.warning(f"[警告] 外部Schema加载失败: {e}")

    # 语义推断 (预测时，目标列可能不存在，这是正常的)
    try:
        from semantic_inference import run_semantic_inference
        # 注意：预测时，我们不知道真实的目标列值，所以传入一个伪列名
        schema_map = run_semantic_inference(all_dataframes, schema_config=schema_config, autofill_fields=False)
        
        # 找到标准目标实体和字段名 (基于训练时的配置)
        # 这是一个简化处理，假设我们能从染色体或一个配置文件中知道标准目标
        # 这里我们硬编码一个常见的模式
        model_gene = next((g for g in champion_chromosome.genes if isinstance(g, ModelGene)), None)
        # 假设目标信息可以从某个地方获取，这里我们从训练数据中获取
        # 在真实场景中，你可能需要一个更鲁棒的方式来传递这个信息
        physical_target_column = "TARGET" # 假设这是训练时的目标列
        target_info = KnowledgeGraphTranslator.get_standard_target_info(
            schema_map, physical_target_table, physical_target_column
        )
        target_variable = f"{target_info['entity']}.{target_info['field']}"
        logging.info(f"推断出的标准目标变量为: {target_variable}")

    except Exception as e:
        logging.error(f"[错误] 语义推断或目标变量确定失败: {e}")
        # 兜底一个默认值
        target_variable = "application_train.TARGET"
        logging.warning(f"[警告] 回退到默认目标变量: {target_variable}")


    # 实例化翻译器和特征引擎
    translator = KnowledgeGraphTranslator(
        inferred_schema=schema_map,
        physical_target_table=physical_target_table,
        physical_target_column="<PREDICTION_MODE>", # 预测模式下，目标列未知
        dataframes=all_dataframes,
        disable_entity_fallback=True
    )
    feature_engine = FeatureEngine(translator, standard_target_variable=target_variable)

    # --- 4. 构建原始特征 ---
    logging.info("正在为新数据构建原始特征...")
    # 预测时，y (目标变量) 是未知的，但特征引擎需要它来定位主表
    # 特征引擎内部会处理 y 不存在的情况
    X_new, _, _, _ = feature_engine.build_features(champion_chromosome)
    logging.info(f"原始特征构建完成。矩阵形状: {X_new.shape}")
    
    # --- 5. 执行预测 ---
    logging.info("正在使用 Pipeline 执行预测 (包含自动预处理)...")
    # Pipeline对象会自动处理新数据的预处理（转换）和预测
    predictions_proba = pipeline.predict_proba(X_new)[:, 1]

    # --- 6. 保存结果 ---
    logging.info(f"正在保存预测结果到: {output_path}")
    main_table_df = all_dataframes.get(physical_target_table)
    if main_table_df is not None and 'SK_ID_CURR' in main_table_df.columns:
        # 假设主键是 SK_ID_CURR
        results_df = pd.DataFrame({
            'SK_ID_CURR': main_table_df['SK_ID_CURR'],
            'PREDICTION': predictions_proba
        })
    else:
        # 如果没有主键，则只保存预测结果
        results_df = pd.DataFrame({'PREDICTION': predictions_proba})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logging.info("--- 预测完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用已保存的模型进行预测。")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./saved_model",
        help="包含 model.joblib 和 champion_chromosome.json 的目录路径。"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="包含待预测数据的CSV文件目录路径。"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./predictions/predictions.csv",
        help="预测结果的输出文件路径。"
    )
    parser.add_argument(
        "--main_table",
        type=str,
        required=True,
        help="物理主表的文件名 (不含.csv扩展名), 例如 'application_train'。"
    )

    args = parser.parse_args()

    predict(
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        physical_target_table=args.main_table,
    )

    # 示例用法:
    # python predict.py --data_dir ./path/to/new_data --main_table application_test --output ./output/my_predictions.csv
