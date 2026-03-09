"""SQL 自动生成器

从训练产出的 `champion_chromosome.json` 解析特征基因，生成用于 DM 平台建宽表的 SQL。

设计要点：
- 自动识别主表（特征中出现次数最多的物理表）。
- 对所有非 GROUP BY 字段统一使用聚合函数，避免 Hive/SparkSQL 的语法错误。
- 生成策略：按表先聚合成"每 key 一行"的子查询，再做 LEFT JOIN，避免 JOIN 后再聚合导致的语法/口径问题。
  - 特别地：`LATEST` 特征也会转成 `MAX(col)`，否则会出现「非聚合列未出现在 group by」的问题。
- 输出：
  - `production_query.sql`
  - `table_mapping_template.json`
  - `expected_feature_columns.json`

注意：若你的业务需要严格"最新一条记录"(按时间戳)的语义，建议在数据侧确保 1:1 关系，
或扩展为 `argMax(col, ts)`（ClickHouse）/ 窗口函数（Hive/Spark）。
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime


SUPPORTED_AGG_OPS = {"AVG", "SUM", "COUNT", "MAX", "MIN"}


@dataclass(frozen=True)
class FeatureExpr:
    sql: str
    alias: str
    table: str


def parse_genes_from_chromosome(chromosome_path: Path) -> List[Dict]:
    """
    从染色体JSON文件解析特征基因
    
    Args:
        chromosome_path: champion_chromosome.json路径
        
    Returns:
        基因列表，每个基因包含 op, path, window 等信息
    """
    logging.info(f"[SQL生成器] 正在解析染色体: {chromosome_path}")
    
    with open(chromosome_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    genes = data.get('genes', [])
    logging.info(f"[SQL生成器] 解析到 {len(genes)} 个特征基因")
    
    return genes


def extract_table_and_field(gene: Dict) -> Tuple[str, str]:
    """
    从基因的path中提取表名和字段名
    
    Args:
        gene: 基因字典，包含 path 字段如 "table_name.field_name"
        
    Returns:
        (表名, 字段名)
    """
    path = gene.get('path', '')
    if '.' in path:
        parts = path.split('.', 1)
        return parts[0], parts[1]
    return '', path


def identify_main_table(genes: List[Dict]) -> str:
    """
    识别主表（出现频率最高的表）
    
    Args:
        genes: 基因列表
        
    Returns:
        主表名
    """
    table_counts = defaultdict(int)
    for gene in genes:
        table_name, _ = extract_table_and_field(gene)
        if table_name:
            table_counts[table_name] += 1
    
    if not table_counts:
        return ""
    
    main_table = max(table_counts, key=table_counts.get)
    logging.info(f"[SQL生成器] 识别主表: {main_table} (出现 {table_counts[main_table]} 次)")
    
    return main_table


def get_all_tables(genes: List[Dict]) -> List[str]:
    """
    获取所有涉及的表名
    
    Args:
        genes: 基因列表
        
    Returns:
        去重后的表名列表
    """
    tables = set()
    for gene in genes:
        table_name, _ = extract_table_and_field(gene)
        if table_name:
            tables.add(table_name)
    return list(tables)


def generate_feature_name(gene: Dict) -> str:
    """
    生成特征名（与architect.py中的逻辑保持一致）
    
    格式: {op}_{table_name}_{field_name}[_{window}d]
    
    Args:
        gene: 基因字典
        
    Returns:
        特征名
    """
    op = gene.get('op', 'LATEST')
    table_name, field_name = extract_table_and_field(gene)
    window = gene.get('window')
    
    feature_name = f"{op}_{table_name}_{field_name}"
    if window:
        feature_name += f"_{window}d"
    
    return feature_name


def _load_chromosome(chromosome_path: Path) -> Dict[str, Any]:
    with open(chromosome_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_path(path: str) -> Tuple[str, str]:
    # format: "table.column"
    if "." not in path:
        raise ValueError(f"Invalid gene path: {path!r} (expected 'table.column')")
    table, column = path.split(".", 1)
    return table, column


def _normalize_identifier(name: str) -> str:
    # 保守处理：DM/Hive 环境通常允许下划线/数字/字母
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def _window_suffix(window: Any) -> str:
    # 兼容旧数据：window 可能是 None / 90 / "90d" / {"days": 90}
    if window is None:
        return ""
    if isinstance(window, (int, float)):
        return f"_{int(window)}d"
    if isinstance(window, str):
        w = window.strip().lower()
        if w.endswith("d") and w[:-1].isdigit():
            return f"_{int(w[:-1])}d"
        if w.isdigit():
            return f"_{int(w)}d"
        return ""
    if isinstance(window, dict):
        days = window.get("days")
        if isinstance(days, (int, float)):
            return f"_{int(days)}d"
    return ""


def _pick_main_table(feature_tables: Iterable[str]) -> str:
    counter = Counter(feature_tables)
    if not counter:
        raise ValueError("No feature tables found in chromosome")
    return counter.most_common(1)[0][0]


def _table_aliases(main_table: str, other_tables: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {main_table: "main"}
    for idx, t in enumerate(other_tables, start=1):
        aliases[t] = f"t{idx}"
    return aliases


def _build_feature_expr(gene: Dict[str, Any], table_to_alias: Dict[str, str]) -> Optional[FeatureExpr]:
    if gene.get("_type") != "FeatureGene":
        return None

    op = str(gene.get("op", "")).upper()
    path = gene.get("path")
    window = gene.get("window")

    if not path:
        return None

    table, col = _parse_path(path)
    alias_table = table_to_alias.get(table)
    if not alias_table:
        # 不在已识别表集合里，跳过
        return None

    col_norm = _normalize_identifier(col)
    table_norm = _normalize_identifier(table)
    suffix = _window_suffix(window)

    # 统一别名格式：OP_table_col[_Nd]
    feature_alias = f"{op}_{table_norm}_{col_norm}{suffix}"

    # 关键修复：只要出现 GROUP BY，就必须让所有非分组列出现在聚合函数中。
    # 这里统一将 LATEST 也映射为 MAX(col) 以保证 SQL 合法。
    qualified = f"{alias_table}.{col_norm}"

    if op == "LATEST":
        sql = f"MAX({qualified})"
    elif op in SUPPORTED_AGG_OPS:
        sql = f"{op}({qualified})"
    else:
        # 未识别算子：保守起见也用 MAX 包裹，避免 GROUP BY 语法报错。
        sql = f"MAX({qualified})"

    return FeatureExpr(sql=sql, alias=feature_alias, table=table)


def generate_sql(
    genes: List[Dict], 
    main_table: str,
    join_key: str = "bill_no",
    schema_config: Optional[Dict] = None,
    use_short_alias: bool = False,
    short_alias_prefix: str = "f",
    short_alias_width: int = 3,
) -> Tuple[str, Dict[str, str]]:
    """
    生成生产环境SQL脚本（子查询聚合策略）
    
    Args:
        genes: 基因列表
        main_table: 主表名
        join_key: 关联字段（默认bill_no）
        schema_config: 可选的schema配置（保留接口兼容）
        
    Returns:
        SQL脚本字符串
    """
    all_tables = get_all_tables(genes)
    aux_tables = [t for t in all_tables if t != main_table]
    
    # 为每个表分配别名
    table_aliases = _table_aliases(main_table, aux_tables)
    
    join_key_norm = _normalize_identifier(join_key)
    
    # 生成特征表达式
    feature_exprs: List[FeatureExpr] = []
    for g in genes:
        expr = _build_feature_expr(g, table_aliases)
        if expr:
            feature_exprs.append(expr)
    
    # 生成策略：每张表先按 join_key 聚合成"每 key 一行"的子查询，再 JOIN。
    # 这样可以：
    # - 彻底规避 Hive/DM 对 SELECT + GROUP BY 的限制
    # - 避免多表 1:N join 造成聚合口径被放大（join fanout）
    feature_by_table: Dict[str, List[FeatureExpr]] = {}
    for fe in feature_exprs:
        feature_by_table.setdefault(fe.table, []).append(fe)

    def _agg_subquery(table: str, table_placeholder: str, sub_alias: str) -> str:
        # 子查询内不需要 table alias；直接用列名即可
        t_features = feature_by_table.get(table, [])

        select_items: List[str] = [f"{_normalize_identifier(join_key)} AS {join_key_norm}"]
        for fe in t_features:
            expr_sql = fe.sql.replace(f"{table_aliases[table]}.", "")
            select_items.append(f"{expr_sql} AS {fe.alias}")

        lines = ["SELECT"]
        for idx, item in enumerate(select_items):
            comma = "," if idx < len(select_items) - 1 else ""
            lines.append(f"    {item}{comma}")

        # 注意：需要在 f-string 中用四个花括号输出字面量 "{{" 和 "}}"
        lines.extend(
            [
                "FROM",
                f"    {table_placeholder}",
                "GROUP BY",
                f"    {_normalize_identifier(join_key)}",
            ]
        )

        return "\n".join(lines)

    # 主表子查询
    main_placeholder = f"{{{{MainTable_{_normalize_identifier(main_table)}}}}}"
    main_sub = _agg_subquery(main_table, main_placeholder, "main_agg")

    # 副表子查询
    other_subs: List[Tuple[str, str]] = []
    for t in aux_tables:
        t_placeholder = f"{{{{Table_{_normalize_identifier(t)}}}}}"
        sub = _agg_subquery(t, t_placeholder, f"{table_aliases[t]}_agg")
        other_subs.append((t, sub))

    # 外层 SELECT：join_key + 各表聚合列（子查询已经起好别名）
    # 可选：使用短字段名 f001/f002... 以规避DM/平台字段名(或字段中文名)长度限制。
    alias_map: Dict[str, str] = {}
    outer_select_lines: List[str] = [f"    main_agg.{join_key_norm} AS {join_key_norm}"]
    for fe in feature_exprs:
        src = "main_agg" if fe.table == main_table else f"{table_aliases[fe.table]}_agg"
        if use_short_alias:
            short_name = f"{short_alias_prefix}{len(alias_map)+1:0{short_alias_width}d}"
            alias_map[short_name] = fe.alias
            outer_select_lines.append(f"    {src}.{fe.alias} AS {short_name}")
        else:
            outer_select_lines.append(f"    {src}.{fe.alias} AS {fe.alias}")

    from_line = "FROM\n    (\n" + "\n".join(["    " + l for l in main_sub.splitlines()]) + "\n    ) AS main_agg"

    join_lines: List[str] = []
    for t, sub in other_subs:
        sub_alias = f"{table_aliases[t]}_agg"
        join_lines.append(
            "\n".join(
                [
                    "LEFT JOIN (",
                    "\n".join(["    " + l for l in sub.splitlines()]),
                    f") AS {sub_alias}",
                    f"    ON main_agg.{join_key_norm} = {sub_alias}.{join_key_norm}",
                ]
            )
        )

    sql_lines = [
        "-- ===============================================",
        "-- 生产环境特征宽表构建SQL",
        f"-- 训练主表: {main_table}",
        f"-- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"-- 特征数量: {len(feature_exprs)}",
        "-- ===============================================",
        "",
        "-- 使用说明:",
        "-- 1. 将 {{TARGET_TABLE}} 替换为你的目标宽表名",
        "-- 2. 将 {{MainTable_xxx}} 替换为实际生产主表名",
        "-- 3. 将 {{Table_xxx}} 替换为实际生产副表名",
        "-- 4. 确认JOIN字段在生产表中存在",
        "-- 5. 在DM数据开发平台执行",
        "",
        "DROP TABLE IF EXISTS {{TARGET_TABLE}};",
        "CREATE TABLE {{TARGET_TABLE}} AS",
        "SELECT",
        ",\n".join(outer_select_lines) + "\n",
        from_line,
        "\n\n".join(join_lines) if join_lines else "",
        ";",
        "",
    ]

    return "\n".join(sql_lines), alias_map


def generate_table_mapping_template(
    genes: List[Dict], 
    main_table: str
) -> Dict[str, Any]:
    """
    生成表名映射模板
    
    Args:
        genes: 基因列表
        main_table: 主表名
        
    Returns:
        映射模板字典
    """
    all_tables = get_all_tables(genes)
    
    mappings = {}
    
    # 主表映射
    mappings[main_table] = {
        "placeholder": f"{{{{MainTable_{main_table}}}}}",
        "role": "主表",
        "production_table": "请替换为生产主表名"
    }
    
    # 副表映射
    for table in all_tables:
        if table != main_table:
            mappings[table] = {
                "placeholder": f"{{{{Table_{table}}}}}",
                "role": "副表",
                "production_table": "请替换为生产副表名"
            }
    
    template = {
        "description": "表名映射模板 - 训练表名 → 生产表名",
        "generated_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "instructions": [
            "1. 将 {{TARGET_TABLE}} 替换为你要创建的目标宽表名",
            "2. 在下方mappings中填写每个训练表对应的生产表名",
            "3. 打开production_query.sql",
            "4. 使用文本编辑器的'全局替换'功能",
            "5. 将placeholder替换为production_table的值"
        ],
        "target_table": {
            "placeholder": "{{TARGET_TABLE}}",
            "description": "你要创建的目标宽表名，如 dm_feature_wide_table_202512"
        },
        "main_table": main_table,
        "table_count": len(all_tables),
        "mappings": mappings,
        "example": {
            "训练表名": "temp_user_info_202511",
            "占位符": "{{Table_temp_user_info_202511}}",
            "生产表名": "prod_user_profile_202512",
            "操作": "在SQL中执行全局替换: {{Table_temp_user_info_202511}} → prod_user_profile_202512"
        }
    }
    
    return template


def generate_feature_list(genes: List[Dict]) -> List[str]:
    """
    生成特征名列表（供手写SQL参考）
    
    Args:
        genes: 基因列表
        
    Returns:
        特征名列表
    """
    feature_names = []
    for gene in genes:
        feature_name = generate_feature_name(gene)
        if feature_name and feature_name not in feature_names:
            feature_names.append(feature_name)
    return feature_names


def generate_production_sql(
    chromosome_path: Path,
    output_dir: Path,
    join_key: str = None,
    schema_config: Optional[Dict] = None
) -> Tuple[Path, Path, Path]:
    """
    主函数：生成所有生产部署文件
    
    Args:
        chromosome_path: 染色体JSON路径
        output_dir: 输出目录
        join_key: 关联字段 (如果为None，自动从染色体元数据中读取)
        schema_config: 可选的schema配置
        
    Returns:
        (sql_path, mapping_path, features_path)
    """
    logging.info("\n--- [SQL生成器] 开始生成生产环境SQL ---")
    
    # [V2.1] 自动从染色体元数据中读取 join_key
    with open(chromosome_path, 'r', encoding='utf-8') as f:
        chromosome_data = json.load(f)
    
    meta = chromosome_data.get('meta', {})
    if join_key is None:
        join_key = meta.get('join_key', 'bill_no')
        logging.info(f"[SQL生成器] 从染色体元数据自动识别主键: {join_key}")
    
    # 1. 解析基因
    genes = chromosome_data.get('genes', [])
    if not genes:
        logging.warning("[SQL生成器] 未找到特征基因，跳过SQL生成")
        return None, None, None
    
    logging.info(f"[SQL生成器] 解析到 {len(genes)} 个特征基因")
    
    # 2. 识别主表 (优先使用元数据，否则自动推断)
    main_table = meta.get('main_table') or identify_main_table(genes)
    if not main_table:
        logging.warning("[SQL生成器] 无法识别主表，跳过SQL生成")
        return None, None, None
    logging.info(f"[SQL生成器] 主表: {main_table}, 主键: {join_key}")
    
    # 3. 生成SQL（长列名版本 + 短列名版本）
    sql_content, _alias_map_long = generate_sql(
        genes,
        main_table,
        join_key,
        schema_config,
        use_short_alias=False,
    )

    sql_content_short, alias_map_short = generate_sql(
        genes,
        main_table,
        join_key,
        schema_config,
        use_short_alias=True,
        short_alias_prefix="f",
        short_alias_width=3,
    )
    
    # 4. 生成表映射模板
    mapping_template = generate_table_mapping_template(genes, main_table)
    
    # 5. 生成特征列表
    feature_list = generate_feature_list(genes)
    
    # 6. 保存文件
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SQL文件（长列名版本）
    sql_path = output_dir / "production_query.sql"
    with open(sql_path, 'w', encoding='utf-8') as f:
        f.write(sql_content)
    logging.info(f"[SQL生成器] ✓ SQL脚本已保存: {sql_path}")

    # SQL文件（短列名版本，用于字段名长度受限场景）
    sql_short_path = output_dir / "production_query_short.sql"
    with open(sql_short_path, 'w', encoding='utf-8') as f:
        f.write(sql_content_short)
    logging.info(f"[SQL生成器] ✓ 短列名SQL脚本已保存: {sql_short_path}")

    # 短列名映射（f001 -> 长特征名），供推理侧 rename 使用
    alias_map_path = output_dir / "feature_alias_map.json"
    with open(alias_map_path, 'w', encoding='utf-8') as f:
        json.dump(alias_map_short, f, indent=2, ensure_ascii=False)
    logging.info(f"[SQL生成器] ✓ 短列名映射已保存: {alias_map_path}")
    
    # 映射模板
    mapping_path = output_dir / "table_mapping_template.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_template, f, indent=2, ensure_ascii=False)
    logging.info(f"[SQL生成器] ✓ 表映射模板已保存: {mapping_path}")
    
    # 特征列表
    features_path = output_dir / "expected_feature_columns.json"
    features_data = {
        "description": "PKL期望的特征列名列表（手写SQL时参考）",
        "generated_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "feature_count": len(feature_list),
        "note": "若使用 production_query.sql（长列名），AS后面的列名必须与此列表完全一致；若使用 production_query_short.sql（短列名），需结合 feature_alias_map.json 在推理前将短列名 rename 回此列表中的长列名。",
        "features": feature_list
    }
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(features_data, f, indent=2, ensure_ascii=False)
    logging.info(f"[SQL生成器] ✓ 特征列表已保存: {features_path}")
    
    # 统计信息
    all_tables = get_all_tables(genes)
    logging.info(f"[SQL生成器] 统计:")
    logging.info(f"  - 主表: {main_table}")
    logging.info(f"  - 副表数量: {len(all_tables) - 1}")
    logging.info(f"  - 特征数量: {len(feature_list)}")
    
    return sql_path, mapping_path, features_path


# ========== 命令行入口 ==========
if __name__ == "__main__":
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # 默认路径
    default_chromosome = Path("saved_model/champion_chromosome.json")
    default_output = Path("saved_model")
    
    # 命令行参数
    if len(sys.argv) > 1:
        chromosome_path = Path(sys.argv[1])
    else:
        chromosome_path = default_chromosome
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = default_output
    
    # 检查文件存在
    if not chromosome_path.exists():
        print(f"[错误] 染色体文件不存在: {chromosome_path}")
        print(f"用法: python sql_generator.py [chromosome_path] [output_dir]")
        sys.exit(1)
    
    # 生成SQL
    sql_path, mapping_path, features_path = generate_production_sql(
        chromosome_path=chromosome_path,
        output_dir=output_dir
    )
    
    if sql_path:
        print(f"\n[完成] 已生成以下文件:")
        print(f"  - {sql_path}")
        print(f"  - {mapping_path}")
        print(f"  - {features_path}")
