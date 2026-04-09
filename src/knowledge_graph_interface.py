"""
对抗性共演化系统 - 知识图谱接口定义

这个文件定义了"翻译官"的抽象接口（规范）。
它规定了上层AI智能体如何与数据层交互，隐藏了所有底层的复杂性，
为上层AI提供统一、干净的数据API。

当前主线架构约定：
- 所有数据访问通过标准业务名称进行
- 物理表名和列名被抽象化，上层只看到业务概念
- 提供统一的DataFrame接口用于特征工程
- 支持跨表关联查询

说明：
- 本接口现在表达的是"翻译官应提供哪些能力"，而不是限定具体数据来源。
- 当前项目主线实现并不要求翻译官自己直连数据库；更常见的方式是主程序先将
  CSV / ClickHouse 数据加载为 DataFrame，再由翻译官负责语义映射与访问。
"""

import abc
from typing import Dict, Any, List
import pandas as pd
from core_structures import FeatureGene


class KnowledgeGraphInterface(abc.ABC):
    """
    "翻译官"的抽象接口（规范）。
    它的实现类将持有"推断模式"以及某种可访问的数据载体。
    它隐藏了所有底层的复杂性，为上层AI智能体提供统一、干净的数据API。

    注意：
    - 历史版本曾将"原始数据库连接"视为默认输入；
    - 当前项目主线实现使用的是"预加载后的 DataFrame 集合"；
    - 因此这里的抽象重点是能力契约，而不是具体的底层存储形态。
    """

    def __init__(self, inferred_schema: Dict[str, Any], raw_db_connection: Any):
        """
        构造函数（在抽象类中可简单实现，用于规范）
        :param inferred_schema: 从 semantic_inference 模块获取的"推断模式"字典
        :param raw_db_connection: 历史命名保留字段。当前实现可将其视为任意底层数据句柄，
            例如数据库连接、DataFrame容器或其他数据访问对象。
        """
        self.inferred_schema = inferred_schema
        self.db_conn = raw_db_connection
        super().__init__()

    @abc.abstractmethod
    def get_standard_schema(self) -> Dict[str, List[str]]:
        """
        [抽象方法] 获取一个简化的"标准模式"。
        这是给"架构师"看的"菜单"，告诉它有哪些标准业务字段可用。

        例如，返回: 
        {
            'UserProfile': ['UserID', 'AnnualIncome', 'Gender', 'RegistrationDate'],
            'UserTransaction': ['TransactionID', 'TransactionAmount', 'Timestamp', 'UserID_FK']
        }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_entity_dataframe(self, entity_name: str) -> pd.DataFrame:
        """
        [抽象方法] 获取一个标准业务实体对应的完整DataFrame。
        这是后续进行特征工程和模型训练的主要数据来源。

        在实现类中，这个方法将负责：
        1. 找到 'entity_name' 对应的物理表名。
        2. 从底层数据载体中取到该表数据（当前主线通常是已预加载的 DataFrame）。
        3. 将所有"物理列名"重命名为"标准业务名称"。
        4. 返回这个"干净"的 DataFrame。

        :param entity_name: 标准业务名称, 例如 'UserProfile'
        :return: 一个包含该实体所有数据的 pandas DataFrame
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_relationship_keys(self) -> Dict[str, str]:
        """
        [抽象方法] 获取实体间的"关联键"（外键）。
        这是V1.0特征工程（跨表聚合）所必需的。

        历史上这个接口只打算返回"关系名 -> 外键名"的简单映射；当前主线实现
        实际返回的是更丰富的关系元数据对象，例如:
        {
            'UserTransaction_to_UserProfile': {
                'from_entity': 'UserTransaction',
                'from_key': 'UserID_FK',
                'to_entity': 'UserProfile',
                'to_key': 'UserID'
            }
        }
        上层模块应关注"可通过该接口拿到跨表关系信息"，而不是依赖早期注释中的
        简化返回形态。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_standard_target_info(self, physical_table: str, physical_column: str) -> Dict[str, str]:
        """
        [抽象方法] 根据物理表名和列名，反向查找标准业务名称。
        这是解决“先有鸡还是先有蛋”问题的核心。

        当前项目中，该能力既可以由实例方法实现，也可以由实现类提供为静态工具方法；
        抽象层只要求"存在这项映射能力"。

        :param physical_table: 目标变量所在的物理表名。
        :param physical_column: 目标变量所在的物理列名。
        :return: 一个包含标准实体名和字段名的字典, e.g., {'entity': 'UserProfile', 'field': 'IsDefault'}
        """
        raise NotImplementedError
