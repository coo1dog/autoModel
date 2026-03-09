"""
ClickHouse数据加载模块

功能：
- 从ClickHouse数据库读取表数据
- 支持指定主表和多个副表
- 返回与CSV加载相同格式的DataFrame字典

使用方式：
    from clickhouse_loader import load_tables_from_clickhouse
    
    config = {
        'host': '192.168.1.100',
        'port': 9000,
        'database': 'prod_db',
        'user': 'admin',
        'password': '******',
        'main_table': 'temp_terminal_usage_202512',
        'aux_tables': ['temp_user_info_202512', 'temp_payment_202512']
    }
    
    dataframes = load_tables_from_clickhouse(config)

依赖安装：
    pip install clickhouse-driver
"""

import logging
import re
from typing import Dict, List, Optional, Any
import pandas as pd

# 尝试导入clickhouse_driver，如果未安装则给出提示
try:
    from clickhouse_driver import Client
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False
    logging.warning("[ClickHouse] clickhouse-driver未安装，请运行: pip install clickhouse-driver")


class ClickHouseLoader:
    """ClickHouse数据加载器"""
    
    def __init__(
        self,
        host: str,
        port: int = 9000,
        database: str = 'default',
        user: str = 'default',
        password: str = '',
        secure: bool = False,
        **kwargs
    ):
        """
        初始化ClickHouse连接
        
        Args:
            host: ClickHouse服务器地址
            port: 端口号（默认9000）
            database: 数据库名
            user: 用户名
            password: 密码
            secure: 是否使用SSL连接
            **kwargs: 其他连接参数
        """
        if not CLICKHOUSE_AVAILABLE:
            raise ImportError(
                "clickhouse-driver未安装，请运行: pip install clickhouse-driver"
            )
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.secure = secure
        self.extra_params = kwargs
        
        self.client = None
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        logging.info(f"[ClickHouse] 正在连接 {self.host}:{self.port}/{self.database}...")
        
        try:
            self.client = Client(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                secure=self.secure,
                **self.extra_params
            )
            
            # 测试连接
            version = self.client.execute("SELECT version()")[0][0]
            logging.info(f"[ClickHouse] ✓ 连接成功，服务器版本: {version}")
            
        except Exception as e:
            logging.error(f"[ClickHouse] ✗ 连接失败: {e}")
            raise
    
    def load_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        加载单张表数据
        
        Args:
            table_name: 表名
            columns: 要查询的列名列表，None表示全部
            where_clause: WHERE条件（不含WHERE关键字）
            limit: 限制返回行数
            
        Returns:
            DataFrame
        """
        # 构建SQL
        if columns:
            cols_str = ", ".join(columns)
        else:
            cols_str = "*"
        
        sql = f"SELECT {cols_str} FROM {table_name}"
        
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        if limit:
            sql += f" LIMIT {limit}"
        
        logging.info(f"[ClickHouse] 正在加载表: {table_name}...")
        
        try:
            # 使用query_dataframe（如果可用）或手动转换
            result = self.client.execute(sql, with_column_types=True)
            data, columns_info = result
            column_names = [col[0] for col in columns_info]
            
            df = pd.DataFrame(data, columns=column_names)
            
            # 清理列名（与CSV加载保持一致）
            df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
            
            logging.info(f"[ClickHouse] ✓ 已加载 {table_name}, shape={df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"[ClickHouse] ✗ 加载表 {table_name} 失败: {e}")
            raise
    
    def load_multiple_tables(
        self,
        main_table: str,
        aux_tables: List[str],
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        加载多张表（主表 + 副表）
        
        Args:
            main_table: 主表名
            aux_tables: 副表名列表
            where_clause: WHERE条件（应用于所有表）
            limit: 限制每张表返回行数
            
        Returns:
            {表名: DataFrame} 字典
        """
        dataframes = {}
        
        # 加载主表
        logging.info(f"[ClickHouse] 开始加载数据，主表: {main_table}, 副表数量: {len(aux_tables)}")
        
        dataframes[main_table] = self.load_table(
            main_table, 
            where_clause=where_clause, 
            limit=limit
        )
        
        # 加载副表
        for table in aux_tables:
            dataframes[table] = self.load_table(
                table, 
                where_clause=where_clause, 
                limit=limit
            )
        
        logging.info(f"[ClickHouse] ✓ 全部加载完成，共 {len(dataframes)} 张表")
        return dataframes
    
    def list_tables(self) -> List[str]:
        """列出当前数据库的所有表"""
        result = self.client.execute("SHOW TABLES")
        return [row[0] for row in result]
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """获取表结构"""
        result = self.client.execute(f"DESCRIBE TABLE {table_name}")
        return [
            {"name": row[0], "type": row[1], "default": row[2]}
            for row in result
        ]
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.disconnect()
            logging.info("[ClickHouse] 连接已关闭")


def load_tables_from_clickhouse(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    便捷函数：从ClickHouse加载表数据
    
    Args:
        config: 配置字典，包含：
            - host: 服务器地址
            - port: 端口（默认9000）
            - database: 数据库名
            - user: 用户名
            - password: 密码
            - main_table: 主表名
            - aux_tables: 副表名列表或逗号分隔字符串
            - where_clause: (可选) WHERE条件
            - limit: (可选) 限制行数
            
    Returns:
        {表名: DataFrame} 字典
    """
    # 解析副表参数（支持列表或逗号分隔字符串）
    aux_tables = config.get('aux_tables', [])
    if isinstance(aux_tables, str):
        aux_tables = [t.strip() for t in aux_tables.split(',') if t.strip()]
    
    # 创建加载器
    loader = ClickHouseLoader(
        host=config['host'],
        port=config.get('port', 9000),
        database=config.get('database', 'default'),
        user=config.get('user', 'default'),
        password=config.get('password', ''),
        secure=config.get('secure', False)
    )
    
    # 加载数据
    try:
        dataframes = loader.load_multiple_tables(
            main_table=config['main_table'],
            aux_tables=aux_tables,
            where_clause=config.get('where_clause'),
            limit=config.get('limit')
        )
        return dataframes
    finally:
        loader.close()


# ========== 命令行测试入口 ==========
if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    parser = argparse.ArgumentParser(description='ClickHouse数据加载测试')
    parser.add_argument('--host', required=True, help='ClickHouse服务器地址')
    parser.add_argument('--port', type=int, default=9000, help='端口号')
    parser.add_argument('--database', default='default', help='数据库名')
    parser.add_argument('--user', default='default', help='用户名')
    parser.add_argument('--password', default='', help='密码')
    parser.add_argument('--main_table', required=True, help='主表名')
    parser.add_argument('--aux_tables', default='', help='副表名（逗号分隔）')
    parser.add_argument('--limit', type=int, help='限制行数（测试用）')
    parser.add_argument('--list_tables', action='store_true', help='列出所有表')
    
    args = parser.parse_args()
    
    # 创建连接
    loader = ClickHouseLoader(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )
    
    try:
        # 列出表
        if args.list_tables:
            tables = loader.list_tables()
            print(f"\n数据库 {args.database} 中的表:")
            for t in tables:
                print(f"  - {t}")
        
        # 加载数据
        else:
            aux_tables = [t.strip() for t in args.aux_tables.split(',') if t.strip()]
            
            dataframes = loader.load_multiple_tables(
                main_table=args.main_table,
                aux_tables=aux_tables,
                limit=args.limit
            )
            
            print("\n加载结果:")
            for name, df in dataframes.items():
                print(f"  - {name}: {df.shape[0]} 行 × {df.shape[1]} 列")
                print(f"    列名: {list(df.columns[:5])}...")
                
    finally:
        loader.close()
