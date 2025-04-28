#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用数据库 FastMCP 服务器
支持连接到 MySQL、PostgreSQL、Oracle、SQL Server 和 SQLite 数据库
提供数据库结构信息和执行只读 SQL 查询的功能
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, text, MetaData, inspect
from fastmcp import FastMCP
import re
import asyncio
import uvicorn
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine
import traceback
# 兼容pydantic v2
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("fast_db_server")

# 设置其他模块的日志级别为警告，减少不必要的日志输出
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

# 敏感字段列表（用于掩码处理）
SENSITIVE_FIELDS = [
    'password', 'pwd', 'secret', 'token', 'ssn', 'social_security',
    'credit_card', 'card_number', 'phone', 'address', 'email',
    'birth', 'birthday', 'id_card', 'identity', 'passport'
]

# 敏感数据掩码模式
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
}

# 敏感列名关键词
SENSITIVE_COLUMN_KEYWORDS = [
    'password', 'passwd', 'secret', 'key', 'token', 'auth', 'credential', 'api_key',
    'credit', 'card', 'ssn', 'social', 'security', 'birth', 'dob', 'license', 'address', 
    'phone', 'email', 'name', 'username', 'user', 'account', 'bank', 'routing', 'salary',
    'income', 'tax', 'private', 'confidential', 'sensitive'
]

# 数据库连接信息，优先从环境变量读取
DB_CONFIG = {
    'type': os.environ.get('DB_TYPE', 'mysql'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': os.environ.get('DB_NAME', 'wenshu_mcp'),
    'port': os.environ.get('DB_PORT', '3306'),
    'sqlite_path': os.environ.get('DB_SQLITE_PATH', ''),
}

# 创建 FastMCP 实例
fast_mcp_server = FastMCP(
    vendor="Universal DB Server",
    display_name="通用数据库 MCP 服务器",
    description="连接到各种数据库并提供数据库结构信息和执行只读查询的工具",
    version="1.0.0"
)

# 定义请求模型 - 使用pydantic v2兼容方式
class DatabaseRequest(BaseModel):
    """数据库连接请求基础模型"""
    host: Optional[str] = Field(default=None, description="数据库主机地址")
    user: Optional[str] = Field(default=None, description="数据库用户名")
    password: Optional[str] = Field(default=None, description="数据库密码")
    database: Optional[str] = Field(default=None, description="要连接的特定数据库名称")
    port: Optional[str] = Field(default=None, description="数据库端口")

class SampleDataRequest(DatabaseRequest):
    """获取样例数据请求模型"""
    limit: int = Field(default=3, description="每个表获取的最大样例数据条数，默认为3")

class ReadonlyQueryRequest(DatabaseRequest):
    """执行只读查询请求模型"""
    query: str = Field(description="要执行的SQL查询语句")
    max_rows: int = Field(default=1000, description="返回的最大行数，默认为1000")

class DBConnector:
    """数据库连接器类，管理不同数据库类型的连接"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.db_type = config.get('type', 'mysql').lower()
        self.engine = None
        
    def get_connection_string(self) -> str:
        """根据配置生成数据库连接字符串"""
        if self.db_type == 'mysql':
            return f"mysql+pymysql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        elif self.db_type == 'postgresql':
            return f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        elif self.db_type == 'oracle':
            dsn = f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
            return f"oracle+cx_oracle://{self.config['user']}:{self.config['password']}@{dsn}"
        elif self.db_type == 'mssql':
            return f"mssql+pyodbc://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}?driver=ODBC+Driver+17+for+SQL+Server"
        elif self.db_type == 'sqlite':
            sqlite_path = self.config.get('sqlite_path', 'sqlite.db')
            return f"sqlite:///{sqlite_path}"
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
    
    def connect(self) -> None:
        """建立数据库连接"""
        try:
            conn_str = self.get_connection_string()
            # 掩盖敏感信息后记录连接字符串
            masked_conn_str = re.sub(r':([^:@]+)@', ':***@', conn_str)
            logger.info(f"连接到数据库: {masked_conn_str}")
            
            self.engine = create_engine(conn_str, echo=False, future=True)
            # 测试连接
            with self.engine.connect() as conn:
                logger.info(f"数据库连接成功: {self.db_type}")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def disconnect(self) -> None:
        """断开数据库连接"""
        if self.engine:
            logger.info("关闭数据库连接")
            self.engine.dispose()
            self.engine = None

# 全局数据库连接器实例
db_connector = DBConnector(DB_CONFIG)

# 辅助函数
def mask_sensitive_data(table_name: str, data: pd.DataFrame) -> pd.DataFrame:
    """对数据进行敏感信息掩码处理"""
    if data.empty:
        return data
    
    # 创建数据副本以避免修改原始数据
    df = data.copy()
    
    # 检查每列是否包含敏感关键词
    for col in df.columns:
        col_lower = str(col).lower()
        
        # 如果列名包含敏感关键词，掩盖整列数据
        if any(keyword in col_lower for keyword in SENSITIVE_COLUMN_KEYWORDS):
            df[col] = df[col].apply(lambda x: "[MASKED]" if pd.notna(x) else x)
            continue
            
        # 仅处理字符串类型的列，对匹配特定模式的内容进行掩码
        if df[col].dtype == 'object':
            for pattern_name, pattern in PII_PATTERNS.items():
                df[col] = df[col].apply(
                    lambda x: re.sub(pattern, f"[MASKED_{pattern_name.upper()}]", str(x)) 
                    if isinstance(x, str) else x
                )
    
    return df

def get_temp_connector(request: DatabaseRequest) -> DBConnector:
    """根据请求创建临时数据库连接器"""
    # 获取请求中的连接参数，如果没有提供则使用环境变量的默认值
    config = {
        'type': os.environ.get('DB_TYPE', 'mysql'),
        'host': request.host or os.environ.get('DB_HOST', 'localhost'),
        'user': request.user or os.environ.get('DB_USER', 'root'),
        'password': request.password or os.environ.get('DB_PASSWORD', ''),
        'database': request.database or os.environ.get('DB_NAME', ''),
        'port': request.port or os.environ.get('DB_PORT', '3306'),
        'sqlite_path': os.environ.get('DB_SQLITE_PATH', 'sqlite.db')
    }
    
    # 记录连接配置（掩盖敏感信息）
    masked_config = config.copy()
    masked_config['password'] = '***' if masked_config['password'] else ''
    logger.debug(f"创建临时数据库连接: {masked_config}")  # 改为DEBUG级别
    
    db = DBConnector(config)
    db.connect()
    return db

def is_read_only_query(query: str) -> bool:
    """检查SQL查询是否为只读查询"""
    if not query:
        return False
        
    # 转换为小写进行不区分大小写的比较
    normalized_query = query.lower().strip()
    
    # 检查是否以SELECT开头且不包含数据修改关键词
    is_select = normalized_query.startswith('select') or normalized_query.startswith('show') or normalized_query.startswith('desc')
    has_modifiers = any(keyword in normalized_query for keyword in [
        'insert', 'update', 'delete', 'drop', 'create', 'alter', 'truncate', 
        'rename', 'replace', 'grant', 'revoke', 'merge', 'call', 'execute'
    ])
    
    # 检查是否包含特殊语法如存储过程调用或多语句执行
    has_special = ';' in normalized_query
    
    return is_select and not (has_modifiers or has_special)

# 工具实现 - 使用新版FastMCP API
@fast_mcp_server.tool("get_database_metadata")
async def get_database_metadata(request: DatabaseRequest):
    """
    获取所有数据库的元数据信息，包括表名、字段名、字段注释、字段类型、字段长度、是否为空、是否主键、外键、索引
    """
    try:
        # 获取数据库连接器
        connector = get_temp_connector(request)
        inspector = inspect(connector.engine)
        
        result = []
        
        # 获取所有表
        tables = inspector.get_table_names()
        
        for table_name in tables:
            table_info = {
                "表名": table_name,
                "中文表名": "",  # 这里需要从表注释中提取，不同数据库实现不同
                "字段信息": []
            }
            
            # 获取表注释
            if connector.db_type == 'mysql':
                # MySQL 获取表注释的特殊处理
                query = text(f"SHOW TABLE STATUS WHERE Name = '{table_name}'")
                with connector.engine.connect() as conn:
                    result_proxy = conn.execute(query)
                    table_status = result_proxy.fetchone()
                    if table_status and table_status.Comment:
                        table_info["中文表名"] = table_status.Comment
            elif connector.db_type == 'postgresql':
                # PostgreSQL获取表注释
                query = text(f"""
                    SELECT obj_description('{table_name}'::regclass, 'pg_class') as comment
                """)
                with connector.engine.connect() as conn:
                    result_proxy = conn.execute(query)
                    comment_row = result_proxy.fetchone()
                    if comment_row and comment_row[0]:
                        table_info["中文表名"] = comment_row[0]
            elif connector.db_type == 'oracle':
                # Oracle获取表注释
                query = text(f"""
                    SELECT comments FROM all_tab_comments 
                    WHERE table_name = '{table_name.upper()}'
                """)
                with connector.engine.connect() as conn:
                    result_proxy = conn.execute(query)
                    comment_row = result_proxy.fetchone()
                    if comment_row and comment_row[0]:
                        table_info["中文表名"] = comment_row[0]
            elif connector.db_type == 'mssql':
                # SQL Server获取表注释
                query = text(f"""
                    SELECT ep.value as table_comment
                    FROM sys.tables t
                    LEFT JOIN sys.extended_properties ep ON ep.major_id = t.object_id
                    AND ep.minor_id = 0
                    AND ep.name = 'MS_Description'
                    WHERE t.name = '{table_name}'
                """)
                with connector.engine.connect() as conn:
                    result_proxy = conn.execute(query)
                    comment_row = result_proxy.fetchone()
                    if comment_row and comment_row[0]:
                        table_info["中文表名"] = comment_row[0]
            
            # 获取列信息
            columns = inspector.get_columns(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)['constrained_columns']
            foreign_keys = inspector.get_foreign_keys(table_name)
            indices = inspector.get_indexes(table_name)
            
            # 记录每个列的信息
            for column in columns:
                column_name = column['name']
                column_type = str(column['type'])
                is_primary = column_name in primary_keys
                
                # 查找外键信息
                fk_info = ""
                for fk in foreign_keys:
                    if column_name in fk['constrained_columns']:
                        referred_table = fk['referred_table']
                        referred_columns = fk['referred_columns']
                        fk_info = f"引用 {referred_table}({', '.join(referred_columns)})"
                        break
                
                # 查找索引信息
                idx_info = []
                for idx in indices:
                    if column_name in idx['column_names']:
                        idx_type = "唯一索引" if idx['unique'] else "普通索引"
                        idx_info.append(f"{idx['name']}({idx_type})")
                
                # 获取列注释（不同数据库处理方式不同）
                column_comment = ""
                if connector.db_type == 'mysql':
                    query = text(f"""
                        SELECT column_comment 
                        FROM information_schema.columns 
                        WHERE table_schema = '{request.database or connector.config['database']}' 
                        AND table_name = '{table_name}' 
                        AND column_name = '{column_name}'
                    """)
                    with connector.engine.connect() as conn:
                        result_proxy = conn.execute(query)
                        comment_row = result_proxy.fetchone()
                        if comment_row and comment_row[0]:
                            column_comment = comment_row[0]
                elif connector.db_type == 'postgresql':
                    # PostgreSQL获取列注释
                    query = text(f"""
                        SELECT col_description('{table_name}'::regclass::oid, 
                            (SELECT ordinal_position 
                             FROM information_schema.columns 
                             WHERE table_name='{table_name}' AND column_name='{column_name}'))
                    """)
                    with connector.engine.connect() as conn:
                        result_proxy = conn.execute(query)
                        comment_row = result_proxy.fetchone()
                        if comment_row and comment_row[0]:
                            column_comment = comment_row[0]
                elif connector.db_type == 'oracle':
                    # Oracle获取列注释
                    query = text(f"""
                        SELECT comments FROM all_col_comments 
                        WHERE table_name = '{table_name.upper()}' 
                        AND column_name = '{column_name.upper()}'
                    """)
                    with connector.engine.connect() as conn:
                        result_proxy = conn.execute(query)
                        comment_row = result_proxy.fetchone()
                        if comment_row and comment_row[0]:
                            column_comment = comment_row[0]
                elif connector.db_type == 'mssql':
                    # SQL Server获取列注释
                    query = text(f"""
                        SELECT ep.value as column_comment
                        FROM sys.columns c
                        LEFT JOIN sys.extended_properties ep ON ep.major_id = c.object_id
                        AND ep.minor_id = c.column_id
                        AND ep.name = 'MS_Description'
                        WHERE c.name = '{column_name}'
                        AND OBJECT_NAME(c.object_id) = '{table_name}'
                    """)
                    with connector.engine.connect() as conn:
                        result_proxy = conn.execute(query)
                        comment_row = result_proxy.fetchone()
                        if comment_row and comment_row[0]:
                            column_comment = comment_row[0]
                
                # 字段长度
                column_length = ""
                if hasattr(column['type'], 'length') and column['type'].length:
                    column_length = str(column['type'].length)
                elif hasattr(column['type'], 'precision') and column['type'].precision:
                    precision = column['type'].precision
                    scale = column['type'].scale if hasattr(column['type'], 'scale') else 0
                    column_length = f"{precision},{scale}"
                
                # 添加字段信息
                field_info = {
                    "字段名": column_name,
                    "字段注释": column_comment,
                    "字段类型": column_type,
                    "字段长度": column_length,
                    "是否为空": "否" if not column.get('nullable', True) else "是",
                    "是否主键": "是" if is_primary else "否",
                    "外键": fk_info,
                    "索引": ", ".join(idx_info) if idx_info else ""
                }
                
                table_info["字段信息"].append(field_info)
            
            result.append(table_info)
        
        # 格式化结果为易读的字符串
        formatted_result = ""
        for table in result:
            formatted_result += f"表名: {table['表名']}\n"
            if table['中文表名']:
                formatted_result += f"中文表名: {table['中文表名']}\n"
            formatted_result += "字段信息:\n"
            
            # 创建字段信息表格
            field_table = "| 字段名 | 字段注释 | 字段类型 | 字段长度 | 是否为空 | 是否主键 | 外键 | 索引 |\n"
            field_table += "| ------ | -------- | -------- | -------- | -------- | -------- | ---- | ---- |\n"
            
            for field in table["字段信息"]:
                field_table += f"| {field['字段名']} | {field['字段注释']} | {field['字段类型']} | {field['字段长度']} | {field['是否为空']} | {field['是否主键']} | {field['外键']} | {field['索引']} |\n"
            
            formatted_result += field_table + "\n\n"
        
        return formatted_result
    
    except Exception as e:
        logger.error(f"获取数据库元数据失败: {str(e)}")
        return {"error": f"获取数据库元数据失败: {str(e)}"}

@fast_mcp_server.tool("get_sample_data")
async def get_sample_data(request: SampleDataRequest):
    """
    获取所有数据库每个表的样例数据（默认最多3条）
    """
    try:
        # 获取数据库连接器
        connector = get_temp_connector(request)
        engine = connector.engine
        inspector = inspect(connector.engine)
        
        result = []
        
        # 获取所有表
        tables = inspector.get_table_names()
        
        # 对每个表获取样例数据
        for table_name in tables:
            try:
                # 根据数据库类型生成不同的查询语句
                if connector.db_type == 'oracle':
                    query = f"SELECT * FROM {table_name} WHERE ROWNUM <= {request.limit}"
                elif connector.db_type == 'mssql':
                    query = f"SELECT TOP {request.limit} * FROM {table_name}"
                else:
                    query = f"SELECT * FROM {table_name} LIMIT {request.limit}"
                
                df = pd.read_sql(query, engine)
                
                # 处理敏感数据
                df = mask_sensitive_data(table_name, df)
                
                # 添加到结果中
                result.append({
                    "表名": table_name,
                    "数据": df.to_dict(orient='records')
                })
            except Exception as e:
                logger.warning(f"获取表 {table_name} 的样例数据失败: {str(e)}")
                result.append({
                    "表名": table_name,
                    "数据": [],
                    "错误": str(e)
                })
        
        # 格式化结果为易读的字符串
        formatted_result = ""
        for table in result:
            formatted_result += f"表名: {table['表名']}\n"
            
            if "错误" in table:
                formatted_result += f"错误: {table['错误']}\n\n"
                continue
            
            if not table["数据"]:
                formatted_result += "无数据\n\n"
                continue
            
            # 创建数据表格
            columns = list(table["数据"][0].keys())
            
            # 表头
            data_table = "| " + " | ".join(columns) + " |\n"
            data_table += "| " + " | ".join(["----" for _ in columns]) + " |\n"
            
            # 表内容
            for row in table["数据"]:
                data_table += "| " + " | ".join([str(row[col]) for col in columns]) + " |\n"
            
            formatted_result += data_table + "\n\n"
        
        return formatted_result
    
    except Exception as e:
        logger.error(f"获取表样例数据失败: {str(e)}")
        return {"error": f"获取表样例数据失败: {str(e)}"}

@fast_mcp_server.tool("execute_readonly_query")
async def execute_readonly_query(request: ReadonlyQueryRequest):
    """
    在只读事务中执行自定义SQL查询，确保查询不会修改数据库
    """
    try:
        # 检查是否是只读查询
        query_upper = request.query.upper()
        if any(keyword in query_upper for keyword in ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE', 'RENAME']):
            return {"error": "只允许执行只读查询，不允许修改数据库"}
        
        # 获取数据库连接器
        connector = get_temp_connector(request)
        engine = connector.engine
        
        # 执行查询
        df = pd.read_sql(request.query, engine, params={})
        
        # 限制返回的行数
        if len(df) > request.max_rows:
            df = df.head(request.max_rows)
            truncated = True
        else:
            truncated = False
        
        # 处理敏感数据
        df = mask_sensitive_data("", df)
        
        # 格式化结果为易读的字符串
        formatted_result = ""
        
        if df.empty:
            formatted_result = "查询返回0行结果"
        else:
            columns = df.columns.tolist()
            
            # 表头
            data_table = "| " + " | ".join(columns) + " |\n"
            data_table += "| " + " | ".join(["----" for _ in columns]) + " |\n"
            
            # 表内容
            for _, row in df.iterrows():
                data_table += "| " + " | ".join([str(row[col]) for col in columns]) + " |\n"
            
            formatted_result = data_table
            
            if truncated:
                formatted_result += f"\n注意: 结果已被截断，只显示前 {request.max_rows} 行"
        
        return formatted_result
    
    except Exception as e:
        logger.error(f"执行SQL查询失败: {str(e)}")
        return {"error": f"执行SQL查询失败: {str(e)}"}

# 初始化和启动服务器
async def initialize():
    """
    初始化服务器，连接数据库
    """
    try:
        # 输出数据库连接信息（不包含密码）
        safe_config = DB_CONFIG.copy()
        safe_config['password'] = '******' if safe_config['password'] else ''
        logger.info(f"数据库配置: {json.dumps(safe_config, ensure_ascii=False, indent=2)}")
        
        # 连接数据库
        db_connector.connect()
        logger.info("数据库连接成功")
    except Exception as e:
        logger.error(f"初始化错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def cleanup():
    """
    清理资源
    """
    try:
        db_connector.disconnect()
        logger.info("数据库连接已断开")
    except Exception as e:
        logger.error(f"清理错误: {str(e)}")
        logger.error(traceback.format_exc())

# 使用FastMCP的SSE功能启动服务器
if __name__ == "__main__":
    try:
        # 解析命令行参数
        import argparse
        parser = argparse.ArgumentParser(description='通用数据库 MCP 服务器')
        parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
        parser.add_argument('--port', type=int, default=int(os.environ.get("SERVER_PORT", "8088")), help='服务器端口')
        parser.add_argument('--db-host', help='数据库主机地址，默认使用环境变量DB_HOST')
        parser.add_argument('--db-port', help='数据库端口，默认使用环境变量DB_PORT')
        parser.add_argument('--db-user', help='数据库用户名，默认使用环境变量DB_USER')
        parser.add_argument('--db-password', help='数据库密码，默认使用环境变量DB_PASSWORD')
        parser.add_argument('--db-name', help='数据库名称，默认使用环境变量DB_NAME')
        parser.add_argument('--db-type', choices=['mysql', 'postgresql', 'oracle', 'mssql', 'sqlite'], 
                            help='数据库类型，默认使用环境变量DB_TYPE')
        parser.add_argument('--verbose', '-v', action='store_true', help='启用详细日志输出')
        args = parser.parse_args()
        
        # 设置日志级别
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("mcp").setLevel(logging.INFO)
            logging.getLogger("fastmcp").setLevel(logging.INFO)
            logging.getLogger("sqlalchemy").setLevel(logging.INFO)
            logger.debug("启用详细日志输出")
        
        # 更新数据库配置（如果提供了命令行参数）
        if args.db_host:
            DB_CONFIG['host'] = args.db_host
        if args.db_port:
            DB_CONFIG['port'] = args.db_port
        if args.db_user:
            DB_CONFIG['user'] = args.db_user
        if args.db_password:
            DB_CONFIG['password'] = args.db_password
        if args.db_name:
            DB_CONFIG['database'] = args.db_name
        if args.db_type:
            DB_CONFIG['type'] = args.db_type
        
        # 初始化数据库连接
        asyncio.run(initialize())
        
        # 简化版本信息日志
        logger.info(f"启动数据库 MCP 服务器，主机：{args.host}，端口：{args.port}")
        logger.info(f"支持的数据库类型：MySQL, PostgreSQL, Oracle, SQL Server, SQLite")
        logger.info(f"当前连接数据库类型：{DB_CONFIG['type']}，主机：{DB_CONFIG['host']}，数据库：{DB_CONFIG['database']}")
        
        # 使用FastMCP实例的run方法启动服务器
        fast_mcp_server.run(transport='sse', host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("收到中断信号，服务器正在关闭...")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # 确保在服务结束时清理资源
        try:
            asyncio.run(cleanup())
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            logger.error(traceback.format_exc())



