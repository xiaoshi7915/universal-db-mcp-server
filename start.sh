#!/bin/bash
# 通用数据库MCP服务器启动脚本

# 默认配置
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8088"}
DB_TYPE=${DB_TYPE:-"mysql"}
DB_HOST=${DB_HOST:-"localhost"}
DB_PORT=${DB_PORT:-"3306"}
DB_USER=${DB_USER:-"root"}
DB_PASSWORD=${DB_PASSWORD:-""}
DB_NAME=${DB_NAME:-""}

# 帮助信息
function show_help {
    echo "通用数据库 MCP 服务器启动脚本"
    echo ""
    echo "用法: ./start.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --help                  显示此帮助信息"
    echo "  --host HOST             设置服务器主机地址 (默认: $HOST)"
    echo "  --port PORT             设置服务器端口 (默认: $PORT)"
    echo "  --db-type TYPE          设置数据库类型 (mysql|postgresql|oracle|mssql|sqlite) (默认: $DB_TYPE)"
    echo "  --db-host HOST          设置数据库主机地址 (默认: $DB_HOST)"
    echo "  --db-port PORT          设置数据库端口 (默认: $DB_PORT)"
    echo "  --db-user USER          设置数据库用户名 (默认: $DB_USER)"
    echo "  --db-password PASSWORD  设置数据库密码 (默认: $DB_PASSWORD)"
    echo "  --db-name NAME          设置数据库名称 (默认: $DB_NAME)"
    echo "  --verbose, -v           启用详细日志输出"
    echo ""
    exit 0
}

# 解析命令行参数
VERBOSE=""
while [ "$1" != "" ]; do
    case $1 in
        --help )                show_help ;;
        --host )                shift; HOST=$1 ;;
        --port )                shift; PORT=$1 ;;
        --db-type )             shift; DB_TYPE=$1 ;;
        --db-host )             shift; DB_HOST=$1 ;;
        --db-port )             shift; DB_PORT=$1 ;;
        --db-user )             shift; DB_USER=$1 ;;
        --db-password )         shift; DB_PASSWORD=$1 ;;
        --db-name )             shift; DB_NAME=$1 ;;
        --verbose | -v )        VERBOSE="--verbose" ;;
        * )                     echo "未知参数: $1"; show_help ;;
    esac
    shift
done

# 设置环境变量
export HOST
export PORT
export DB_TYPE
export DB_HOST
export DB_PORT
export DB_USER
export DB_PASSWORD
export DB_NAME

# 输出配置信息
echo "启动通用数据库 MCP 服务器..."
echo "主机：$HOST"
echo "端口：$PORT"
echo "数据库类型：$DB_TYPE"
echo "数据库主机：$DB_HOST"
echo "数据库端口：$DB_PORT"
echo "数据库名称：$DB_NAME"
echo "数据库用户：$DB_USER"
echo ""

# 启动服务器
python fast_db_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --db-type "$DB_TYPE" \
    --db-host "$DB_HOST" \
    --db-port "$DB_PORT" \
    --db-user "$DB_USER" \
    --db-password "$DB_PASSWORD" \
    --db-name "$DB_NAME" \
    $VERBOSE 