# 通用数据库 MCP 服务器

这是一个通用的数据库 MCP (Model Context Protocol) 服务器，支持连接到各种关系型数据库系统并提供数据库结构信息和执行只读 SQL 查询。

## 功能特点

- **多数据库支持**：MySQL、PostgreSQL、Oracle、SQL Server和SQLite
- **元数据访问**：获取数据库表结构、字段注释、主键、索引和外键信息
- **样例数据查看**：快速查看各表数据样例
- **只读SQL查询**：安全执行任意只读SQL查询
- **数据隐私保护**：自动掩码敏感信息，保护隐私数据
- **MCP协议通信**：通过标准MCP协议与大语言模型通信
- **SSE传输**：使用服务器推送事件(SSE)传输，无WebSocket依赖

## 系统要求

- Python 3.10 或更高版本
- 适用的数据库服务器(根据需要): MySQL、PostgreSQL、Oracle、SQL Server或SQLite

## 快速开始

### 直接安装

1. 克隆仓库:
   ```
   git clone https://github.com/xiaoshi7915/universal-db-mcp-server.git
   cd universal-db-mcp-server
   ```

2. 安装依赖:
   ```
   pip install -r requirements.txt
   ```

3. 启动服务器:
   ```
   ./start.sh --db-host localhost --db-user root --db-password password --db-name mydatabase
   ```

### Docker部署

1. 构建Docker镜像:
   ```
   docker build -t universal-db-mcp-server .
   ```

2. 运行Docker容器:
   ```
   docker run -p 8088:8088 -e DB_HOST=host.docker.internal -e DB_USER=root -e DB_PASSWORD=password -e DB_NAME=mydatabase universal-db-mcp-server
   ```

## 配置选项

可以通过环境变量或命令行参数配置服务器:

| 参数 | 环境变量 | 说明 | 默认值 |
|------|----------|------|--------|
| --host | HOST | 服务器主机地址 | 0.0.0.0 |
| --port | PORT | 服务器端口 | 8088 |
| --db-type | DB_TYPE | 数据库类型 | mysql |
| --db-host | DB_HOST | 数据库主机地址 | localhost |
| --db-port | DB_PORT | 数据库端口 | 3306 |
| --db-user | DB_USER | 数据库用户名 | root |
| --db-password | DB_PASSWORD | 数据库密码 | (空) |
| --db-name | DB_NAME | 数据库名称 | (空) |
| --verbose | - | 启用详细日志输出 | false |

## 与Claude集成

要将此服务器与Claude集成，请在`claude_desktop_config.json`文件中添加以下配置:

```json
{
  "mcpServers": {
    "db": {
      "command": "/path/to/start.sh",
      "args": [
        "--db-host", "localhost",
        "--db-user", "root",
        "--db-password", "password",
        "--db-name", "mydatabase"
      ],
      "env": {
        "DB_TYPE": "mysql"
      }
    }
  }
}
```

## Docker Compose部署

对于更完整的部署环境，可以使用Docker Compose:

```yaml
version: '3'

services:
  db-mcp-server:
    build: .
    ports:
      - "8088:8088"
    environment:
      - DB_TYPE=mysql
      - DB_HOST=mysql
      - DB_PORT=3306
      - DB_USER=root
      - DB_PASSWORD=rootpassword
      - DB_NAME=testdb
    depends_on:
      - mysql

  mysql:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=rootpassword
      - MYSQL_DATABASE=testdb
    volumes:
      - mysql-data:/var/lib/mysql

volumes:
  mysql-data:
```

## 安全注意事项

- 服务器默认监听所有网络接口 (0.0.0.0)，生产环境中建议只监听localhost或使用反向代理
- 敏感信息(如密码、个人身份信息)会自动进行掩码处理，但建议在生产环境中使用只有读取权限的数据库用户
- 仅支持只读查询，禁止修改数据库操作

## 许可

[MIT License](LICENSE)

## 贡献

欢迎提交问题和拉取请求! 