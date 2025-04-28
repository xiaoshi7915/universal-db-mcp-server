FROM python:3.11-slim

WORKDIR /app

# 安装必要的依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY fast_db_server.py .
COPY start.sh .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置默认环境变量
ENV HOST=0.0.0.0
ENV PORT=8088
ENV DB_TYPE=mysql
ENV DB_HOST=localhost
ENV DB_PORT=3306
ENV DB_USER=root
ENV DB_PASSWORD=rootpassword
ENV DB_NAME=testdb

# 给启动脚本添加执行权限
RUN chmod +x start.sh

# 暴露服务端口
EXPOSE 8088

# 定义启动命令
CMD ["./start.sh"]