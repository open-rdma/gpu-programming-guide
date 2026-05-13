# CUDA在线评测系统 — 架构设计

## 概述

为GPU编程课程配备的在线代码提交与性能评测系统，部署在配备NVIDIA RTX 3060的服务器上，支持多用户提交CUDA代码、自动化编译、NCU性能分析，并返回分析结果。

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    HTTP API Server                    │
│                   (FastAPI + Jinja2)                  │
├──────────────────────────────────────────────────────┤
│  认证模块    │  速率限制   │  队列管理  │  管理后台  │
├──────────────────────────────────────────────────────┤
│              安全检测 → nvcc编译 → ncu评测           │
├──────────────────────────────────────────────────────┤
│              SQLite (用户/任务/结果)                  │
└──────────────────────────────────────────────────────┘
```

## 核心模块

### 1. 认证模块 (`auth.py`)
- 基于API Key的鉴权机制
- 管理员账号拥有独立的管理后台访问权限
- 用户注册需要管理员审批（或由管理员直接创建）
- JWT token用于会话管理，API Key用于程序化提交

### 2. 速率限制 (`rate_limiter.py`)
- 令牌桶算法
- 默认：每用户每小时5次提交（管理员可配置）
- 每次提交消耗一个令牌，令牌按固定速率恢复

### 3. 队列管理 (`queue_manager.py`)
- 内存队列 + SQLite持久化
- 先进先出（FIFO）
- 用户可查看队列中自己的任务状态和位置
- 任务状态：等待中 → 编译中 → 评测中 → 已完成 / 失败

### 4. 安全检测 (`security.py`)
- 禁止系统调用：`system()`, `popen()`, `exec()`, `fork()`, `spawn()`
- 禁止文件系统危险操作：删除/修改系统文件
- 禁止网络调用：`socket`, `curl`, `wget`
- 白名单允许：CUDA kernel函数、标准C++库、cuda_runtime API
- 代码长度限制：最大64KB

### 5. 编译模块 (`compiler.py`)
- 调用nvcc进行编译
- 设置编译超时（默认60秒）
- 捕获编译错误并返回给用户
- 支持编译选项配置

### 6. 评测模块 (`profiler.py`)
- 调用ncu (NVIDIA Compute Utility) 进行性能分析
- 设置执行超时（默认300秒）
- 生成NCU原始报告文件（.ncu-rep）
- 返回关键性能指标摘要

### 7. 管理后台 (`admin.py`)
- 用户管理：创建/禁用/删除用户，调整速率限制
- 队列监控：查看所有任务状态
- 系统配置：调整编译参数、速率限制默认值

## API设计

### 公开接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/login` | 用户登录，获取token |
| POST | `/api/submit` | 提交代码 |
| GET | `/api/queue` | 查看排队状态 |
| GET | `/api/result/{job_id}` | 获取评测结果 |
| GET | `/api/download/{job_id}` | 下载NCU报告文件 |

### 管理后台接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/POST | `/admin/users` | 用户管理 |
| GET | `/admin/queue` | 查看全部队列 |
| DELETE | `/admin/queue/{job_id}` | 取消任务 |
| GET/POST | `/admin/config` | 系统配置 |

### Web页面

| 路径 | 说明 |
|------|------|
| `/` | 首页/登录页 |
| `/dashboard` | 用户控制台 |
| `/submit` | 代码提交页面 |
| `/queue` | 队列状态查看 |
| `/result/{job_id}` | 评测结果详情 |
| `/admin/` | 管理后台首页 |

## 数据模型

### 用户表 (users)
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| username | TEXT | 用户名，唯一 |
| password_hash | TEXT | 密码哈希 |
| api_key | TEXT | API密钥，唯一 |
| is_admin | BOOLEAN | 是否管理员 |
| is_active | BOOLEAN | 是否启用 |
| rate_limit | INTEGER | 每小时间隔内最大提交数 |
| created_at | DATETIME | 创建时间 |

### 任务表 (jobs)
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| user_id | INTEGER | 外键 |
| status | TEXT | waiting/compiling/profiling/completed/failed |
| source_code | TEXT | 提交的代码 |
| compile_output | TEXT | 编译输出 |
| profile_summary | TEXT | 评测摘要 |
| ncu_report_path | TEXT | NCU报告文件路径 |
| queue_position | INTEGER | 队列位置 |
| error_message | TEXT | 错误信息 |
| created_at | DATETIME | 提交时间 |
| completed_at | DATETIME | 完成时间 |

### 配置表 (config)
| 字段 | 类型 | 说明 |
|------|------|------|
| key | TEXT | 配置键 |
| value | TEXT | 配置值 |

## 部署方案

### Docker容器化
- 基础镜像：`nvidia/cuda:12.0-devel-ubuntu22.04`
- Python 3.10 + FastAPI + uvicorn
- 内置nvcc和ncu工具
- 挂载卷：uploads目录（存储提交代码和NCU报告）
- 端口映射：8080
- GPU访问：`--gpus all`

### Dockerfile结构
```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY app/ ./app/
COPY static/ ./static/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 启动命令
```bash
docker build -t cuda-eval-system .
docker run --gpus all -p 8080:8080 -v $(pwd)/uploads:/app/uploads cuda-eval-system
```
