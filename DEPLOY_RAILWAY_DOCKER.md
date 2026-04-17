# Railway + Docker 部署（前端 + 后端）

## 目标

- 后端：FastAPI（`backend.main:app`）
- 前端：Next.js（`frontend/`）
- 两个 Railway Service：`backend` + `frontend`

## 0. 前置条件

- 代码已 push 到 GitHub
- 仓库包含：
  - `backend/Dockerfile`
  - `frontend/Dockerfile`

## 1. 本地用 Docker Compose 先跑通（推荐）

在项目根目录执行：

```bash
docker compose up --build
```

验证：

- 前端：http://localhost:3000
- 后端健康检查：http://localhost:8000/health
- 后端文档：http://localhost:8000/docs

## 2. Railway 创建后端 Service（Dockerfile）

1. Railway → New Project
2. Add Service → GitHub Repo → 选择你的仓库
3. 设置 Service 为 Docker build，并指定 Dockerfile 路径：
   - `backend/Dockerfile`
4. 设置后端环境变量（Service → Variables）：
   - `PORT`：Railway 会自动注入（一般无需手动）
   - `FORECASTPRO_FAST=1`
   - `FORECASTPRO_ARIMA_MAX_SECONDS=8`
   - `FORECASTPRO_ARIMA_MAXITER=25`
5. 部署完成后，记录后端域名，例如：
   - `https://<backend>.up.railway.app`

验证：

- `https://<backend>/health` → `{"ok": true, ...}`
- `https://<backend>/docs` 可打开

## 3. Railway 创建前端 Service（Dockerfile）

1. Add Service → GitHub Repo → 选择同一个仓库
2. Dockerfile 路径：
   - `frontend/Dockerfile`
3. 设置前端环境变量：
   - `API_BASE_URL=https://<backend>.up.railway.app`
   - `NEXT_PUBLIC_API_BASE_URL=https://<backend>.up.railway.app`
   - `NODE_ENV=production`
4. 部署完成后打开前端域名：
   - `https://<frontend>.up.railway.app`

## 4. 常见问题

### 4.1 前端能打开但请求报错 / 无法连接后端

检查前端 Service 的变量是否指向了正确的后端域名（必须是根域名，不带 `/health`、`/docs`）。

### 4.2 后端计算太慢导致前端超时

保持：

- `FORECASTPRO_FAST=1`
- `FORECASTPRO_ARIMA_MAX_SECONDS=8`
- `FORECASTPRO_ARIMA_MAXITER=25`

并在前端少勾选重模型（ARIMA/RF/XGB）。

