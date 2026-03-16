# ai-id-photo-service

面向主业务后端（server）的下游 AI 图像处理微服务，专注证件照生成：检测、抠图、换底、裁剪、增强、预览图、高清图、排版图。

## 1. 项目说明

- 技术栈：Python 3.10+、FastAPI、OpenCV、Pillow、NumPy、rembg、loguru。
- 服务定位：**仅负责图像处理**，不负责用户/订单/支付/权限/历史记录。
- 调用方式：**仅通过 HTTP API 被 server 调用**，前端不直接调用。
- 部署方式：独立 Python 项目，运行于独立 Docker 容器。
- 存储约定：与 server 共享宿主机 `uploads` 挂载目录。

## 2. 与主业务后端的对接方式

1. 主后端创建业务任务并准备原图路径（例如 `uploads/original/img_001.jpg`）。
2. 主后端调用 `POST /ai/generate-id-photo`。
3. 本服务返回 `previewUrl / hdUrl / printUrl / widthMm / heightMm / pixelWidth / pixelHeight / qualityStatus`。
4. 主后端将返回数据落库，并继续订单流程。

> 注意：`printUrl` 允许为空（未请求排版时）；返回路径统一为 `uploads/...` 风格，便于直接入库。

## 3. API 列表

- `GET /ai/health`
- `POST /ai/detect`
- `POST /ai/generate-id-photo`
- `POST /ai/generate-print-layout`

统一返回：

```json
{
  "success": true,
  "message": "OK",
  "data": {}
}
```

失败：

```json
{
  "success": false,
  "message": "Error message",
  "data": null
}
```

## 4. 本地运行方式（非 Docker）

```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. Docker / docker-compose 运行

### 5.1 准备共享 uploads

确保宿主机有共享目录（需与主 server 使用同一路径）：

```bash
mkdir -p ../uploads/{original,preview,hd,print,temp}
```

### 5.2 启动

```bash
cd ai-service
cp .env.example .env
docker compose up -d --build
```

### 5.3 关键环境变量

- `UPLOAD_BASE_DIR=/data/uploads`：容器内共享挂载目录。
- `UPLOAD_PUBLIC_PREFIX=uploads`：接口返回路径前缀（与 server 入库规则一致）。

## 6. 示例请求

### 6.1 检测

```bash
curl -X POST 'http://127.0.0.1:8000/ai/detect' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId":"img_001",
    "originalImagePath":"uploads/original/img_001.jpg"
  }'
```

### 6.2 生成证件照

```bash
curl -X POST 'http://127.0.0.1:8000/ai/generate-id-photo' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId":"img_001",
    "sourceType":"scene",
    "sceneKey":"passport",
    "customWidthMm":null,
    "customHeightMm":null,
    "backgroundColor":"white",
    "beautyEnabled":false,
    "printLayoutType":"six",
    "originalImagePath":"uploads/original/img_001.jpg"
  }'
```

### 6.3 生成排版图

```bash
curl -X POST 'http://127.0.0.1:8000/ai/generate-print-layout' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId":"img_001",
    "hdImagePath":"uploads/hd/img_001_hd.jpg",
    "layoutType":"six"
  }'
```

## 7. 输出目录说明（共享 uploads）

- `uploads/preview/`：预览图（压缩）
- `uploads/hd/`：高清图
- `uploads/print/`：排版图
- `uploads/temp/`：处理中间产物（如抠图透明图）

命名约定：

- `{imageId}_preview.jpg`
- `{imageId}_hd.jpg`
- `{imageId}_print_6.jpg`
- `{imageId}_print_8.jpg`
- `{imageId}_print_12.jpg`

## 8. 字段对齐约定（与 server）

请求字段保持：

- `imageId`
- `sourceType`
- `sceneKey`
- `customWidthMm`
- `customHeightMm`
- `backgroundColor`
- `beautyEnabled`
- `printLayoutType`
- `originalImagePath`

核心返回字段保持：

- `previewUrl`
- `hdUrl`
- `printUrl`
- `backgroundColor`
- `widthMm`
- `heightMm`
- `pixelWidth`
- `pixelHeight`
- `qualityStatus`

## 9. 第一版模块说明（mock 与真实）

- detect：基础真实实现 + 规则（OpenCV Haar 人脸 + 拉普拉斯模糊评分）
- quality：规则版（可扩展为模型评分）
- enhance：轻量真实实现（锐化 + 亮度 + 对比）
- segment：真实实现（rembg）
- crop/background/preview/print：真实实现
- 第一版不使用大模型。

## 10. 目录结构

```text
ai-service/
  main.py
  requirements.txt
  .env.example
  Dockerfile
  docker-compose.yml
  README.md
  uploads/
    preview/
    hd/
    print/
    temp/
  api/
  services/
  pipeline/
  models/
  utils/
  constants/
```
