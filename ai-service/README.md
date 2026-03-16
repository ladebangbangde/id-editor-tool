# ai-id-photo-service

面向主业务后端（server）的下游 AI 图像处理微服务，专注证件照生成：检测、抠图、换底、裁剪、增强、预览图、高清图、排版图。

## 1. 项目说明

- 技术栈：Python 3.10+、FastAPI、OpenCV、Pillow、NumPy、rembg、loguru。
- 服务定位：**仅负责图像处理**，不负责用户/订单/支付/权限/历史记录。
- 与主后端协作方式：server 通过 HTTP 调用本服务接口，并将返回字段直接入库到 `image_results`（字段已对齐）。

## 2. 与主业务后端的对接方式

1. 主后端创建业务任务并准备原图路径（如 `uploads/original/img_001.jpg`）。
2. 主后端调用 `POST /ai/generate-id-photo`。
3. 本服务返回 `previewUrl / hdUrl / printUrl / widthMm / heightMm / pixelWidth / pixelHeight / qualityStatus`。
4. 主后端将返回数据落库，并继续订单流转。

> 注意：`printUrl` 允许为空（未请求排版时）。

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

## 4. 本地运行方式

```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. 示例请求

### 5.1 检测

```bash
curl -X POST 'http://127.0.0.1:8000/ai/detect' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId":"img_001",
    "originalImagePath":"uploads/original/img_001.jpg"
  }'
```

### 5.2 生成证件照

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

### 5.3 生成排版图

```bash
curl -X POST 'http://127.0.0.1:8000/ai/generate-print-layout' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId":"img_001",
    "hdImagePath":"uploads/hd/img_001_hd.jpg",
    "layoutType":"six"
  }'
```

## 6. 输出目录说明

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

## 7. 字段对齐约定（与 server）

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

## 8. 第一版模块说明（mock 与真实）

- detect：**基础真实实现 + 规则**（OpenCV Haar 人脸 + 拉普拉斯模糊评分）
- quality：**规则版**（可扩展为模型评分）
- enhance：**轻量真实实现**（锐化 + 亮度 + 对比）
- segment：**真实实现**（rembg 抠图）
- crop/background/preview/print：**真实实现**

## 9. 目录结构

```text
ai-service/
  main.py
  requirements.txt
  .env.example
  README.md
  uploads/
    preview/
    hd/
    print/
    temp/
  api/
    detect_api.py
    generate_api.py
    print_api.py
    health_api.py
  services/
    detect_service.py
    segment_service.py
    background_service.py
    crop_service.py
    enhance_service.py
    print_service.py
    quality_service.py
  pipeline/
    generate_id_photo.py
    generate_print_layout.py
    build_preview.py
  models/
    request_models.py
    response_models.py
  utils/
    file_utils.py
    image_utils.py
    response_utils.py
    logger.py
    config.py
  constants/
    photo_sizes.py
    colors.py
    status.py
```
