# ai-id-photo-service

`ai-service` 是 `id-editor-tool` 中负责证件照 AI 处理的微服务。它保留了给未来 `id-editor-server` 使用的**路径传参接口**，同时新增了适合本地单独调试的**文件上传接口**，这样你只准备一张测试图，就可以本地完成检测、证件照生成、排版图生成，并直接通过浏览器查看输出结果。

---

## 1. 项目定位

本服务只负责图像处理，不处理业务账户、订单、权限或历史记录。

核心能力：

- 人脸检测
- 抠图
- 换底色
- 证件照裁剪
- 简单美化增强
- 预览图 / 高清图 / 排版图输出
- 本地静态文件浏览

### 两类接口的定位

#### A. 保留的原始接口（给 `id-editor-server` 调用）

这些接口继续保留，仍然使用 **JSON + 文件路径** 的模式：

- `GET /ai/health`
- `POST /ai/detect`
- `POST /ai/generate-id-photo`
- `POST /ai/generate-print-layout`

> 这些接口**没有删除，也没有被强制改成 multipart**，后续 server 仍可按路径方式调用。

#### B. 新增的上传调试接口（给本地开发/调试使用）

这些接口使用 **multipart/form-data** 上传本地图片：

- `POST /ai/detect-upload`
- `POST /ai/generate-id-photo-upload`
- `POST /ai/generate-print-layout-upload`

> 这些接口是为了本地快速调试新增的，不替代原有 server 接口。

---

## 2. 目录结构

```text
ai-service/
  api/
    detect_api.py
    generate_api.py
    health_api.py
    print_api.py
    upload_debug_api.py
  constants/
  core/
    config.py
    exceptions.py
  models/
  pipeline/
  services/
    storage_service.py
  scripts/
    test_detect_upload.py
    test_generate_upload.py
    test_print_upload.py
  uploads/
    original/
    preview/
    hd/
    print/
    temp/
  .env.example
  Dockerfile
  main.py
  README.md
  requirements.txt
```

---

## 3. 环境要求

推荐环境：

- Python 3.10+（本仓库已用 Python 3.11 Docker 镜像验证）
- Linux / macOS
- 能安装 `opencv-python`、`rembg`、`Pillow`

如在 Linux 本地运行，常见需要系统依赖：

- `libgl1`
- `libglib2.0-0`
- `libgomp1`

---

## 4. 配置说明

服务读取 `.env`，默认可从 `.env.example` 复制。

### 核心配置项

```env
APP_NAME=ai-id-photo-service
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
UPLOAD_ROOT=uploads
ORIGINAL_DIR=original
PREVIEW_DIR=preview
HD_DIR=hd
PRINT_DIR=print
TEMP_DIR=temp
MAX_UPLOAD_MB=15
DEFAULT_BG_COLOR=white
DEFAULT_LAYOUT_TYPE=six
SAVE_INTERMEDIATE=true
PREVIEW_QUALITY=88
HD_QUALITY=95
JPEG_DPI=300
MIN_VALID_FACE_WIDTH=60
MIN_VALID_FACE_HEIGHT=60
MULTI_FACE_MIN_AREA_RATIO=0.25
FACE_BOX_IOU_THRESHOLD=0.35
```

### 配置含义

- `HOST` / `PORT`：服务监听地址
- `UPLOAD_ROOT`：上传与输出根目录
- `ORIGINAL_DIR`：原图目录
- `PREVIEW_DIR`：预览图目录
- `HD_DIR`：高清图目录
- `PRINT_DIR`：排版图目录
- `TEMP_DIR`：处理中间文件目录
- `LOG_LEVEL`：日志级别
- `MAX_UPLOAD_MB`：上传大小上限（MB）
- `DEFAULT_BG_COLOR`：上传接口默认底色
- `DEFAULT_LAYOUT_TYPE`：上传排版接口默认排版类型
- `SAVE_INTERMEDIATE`：是否保留中间文件（如抠图透明 PNG）
- `MIN_VALID_FACE_WIDTH` / `MIN_VALID_FACE_HEIGHT`：有效人脸框的最小宽高阈值
- `MULTI_FACE_MIN_AREA_RATIO`：次级候选框相对主脸面积的最小比例，低于该比例默认忽略
- `FACE_BOX_IOU_THRESHOLD`：候选框去重时使用的 IoU 阈值

> 兼容说明：服务仍兼容旧变量名 `APP_HOST` / `APP_PORT` / `UPLOAD_BASE_DIR`，方便和现有调用方式共存。

---

## 5. 本地启动步骤

```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

- 健康检查：`http://127.0.0.1:8000/ai/health`
- 静态文件目录：`http://127.0.0.1:8000/uploads/...`

---

## 6. Docker 启动步骤

### 构建镜像

```bash
docker build -t id-editor-tool-ai ./ai-service
```

### 运行容器

```bash
docker run --rm -p 8000:8000 -v $(pwd)/uploads:/app/uploads id-editor-tool-ai
```

如果希望用 `ai-service` 目录自己的 `.env`：

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  --env-file ./ai-service/.env.example \
  id-editor-tool-ai
```

---

## 7. 输出目录说明

默认都在 `ai-service/uploads/` 下：

- `uploads/original/`：上传后的原图
- `uploads/preview/`：预览图
- `uploads/hd/`：高清图
- `uploads/print/`：排版图
- `uploads/temp/`：中间文件（如分割透明图）

常见命名：

- `{imageId}_preview.jpg`
- `{imageId}_hd.jpg`
- `{imageId}_print_6.jpg`
- `{imageId}_print_8.jpg`
- `{imageId}_print_12.jpg`

---

## 8. 静态访问说明

服务已把 `uploads/` 挂到静态资源：

- `http://127.0.0.1:8000/uploads/original/...`
- `http://127.0.0.1:8000/uploads/preview/...`
- `http://127.0.0.1:8000/uploads/hd/...`
- `http://127.0.0.1:8000/uploads/print/...`

因此返回中的：

- `previewUrl`
- `hdUrl`
- `printUrl`

都会尽量返回成可直接访问的 URL 路径，例如：

```text
/uploads/preview/demo_preview.jpg
/uploads/hd/demo_hd.jpg
/uploads/print/demo_print_6.jpg
```

浏览器直接打开即可查看结果。

---

## 9. 统一返回格式

### 成功

```json
{
  "success": true,
  "message": "OK",
  "data": {}
}
```

### 失败

```json
{
  "success": false,
  "message": "Error message",
  "errorCode": "PROCESS_FAILED",
  "data": null
}
```

### 当前支持的错误码

- `INVALID_IMAGE`
- `NO_FACE_DETECTED`
- `MULTIPLE_FACES_DETECTED`
- `IMAGE_TOO_BLURRY`
- `FACE_TOO_SMALL`
- `POSE_INVALID`
- `FACE_OCCLUDED`
- `HEAD_CROPPED`
- `IMAGE_TOO_SMALL`
- `FILE_NOT_FOUND`
- `INVALID_ARGUMENT`
- `PROCESS_FAILED`

---

## 10. 原始接口说明（给 server 调用）

这些接口保留原有 JSON 路径传参风格。

### 10.1 `GET /ai/health`

```bash
curl http://127.0.0.1:8000/ai/health
```

### 10.2 `POST /ai/detect`

```bash
curl -X POST 'http://127.0.0.1:8000/ai/detect' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId": "img_001",
    "originalImagePath": "uploads/original/img_001.jpg"
  }'
```

成功返回中的 `data` 会明确区分“检测到人脸”和“是否通过证件照准入校验”：

```json
{
  "success": true,
  "message": "OK",
  "data": {
    "imageId": "img_001",
    "hasFace": true,
    "faceCount": 1,
    "pass": true,
    "reasons": [],
    "message": "照片符合证件照制作要求",
    "primaryFaceBox": {
      "x": 120,
      "y": 80,
      "width": 260,
      "height": 260
    }
  }
}
```

多人图示例：

```json
{
  "success": true,
  "message": "OK",
  "data": {
    "imageId": "img_multi",
    "hasFace": true,
    "faceCount": 2,
    "pass": false,
    "reasons": ["MULTIPLE_FACES_DETECTED"],
    "message": "检测到多张人脸，不符合证件照制作要求",
    "primaryFaceBox": {
      "x": 120,
      "y": 80,
      "width": 220,
      "height": 220
    }
  }
}
```

### 10.3 `POST /ai/generate-id-photo`

```bash
curl -X POST 'http://127.0.0.1:8000/ai/generate-id-photo' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId": "img_001",
    "sourceType": "scene",
    "sceneKey": "passport",
    "backgroundColor": "white",
    "beautyEnabled": false,
    "printLayoutType": "six",
    "originalImagePath": "uploads/original/img_001.jpg"
  }'
```

> 生成前会先执行和 `/ai/detect` 完全一致的准入校验。如果 `pass=false`，接口会直接拒绝继续抠图、换底、裁剪或排版。

### 10.4 `POST /ai/generate-print-layout`

```bash
curl -X POST 'http://127.0.0.1:8000/ai/generate-print-layout' \
  -H 'Content-Type: application/json' \
  -d '{
    "imageId": "img_001",
    "hdImagePath": "uploads/hd/img_001_hd.jpg",
    "layoutType": "six"
  }'
```

---

## 11. 上传接口说明（给本地调试）

### 11.1 `POST /ai/detect-upload`

上传一张原图，服务会自动保存到 `uploads/original/` 后再检测。

```bash
curl -X POST 'http://127.0.0.1:8000/ai/detect-upload' \
  -F 'image=@./tests/demo.jpg' \
  -F 'imageId=demo_detect'
```

`/ai/detect-upload` 与 `/ai/detect` 使用同一套检测与准入规则，返回字段保持一致：`hasFace`、`faceCount`、`pass`、`reasons`、`message`、`primaryFaceBox`。其中 `faceCount` 表示**过滤、去重后的有效独立人脸数量**，不是原始候选框数量。

### 11.2 `POST /ai/generate-id-photo-upload`

上传一张原图，直接生成证件照。

```bash
curl -X POST 'http://127.0.0.1:8000/ai/generate-id-photo-upload' \
  -F 'image=@./tests/demo.jpg' \
  -F 'imageId=demo_generate' \
  -F 'sourceType=scene' \
  -F 'sceneKey=passport' \
  -F 'backgroundColor=white' \
  -F 'beautyEnabled=false' \
  -F 'printLayoutType=six'
```

> 上传生成接口与路径接口同样会先执行准入校验；多人图、无人脸图、模糊图、姿态不合格图不会继续生成。

### 11.3 `POST /ai/generate-print-layout-upload`

上传一张原图，服务会先生成证件照，再输出排版图。

```bash
curl -X POST 'http://127.0.0.1:8000/ai/generate-print-layout-upload' \
  -F 'image=@./tests/demo.jpg' \
  -F 'imageId=demo_layout' \
  -F 'layoutType=six' \
  -F 'sourceType=scene' \
  -F 'sceneKey=passport' \
  -F 'backgroundColor=white' \
  -F 'beautyEnabled=false'
```

> 排版图上传接口会先走证件照生成链路，因此也会复用同一套准入校验并在不合格时直接拒绝。

---

## 12. Python 测试脚本

已提供：

- `scripts/test_detect_upload.py`
- `scripts/test_generate_upload.py`
- `scripts/test_print_upload.py`

### 示例

```bash
cd ai-service
python scripts/test_detect_upload.py ./tests/demo.jpg
python scripts/test_generate_upload.py ./tests/demo.jpg --print-layout-type six
python scripts/test_print_upload.py ./tests/demo.jpg --layout-type eight
```

脚本会直接打印 JSON 响应，成功时可拿到：

- `originalImagePath`
- `originalImageUrl`
- `previewUrl`
- `hdUrl`
- `printUrl`

---

## 13. 处理结果在哪里查看

### 方式一：看接口返回 JSON

返回里会有：

- `previewUrl`
- `hdUrl`
- `printUrl`

### 方式二：浏览器直接打开

例如：

- `http://127.0.0.1:8000/uploads/preview/demo_generate_preview.jpg`
- `http://127.0.0.1:8000/uploads/hd/demo_generate_hd.jpg`
- `http://127.0.0.1:8000/uploads/print/demo_generate_print_6.jpg`

### 方式三：直接看本地目录

- `ai-service/uploads/original/`
- `ai-service/uploads/preview/`
- `ai-service/uploads/hd/`
- `ai-service/uploads/print/`

---

## 14. 如何准备测试图片

### 输入图片要求

建议使用：

- 单人正面照
- 头部完整、不被裁切
- 光线均匀
- 背景尽量简单
- 分辨率不要太低

尽量避免：

- 多人合照
- 侧脸过大
- 遮挡严重
- 头像太小
- 模糊图

### 准入规则说明

- 仅支持**单人正脸照片**。
- 系统会先对候选人脸框做**过滤、去重、主脸优先**后处理，再统计有效独立人脸数量。
- `faceCount` 表示**有效独立人脸数量**，不是检测器直接返回的原始候选框数量。
- 帽子、头发边缘、阴影、耳机、背景高对比纹理等导致的小误检框，会尽量在后处理阶段被过滤掉。
- 过滤后若仍有 2 张及以上有效独立人脸，才会返回 `MULTIPLE_FACES_DETECTED`。
- 无人脸图会返回 `hasFace=false`、`pass=false`，并给出 `NO_FACE_DETECTED`。
- 模糊图可能返回 `IMAGE_TOO_BLURRY`。
- 人脸过小可能返回 `FACE_TOO_SMALL`。
- 侧脸过大、人物偏转明显可能返回 `POSE_INVALID`。
- 遮挡严重可能返回 `FACE_OCCLUDED`。
- 头顶、下巴或左右边缘裁切过重可能返回 `HEAD_CROPPED`。
- `generate-id-photo` / `generate-id-photo-upload` / `generate-print-layout` / `generate-print-layout-upload` 在输入不合格时会**直接拒绝**，不会继续生成错误结果。

> 当前裁剪逻辑已经增强：优先参考检测到的人脸框，尽量保留头顶留白、让脸部占比更接近常见证件照；如果没有检测框，会自动回退到原有中心裁剪策略。

---

## 15. 常见报错排查

### 1) `FILE_NOT_FOUND`

说明路径接口传入的文件不存在，或 `hdImagePath` 指向了错误位置。

### 2) `INVALID_IMAGE`

说明文件存在，但不是可识别图像，或者图像内容损坏。

### 3) `NO_FACE_DETECTED`

说明没有检测到可用人脸。建议换单人正面图再试。

### 4) `MULTIPLE_FACES_DETECTED`

说明图里在**候选框过滤、去重、主脸优先**之后，仍检测到了多张有效独立人脸，不符合证件照制作要求。生成接口会直接拒绝继续处理。

### 5) `IMAGE_TOO_BLURRY`

说明图片过于模糊，无法满足证件照输入要求。建议上传更清晰原图。

### 6) `FACE_TOO_SMALL`

说明人脸在整张图中占比过小，后续裁剪难以保证证件照质量。建议上传人像主体更明显的照片。

### 7) `POSE_INVALID`

说明姿态偏转过大或不符合正脸要求。建议上传单人正脸照片。

### 8) `FACE_OCCLUDED`

说明人脸存在严重遮挡。建议去掉口罩、手部、头发等遮挡后重试。

### 9) `HEAD_CROPPED`

说明头部被裁切过多。建议上传头部完整、四周留白更合理的照片。

### 10) `IMAGE_TOO_SMALL`

说明输出分辨率过低，建议上传更高分辨率原图。

### 11) `PROCESS_FAILED`

通常是抠图、图像处理、依赖环境或系统库问题。优先检查：

- 是否成功安装 `requirements.txt`
- 系统是否有 OpenCV 运行所需依赖
- Docker 中是否正常安装系统库

---

## 16. 推荐本地调试流程

### 只做检测

1. 启动服务
2. 调 `POST /ai/detect-upload`
3. 看 JSON 里的人脸检测结果

### 生成证件照

1. 调 `POST /ai/generate-id-photo-upload`
2. 拿到 `previewUrl` 和 `hdUrl`
3. 浏览器打开这些 URL 查看效果

### 生成排版图

1. 调 `POST /ai/generate-print-layout-upload`
2. 拿到 `printUrl`
3. 浏览器直接打开排版图

这样即使没有 `id-editor-server`，也能独立完成整个调试流程。
